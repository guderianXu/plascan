#include "cli_common.h"
#include "CliJsonIO.h"
#include "CliPathUtils.h"

#include "DenseCloudRefinementService.h"
#include "io/PathIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QLockFile>
#include <QStringList>
#include <QUuid>

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

QJsonObject passReportToJson(const xjw::mvs::TerrainHeightSpikeFilterReport &report)
{
    return QJsonObject{
        {QStringLiteral("input_points"), static_cast<double>(report.inputPoints)},
        {QStringLiteral("output_points"), static_cast<double>(report.outputPoints)},
        {QStringLiteral("removed_points"), static_cast<double>(report.removedPoints)},
        {QStringLiteral("local_plane_removed_points"),
         static_cast<double>(report.localPlaneRemovedPoints)},
        {QStringLiteral("median_cell_z_range_before"), report.medianCellZRangeBefore},
        {QStringLiteral("p95_cell_z_range_before"), report.p95CellZRangeBefore},
        {QStringLiteral("median_cell_z_range_after"), report.medianCellZRangeAfter},
        {QStringLiteral("p95_cell_z_range_after"), report.p95CellZRangeAfter}
    };
}

struct NormalizedRefinementPaths
{
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path report;
};

NormalizedRefinementPaths normalizeAndValidatePaths(
    const std::string &input,
    const std::string &output,
    const std::string &report)
{
    const auto compare = [](const std::filesystem::path &first,
                            const std::filesystem::path &second,
                            const std::string &firstName,
                            const std::string &secondName) {
        const xjw::common::io::SafePathComparison comparison =
            xjw::common::io::comparePathsSafely(first, second);
        if (!comparison.valid)
        {
            cli::fatal("Cannot safely compare " + firstName + " and "
                           + secondName + " paths: "
                           + comparison.error.message(),
                       cli::EXIT_ARG_ERR);
        }
        if (comparison.equivalent)
        {
            cli::fatal(firstName + " and " + secondName
                           + " paths must be different",
                       cli::EXIT_ARG_ERR);
        }
        return comparison;
    };

    const std::filesystem::path rawInput =
        xjw::common::io::toFilesystemPath(input);
    const std::filesystem::path rawOutput =
        xjw::common::io::toFilesystemPath(output);
    const std::filesystem::path rawReport =
        xjw::common::io::toFilesystemPath(report);
    const auto rejectLinkedOutput = [](const std::filesystem::path &path,
                                       const std::string &name) {
        const QFileInfo info(xjw::common::io::fromFilesystemPath(path));
        if (info.isSymLink() || info.isJunction())
        {
            cli::fatal(name + " path must not be a symbolic link or junction",
                       cli::EXIT_ARG_ERR);
        }
    };
    rejectLinkedOutput(rawOutput, "output");
    rejectLinkedOutput(rawReport, "report");

    const xjw::common::io::SafePathComparison inputOutput =
        compare(rawInput, rawOutput, "input", "output");
    const xjw::common::io::SafePathComparison inputReport =
        compare(rawInput, rawReport, "input", "report");
    const xjw::common::io::SafePathComparison outputReport =
        compare(rawOutput, rawReport, "output", "report");

    if (inputOutput.normalizedFirst != inputReport.normalizedFirst
        || inputOutput.normalizedSecond != outputReport.normalizedFirst
        || inputReport.normalizedSecond != outputReport.normalizedSecond)
    {
        cli::fatal("Path normalization produced inconsistent results",
                   cli::EXIT_ARG_ERR);
    }
    return {inputOutput.normalizedFirst,
            inputOutput.normalizedSecond,
            inputReport.normalizedSecond};
}

std::filesystem::path filesystemPath(const QString &path)
{
    return xjw::common::io::toFilesystemPath(path);
}

QString uniqueSiblingPath(const QString &finalPath, const QString &purpose)
{
    const QFileInfo info(finalPath);
    for (int attempt = 0; attempt < 256; ++attempt)
    {
        const QString candidate = QDir(info.absolutePath()).filePath(
            QStringLiteral(".%1.%2-%3.tmp")
                .arg(info.fileName(),
                     purpose,
                     QUuid::createUuid().toString(QUuid::WithoutBraces)));
        const QFileInfo candidateInfo(candidate);
        if (!candidateInfo.exists() && !candidateInfo.isSymLink())
        {
            return candidate;
        }
    }
    throw std::runtime_error(
        QStringLiteral("无法为事务产物分配唯一同级路径: %1")
            .arg(finalPath).toStdString());
}

struct PublishedArtifact
{
    QString stagedPath;
    QString finalPath;
    QString backupPath;
    bool backupActive = false;
    bool installed = false;
    bool preserveBackup = false;
};

class RefinementArtifactTransaction
{
public:
    RefinementArtifactTransaction(QString stagedOutput,
                                  QString finalOutput,
                                  QString stagedReport,
                                  QString finalReport)
        : _artifacts{{
              {std::move(stagedOutput), std::move(finalOutput), {}, false, false, false},
              {std::move(stagedReport), std::move(finalReport), {}, false, false, false}}}
    {
        for (PublishedArtifact &artifact : _artifacts)
        {
            artifact.backupPath = uniqueSiblingPath(
                artifact.finalPath, QStringLiteral("backup"));
        }
    }

    RefinementArtifactTransaction(const RefinementArtifactTransaction &) = delete;
    RefinementArtifactTransaction &operator=(
        const RefinementArtifactTransaction &) = delete;

    ~RefinementArtifactTransaction()
    {
        if (!_committed)
        {
            rollbackNoThrow(nullptr);
        }
        for (const PublishedArtifact &artifact : _artifacts)
        {
            QFile::remove(artifact.stagedPath);
        }
    }

    bool publish(QString *errorMessage, QStringList *warnings)
    {
        for (const PublishedArtifact &artifact : _artifacts)
        {
            const QFileInfo stagedInfo(artifact.stagedPath);
            const QFileInfo finalInfo(artifact.finalPath);
            if (!stagedInfo.isFile() || stagedInfo.isSymLink())
            {
                setError(errorMessage,
                         QStringLiteral("事务暂存产物无效: %1")
                             .arg(artifact.stagedPath));
                return false;
            }
            if (finalInfo.isSymLink()
                || (finalInfo.exists() && !finalInfo.isFile()))
            {
                setError(errorMessage,
                         QStringLiteral("正式产物路径已存在且不是普通文件: %1")
                             .arg(artifact.finalPath));
                return false;
            }
        }

        for (PublishedArtifact &artifact : _artifacts)
        {
            if (!QFileInfo::exists(artifact.finalPath))
            {
                continue;
            }
            std::error_code error;
            std::filesystem::rename(filesystemPath(artifact.finalPath),
                                    filesystemPath(artifact.backupPath),
                                    error);
            if (error)
            {
                setError(errorMessage,
                         QStringLiteral("无法备份已有正式产物 %1: %2")
                             .arg(artifact.finalPath,
                                  QString::fromStdString(error.message())));
                rollbackNoThrow(errorMessage);
                return false;
            }
            artifact.backupActive = true;
        }

        for (PublishedArtifact &artifact : _artifacts)
        {
            std::error_code error;
            std::filesystem::rename(filesystemPath(artifact.stagedPath),
                                    filesystemPath(artifact.finalPath),
                                    error);
            if (error)
            {
                setError(errorMessage,
                         QStringLiteral("无法发布正式产物 %1: %2")
                             .arg(artifact.finalPath,
                                  QString::fromStdString(error.message())));
                rollbackNoThrow(errorMessage);
                return false;
            }
            artifact.installed = true;
        }

        _committed = true;
        for (PublishedArtifact &artifact : _artifacts)
        {
            if (!artifact.backupActive)
            {
                continue;
            }
            std::error_code error;
            std::filesystem::remove(filesystemPath(artifact.backupPath), error);
            if (error)
            {
                if (warnings)
                {
                    warnings->append(
                        QStringLiteral("新产物已发布，但旧备份清理失败: %1 (%2)")
                            .arg(artifact.backupPath,
                                 QString::fromStdString(error.message())));
                }
                continue;
            }
            artifact.backupActive = false;
        }
        return true;
    }

private:
    static void setError(QString *errorMessage, const QString &message)
    {
        if (errorMessage)
        {
            *errorMessage = message;
        }
    }

    void rollbackNoThrow(QString *errorMessage) noexcept
    {
        if (_rollbackAttempted)
        {
            return;
        }
        _rollbackAttempted = true;
        try
        {
            QStringList recoveryFailures;
            for (auto it = _artifacts.rbegin(); it != _artifacts.rend(); ++it)
            {
                PublishedArtifact &artifact = *it;
                if (artifact.installed)
                {
                    std::error_code error;
                    std::filesystem::remove(
                        filesystemPath(artifact.finalPath), error);
                    if (error)
                    {
                        artifact.preserveBackup = true;
                        recoveryFailures.append(
                            artifact.backupActive
                                ? artifact.backupPath
                                : artifact.finalPath);
                        continue;
                    }
                    artifact.installed = false;
                }
                if (artifact.backupActive && !artifact.preserveBackup)
                {
                    std::error_code error;
                    std::filesystem::rename(
                        filesystemPath(artifact.backupPath),
                        filesystemPath(artifact.finalPath),
                        error);
                    if (error)
                    {
                        artifact.preserveBackup = true;
                        recoveryFailures.append(artifact.backupPath);
                        continue;
                    }
                    artifact.backupActive = false;
                }
            }
            if (errorMessage && !recoveryFailures.isEmpty())
            {
                errorMessage->append(
                    QStringLiteral("；部分产物无法自动恢复，请人工检查: %1")
                        .arg(recoveryFailures.join(QStringLiteral(", "))));
            }
        }
        catch (...)
        {
            // 析构和错误恢复路径不得继续抛出；原正式产物备份不会在这里删除。
        }
    }

    std::array<PublishedArtifact, 2> _artifacts;
    bool _committed = false;
    bool _rollbackAttempted = false;
};

QJsonObject resultToJson(const xjw::mvs::DenseCloudRefinementRequest &request,
                         const xjw::mvs::DenseCloudRefinementResult &result)
{
    QJsonObject localPlane{
        {QStringLiteral("enabled"), request.filterOptions.localPlaneFilterEnabled},
        {QStringLiteral("min_points"), request.filterOptions.localPlaneMinPoints},
        {QStringLiteral("min_residual_threshold"),
         request.filterOptions.localPlaneMinResidualThreshold},
        {QStringLiteral("mad_multiplier"), request.filterOptions.localPlaneMadMultiplier},
        {QStringLiteral("removed_points"),
         static_cast<double>(result.report.localPlaneRemovedPoints)}
    };
    QJsonObject filter = passReportToJson(result.report);
    filter[QStringLiteral("enabled")] = request.filterOptions.enabled;
    filter[QStringLiteral("grid_cells")] = request.filterOptions.gridResolution;
    filter[QStringLiteral("min_cell_points")] = request.filterOptions.minCellPoints;
    filter[QStringLiteral("min_height_threshold")] = request.filterOptions.minHeightThreshold;
    filter[QStringLiteral("mad_multiplier")] = request.filterOptions.madMultiplier;
    filter[QStringLiteral("local_plane_filter")] = localPlane;

    QJsonArray passReports;
    for (std::size_t index = 0; index < result.passReports.size(); ++index)
    {
        QJsonObject pass = passReportToJson(result.passReports[index]);
        pass[QStringLiteral("pass_index")] = static_cast<int>(index + 1);
        passReports.append(pass);
    }

    QJsonArray warnings;
    for (const std::string &warning : result.warnings)
    {
        warnings.append(xjw::cli::fromStdString(warning));
    }

    return QJsonObject{
        {QStringLiteral("status"), QStringLiteral("ok")},
        {QStringLiteral("input"), xjw::cli::fromStdString(request.inputPath)},
        {QStringLiteral("output"), xjw::cli::fromStdString(request.outputPath)},
        {QStringLiteral("mode"), xjw::cli::fromStdString(result.mode)},
        {QStringLiteral("streaming_chunk_mb"), request.streamingChunkMb},
        {QStringLiteral("terrain_filter_passes"), request.filterPasses},
        {QStringLiteral("input_points"), static_cast<double>(result.report.inputPoints)},
        {QStringLiteral("output_points"), static_cast<double>(result.report.outputPoints)},
        {QStringLiteral("terrain_spike_filter"), filter},
        {QStringLiteral("pass_reports"), passReports},
        {QStringLiteral("warnings"), warnings}
    };
}

int executeRefinementTransaction(
    const xjw::mvs::DenseCloudRefinementRequest &request,
    const QString &reportPath,
    xjw::mvs::DenseCloudRefinementResult *result,
    std::string *errorMessage)
{
    const QString outputPath = xjw::cli::fromStdString(request.outputPath);
    const QFileInfo outputInfo(outputPath);
    const QFileInfo reportInfo(reportPath);
    if (!QDir().mkpath(outputInfo.absolutePath())
        || !QDir().mkpath(reportInfo.absolutePath()))
    {
        *errorMessage = "无法创建细化点云或报告输出目录";
        return cli::EXIT_IO_ERR;
    }
    if (outputInfo.isSymLink() || outputInfo.isJunction()
        || reportInfo.isSymLink() || reportInfo.isJunction()
        || (outputInfo.exists() && !outputInfo.isFile())
        || (reportInfo.exists() && !reportInfo.isFile()))
    {
        *errorMessage = "细化点云与报告输出必须是普通文件路径，不能是链接或目录";
        return cli::EXIT_IO_ERR;
    }

    std::array<QString, 2> lockPaths{
        QDir(outputInfo.absolutePath()).filePath(
            QStringLiteral(".%1.refine.lock").arg(outputInfo.fileName())),
        QDir(reportInfo.absolutePath()).filePath(
            QStringLiteral(".%1.refine.lock").arg(reportInfo.fileName()))};
    std::sort(lockPaths.begin(), lockPaths.end(), [](const QString &first,
                                                     const QString &second) {
        return filesystemPath(first) < filesystemPath(second);
    });
    std::vector<std::unique_ptr<QLockFile>> artifactLocks;
    artifactLocks.reserve(lockPaths.size());
    for (const QString &lockPath : lockPaths)
    {
        auto lock = std::make_unique<QLockFile>(lockPath);
        if (!lock->tryLock(0))
        {
            *errorMessage = lock->error() == QLockFile::LockFailedError
                ? "细化点云或报告输出正在被另一个任务写入"
                : "无法创建或访问细化点云或报告输出锁文件";
            return cli::EXIT_IO_ERR;
        }
        artifactLocks.push_back(std::move(lock));
    }

    const QString stagedOutput = uniqueSiblingPath(
        outputPath, QStringLiteral("stage"));
    const QString stagedReport = uniqueSiblingPath(
        reportPath, QStringLiteral("stage"));
    RefinementArtifactTransaction transaction(
        stagedOutput, outputPath, stagedReport, reportPath);

    xjw::mvs::DenseCloudRefinementRequest stagedRequest = request;
    stagedRequest.outputPath = xjw::common::io::toUtf8Path(stagedOutput);
    if (!xjw::mvs::refineDenseCloud(stagedRequest, result, errorMessage))
    {
        return cli::EXIT_ALGO_ERR;
    }

    QString jsonError;
    if (!xjw::cli::writeJsonFile(
            stagedReport, resultToJson(request, *result), &jsonError))
    {
        *errorMessage = jsonError.toStdString();
        return cli::EXIT_IO_ERR;
    }

    QString publishError;
    QStringList warnings;
    if (!transaction.publish(&publishError, &warnings))
    {
        *errorMessage = publishError.toStdString();
        return cli::EXIT_IO_ERR;
    }
    for (const QString &warning : warnings)
    {
        std::cerr << "警告: " << warning.toStdString() << '\n';
    }
    return cli::EXIT_OK;
}

} // namespace

int main(int argc, char **argv)
{
    CLI::App app{"Refine dense point clouds and write a quality report"};

    std::string inputPath;
    std::string outputPath;
    std::string reportPath;
    bool disableTerrainSpikeFilter = false;
    int terrainGridCells = 260;
    int terrainMinCellPoints = 32;
    float terrainMinHeightThreshold = 0.25f;
    float terrainMadMultiplier = 3.0f;
    bool terrainLocalPlaneFilter = true;
    int terrainLocalPlaneMinPoints = 12;
    float terrainLocalPlaneMinResidualThreshold = 0.12f;
    float terrainLocalPlaneMadMultiplier = 4.0f;
    int terrainFilterPasses = 2;
    int streamingChunkMb = 128;

    app.add_option("--input", inputPath, "Input dense cloud PLY path")->required()->check(CLI::ExistingFile);
    app.add_option("--output", outputPath, "Output refined dense cloud PLY path")->required();
    app.add_option("--report-json", reportPath, "Output JSON quality report path")->required();
    app.add_flag("--disable-terrain-spike-filter", disableTerrainSpikeFilter,
                 "Copy finite points without terrain height spike filtering");
    app.add_option("--terrain-grid-cells", terrainGridCells,
                   "XY grid resolution used by the terrain spike filter")->check(CLI::Range(1, 1024));
    app.add_option("--terrain-min-cell-points", terrainMinCellPoints,
                   "Minimum point count for robust local terrain statistics")
        ->check(CLI::Range(1, 1000000));
    app.add_option("--terrain-min-height-threshold", terrainMinHeightThreshold,
                   "Minimum absolute height threshold in local terrain cells")
        ->check(CLI::PositiveNumber);
    app.add_option("--terrain-mad-multiplier", terrainMadMultiplier,
                   "MAD multiplier used by the robust local terrain height gate")
        ->check(CLI::PositiveNumber);
    app.add_flag("--terrain-local-plane-filter,!--disable-terrain-local-plane-filter",
                 terrainLocalPlaneFilter, "Enable the second-pass local plane residual filter");
    app.add_option("--terrain-local-plane-min-points", terrainLocalPlaneMinPoints,
                   "Minimum neighborhood point count for local plane residual filtering")
        ->check(CLI::Range(3, 1000000));
    app.add_option("--terrain-local-plane-min-residual-threshold",
                   terrainLocalPlaneMinResidualThreshold,
                   "Minimum absolute residual threshold for local plane filtering")
        ->check(CLI::PositiveNumber);
    app.add_option("--terrain-local-plane-mad-multiplier", terrainLocalPlaneMadMultiplier,
                   "MAD multiplier used by the local plane residual gate")
        ->check(CLI::PositiveNumber);
    app.add_option("--terrain-filter-passes", terrainFilterPasses,
                   "Number of terrain cleanup passes")
        ->check(CLI::Range(1, 8));
    app.add_option("--streaming-chunk-mb", streamingChunkMb,
                   "Chunk size for binary PLY streaming mode")
        ->check(CLI::Range(1, 2048));

    CLI11_PARSE(app, argc, argv);

    const NormalizedRefinementPaths normalizedPaths =
        normalizeAndValidatePaths(inputPath, outputPath, reportPath);

    xjw::mvs::DenseCloudRefinementRequest request;
    request.inputPath = xjw::common::io::toUtf8Path(normalizedPaths.input);
    request.outputPath = xjw::common::io::toUtf8Path(normalizedPaths.output);
    request.filterPasses = std::clamp(terrainFilterPasses, 1, 8);
    request.streamingChunkMb = std::clamp(streamingChunkMb, 1, 2048);
    request.filterOptions.enabled = !disableTerrainSpikeFilter;
    request.filterOptions.gridResolution = std::clamp(terrainGridCells, 1, 1024);
    request.filterOptions.minCellPoints = std::max(1, terrainMinCellPoints);
    request.filterOptions.minHeightThreshold = terrainMinHeightThreshold;
    request.filterOptions.madMultiplier = terrainMadMultiplier;
    request.filterOptions.localPlaneFilterEnabled = terrainLocalPlaneFilter;
    request.filterOptions.localPlaneMinPoints = std::max(3, terrainLocalPlaneMinPoints);
    request.filterOptions.localPlaneMinResidualThreshold = terrainLocalPlaneMinResidualThreshold;
    request.filterOptions.localPlaneMadMultiplier = terrainLocalPlaneMadMultiplier;

    xjw::mvs::DenseCloudRefinementResult result;
    std::string error;
    int executionCode = cli::EXIT_IO_ERR;
    try
    {
        executionCode = executeRefinementTransaction(
            request,
            xjw::common::io::fromFilesystemPath(normalizedPaths.report),
            &result,
            &error);
    }
    catch (const std::exception &exception)
    {
        error = exception.what();
    }
    if (executionCode != cli::EXIT_OK)
    {
        cli::fatal(error, executionCode);
    }

    std::cout << "dense_cloud_refine_cli(" << result.mode << "): " << result.report.inputPoints
              << " -> " << result.report.outputPoints << " points, removed "
              << result.report.removedPoints << " points\n";
    return cli::EXIT_OK;
}
