#include "cli_common.h"

#include "DepthMapGenerator.h"
#include "MvsSourcePlanner.h"
#include "MvsWorkspaceReplay.h"
#include "ProjectDenseWorkflowConfig.h"
#include "SparseCloudPreprocessor.h"
#include "io/PathIO.h"

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QTimer>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

bool ensureFreshOutputDirectory(const QString &path, QString *errorMessage)
{
    QDir directory(path);
    if (directory.exists())
    {
        const QStringList entries = directory.entryList(
            QDir::AllEntries | QDir::NoDotAndDotDot);
        if (!entries.isEmpty())
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "输出目录非空，MVS replay 拒绝覆盖已有结果：%1")
                                    .arg(QDir::toNativeSeparators(path));
            }
            return false;
        }
        return true;
    }
    if (!QDir().mkpath(path))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建输出目录：%1")
                                .arg(QDir::toNativeSeparators(path));
        }
        return false;
    }
    return true;
}

bool writeReplayReport(const QString &path,
                       const QJsonObject &report,
                       QString *errorMessage)
{
    QSaveFile output(path);
    const QByteArray json = QJsonDocument(report).toJson(QJsonDocument::Indented);
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(json) != json.size() ||
        !output.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入 MVS replay 报告：%1")
                                .arg(QDir::toNativeSeparators(path));
        }
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    QCoreApplication qtApp(argc, argv);
    CLI::App app{"PlaScan 现有 MVS 工作区深度重放工具"};

    std::string inputManifest;
    std::string pairAuditReport;
    std::string sparseCloudPath;
    std::string maskDirectory;
    std::string outputDirectory;
    std::string quality = "highest";
    std::string sceneProfile = "orbital_object";
    std::string depthFilter = "mild";
    std::string device = "cuda";
    int sourceViews = 4;
    int threads = 7;
    int gpuFrameWorkers = 1;
    int cpuFrameWorkers = 0;
    bool saveLevels = false;

    app.add_option("--input-manifest", inputManifest,
                   "现有 mvs_manifest.json，用于读取影像顺序和相机")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--pair-audit", pairAuditReport,
                   "mvs_pair_audit_cli 生成的 JSON 报告")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--sparse-cloud", sparseCloudPath,
                   "用于深度范围与可见性预处理的 SFM 稀疏点云 PLY")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--mask-dir", maskDirectory,
                   "可选项目排除蒙版目录，要求每张影像均有 <stem>_mask.png");
    app.add_option("-o,--output-dir", outputDirectory,
                   "必须为空或不存在的独立 MVS 输出目录")
        ->required();
    app.add_option("--quality", quality, "深度质量：highest/high/medium/low/lowest")
        ->check(CLI::IsMember({"highest", "high", "medium", "low", "lowest"}));
    app.add_option("--scene-profile", sceneProfile,
                   "场景：auto/orbital_object/aerial_terrain")
        ->check(CLI::IsMember({"auto", "orbital_object", "aerial_terrain"}));
    app.add_option("--depth-filter", depthFilter,
                   "过滤：auto/mild/moderate/aggressive")
        ->check(CLI::IsMember({"auto", "mild", "moderate", "aggressive"}));
    app.add_option("--device", device, "设备：cuda/cpu")
        ->check(CLI::IsMember({"cuda", "cpu"}));
    app.add_option("--source-views", sourceViews, "请求源视图数")
        ->check(CLI::Range(1, 16));
    app.add_option("--threads", threads, "CPU 线程预算")
        ->check(CLI::Range(1, 64));
    app.add_option("--gpu-frame-workers", gpuFrameWorkers, "GPU 帧并发数")
        ->check(CLI::Range(0, 2));
    app.add_option("--cpu-frame-workers", cpuFrameWorkers, "CPU 帧并发数")
        ->check(CLI::Range(0, 4));
    app.add_flag("--save-levels", saveLevels, "保存中间深度金字塔层");
    CLI11_PARSE(app, argc, argv);

    const QString manifestPath = QFileInfo(
        QString::fromUtf8(inputManifest.c_str())).absoluteFilePath();
    const QString auditPath = QFileInfo(
        QString::fromUtf8(pairAuditReport.c_str())).absoluteFilePath();
    const QString sparsePath = QFileInfo(
        QString::fromUtf8(sparseCloudPath.c_str())).absoluteFilePath();
    const QString maskDir = maskDirectory.empty()
        ? QString()
        : QFileInfo(QString::fromUtf8(maskDirectory.c_str())).absoluteFilePath();
    const QString outputDir = QFileInfo(
        QString::fromUtf8(outputDirectory.c_str())).absoluteFilePath();

    QString error;
    if (!ensureFreshOutputDirectory(outputDir, &error))
    {
        std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return cli::EXIT_IO_ERR;
    }

    std::vector<xjw::mvs::CameraView> views;
    if (!xjw::mvs::loadMvsReplayViews(
            manifestPath, maskDir, &views, &error))
    {
        std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return cli::EXIT_IO_ERR;
    }

    std::vector<xjw::mvs::MvsSourcePairQuality> pairQualities;
    xjw::mvs::MvsPairAuditSummary pairSummary;
    if (!xjw::mvs::loadMvsPairAuditReport(
            auditPath, &pairQualities, &pairSummary, &error))
    {
        std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return cli::EXIT_IO_ERR;
    }

    std::vector<std::string> currentImages;
    currentImages.reserve(views.size());
    for (const xjw::mvs::CameraView &view : views)
    {
        currentImages.push_back(view.imagePath);
    }
    pairQualities = xjw::mvs::filterMvsSourcePairQualitiesForImages(
        pairQualities, currentImages);
    const int verifiedCurrentPairs = static_cast<int>(std::count_if(
        pairQualities.cbegin(),
        pairQualities.cend(),
        [](const xjw::mvs::MvsSourcePairQuality &quality)
        {
            return quality.verified && quality.hasVerificationStatistics;
        }));
    if (verifiedCurrentPairs <= 0)
    {
        std::fprintf(stderr,
                     "当前影像集合没有通过几何验证的 MVS 源像对，拒绝重放。\n");
        return cli::EXIT_ALGO_ERR;
    }

    xjw::mvs::SparseCloudPreprocessor preprocessor(
        plapoint::ProcessingDevice::CPU);
    xjw::mvs::PreprocessResult preprocessResult;
    std::string preprocessError;
    if (!preprocessor.run(xjw::common::io::toUtf8Path(sparsePath),
                          views,
                          preprocessResult,
                          &preprocessError))
    {
        std::fprintf(stderr,
                     "稀疏点云预处理失败：%s\n",
                     preprocessError.c_str());
        return cli::EXIT_ALGO_ERR;
    }

    const QJsonObject settingsJson{
        {QStringLiteral("qualityProfile"), QString::fromStdString(quality)},
        {QStringLiteral("sceneProfile"), QString::fromStdString(sceneProfile)},
        {QStringLiteral("depthFilterMode"), QString::fromStdString(depthFilter)},
        {QStringLiteral("minViews"), sourceViews},
        {QStringLiteral("threads"), threads},
        {QStringLiteral("gpu_frame_workers"), gpuFrameWorkers},
        {QStringLiteral("cpu_frame_workers"), cpuFrameWorkers},
        {QStringLiteral("cuda"), device == "cuda"},
        {QStringLiteral("saveIntermediatePyramidLevels"), saveLevels},
        {QStringLiteral("pipeline_mode"), true}
    };
    const xjw::gui::project::DenseGenerationSettings settings =
        xjw::gui::project::denseGenerationSettingsFromJson(settingsJson);
    xjw::mvs::DepthGenConfig config =
        xjw::gui::project::buildDepthGenConfig(
            settings, static_cast<int>(views.size()));
    config.runFusion = false;
    config.saveIntermediateDepthMaps = true;
    config.intermediateDir = xjw::common::io::toUtf8Path(outputDir);
    config.inputSignature =
        QStringLiteral("mvs-replay:%1:%2")
            .arg(QFileInfo(manifestPath).lastModified().toMSecsSinceEpoch())
            .arg(QFileInfo(auditPath).lastModified().toMSecsSinceEpoch())
            .toStdString();
    config.sourcePairQualities = std::move(pairQualities);
    config.requireVerifiedSourcePairs = true;

    std::fprintf(stdout,
                 "views=%zu verified_pairs=%d failed_pairs=%d "
                 "missing_stats_pairs=%d requested_sources=%d\n",
                 views.size(),
                 verifiedCurrentPairs,
                 pairSummary.failedPairCount,
                 pairSummary.missingStatisticsPairCount,
                 sourceViews);
    std::fflush(stdout);

    xjw::mvs::DepthMapGenerator generator;
    generator.setViews(views);
    generator.setSparseCloud(preprocessResult.cloud);
    generator.setConfig(config);
    generator.setOutputDir(xjw::common::io::toUtf8Path(outputDir));

    QEventLoop loop;
    bool success = false;
    QString generatorError;
    QJsonArray artifacts;
    int lastProgress = -1;
    QObject::connect(
        &generator,
        &xjw::mvs::DepthMapGenerator::progressChanged,
        &loop,
        [&lastProgress](const QString &stage, float ratio)
        {
            const int progress = std::clamp(
                static_cast<int>(ratio * 100.0f), 0, 100);
            if (progress >= lastProgress + 5 || progress == 100)
            {
                lastProgress = progress;
                std::fprintf(stdout,
                             "progress=%d stage=%s\n",
                             progress,
                             stage.toUtf8().constData());
                std::fflush(stdout);
            }
        });
    QObject::connect(
        &generator,
        &xjw::mvs::DepthMapGenerator::errorOccurred,
        &loop,
        [&generatorError](const QString &message)
        {
            generatorError = message;
            std::fprintf(stderr, "mvs_error=%s\n", message.toUtf8().constData());
        });
    QObject::connect(
        &generator,
        &xjw::mvs::DepthMapGenerator::depthMapArtifactSaved,
        &loop,
        [&artifacts](const QJsonObject &artifact)
        {
            artifacts.append(artifact);
        });
    QObject::connect(
        &generator,
        &xjw::mvs::DepthMapGenerator::finished,
        &loop,
        [&loop, &success](bool ok)
        {
            success = ok;
            loop.quit();
        });
    QTimer::singleShot(0, &generator, &xjw::mvs::DepthMapGenerator::start);
    loop.exec();

    const QJsonObject report{
        {QStringLiteral("schema"), QStringLiteral("plascan_mvs_depth_replay_v1")},
        {QStringLiteral("status"),
         success ? QStringLiteral("ok") : QStringLiteral("failed")},
        {QStringLiteral("input_manifest"), manifestPath},
        {QStringLiteral("pair_audit"), auditPath},
        {QStringLiteral("sparse_cloud"), sparsePath},
        {QStringLiteral("mask_directory"), maskDir},
        {QStringLiteral("output_directory"), outputDir},
        {QStringLiteral("view_count"), static_cast<int>(views.size())},
        {QStringLiteral("audited_pair_count"), pairSummary.auditedPairCount},
        {QStringLiteral("verified_pair_count"), verifiedCurrentPairs},
        {QStringLiteral("failed_pair_count"), pairSummary.failedPairCount},
        {QStringLiteral("missing_statistics_pair_count"),
         pairSummary.missingStatisticsPairCount},
        {QStringLiteral("requested_source_view_count"), sourceViews},
        {QStringLiteral("quality"), QString::fromStdString(quality)},
        {QStringLiteral("scene_profile"), QString::fromStdString(sceneProfile)},
        {QStringLiteral("depth_filter"), QString::fromStdString(depthFilter)},
        {QStringLiteral("depth_artifact_count"), artifacts.size()},
        {QStringLiteral("depth_artifacts"), artifacts},
        {QStringLiteral("error"), generatorError}
    };
    const QString reportPath =
        QDir(outputDir).filePath(QStringLiteral("mvs_replay_report.json"));
    if (!writeReplayReport(reportPath, report, &error))
    {
        std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return cli::EXIT_IO_ERR;
    }

    std::fprintf(stdout,
                 "status=%s artifacts=%d report=%s\n",
                 success ? "ok" : "failed",
                 static_cast<int>(artifacts.size()),
                 QDir::toNativeSeparators(reportPath).toUtf8().constData());
    return success ? cli::EXIT_OK : cli::EXIT_ALGO_ERR;
}
