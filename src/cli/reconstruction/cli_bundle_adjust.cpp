// =============================================================================
// 文件: cli_bundle_adjust.cpp
// 功能: PlaScan 光束法平差 CLI，区分扫描点云点到面与行星稀疏激光测距
// =============================================================================
#include "cli_common.h"
#include "CliConsole.h"
#include "CliJsonIO.h"
#include "CliPathUtils.h"

#include "project/BaInputBuilder.h"
#include "BundleAdjustSolver.h"
#include "BundleAdjustService.h"
#include "project/ProjectDocumentModel.h"
#include "project/ProjectIO.h"
#include "project/ProjectSession.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QTextStream>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

    constexpr double kMaxReprojectionRmsRegressionPx = 0.25;
    constexpr double kMaxReprojectionRmsRegressionPercent = 10.0;
    constexpr double kMaxOptimizedTrackDropPercent = 5.0;
    constexpr double kQualityGateEpsilon = 1e-9;

    void fatalQt(const QString& message, int code = cli::EXIT_ARG_ERR)
    {
        xjw::cli::printError(message);
        std::exit(code);
    }

    xjw::BABackend parseBaBackendName(const QString& raw)
    {
        const QString value = raw.trimmed().toLower();
        if (value == QLatin1String("auto"))
        {
            return xjw::BABackend::Auto;
        }
        if (value == QLatin1String("legacy_cpu"))
        {
            return xjw::BABackend::LegacyCpu;
        }
        if (value == QLatin1String("plamatrix_cpu"))
        {
            return xjw::BABackend::PlaMatrixCpu;
        }
        if (value == QLatin1String("plamatrix_cuda"))
        {
            return xjw::BABackend::PlaMatrixCuda;
        }
        if (value == QLatin1String("plamatrix_opencl"))
        {
            return xjw::BABackend::PlaMatrixOpenCl;
        }
        fatalQt(QStringLiteral("未知 BA 后端: %1，可选 auto / legacy_cpu / plamatrix_cpu / "
                               "plamatrix_cuda / plamatrix_opencl")
                    .arg(raw));
        return xjw::BABackend::LegacyCpu;
    }

    xjw::lidar::PlanetaryLaserSensorModel parsePlanetaryLaserSensorModel(const QString& raw)
    {
        const QString value = raw.trimmed().toLower();
        if (value == QLatin1String("frame"))
        {
            return xjw::lidar::PlanetaryLaserSensorModel::Frame;
        }
        if (value == QLatin1String("line_scan"))
        {
            return xjw::lidar::PlanetaryLaserSensorModel::LineScan;
        }
        if (value == QLatin1String("unknown"))
        {
            return xjw::lidar::PlanetaryLaserSensorModel::Unknown;
        }
        fatalQt(QStringLiteral("未知行星激光 sensor model: %1，可选 frame / line_scan / unknown").arg(raw));
        return xjw::lidar::PlanetaryLaserSensorModel::Unknown;
    }

    xjw::lidar::PlanetaryLaserRangeType parsePlanetaryLaserRangeType(const QString& raw)
    {
        const QString value = raw.trimmed().toLower();
        if (value == QLatin1String("one_way"))
        {
            return xjw::lidar::PlanetaryLaserRangeType::OneWay;
        }
        if (value == QLatin1String("round_trip"))
        {
            return xjw::lidar::PlanetaryLaserRangeType::RoundTrip;
        }
        if (value == QLatin1String("unknown"))
        {
            return xjw::lidar::PlanetaryLaserRangeType::Unknown;
        }
        fatalQt(QStringLiteral("未知行星激光 range type: %1，可选 one_way / round_trip / unknown").arg(raw));
        return xjw::lidar::PlanetaryLaserRangeType::Unknown;
    }

    QStringList resolveSelectedImages(const QJsonObject& meta, const std::vector<std::string>& tokens)
    {
        QStringList selected;
        if (tokens.empty())
        {
            selected = xjw::common::project::projectImagePaths(meta);
        }
        else
        {
            for (const std::string& rawToken : tokens)
            {
                const QString token = xjw::cli::fromStdString(rawToken).trimmed();
                const QString path = xjw::common::project::resolveProjectImagePathFromToken(token, meta);
                if (path.isEmpty())
                {
                    fatalQt(QStringLiteral("无法在项目中解析影像: %1").arg(token), cli::EXIT_ARG_ERR);
                }
                if (!selected.contains(path))
                {
                    selected.append(path);
                }
            }
        }

        if (selected.size() < 2)
        {
            fatalQt(QStringLiteral("至少需要 2 张影像参与 BA，当前为 %1").arg(selected.size()), cli::EXIT_ARG_ERR);
        }
        return selected;
    }

    QString defaultOutputDir(const QString& projectPath)
    {
        const QString stamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"));
        return QDir(xjw::common::project::ProjectIO::projectBundleAdjustDir(projectPath)).filePath(stamp);
    }

    QJsonObject
    bundleAdjustRecord(const QString& mode, const QString& outputDir, const xjw::gui::BaServiceResult& result)
    {
        QJsonObject record = result.resultJson;
        record[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
        record[QStringLiteral("source")] = QStringLiteral("bundle_adjust_cli");
        record[QStringLiteral("mode")] = mode;
        record[QStringLiteral("output_dir")] = outputDir;
        return record;
    }

    bool directoryHasEntries(const QString& path)
    {
        QDir dir(path);
        return dir.exists() && !dir.entryList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty();
    }

    struct PlanetaryLaserCliOptions
    {
        bool enabled = false;
        QString dataPath;
        QString cameraCoordinateFrame;
        QString cameraSensorFrame;
        xjw::lidar::PlanetaryLaserJsonParseOptions parseOptions;
        QVector<QStringList> imageAliasesByCameraIndex;
        bool confirmUnknownSensorIsFrame = false;
        bool confirmUnknownRangeIsOneWay = false;
        bool allowUnmappedShots = false;
        bool allowUnmappedMeasuredImages = false;
        double rangeWeight = 1.0;
        double rangeHuberDeltaSigma = 3.0;
    };

    QVector<QStringList> parsePlanetaryLaserImageAliases(const std::vector<std::string>& assignments, int cameraCount)
    {
        QVector<QStringList> aliases(cameraCount);
        for (const std::string& rawAssignment : assignments)
        {
            const QString assignment = xjw::cli::fromStdString(rawAssignment).trimmed();
            const int separator = assignment.indexOf(QLatin1Char('='));
            bool indexOk = false;
            const int cameraIndex = separator > 0 ? assignment.left(separator).trimmed().toInt(&indexOk) : -1;
            const QString alias = separator > 0 ? assignment.mid(separator + 1).trimmed() : QString();
            if (!indexOk || cameraIndex < 0 || cameraIndex >= cameraCount || alias.isEmpty())
            {
                fatalQt(QStringLiteral("非法 --laser-range-image-alias '%1'；格式必须为 CAMERA_INDEX=IMAGE_ID，"
                                       "且索引位于 [0, %2)")
                            .arg(assignment)
                            .arg(cameraCount));
            }
            if (!aliases[cameraIndex].contains(alias))
            {
                aliases[cameraIndex].append(alias);
            }
        }
        return aliases;
    }

    void ensureOutputDirAllowed(const QString& outputDir, bool force)
    {
        if (directoryHasEntries(outputDir) && !force)
        {
            fatalQt(QStringLiteral("输出目录已存在且非空，请更换目录或使用 --force: %1").arg(outputDir),
                    cli::EXIT_ARG_ERR);
        }
    }

    xjw::gui::BaServiceOptions makeServiceOptions(const QStringList& selectedImages,
                                                  const QString& outputDir,
                                                  int threads,
                                                  bool dryRun,
                                                  const xjw::BAOptions& baOptions,
                                                  bool enableLaser,
                                                  const QString& laserCloud,
                                                  double laserMaxDistance,
                                                  double laserVoxelSize,
                                                  double laserMaxCurvature,
                                                  int laserMaxSamples,
                                                  bool laserUseMissingNormalsAsHeightPlanes,
                                                  double laserWeight,
                                                  double laserSigma,
                                                  double laserHuberDelta,
                                                  const PlanetaryLaserCliOptions& planetaryLaser,
                                                  bool exportEvalPlot)
    {
        xjw::gui::BaServiceOptions options;
        options.selectedImages = selectedImages;
        options.outputDir = outputDir;
        options.threads = threads;
        options.dryRun = dryRun;
        options.baOpt = baOptions;
        options.enableLaserConstraints = enableLaser;
        options.laserConstraintCloudPath = laserCloud;
        options.laserAssociationMaxDistanceMeters = laserMaxDistance;
        options.laserVoxelSizeMeters = laserVoxelSize;
        options.laserMaxCurvature = laserMaxCurvature;
        options.laserMaxSamples = laserMaxSamples;
        options.laserUseMissingNormalsAsHeightPlanes = laserUseMissingNormalsAsHeightPlanes;
        options.laserWeight = laserWeight;
        options.laserSigmaMeters = laserSigma;
        options.laserHuberDeltaMeters = laserHuberDelta;
        options.enablePlanetaryLaserRangeConstraints = planetaryLaser.enabled;
        options.planetaryLaserDataPath = planetaryLaser.dataPath;
        options.planetaryLaserCameraCoordinateFrame = planetaryLaser.cameraCoordinateFrame;
        options.planetaryLaserCameraSensorFrame = planetaryLaser.cameraSensorFrame;
        options.planetaryLaserParseOptions = planetaryLaser.parseOptions;
        options.planetaryLaserImageAliasesByCameraIndex = planetaryLaser.imageAliasesByCameraIndex;
        options.planetaryLaserConfirmUnknownSensorIsFrame = planetaryLaser.confirmUnknownSensorIsFrame;
        options.planetaryLaserConfirmUnknownRangeIsOneWay = planetaryLaser.confirmUnknownRangeIsOneWay;
        options.planetaryLaserAllowUnmappedShots = planetaryLaser.allowUnmappedShots;
        options.planetaryLaserAllowUnmappedMeasuredImages = planetaryLaser.allowUnmappedMeasuredImages;
        options.planetaryLaserRangeWeight = planetaryLaser.rangeWeight;
        options.planetaryLaserRangeHuberDeltaSigma = planetaryLaser.rangeHuberDeltaSigma;
        options.exportEvalPlot = exportEvalPlot;
        options.exportObservationDetails = false;
        return options;
    }

    QString buildStatusMessage(xjw::core::project::BaInputBuildStatus status,
                               const xjw::core::project::BaInputBuildResult& input)
    {
        switch (status)
        {
        case xjw::core::project::BaInputBuildStatus::Ok:
            return QStringLiteral("OK");
        case xjw::core::project::BaInputBuildStatus::NotEnoughCameras:
            return QStringLiteral("项目中可用相机少于 2 台");
        case xjw::core::project::BaInputBuildStatus::NoTracks:
        default:
        {
            const auto& diagnostics = input.matchDiagnostics;
            QString message = QStringLiteral("没有可用于 BA 的 tracks：相机 %1，匹配记录 %2，存在/可读分片 %3/%4，"
                                             "owner 已解析 %5，几何通过/peer 已解析块 %6/%7，候选对 %8，索引观测 %9，"
                                             "多视轨迹 %10，最小匹配数拒绝 %11。")
                                  .arg(input.cameras.size())
                                  .arg(diagnostics.matchResultRecordCount)
                                  .arg(diagnostics.existingShardCount)
                                  .arg(diagnostics.readableShardCount)
                                  .arg(diagnostics.resolvedOwnerShardCount)
                                  .arg(diagnostics.geometryPassedBlockCount)
                                  .arg(diagnostics.resolvedPeerBlockCount)
                                  .arg(diagnostics.acceptedPairCount)
                                  .arg(input.indexedObservationCount)
                                  .arg(input.multiViewTrackCount)
                                  .arg(diagnostics.rejectedByMinMatchesCount);
            if (!diagnostics.firstShardReadError.isEmpty())
            {
                message += QStringLiteral(" 首个分片错误：%1").arg(diagnostics.firstShardReadError);
            }
            return message;
        }
        }
    }

    xjw::gui::BaServiceResult runOneBa(const std::vector<xjw::FramePinholeCamera>& cameras,
                                       const std::vector<xjw::BATrack>& tracks,
                                       const xjw::core::project::BaInputBuildResult& input,
                                       xjw::gui::BaServiceOptions options)
    {
        std::vector<xjw::BATrack> runTracks = tracks;
        options.imagePathByIndex = input.imagePathByIndex;
        options.beforeCamMeta = input.beforeCamMeta;
        return xjw::gui::BundleAdjustService::run(cameras, runTracks, options);
    }

    double jsonDouble(const QJsonObject& object, const QString& key)
    {
        return object.value(key).toDouble(0.0);
    }

    int jsonInt(const QJsonObject& object, const QString& key)
    {
        return object.value(key).toInt(0);
    }

    double percentIncrease(double before, double after)
    {
        if (!(before > 0.0))
        {
            return after > before ? 100.0 : 0.0;
        }
        return std::max(0.0, (after - before) * 100.0 / before);
    }

    QJsonObject buildQualityGate(const QJsonObject& baseObj, const QJsonObject& laserObj)
    {
        const QJsonObject laserSummary = laserObj.value(QStringLiteral("laser_constraints_summary")).toObject();
        const double baselineAfter = jsonDouble(baseObj, QStringLiteral("mean_rms_after"));
        const double laserAfter = jsonDouble(laserObj, QStringLiteral("mean_rms_after"));
        const double laserBeforeMeters = jsonDouble(laserSummary, QStringLiteral("laser_rms_before_m"));
        const double laserAfterMeters = jsonDouble(laserSummary, QStringLiteral("laser_rms_after_m"));
        const double laserReductionMeters = laserBeforeMeters - laserAfterMeters;
        const int associatedTracks = jsonInt(laserSummary, QStringLiteral("associated_tracks"));
        const int laserConstraintCount = jsonInt(laserSummary, QStringLiteral("laser_constraint_count"));
        const int baselineOptimized = jsonInt(baseObj, QStringLiteral("optimized_count"));
        const int laserOptimized = jsonInt(laserObj, QStringLiteral("optimized_count"));

        const double reprojectionRegressionPx = std::max(0.0, laserAfter - baselineAfter);
        const double reprojectionRegressionPercent = percentIncrease(baselineAfter, laserAfter);
        const int optimizedTrackDrop = std::max(0, baselineOptimized - laserOptimized);
        const double optimizedTrackDropPercent =
            baselineOptimized > 0
                ? static_cast<double>(optimizedTrackDrop) * 100.0 / static_cast<double>(baselineOptimized)
                : 0.0;

        QJsonArray failureCodes;
        QJsonArray failureReasons;
        auto appendFailure = [&failureCodes, &failureReasons](const QString& code, const QString& reason)
        {
            failureCodes.append(code);
            failureReasons.append(reason);
        };

        if (laserConstraintCount <= 0 || associatedTracks <= 0)
        {
            appendFailure(QStringLiteral("no_laser_constraints"),
                          QStringLiteral("LiDAR BA did not keep any active laser constraints."));
        }
        if (laserReductionMeters <= 0.0)
        {
            appendFailure(QStringLiteral("laser_rms_not_reduced"),
                          QStringLiteral("LiDAR point-to-plane RMS did not decrease."));
        }
        if (reprojectionRegressionPx > kMaxReprojectionRmsRegressionPx + kQualityGateEpsilon ||
            reprojectionRegressionPercent > kMaxReprojectionRmsRegressionPercent + kQualityGateEpsilon)
        {
            appendFailure(QStringLiteral("reprojection_rms_regressed"),
                          QStringLiteral("LiDAR BA increased reprojection RMS beyond the acceptance threshold."));
        }
        if (optimizedTrackDropPercent > kMaxOptimizedTrackDropPercent + kQualityGateEpsilon)
        {
            appendFailure(QStringLiteral("optimized_tracks_dropped"),
                          QStringLiteral("LiDAR BA optimized materially fewer tracks than the baseline run."));
        }

        QJsonObject thresholds;
        thresholds[QStringLiteral("max_reprojection_rms_regression_px")] = kMaxReprojectionRmsRegressionPx;
        thresholds[QStringLiteral("max_reprojection_rms_regression_percent")] = kMaxReprojectionRmsRegressionPercent;
        thresholds[QStringLiteral("max_optimized_track_drop_percent")] = kMaxOptimizedTrackDropPercent;

        QJsonObject metrics;
        metrics[QStringLiteral("reprojection_rms_regression_px")] = reprojectionRegressionPx;
        metrics[QStringLiteral("reprojection_rms_regression_percent")] = reprojectionRegressionPercent;
        metrics[QStringLiteral("optimized_track_drop")] = optimizedTrackDrop;
        metrics[QStringLiteral("optimized_track_drop_percent")] = optimizedTrackDropPercent;

        QJsonObject gate;
        gate[QStringLiteral("passed")] = failureCodes.isEmpty();
        gate[QStringLiteral("status")] = failureCodes.isEmpty() ? QStringLiteral("pass") : QStringLiteral("fail");
        gate[QStringLiteral("failure_codes")] = failureCodes;
        gate[QStringLiteral("failure_reasons")] = failureReasons;
        gate[QStringLiteral("thresholds")] = thresholds;
        gate[QStringLiteral("metrics")] = metrics;
        return gate;
    }

    QJsonObject buildCompareJson(const QString& outputDir,
                                 const xjw::gui::BaServiceResult& baseline,
                                 const xjw::gui::BaServiceResult& laser)
    {
        const QJsonObject baseObj = baseline.resultJson;
        const QJsonObject laserObj = laser.resultJson;

        QJsonObject compare;
        compare[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
        compare[QStringLiteral("output_dir")] = outputDir;
        compare[QStringLiteral("baseline_run_json")] =
            baseObj.value(QStringLiteral("files")).toObject().value(QStringLiteral("run_json")).toString();
        compare[QStringLiteral("laser_run_json")] =
            laserObj.value(QStringLiteral("files")).toObject().value(QStringLiteral("run_json")).toString();
        compare[QStringLiteral("baseline_mean_rms_before")] = jsonDouble(baseObj, QStringLiteral("mean_rms_before"));
        compare[QStringLiteral("baseline_mean_rms_after")] = jsonDouble(baseObj, QStringLiteral("mean_rms_after"));
        compare[QStringLiteral("laser_mean_rms_before")] = jsonDouble(laserObj, QStringLiteral("mean_rms_before"));
        compare[QStringLiteral("laser_mean_rms_after")] = jsonDouble(laserObj, QStringLiteral("mean_rms_after"));
        compare[QStringLiteral("baseline_improvement")] = jsonDouble(baseObj, QStringLiteral("mean_rms_before")) -
                                                          jsonDouble(baseObj, QStringLiteral("mean_rms_after"));
        compare[QStringLiteral("laser_improvement")] = jsonDouble(laserObj, QStringLiteral("mean_rms_before")) -
                                                       jsonDouble(laserObj, QStringLiteral("mean_rms_after"));
        compare[QStringLiteral("laser_after_minus_baseline_after")] =
            jsonDouble(laserObj, QStringLiteral("mean_rms_after")) -
            jsonDouble(baseObj, QStringLiteral("mean_rms_after"));
        compare[QStringLiteral("laser_constraints_summary")] =
            laserObj.value(QStringLiteral("laser_constraints_summary")).toObject();
        compare[QStringLiteral("quality_gate")] = buildQualityGate(baseObj, laserObj);
        return compare;
    }

    void printRunSummary(const QString& label, const xjw::gui::BaServiceResult& result)
    {
        const QJsonObject obj = result.resultJson;
        const QString usedBackend = obj.value(QStringLiteral("ba_used_backend")).toString(QStringLiteral("unknown"));
        const bool usedGpu = obj.value(QStringLiteral("ba_used_gpu")).toBool(false);
        const bool backendFallback = obj.value(QStringLiteral("ba_backend_fallback")).toBool(false);
        const bool qualityRejected = obj.value(QStringLiteral("ba_quality_gate_rejected")).toBool(false);
        const int observationCount = obj.value(QStringLiteral("ba_observation_count")).toInt(0);
        const double totalSeconds = obj.value(QStringLiteral("ba_total_seconds")).toDouble(0.0);
        const double validTrackRatio = obj.value(QStringLiteral("ba_valid_track_ratio")).toDouble(0.0);
        const QString solveStatus = obj.value(QStringLiteral("ba_solve_status")).toString(QStringLiteral("unknown"));
        const QString backendReason = obj.value(QStringLiteral("ba_backend_selection_reason")).toString();
        const QString qualityMessage = obj.value(QStringLiteral("ba_quality_gate_message")).toString();
        xjw::cli::printUtf8(stdout,
                            QStringLiteral("%1: tracks=%2 optimized=%3 observations=%4 rms_before=%5 rms_after=%6 "
                                           "backend=%7 gpu=%8 fallback=%9 valid_ratio=%10 "
                                           "quality_rejected=%11 status=%12 total_s=%13")
                                .arg(label)
                                .arg(obj.value(QStringLiteral("track_count")).toInt())
                                .arg(obj.value(QStringLiteral("optimized_count")).toInt())
                                .arg(observationCount)
                                .arg(obj.value(QStringLiteral("mean_rms_before")).toDouble(), 0, 'f', 6)
                                .arg(obj.value(QStringLiteral("mean_rms_after")).toDouble(), 0, 'f', 6)
                                .arg(usedBackend,
                                     usedGpu ? QStringLiteral("true") : QStringLiteral("false"),
                                     backendFallback ? QStringLiteral("true") : QStringLiteral("false"))
                                .arg(validTrackRatio, 0, 'f', 4)
                                .arg(qualityRejected ? QStringLiteral("true") : QStringLiteral("false"))
                                .arg(solveStatus)
                                .arg(totalSeconds, 0, 'f', 3));
        if (!backendReason.isEmpty())
        {
            xjw::cli::printUtf8(stdout, QStringLiteral("%1 BA 后端说明: %2").arg(label, backendReason));
        }
        if (!qualityMessage.isEmpty())
        {
            xjw::cli::printUtf8(stdout, QStringLiteral("%1 BA 质量门控: %2").arg(label, qualityMessage));
        }
    }

} // namespace

int main(int argc, char* argv[])
{
    QCoreApplication qtApp(argc, argv);
    Q_UNUSED(qtApp);

    CLI::App app{"PlaScan 光束法平差 CLI — 支持扫描 LiDAR 点到面与行星稀疏 laser-range shot"};
    cli::configureApp(app);

    std::string projectPathRaw;
    std::string chunkIdRaw;
    std::string chunkNameRaw;
    std::string outputDirRaw;
    std::vector<std::string> imageTokens;
    std::string laserCloudRaw;
    std::string planetaryLaserDataRaw;
    std::string planetaryLaserCameraFrameRaw;
    std::string planetaryLaserCameraSensorFrameRaw;
    std::string planetaryLaserIsisTargetRaw;
    std::string planetaryLaserIsisBodyFrameRaw;
    std::string planetaryLaserIsisLaserFrameRaw;
    std::string planetaryLaserIsisSensorModelRaw = "unknown";
    std::string planetaryLaserIsisRangeTypeRaw = "unknown";
    std::vector<double> planetaryLaserIsisLeverArm;
    std::vector<std::string> planetaryLaserImageAliasRaw;
    std::string baBackendRaw = "auto";

    int minMatches = 15;
    int threads = 4;
    int maxIterations = 20;
    int maxPointIterations = 12;
    int maxCameraIterations = 10;
    int chunkSize = 20000;
    int baPlaMatrixDevice = 0;
    int baMinCudaCameras = xjw::BAOptions::kDefaultMinPlaMatrixGpuCameras;
    int baMinCudaObservations = xjw::BAOptions::kDefaultMinPlaMatrixGpuObservations;
    double baMaxInitialTrackRms = 100.0;
    int baMaxDenseSchurCameras = 200;
    double huberDelta = 3.0;
    double damping = 1e-3;
    double finiteDiffEps = 1e-6;
    double stepTolerance = 1e-8;
    double baMaxAcceptedRmsGrowth = 1.25;
    double baMinAcceptedValidTrackRatio = 0.60;
    double baMaxConstraintRmsGrowth = 1.25;
    bool refinePose = true;
    bool dryRun = false;
    bool force = false;
    bool abCompare = false;
    bool exportEvalPlot = false;
    bool failOnQualityGate = false;
    bool laserUseMissingNormalsAsHeightPlanes = false;
    bool planetaryLaserConfirmUnknownSensorIsFrame = false;
    bool planetaryLaserConfirmUnknownRangeIsOneWay = false;
    bool planetaryLaserAllowUnmappedShots = false;
    bool planetaryLaserAllowUnmappedMeasuredImages = false;
    bool baBackendFallback = true;
    bool baEnableQualityGate = true;
    bool baCompareAutoWithLegacy = true;

    double laserMaxDistance = 0.05;
    double laserVoxelSize = 0.0;
    double laserMaxCurvature = 0.2;
    int laserMaxSamples = 500000;
    double laserWeight = 0.0;
    double laserSigma = 0.0025;
    double laserHuberDelta = 0.05;
    double planetaryLaserRangeWeight = 1.0;
    double planetaryLaserRangeHuberDeltaSigma = 3.0;

    app.add_option("project", projectPathRaw, ".plascan 项目文件")->required();
    app.add_option("--chunk-id", chunkIdRaw, "使用指定 UUID 的 Chunk");
    app.add_option("--chunk-name", chunkNameRaw, "使用指定名称的 Chunk");
    app.add_option("-o,--output-dir", outputDirRaw, "输出目录；未指定时写入当前 Chunk 的 bundle_adjust/<timestamp>");
    app.add_option("--image,--images", imageTokens, "限定参与 BA 的影像路径/文件名/去扩展名，可重复或逗号分隔")
        ->delimiter(',');
    app.add_option("--min-matches", minMatches, "构建 BA tracks 时单对匹配最少点数");
    app.add_option("--threads", threads, "线程数记录字段");
    app.add_option("--max-iterations", maxIterations, "BA 全局最大迭代次数");
    app.add_option("--max-point-iterations", maxPointIterations, "单点优化最大迭代次数");
    app.add_option("--max-camera-iterations", maxCameraIterations, "相机位姿优化最大迭代次数");
    app.add_option("--chunk-size", chunkSize, "BA 分块大小");
    app.add_option("--huber-delta", huberDelta, "影像重投影残差 Huber 阈值");
    app.add_option("--damping", damping, "LM 阻尼初值");
    app.add_option("--finite-diff-eps", finiteDiffEps, "有限差分步长");
    app.add_option("--step-tolerance", stepTolerance, "收敛步长阈值");
    app.add_option("--ba-backend",
                   baBackendRaw,
                   "BA 求解后端: auto / legacy_cpu / plamatrix_cpu / plamatrix_cuda / "
                   "plamatrix_opencl");
    app.add_option("--ba-plamatrix-device", baPlaMatrixDevice, "PlaMatrix CUDA/OpenCL Schur PCG 使用的设备索引");
    app.add_option("--ba-min-cuda-cameras", baMinCudaCameras, "Auto 选择 PlaMatrix CUDA/OpenCL 所需的最小相机数");
    app.add_option(
        "--ba-min-cuda-observations", baMinCudaObservations, "Auto 选择 PlaMatrix CUDA/OpenCL 所需的最小观测数");
    app.add_option("--ba-max-initial-track-rms",
                   baMaxInitialTrackRms,
                   "联合 BA 装配前初始 track RMS 粗差上限（像素），0 表示关闭");
    app.add_option(
        "--ba-max-dense-schur-cameras", baMaxDenseSchurCameras, "PlaMatrix CPU 使用 dense Schur 的最大可变相机数");
    app.add_flag("--ba-backend-fallback{true},--no-ba-backend-fallback{false}",
                 baBackendFallback,
                 "请求的 BA 后端不可用时是否允许回退到可用后端");
    app.add_flag("--ba-quality-gate{true},--no-ba-quality-gate{false}",
                 baEnableQualityGate,
                 "Auto BA 是否启用质量门控，候选后端质量异常时回退 legacy_cpu");
    app.add_option("--ba-max-rms-growth", baMaxAcceptedRmsGrowth, "Auto BA 候选后端允许的最大 RMS 增长倍率");
    app.add_option(
        "--ba-max-constraint-rms-growth", baMaxConstraintRmsGrowth, "Auto BA 物方约束允许的最大 RMS 增长倍率");
    app.add_option(
        "--ba-min-valid-track-ratio", baMinAcceptedValidTrackRatio, "Auto BA 候选后端允许的最小有效 track 比例");
    app.add_flag("--ba-compare-legacy{true},--no-ba-compare-legacy{false}",
                 baCompareAutoWithLegacy,
                 "Auto BA 选择加速后端时是否运行 legacy 对照用于质量门控");
    app.add_flag("--refine-pose{true},--no-refine-pose{false}", refinePose, "是否优化相机姿态");
    app.add_flag("--dry-run", dryRun, "仅检查输入并统计 tracks，不执行优化");
    app.add_flag("--force", force, "允许使用非空输出目录");
    app.add_flag("--export-eval-plot", exportEvalPlot, "导出 BA 评估 PNG；无头命令行默认关闭");

    app.add_option("--laser-cloud", laserCloudRaw, "带 normal_x/normal_y/normal_z 的 LiDAR PLY 点云");
    app.add_flag("--laser-missing-normals-as-height-planes",
                 laserUseMissingNormalsAsHeightPlanes,
                 "将无 normal 字段的 LiDAR XYZ PLY 按水平高度面约束使用");
    app.add_option("--laser-max-distance", laserMaxDistance, "track 到 LiDAR 平面最大关联距离（米）");
    app.add_option("--laser-voxel-size", laserVoxelSize, "LiDAR 点云体素降采样尺寸（米），0 表示关闭");
    app.add_option("--laser-max-curvature", laserMaxCurvature, "允许参与约束的最大曲率");
    app.add_option("--laser-max-samples", laserMaxSamples, "LiDAR map 最大采样点数");
    app.add_option("--laser-weight", laserWeight, "LiDAR 统计权重；0 表示按 --laser-sigma 自动取 1/sigma^2");
    app.add_option("--laser-sigma", laserSigma, "LiDAR 点到面标准差（米），用于自动统计权重");
    app.add_option("--laser-huber-delta", laserHuberDelta, "LiDAR 残差 Huber 阈值（米）");
    app.add_option("--laser-range-data",
                   planetaryLaserDataRaw,
                   "行星稀疏激光测距 shot：PlaScan SI JSON v1 或 ISIS LidarData JSON");
    app.add_option("--laser-range-camera-frame",
                   planetaryLaserCameraFrameRaw,
                   "当前 BA 相机/track 所在坐标系；必须与数据 body_fixed_frame 一致");
    app.add_option("--laser-range-camera-sensor-frame",
                   planetaryLaserCameraSensorFrameRaw,
                   "lever arm 所在相机传感器框架；非零杆臂时必须与数据 laser_frame 一致");
    auto* planetaryLaserRangeWeightOption =
        app.add_option("--laser-range-weight", planetaryLaserRangeWeight, "行星激光测距残差全局权重");
    auto* planetaryLaserHuberOption = app.add_option(
        "--laser-range-huber-delta-sigma", planetaryLaserRangeHuberDeltaSigma, "按 range sigma 归一化后的 Huber 阈值");
    app.add_flag("--laser-range-confirm-frame-camera",
                 planetaryLaserConfirmUnknownSensorIsFrame,
                 "数据 sensor_model=unknown 时，显式确认按静态 frame camera 处理");
    app.add_flag("--laser-range-confirm-one-way",
                 planetaryLaserConfirmUnknownRangeIsOneWay,
                 "数据 range_type=unknown 时，显式确认 range 已是单程几何距离");
    app.add_flag("--laser-range-allow-unmapped-shots",
                 planetaryLaserAllowUnmappedShots,
                 "显式允许跳过不属于当前选中影像集的 shot");
    app.add_flag("--laser-range-allow-unmapped-measures",
                 planetaryLaserAllowUnmappedMeasuredImages,
                 "显式允许丢弃未映射到当前工程相机的真实 measured 像点");
    app.add_option("--laser-range-image-alias",
                   planetaryLaserImageAliasRaw,
                   "额外影像别名，格式 CAMERA_INDEX=IMAGE_ID，可重复；"
                   "用于把 ISIS serialNumber 映射到工程相机");
    app.add_option("--laser-range-isis-target", planetaryLaserIsisTargetRaw, "ISIS JSON 缺失的目标天体名称，例如 MOON");
    app.add_option(
        "--laser-range-isis-body-frame", planetaryLaserIsisBodyFrameRaw, "ISIS JSON 缺失的天体固连框架，例如 IAU_MOON");
    app.add_option("--laser-range-isis-laser-frame",
                   planetaryLaserIsisLaserFrameRaw,
                   "ISIS JSON 缺失的 lever arm 表达框架；零杆臂也必须显式提供");
    app.add_option("--laser-range-isis-sensor-model",
                   planetaryLaserIsisSensorModelRaw,
                   "ISIS 数据相机模型: frame / line_scan / unknown");
    app.add_option("--laser-range-isis-range-type",
                   planetaryLaserIsisRangeTypeRaw,
                   "ISIS range 语义: one_way / round_trip / unknown");
    app.add_option("--laser-range-isis-lever-arm",
                   planetaryLaserIsisLeverArm,
                   "ISIS JSON 缺失的相机原点到激光发射中心杆臂，按 laser frame 表达（米，3 个数）")
        ->expected(3);
    app.add_flag("--ab-compare", abCompare, "一次性运行 baseline 与 LiDAR BA，并写 ba_ab_compare.json");
    app.add_flag("--fail-on-quality-gate",
                 failOnQualityGate,
                 "仅在 --ab-compare 下生效；LiDAR BA 质量门禁失败时写出对比 JSON 后返回非零");

    CLI11_PARSE(app, argc, argv);

    const QString projectPath = xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(projectPathRaw));
    if (!QFileInfo::exists(projectPath))
    {
        fatalQt(QStringLiteral("项目文件不存在: %1").arg(projectPath), cli::EXIT_IO_ERR);
    }

    xjw::common::project::ProjectSession projectSession;
    QString projectError;
    if (!projectSession.open(projectPath, &projectError))
    {
        fatalQt(QStringLiteral("无法打开 .plascan Chunk 工程: %1").arg(projectError), cli::EXIT_IO_ERR);
    }
    if (!chunkIdRaw.empty() && !chunkNameRaw.empty())
    {
        fatalQt(QStringLiteral("--chunk-id 与 --chunk-name 不能同时使用"), cli::EXIT_ARG_ERR);
    }
    if (!projectSession.selectChunk(
            xjw::cli::fromStdString(chunkIdRaw), xjw::cli::fromStdString(chunkNameRaw), &projectError))
    {
        fatalQt(QStringLiteral("Chunk 选择失败: %1").arg(projectError), cli::EXIT_IO_ERR);
    }

    const QString outputDir = outputDirRaw.empty() ? defaultOutputDir(projectPath)
                                                   : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(outputDirRaw));
    const QString laserCloud = xjw::cli::fromStdString(laserCloudRaw).trimmed();
    const bool enableLaser = !laserCloud.isEmpty();
    const QString planetaryLaserDataInput = xjw::cli::fromStdString(planetaryLaserDataRaw).trimmed();
    const QString planetaryLaserData =
        planetaryLaserDataInput.isEmpty() ? QString() : xjw::cli::cleanAbsolutePath(planetaryLaserDataInput);
    const bool enablePlanetaryLaser = !planetaryLaserData.isEmpty();

    PlanetaryLaserCliOptions planetaryLaserOptions;
    planetaryLaserOptions.enabled = enablePlanetaryLaser;
    planetaryLaserOptions.dataPath = planetaryLaserData;
    planetaryLaserOptions.cameraCoordinateFrame = xjw::cli::fromStdString(planetaryLaserCameraFrameRaw).trimmed();
    planetaryLaserOptions.cameraSensorFrame = xjw::cli::fromStdString(planetaryLaserCameraSensorFrameRaw).trimmed();
    planetaryLaserOptions.confirmUnknownSensorIsFrame = planetaryLaserConfirmUnknownSensorIsFrame;
    planetaryLaserOptions.confirmUnknownRangeIsOneWay = planetaryLaserConfirmUnknownRangeIsOneWay;
    planetaryLaserOptions.allowUnmappedShots = planetaryLaserAllowUnmappedShots;
    planetaryLaserOptions.allowUnmappedMeasuredImages = planetaryLaserAllowUnmappedMeasuredImages;
    planetaryLaserOptions.rangeWeight = planetaryLaserRangeWeight;
    planetaryLaserOptions.rangeHuberDeltaSigma = planetaryLaserRangeHuberDeltaSigma;

    const QString isisTarget = xjw::cli::fromStdString(planetaryLaserIsisTargetRaw).trimmed();
    const QString isisBodyFrame = xjw::cli::fromStdString(planetaryLaserIsisBodyFrameRaw).trimmed();
    const QString isisLaserFrame = xjw::cli::fromStdString(planetaryLaserIsisLaserFrameRaw).trimmed();
    const QString isisSensorModel = xjw::cli::fromStdString(planetaryLaserIsisSensorModelRaw).trimmed().toLower();
    const QString isisRangeType = xjw::cli::fromStdString(planetaryLaserIsisRangeTypeRaw).trimmed().toLower();
    const bool hasAnyIsisContext = !isisTarget.isEmpty() || !isisBodyFrame.isEmpty() || !isisLaserFrame.isEmpty() ||
                                   !planetaryLaserIsisLeverArm.empty() || isisSensorModel != QLatin1String("unknown") ||
                                   isisRangeType != QLatin1String("unknown");
    if (hasAnyIsisContext && (isisTarget.isEmpty() || isisBodyFrame.isEmpty() || isisLaserFrame.isEmpty()))
    {
        fatalQt(QStringLiteral("ISIS LidarData JSON 上下文必须同时指定 --laser-range-isis-target、"
                               "--laser-range-isis-body-frame 和 --laser-range-isis-laser-frame"));
    }
    if (hasAnyIsisContext && planetaryLaserIsisLeverArm.size() != 3)
    {
        fatalQt(QStringLiteral("ISIS LidarData JSON 不记录杆臂；必须显式提供 "
                               "--laser-range-isis-lever-arm x y z，确认零杆臂时请写 0 0 0"));
    }
    if (hasAnyIsisContext)
    {
        xjw::lidar::PlanetaryLaserIsisContext context;
        context.reference.targetName = isisTarget.toStdString();
        context.reference.bodyFixedFrame = isisBodyFrame.toStdString();
        context.reference.laserFrame = isisLaserFrame.toStdString();
        context.reference.timeSystem = xjw::lidar::PlanetaryLaserTimeSystem::TdbEtSeconds;
        context.reference.latitudeType = "planetocentric";
        context.reference.longitudeDirection = "positive_east";
        context.sensorModel = parsePlanetaryLaserSensorModel(isisSensorModel);
        context.rangeType = parsePlanetaryLaserRangeType(isisRangeType);
        context.leverArmSensorMeters = {{
            planetaryLaserIsisLeverArm[0],
            planetaryLaserIsisLeverArm[1],
            planetaryLaserIsisLeverArm[2],
        }};
        planetaryLaserOptions.parseOptions.isisContext = context;
    }

    const bool hasPlanetaryLaserOnlyArguments =
        hasAnyIsisContext || !planetaryLaserOptions.cameraCoordinateFrame.isEmpty() ||
        !planetaryLaserOptions.cameraSensorFrame.isEmpty() || planetaryLaserConfirmUnknownSensorIsFrame ||
        planetaryLaserConfirmUnknownRangeIsOneWay || planetaryLaserAllowUnmappedShots ||
        planetaryLaserAllowUnmappedMeasuredImages || planetaryLaserRangeWeightOption->count() > 0 ||
        planetaryLaserHuberOption->count() > 0 || !planetaryLaserImageAliasRaw.empty();
    if (!enablePlanetaryLaser && hasPlanetaryLaserOnlyArguments)
    {
        fatalQt(QStringLiteral("--laser-range-* 参数要求同时指定 --laser-range-data"));
    }
    if (abCompare && enablePlanetaryLaser)
    {
        fatalQt(QStringLiteral("当前 --ab-compare 只支持 --laser-cloud；行星测距请单独运行并查看"
                               " planetary_laser_range_summary"));
    }
    if (abCompare && !enableLaser)
    {
        fatalQt(QStringLiteral("--ab-compare 需要同时指定 --laser-cloud"), cli::EXIT_ARG_ERR);
    }
    if (enableLaser && !QFileInfo::exists(laserCloud))
    {
        fatalQt(QStringLiteral("LiDAR 点云不存在: %1").arg(laserCloud), cli::EXIT_IO_ERR);
    }
    if (enableLaser && laserWeight <= 0.0 && !(laserSigma > 0.0))
    {
        fatalQt(QStringLiteral("自动 LiDAR 权重要求 --laser-sigma > 0"), cli::EXIT_ARG_ERR);
    }
    if (enableLaser && enablePlanetaryLaser)
    {
        fatalQt(QStringLiteral("--laser-cloud（扫描点云点到面）与 --laser-range-data（行星稀疏测距）"
                               "是不同观测模型，当前不能在一次运行中同时指定"));
    }
    if (enablePlanetaryLaser && !QFileInfo::exists(planetaryLaserData))
    {
        fatalQt(QStringLiteral("行星激光测距 JSON 不存在: %1").arg(planetaryLaserData), cli::EXIT_IO_ERR);
    }
    if (enablePlanetaryLaser && planetaryLaserOptions.cameraCoordinateFrame.isEmpty())
    {
        fatalQt(QStringLiteral("--laser-range-data 要求显式指定 --laser-range-camera-frame，"
                               "确认当前相机与激光落点位于同一坐标系"));
    }
    if (enablePlanetaryLaser &&
        (!std::isfinite(planetaryLaserRangeWeight) || planetaryLaserRangeWeight <= 0.0 ||
         !std::isfinite(planetaryLaserRangeHuberDeltaSigma) || planetaryLaserRangeHuberDeltaSigma < 0.0))
    {
        fatalQt(QStringLiteral("--laser-range-weight 必须为正，--laser-range-huber-delta-sigma 必须有限且非负"));
    }

    const QJsonObject meta = projectSession.mergedMetadata();
    const QStringList selectedImages = resolveSelectedImages(meta, imageTokens);

    ensureOutputDirAllowed(outputDir, force);

    xjw::core::project::BaInputBuildResult baInput;
    const xjw::core::project::BaInputBuildStatus buildStatus =
        xjw::core::project::buildBaInputFromMeta(meta, selectedImages, minMatches, &baInput);
    if (buildStatus != xjw::core::project::BaInputBuildStatus::Ok)
    {
        fatalQt(buildStatusMessage(buildStatus, baInput), cli::EXIT_ALGO_ERR);
    }
    if (!planetaryLaserImageAliasRaw.empty())
    {
        planetaryLaserOptions.imageAliasesByCameraIndex =
            parsePlanetaryLaserImageAliases(planetaryLaserImageAliasRaw, static_cast<int>(baInput.cameras.size()));
    }
    if (enablePlanetaryLaser)
    {
        QString aliasError;
        if (!xjw::gui::mergePlanetaryLaserProjectImageAliases(
                meta, baInput.imagePathByIndex, &planetaryLaserOptions.imageAliasesByCameraIndex, &aliasError))
        {
            fatalQt(aliasError, cli::EXIT_ARG_ERR);
        }
    }

    xjw::BAOptions baOptions;
    baOptions.maxIterations = maxIterations;
    baOptions.maxPointIterations = maxPointIterations;
    baOptions.maxCameraIterations = maxCameraIterations;
    Q_UNUSED(chunkSize);
    baOptions.huberDelta = huberDelta;
    baOptions.damping = damping;
    baOptions.finiteDiffEps = finiteDiffEps;
    baOptions.stepTolerance = stepTolerance;
    baOptions.refineCameraPose = refinePose;
    baOptions.numThreads = threads;
    baOptions.backend = parseBaBackendName(xjw::cli::fromStdString(baBackendRaw));
    baOptions.plaMatrixDevice = std::max(0, baPlaMatrixDevice);
    baOptions.minPlaMatrixGpuCameras = std::max(1, baMinCudaCameras);
    baOptions.minPlaMatrixGpuObservations = std::max(1, baMinCudaObservations);
    baOptions.maxInitialTrackRms = std::max(0.0, baMaxInitialTrackRms);
    baOptions.maxDenseSchurCameras = std::max(1, baMaxDenseSchurCameras);
    baOptions.allowBackendFallback = baBackendFallback;
    baOptions.enableBackendQualityGate = baEnableQualityGate;
    baOptions.maxAcceptedRmsGrowth = std::max(0.0, baMaxAcceptedRmsGrowth);
    baOptions.minAcceptedValidTrackRatio = std::max(0.0, baMinAcceptedValidTrackRatio);
    baOptions.maxAcceptedConstraintRmsGrowth = std::max(1.0, baMaxConstraintRmsGrowth);
    baOptions.compareAutoBackendWithLegacy = baCompareAutoWithLegacy;
    if (baInput.surveyControlTrackCount > 0)
    {
        baOptions.enableControlPointConstraints = true;
    }
    if (!baInput.scaleBarConstraints.empty())
    {
        baOptions.enableScaleBarConstraints = true;
        baOptions.scaleBarConstraints = baInput.scaleBarConstraints;
    }

    const QString ba_input_summary =
        QStringLiteral("BA 输入: cameras=%1 tracks=%2 selected_images=%3 indexed_observations=%4 ") +
        QStringLiteral("multiview_tracks=%5 survey_control_tracks=%6 scale_bars=%7");
    xjw::cli::printUtf8(stdout,
                        ba_input_summary.arg(static_cast<int>(baInput.cameras.size()))
                            .arg(static_cast<int>(baInput.tracks.size()))
                            .arg(selectedImages.size())
                            .arg(baInput.indexedObservationCount)
                            .arg(baInput.multiViewTrackCount)
                            .arg(baInput.surveyControlTrackCount)
                            .arg(baInput.surveyScaleBarConstraintCount));

    if (abCompare)
    {
        const QString baselineDir = QDir(outputDir).filePath(QStringLiteral("baseline"));
        const QString laserDir = QDir(outputDir).filePath(QStringLiteral("laser"));
        ensureOutputDirAllowed(baselineDir, force);
        ensureOutputDirAllowed(laserDir, force);

        xjw::gui::BaServiceOptions baselineOptions = makeServiceOptions(selectedImages,
                                                                        baselineDir,
                                                                        threads,
                                                                        dryRun,
                                                                        baOptions,
                                                                        false,
                                                                        QString(),
                                                                        laserMaxDistance,
                                                                        laserVoxelSize,
                                                                        laserMaxCurvature,
                                                                        laserMaxSamples,
                                                                        false,
                                                                        laserWeight,
                                                                        laserSigma,
                                                                        laserHuberDelta,
                                                                        PlanetaryLaserCliOptions{},
                                                                        exportEvalPlot);
        const xjw::gui::BaServiceResult baseline = runOneBa(baInput.cameras, baInput.tracks, baInput, baselineOptions);
        if (!baseline.success)
        {
            fatalQt(QStringLiteral("baseline BA 失败: %1").arg(baseline.errorMessage), cli::EXIT_ALGO_ERR);
        }
        printRunSummary(QStringLiteral("baseline"), baseline);

        xjw::gui::BaServiceOptions laserOptions = makeServiceOptions(selectedImages,
                                                                     laserDir,
                                                                     threads,
                                                                     dryRun,
                                                                     baOptions,
                                                                     true,
                                                                     laserCloud,
                                                                     laserMaxDistance,
                                                                     laserVoxelSize,
                                                                     laserMaxCurvature,
                                                                     laserMaxSamples,
                                                                     laserUseMissingNormalsAsHeightPlanes,
                                                                     laserWeight,
                                                                     laserSigma,
                                                                     laserHuberDelta,
                                                                     PlanetaryLaserCliOptions{},
                                                                     exportEvalPlot);
        const xjw::gui::BaServiceResult laser = runOneBa(baInput.cameras, baInput.tracks, baInput, laserOptions);
        if (!laser.success)
        {
            fatalQt(QStringLiteral("LiDAR BA 失败: %1").arg(laser.errorMessage), cli::EXIT_ALGO_ERR);
        }
        printRunSummary(QStringLiteral("laser"), laser);

        const QString comparePath = QDir(outputDir).filePath(QStringLiteral("ba_ab_compare.json"));
        const QJsonObject compareJson = buildCompareJson(outputDir, baseline, laser);
        QString jsonError;
        if (!xjw::cli::writeJsonFile(comparePath, compareJson, &jsonError))
        {
            fatalQt(jsonError, cli::EXIT_IO_ERR);
        }
        xjw::cli::printUtf8(stdout, QStringLiteral("A/B 对比已写入: %1").arg(comparePath));
        const bool quality_gate_passed =
            compareJson.value(QStringLiteral("quality_gate")).toObject().value(QStringLiteral("passed")).toBool(false);
        projectSession.appendResult(QStringLiteral("bundle_adjust_results"),
                                    bundleAdjustRecord(QStringLiteral("baseline"), baselineDir, baseline));
        projectSession.appendResult(QStringLiteral("bundle_adjust_results"),
                                    bundleAdjustRecord(QStringLiteral("laser"), laserDir, laser));
        QJsonObject compareRecord = compareJson;
        compareRecord[QStringLiteral("path")] = comparePath;
        compareRecord[QStringLiteral("kind")] = QStringLiteral("bundle_adjust_comparison");
        projectSession.upsertResultByPath(QStringLiteral("report_results"), QStringLiteral("path"), compareRecord);
        if (failOnQualityGate && !quality_gate_passed)
        {
            if (!projectSession.save(&projectError))
            {
                fatalQt(QStringLiteral("A/B BA 已完成，但质量报告保存失败: %1").arg(projectError), cli::EXIT_IO_ERR);
            }
            xjw::cli::printError(QStringLiteral("LiDAR BA 质量门禁失败，未写回相机；详情见: %1").arg(comparePath));
            return cli::EXIT_ALGO_ERR;
        }

        int updatedCameraCount = 0;
        if (!dryRun && !projectSession.updateImageCameras(laser.pendingCamUpdates, &updatedCameraCount, &projectError))
        {
            fatalQt(QStringLiteral("A/B BA 已完成，但相机写回失败: %1").arg(projectError), cli::EXIT_IO_ERR);
        }
        if (!projectSession.save(&projectError))
        {
            fatalQt(QStringLiteral("A/B BA 已完成，但 Chunk 写回失败: %1").arg(projectError), cli::EXIT_IO_ERR);
        }
        xjw::cli::printUtf8(stdout,
                            QStringLiteral("已写回 Chunk %1，相机=%2")
                                .arg(projectSession.activeChunk().directory)
                                .arg(updatedCameraCount));
        return cli::EXIT_OK;
    }

    xjw::gui::BaServiceOptions options = makeServiceOptions(selectedImages,
                                                            outputDir,
                                                            threads,
                                                            dryRun,
                                                            baOptions,
                                                            enableLaser,
                                                            laserCloud,
                                                            laserMaxDistance,
                                                            laserVoxelSize,
                                                            laserMaxCurvature,
                                                            laserMaxSamples,
                                                            laserUseMissingNormalsAsHeightPlanes,
                                                            laserWeight,
                                                            laserSigma,
                                                            laserHuberDelta,
                                                            planetaryLaserOptions,
                                                            exportEvalPlot);
    const xjw::gui::BaServiceResult result = runOneBa(baInput.cameras, baInput.tracks, baInput, options);
    if (!result.success)
    {
        fatalQt(QStringLiteral("BA 失败: %1").arg(result.errorMessage), cli::EXIT_ALGO_ERR);
    }

    projectSession.appendResult(QStringLiteral("bundle_adjust_results"),
                                bundleAdjustRecord(enablePlanetaryLaser ? QStringLiteral("planetary_laser_range")
                                                                        : (enableLaser ? QStringLiteral("laser_surface")
                                                                                       : QStringLiteral("baseline")),
                                                   outputDir,
                                                   result));
    int updatedCameraCount = 0;
    if (!dryRun && !projectSession.updateImageCameras(result.pendingCamUpdates, &updatedCameraCount, &projectError))
    {
        fatalQt(QStringLiteral("BA 已完成，但相机写回失败: %1").arg(projectError), cli::EXIT_IO_ERR);
    }
    if (!projectSession.save(&projectError))
    {
        fatalQt(QStringLiteral("BA 已完成，但 Chunk 写回失败: %1").arg(projectError), cli::EXIT_IO_ERR);
    }

    printRunSummary(enablePlanetaryLaser ? QStringLiteral("planetary_laser_range")
                                         : (enableLaser ? QStringLiteral("laser_surface") : QStringLiteral("baseline")),
                    result);
    xjw::cli::printUtf8(
        stdout,
        QStringLiteral("已写回 Chunk %1，相机=%2").arg(projectSession.activeChunk().directory).arg(updatedCameraCount));
    xjw::cli::printUtf8(stdout, QStringLiteral("输出目录: %1").arg(outputDir));
    return cli::EXIT_OK;
}
