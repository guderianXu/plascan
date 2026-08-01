// =============================================================================
// 文件: cli_bundle_adjust.cpp
// 功能: PlaScan 光束法平差 CLI，可选 LiDAR 点到面约束和 A/B 对比
// =============================================================================
#include "cli_common.h"
#include "CliConsole.h"
#include "CliJsonIO.h"
#include "CliPathUtils.h"

#include "project/BaInputBuilder.h"
#include "BundleAdjust.h"
#include "BundleAdjustService.h"
#include "ProjectFilesManager.h"
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

void fatalQt(const QString &message, int code = cli::EXIT_ARG_ERR)
{
    xjw::cli::printError(message);
    std::exit(code);
}

xjw::BABackend parseBaBackendName(const QString &raw)
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
    if (value == QLatin1String("ceres_cpu"))
    {
        return xjw::BABackend::CeresCpu;
    }
    if (value == QLatin1String("ceres_cuda"))
    {
        return xjw::BABackend::CeresCuda;
    }
    if (value == QLatin1String("native_cuda"))
    {
        return xjw::BABackend::NativeCuda;
    }

    fatalQt(QStringLiteral("未知 BA 后端: %1，可选 auto / legacy_cpu / ceres_cpu / ceres_cuda / native_cuda")
                .arg(raw));
    return xjw::BABackend::LegacyCpu;
}

QStringList resolveSelectedImages(const QJsonObject &meta, const std::vector<std::string> &tokens)
{
    QStringList selected;
    if (tokens.empty())
    {
        selected = xjw::common::project::projectImagePaths(meta);
    }
    else
    {
        for (const std::string &rawToken : tokens)
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

QString defaultOutputDir(const QString &projectPath)
{
    const QString stamp = QDateTime::currentDateTime().toString(
        QStringLiteral("yyyyMMdd_HHmmss_zzz"));
    return QDir(
        xjw::common::project::ProjectIO::projectBundleAdjustDir(projectPath))
        .filePath(stamp);
}

QJsonObject bundleAdjustRecord(const QString &mode,
                               const QString &outputDir,
                               const xjw::gui::BaServiceResult &result)
{
    QJsonObject record = result.resultJson;
    record[QStringLiteral("created_at")] =
        QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    record[QStringLiteral("source")] = QStringLiteral("bundle_adjust_cli");
    record[QStringLiteral("mode")] = mode;
    record[QStringLiteral("output_dir")] = outputDir;
    return record;
}

bool directoryHasEntries(const QString &path)
{
    QDir dir(path);
    return dir.exists() && !dir.entryList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty();
}

void ensureOutputDirAllowed(const QString &outputDir, bool force)
{
    if (directoryHasEntries(outputDir) && !force)
    {
        fatalQt(QStringLiteral("输出目录已存在且非空，请更换目录或使用 --force: %1").arg(outputDir),
                cli::EXIT_ARG_ERR);
    }
}

xjw::gui::BaServiceOptions makeServiceOptions(const QStringList &selectedImages,
                                              const QString &outputDir,
                                              int threads,
                                              bool dryRun,
                                              const xjw::BAOptions &baOptions,
                                              bool enableLaser,
                                              const QString &laserCloud,
                                              double laserMaxDistance,
                                              double laserVoxelSize,
                                              double laserMaxCurvature,
                                              int laserMaxSamples,
                                              bool laserUseMissingNormalsAsHeightPlanes,
                                              double laserWeight,
                                              double laserHuberDelta,
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
    options.laserHuberDeltaMeters = laserHuberDelta;
    options.exportEvalPlot = exportEvalPlot;
    return options;
}

QString buildStatusMessage(xjw::core::project::BaInputBuildStatus status)
{
    switch (status)
    {
    case xjw::core::project::BaInputBuildStatus::Ok:
        return QStringLiteral("OK");
    case xjw::core::project::BaInputBuildStatus::NotEnoughCameras:
        return QStringLiteral("项目中可用相机少于 2 台");
    case xjw::core::project::BaInputBuildStatus::NoTracks:
    default:
        return QStringLiteral("没有可用于 BA 的 tracks，请确认匹配 sidecar 包含 matched_points 和 matched_indices");
    }
}

xjw::gui::BaServiceResult runOneBa(const std::vector<xjw::Camera> &cameras,
                                   const std::vector<xjw::BATrack> &tracks,
                                   const xjw::core::project::BaInputBuildResult &input,
                                   xjw::gui::BaServiceOptions options)
{
    std::vector<xjw::BATrack> runTracks = tracks;
    options.imagePathByIndex = input.imagePathByIndex;
    options.beforeCamMeta = input.beforeCamMeta;
    return xjw::gui::BundleAdjustService::run(cameras, runTracks, options);
}

double jsonDouble(const QJsonObject &object, const QString &key)
{
    return object.value(key).toDouble(0.0);
}

int jsonInt(const QJsonObject &object, const QString &key)
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

QJsonObject buildQualityGate(const QJsonObject &baseObj, const QJsonObject &laserObj)
{
    const QJsonObject laserSummary =
        laserObj.value(QStringLiteral("laser_constraints_summary")).toObject();
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
    const double optimizedTrackDropPercent = baselineOptimized > 0
        ? static_cast<double>(optimizedTrackDrop) * 100.0 / static_cast<double>(baselineOptimized)
        : 0.0;

    QJsonArray failureCodes;
    QJsonArray failureReasons;
    auto appendFailure = [&failureCodes, &failureReasons](const QString &code, const QString &reason)
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
    if (reprojectionRegressionPx > kMaxReprojectionRmsRegressionPx + kQualityGateEpsilon
        || reprojectionRegressionPercent > kMaxReprojectionRmsRegressionPercent + kQualityGateEpsilon)
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
    thresholds[QStringLiteral("max_reprojection_rms_regression_percent")] =
        kMaxReprojectionRmsRegressionPercent;
    thresholds[QStringLiteral("max_optimized_track_drop_percent")] = kMaxOptimizedTrackDropPercent;

    QJsonObject metrics;
    metrics[QStringLiteral("reprojection_rms_regression_px")] = reprojectionRegressionPx;
    metrics[QStringLiteral("reprojection_rms_regression_percent")] = reprojectionRegressionPercent;
    metrics[QStringLiteral("optimized_track_drop")] = optimizedTrackDrop;
    metrics[QStringLiteral("optimized_track_drop_percent")] = optimizedTrackDropPercent;

    QJsonObject gate;
    gate[QStringLiteral("passed")] = failureCodes.isEmpty();
    gate[QStringLiteral("status")] = failureCodes.isEmpty()
        ? QStringLiteral("pass")
        : QStringLiteral("fail");
    gate[QStringLiteral("failure_codes")] = failureCodes;
    gate[QStringLiteral("failure_reasons")] = failureReasons;
    gate[QStringLiteral("thresholds")] = thresholds;
    gate[QStringLiteral("metrics")] = metrics;
    return gate;
}

QJsonObject buildCompareJson(const QString &outputDir,
                             const xjw::gui::BaServiceResult &baseline,
                             const xjw::gui::BaServiceResult &laser)
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
    compare[QStringLiteral("baseline_improvement")] =
        jsonDouble(baseObj, QStringLiteral("mean_rms_before")) - jsonDouble(baseObj, QStringLiteral("mean_rms_after"));
    compare[QStringLiteral("laser_improvement")] =
        jsonDouble(laserObj, QStringLiteral("mean_rms_before"))
        - jsonDouble(laserObj, QStringLiteral("mean_rms_after"));
    compare[QStringLiteral("laser_after_minus_baseline_after")] =
        jsonDouble(laserObj, QStringLiteral("mean_rms_after")) - jsonDouble(baseObj, QStringLiteral("mean_rms_after"));
    compare[QStringLiteral("laser_constraints_summary")] =
        laserObj.value(QStringLiteral("laser_constraints_summary")).toObject();
    compare[QStringLiteral("quality_gate")] = buildQualityGate(baseObj, laserObj);
    return compare;
}

void printRunSummary(const QString &label, const xjw::gui::BaServiceResult &result)
{
    const QJsonObject obj = result.resultJson;
    const QString usedBackend = obj.value(QStringLiteral("ba_used_backend")).toString(QStringLiteral("unknown"));
    const QString solver =
        obj.value(QStringLiteral("ba_ceres_linear_solver")).toString(QStringLiteral("none"));
    const bool usedGpu = obj.value(QStringLiteral("ba_used_gpu")).toBool(false);
    const bool backendFallback = obj.value(QStringLiteral("ba_backend_fallback")).toBool(false);
    const bool qualityRejected = obj.value(QStringLiteral("ba_quality_gate_rejected")).toBool(false);
    const int observationCount = obj.value(QStringLiteral("ba_observation_count")).toInt(0);
    const double totalSeconds = obj.value(QStringLiteral("ba_total_seconds")).toDouble(0.0);
    const double validTrackRatio = obj.value(QStringLiteral("ba_valid_track_ratio")).toDouble(0.0);
    const QString backendReason =
        obj.value(QStringLiteral("ba_backend_selection_reason")).toString();
    const QString qualityMessage =
        obj.value(QStringLiteral("ba_quality_gate_message")).toString();
    xjw::cli::printUtf8(stdout,
              QStringLiteral("%1: tracks=%2 optimized=%3 observations=%4 rms_before=%5 rms_after=%6 "
                             "backend=%7 solver=%8 gpu=%9 fallback=%10 valid_ratio=%11 "
                             "quality_rejected=%12 total_s=%13")
                  .arg(label)
                  .arg(obj.value(QStringLiteral("track_count")).toInt())
                  .arg(obj.value(QStringLiteral("optimized_count")).toInt())
                  .arg(observationCount)
                  .arg(obj.value(QStringLiteral("mean_rms_before")).toDouble(), 0, 'f', 6)
                  .arg(obj.value(QStringLiteral("mean_rms_after")).toDouble(), 0, 'f', 6)
                  .arg(usedBackend,
                       solver,
                       usedGpu ? QStringLiteral("true") : QStringLiteral("false"),
                       backendFallback ? QStringLiteral("true") : QStringLiteral("false"))
                  .arg(validTrackRatio, 0, 'f', 4)
                  .arg(qualityRejected ? QStringLiteral("true") : QStringLiteral("false"))
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

int main(int argc, char *argv[])
{
    QCoreApplication qtApp(argc, argv);
    Q_UNUSED(qtApp);

    CLI::App app{"PlaScan 光束法平差 CLI — 支持 LiDAR 点到面约束与 A/B 对比"};

    std::string projectPathRaw;
    std::string chunkIdRaw;
    std::string chunkNameRaw;
    std::string outputDirRaw;
    std::vector<std::string> imageTokens;
    std::string laserCloudRaw;
    std::string baBackendRaw = "auto";

    int minMatches = 15;
    int threads = 4;
    int maxIterations = 20;
    int maxPointIterations = 12;
    int maxCameraIterations = 10;
    int chunkSize = 20000;
    int baCudaDevice = 0;
    int baMinCudaCameras = 50;
    int baMinCudaObservations = 500000;
    int baNativeCudaDevice = 0;
    int baMinCpuObservations = 50000;
    int baMaxCeresPointOnlyObservations = 100000;
    int baMaxDenseSchurCameras = 200;
    int baMaxSparseSchurCameras = 2000;
    double huberDelta = 3.0;
    double damping = 1e-3;
    double finiteDiffEps = 1e-6;
    double stepTolerance = 1e-8;
    double baNativeCudaMaxPointStep = 1.0;
    double baMaxAcceptedRmsGrowth = 1.25;
    double baMinAcceptedValidTrackRatio = 0.60;
    double baMaxConstraintRmsGrowth = 1.25;
    double baMaxCudaMemoryFraction = 0.70;
    bool refinePose = true;
    bool dryRun = false;
    bool force = false;
    bool abCompare = false;
    bool exportEvalPlot = false;
    bool failOnQualityGate = false;
    bool laserUseMissingNormalsAsHeightPlanes = false;
    bool baBackendFallback = true;
    bool baEnableQualityGate = true;
    bool baCompareAutoWithLegacy = true;

    double laserMaxDistance = 1.0;
    double laserVoxelSize = 0.0;
    double laserMaxCurvature = 0.2;
    int laserMaxSamples = 500000;
    double laserWeight = 1.0;
    double laserHuberDelta = 0.2;

    app.add_option("project", projectPathRaw, ".plascan 项目文件")->required();
    app.add_option("--chunk-id", chunkIdRaw, "使用指定 UUID 的 Chunk");
    app.add_option("--chunk-name", chunkNameRaw, "使用指定名称的 Chunk");
    app.add_option(
        "-o,--output-dir",
        outputDirRaw,
        "输出目录；未指定时写入当前 Chunk 的 bundle_adjust/<timestamp>");
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
                   "BA 求解后端: auto / legacy_cpu / ceres_cpu / ceres_cuda / native_cuda");
    app.add_option("--ba-cuda-device", baCudaDevice, "Ceres CUDA BA 使用的 GPU 设备 ID");
    app.add_option("--ba-min-cuda-cameras", baMinCudaCameras, "低于该相机数时 Ceres CUDA BA 回退到 CPU");
    app.add_option("--ba-min-cuda-observations",
                   baMinCudaObservations,
                   "低于该观测数时自动 BA 不选择 Ceres CUDA");
    app.add_option("--ba-min-cpu-observations",
                   baMinCpuObservations,
                   "低于该观测数时自动 BA 不选择 Ceres CPU");
    app.add_option("--ba-native-cuda-device", baNativeCudaDevice, "native_cuda BA 使用的 GPU 设备 ID");
    app.add_option("--ba-native-cuda-max-point-step",
                   baNativeCudaMaxPointStep,
                   "native_cuda 单次点块更新的最大三维位移范数");
    app.add_option("--ba-max-ceres-point-only-observations",
                   baMaxCeresPointOnlyObservations,
                   "point-only Ceres BA 超过该观测数时回退 legacy_cpu");
    app.add_option("--ba-max-dense-schur-cameras",
                   baMaxDenseSchurCameras,
                   "Ceres Auto 使用 dense Schur 的最大可变相机数");
    app.add_option("--ba-max-sparse-schur-cameras",
                   baMaxSparseSchurCameras,
                   "Ceres Auto 使用 sparse Schur 的最大可变相机数");
    app.add_option("--ba-max-cuda-memory-fraction",
                   baMaxCudaMemoryFraction,
                   "Ceres CUDA 最多使用当前空闲显存的比例");
    app.add_flag("--ba-backend-fallback{true},--no-ba-backend-fallback{false}",
                 baBackendFallback,
                 "请求的 BA 后端不可用时是否允许回退到可用后端");
    app.add_flag("--ba-quality-gate{true},--no-ba-quality-gate{false}",
                 baEnableQualityGate,
                 "Auto BA 是否启用质量门控，候选后端质量异常时回退 legacy_cpu");
    app.add_option("--ba-max-rms-growth",
                   baMaxAcceptedRmsGrowth,
                   "Auto BA 候选后端允许的最大 RMS 增长倍率");
    app.add_option("--ba-max-constraint-rms-growth",
                   baMaxConstraintRmsGrowth,
                   "Auto BA 物方约束允许的最大 RMS 增长倍率");
    app.add_option("--ba-min-valid-track-ratio",
                   baMinAcceptedValidTrackRatio,
                   "Auto BA 候选后端允许的最小有效 track 比例");
    app.add_flag("--ba-compare-legacy{true},--no-ba-compare-legacy{false}",
                 baCompareAutoWithLegacy,
                 "Auto BA 选择 Ceres/CUDA 时是否运行 legacy 对照用于质量门控");
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
    app.add_option("--laser-weight", laserWeight, "LiDAR 点到面残差权重");
    app.add_option("--laser-huber-delta", laserHuberDelta, "LiDAR 残差 Huber 阈值（米）");
    app.add_flag("--ab-compare", abCompare, "一次性运行 baseline 与 LiDAR BA，并写 ba_ab_compare.json");
    app.add_flag("--fail-on-quality-gate",
                 failOnQualityGate,
                 "仅在 --ab-compare 下生效；LiDAR BA 质量门禁失败时写出对比 JSON 后返回非零");

    CLI11_PARSE(app, argc, argv);

    const QString projectPath = xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(projectPathRaw));
    if (!QFileInfo::exists(projectPath))
    {
        fatalQt(QStringLiteral("项目文件不存在: %1").arg(projectPath),
                cli::EXIT_IO_ERR);
    }

    xjw::common::project::ProjectSession projectSession;
    QString projectError;
    if (!projectSession.open(projectPath, &projectError))
    {
        fatalQt(QStringLiteral("无法打开 .plascan Chunk 工程: %1")
                    .arg(projectError),
                cli::EXIT_IO_ERR);
    }
    if (!chunkIdRaw.empty() && !chunkNameRaw.empty())
    {
        fatalQt(QStringLiteral("--chunk-id 与 --chunk-name 不能同时使用"),
                cli::EXIT_ARG_ERR);
    }
    if (!projectSession.selectChunk(
            xjw::cli::fromStdString(chunkIdRaw),
            xjw::cli::fromStdString(chunkNameRaw),
            &projectError))
    {
        fatalQt(QStringLiteral("Chunk 选择失败: %1").arg(projectError),
                cli::EXIT_IO_ERR);
    }

    const QString outputDir = outputDirRaw.empty()
        ? defaultOutputDir(projectPath)
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(outputDirRaw));
    const QString laserCloud = xjw::cli::fromStdString(laserCloudRaw).trimmed();
    const bool enableLaser = !laserCloud.isEmpty();

    if (abCompare && !enableLaser)
    {
        fatalQt(QStringLiteral("--ab-compare 需要同时指定 --laser-cloud"), cli::EXIT_ARG_ERR);
    }
    if (enableLaser && !QFileInfo::exists(laserCloud))
    {
        fatalQt(QStringLiteral("LiDAR 点云不存在: %1").arg(laserCloud), cli::EXIT_IO_ERR);
    }

    const QJsonObject meta = projectSession.mergedMetadata();
    const QStringList selectedImages = resolveSelectedImages(meta, imageTokens);

    ensureOutputDirAllowed(outputDir, force);

    xjw::core::project::BaInputBuildResult baInput;
    const xjw::core::project::BaInputBuildStatus buildStatus =
        xjw::core::project::buildBaInputFromMeta(meta, selectedImages, minMatches, &baInput);
    if (buildStatus != xjw::core::project::BaInputBuildStatus::Ok)
    {
        fatalQt(buildStatusMessage(buildStatus), cli::EXIT_ALGO_ERR);
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
    baOptions.ceresCudaDevice = std::max(0, baCudaDevice);
    baOptions.minCeresCudaCameras = std::max(1, baMinCudaCameras);
    baOptions.minCeresCudaObservations = std::max(1, baMinCudaObservations);
    baOptions.nativeCudaDevice = std::max(0, baNativeCudaDevice);
    baOptions.nativeCudaMaxPointStepNorm =
        std::max(1e-12, baNativeCudaMaxPointStep);
    baOptions.minCeresCpuObservations = std::max(1, baMinCpuObservations);
    baOptions.maxCeresPointOnlyObservations = std::max(1, baMaxCeresPointOnlyObservations);
    baOptions.maxDenseSchurCameras =
        std::max(1, baMaxDenseSchurCameras);
    baOptions.maxSparseSchurCameras =
        std::max(
            baOptions.maxDenseSchurCameras,
            baMaxSparseSchurCameras);
    baOptions.maxCeresCudaMemoryFraction =
        std::clamp(baMaxCudaMemoryFraction, 0.01, 1.0);
    baOptions.allowBackendFallback = baBackendFallback;
    baOptions.enableBackendQualityGate = baEnableQualityGate;
    baOptions.maxAcceptedRmsGrowth = std::max(0.0, baMaxAcceptedRmsGrowth);
    baOptions.minAcceptedValidTrackRatio = std::max(0.0, baMinAcceptedValidTrackRatio);
    baOptions.maxAcceptedConstraintRmsGrowth =
        std::max(1.0, baMaxConstraintRmsGrowth);
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
        QStringLiteral("BA 输入: cameras=%1 tracks=%2 selected_images=%3 indexed_observations=%4 ")
        + QStringLiteral("multiview_tracks=%5 survey_control_tracks=%6 scale_bars=%7");
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

        xjw::gui::BaServiceOptions baselineOptions = makeServiceOptions(
            selectedImages, baselineDir, threads, dryRun, baOptions,
            false, QString(), laserMaxDistance, laserVoxelSize, laserMaxCurvature,
            laserMaxSamples, false, laserWeight, laserHuberDelta, exportEvalPlot);
        const xjw::gui::BaServiceResult baseline = runOneBa(
            baInput.cameras, baInput.tracks, baInput, baselineOptions);
        if (!baseline.success)
        {
            fatalQt(QStringLiteral("baseline BA 失败: %1").arg(baseline.errorMessage), cli::EXIT_ALGO_ERR);
        }
        printRunSummary(QStringLiteral("baseline"), baseline);

        xjw::gui::BaServiceOptions laserOptions = makeServiceOptions(
            selectedImages, laserDir, threads, dryRun, baOptions,
            true, laserCloud, laserMaxDistance, laserVoxelSize, laserMaxCurvature,
            laserMaxSamples,
            laserUseMissingNormalsAsHeightPlanes,
            laserWeight,
            laserHuberDelta,
            exportEvalPlot);
        const xjw::gui::BaServiceResult laser = runOneBa(
            baInput.cameras, baInput.tracks, baInput, laserOptions);
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
        const bool quality_gate_passed = compareJson.value(QStringLiteral("quality_gate"))
                                             .toObject()
                                             .value(QStringLiteral("passed"))
                                             .toBool(false);
        projectSession.appendResult(
            QStringLiteral("bundle_adjust_results"),
            bundleAdjustRecord(
                QStringLiteral("baseline"), baselineDir, baseline));
        projectSession.appendResult(
            QStringLiteral("bundle_adjust_results"),
            bundleAdjustRecord(QStringLiteral("laser"), laserDir, laser));
        int updatedCameraCount = 0;
        if (!dryRun
            && !projectSession.updateImageCameras(
                laser.pendingCamUpdates,
                &updatedCameraCount,
                &projectError))
        {
            fatalQt(QStringLiteral("A/B BA 已完成，但相机写回失败: %1")
                        .arg(projectError),
                    cli::EXIT_IO_ERR);
        }
        QJsonObject compareRecord = compareJson;
        compareRecord[QStringLiteral("path")] = comparePath;
        compareRecord[QStringLiteral("kind")] =
            QStringLiteral("bundle_adjust_comparison");
        projectSession.upsertResultByPath(
            QStringLiteral("report_results"),
            QStringLiteral("path"),
            compareRecord);
        if (!projectSession.save(&projectError))
        {
            fatalQt(QStringLiteral("A/B BA 已完成，但 Chunk 写回失败: %1")
                        .arg(projectError),
                    cli::EXIT_IO_ERR);
        }
        xjw::cli::printUtf8(
            stdout,
            QStringLiteral("已写回 Chunk %1，相机=%2")
                .arg(projectSession.activeChunk().directory)
                .arg(updatedCameraCount));
        if (failOnQualityGate && !quality_gate_passed)
        {
            xjw::cli::printError(
                QStringLiteral("LiDAR BA 质量门禁失败，详情见: %1")
                    .arg(comparePath));
            return cli::EXIT_ALGO_ERR;
        }
        return cli::EXIT_OK;
    }

    xjw::gui::BaServiceOptions options = makeServiceOptions(
        selectedImages, outputDir, threads, dryRun, baOptions,
        enableLaser, laserCloud, laserMaxDistance, laserVoxelSize, laserMaxCurvature,
        laserMaxSamples,
        laserUseMissingNormalsAsHeightPlanes,
        laserWeight,
        laserHuberDelta,
        exportEvalPlot);
    const xjw::gui::BaServiceResult result = runOneBa(baInput.cameras, baInput.tracks, baInput, options);
    if (!result.success)
    {
        fatalQt(QStringLiteral("BA 失败: %1").arg(result.errorMessage), cli::EXIT_ALGO_ERR);
    }

    projectSession.appendResult(
        QStringLiteral("bundle_adjust_results"),
        bundleAdjustRecord(
            enableLaser ? QStringLiteral("laser")
                        : QStringLiteral("baseline"),
            outputDir,
            result));
    int updatedCameraCount = 0;
    if (!dryRun
        && !projectSession.updateImageCameras(
            result.pendingCamUpdates,
            &updatedCameraCount,
            &projectError))
    {
        fatalQt(QStringLiteral("BA 已完成，但相机写回失败: %1")
                    .arg(projectError),
                cli::EXIT_IO_ERR);
    }
    if (!projectSession.save(&projectError))
    {
        fatalQt(QStringLiteral("BA 已完成，但 Chunk 写回失败: %1")
                    .arg(projectError),
                cli::EXIT_IO_ERR);
    }

    printRunSummary(enableLaser ? QStringLiteral("laser") : QStringLiteral("baseline"), result);
    xjw::cli::printUtf8(
        stdout,
        QStringLiteral("已写回 Chunk %1，相机=%2")
            .arg(projectSession.activeChunk().directory)
            .arg(updatedCameraCount));
    xjw::cli::printUtf8(stdout, QStringLiteral("输出目录: %1").arg(outputDir));
    return cli::EXIT_OK;
}
