// =============================================================================
// 文件: cli_bundle_adjust.cpp
// 功能: PlaScan 光束法平差 CLI，可选 LiDAR 点到面约束和 A/B 对比
// =============================================================================
#include "cli_common.h"

#include "BaInputBuilder.h"
#include "BundleAdjustService.h"
#include "PlascanArchive.h"
#include "ProjectFilesManager.h"
#include "ProjectIO.h"
#include "ProjectSupportUtils.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QTextStream>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace
{

QString toQString(const std::string &value)
{
    return QString::fromLocal8Bit(value.c_str()).trimmed();
}

void printUtf8(FILE *stream, const QString &message)
{
    const QByteArray bytes = message.toUtf8();
    std::fwrite(bytes.constData(), 1, static_cast<std::size_t>(bytes.size()), stream);
    std::fwrite("\n", 1, 1, stream);
    std::fflush(stream);
}

void fatalQt(const QString &message, int code = cli::EXIT_ARG_ERR)
{
    printUtf8(stderr, QStringLiteral("错误: %1").arg(message));
    std::exit(code);
}

QJsonObject readJsonObject(const QByteArray &bytes, const QString &label)
{
    QJsonParseError error;
    const QJsonDocument doc = QJsonDocument::fromJson(bytes, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject())
    {
        fatalQt(QStringLiteral("%1 不是有效 JSON 对象: %2").arg(label, error.errorString()), cli::EXIT_IO_ERR);
    }
    return doc.object();
}

QJsonObject loadProjectMeta(const QString &projectPath)
{
    if (!QFileInfo::exists(projectPath))
    {
        fatalQt(QStringLiteral("项目文件不存在: %1").arg(projectPath), cli::EXIT_IO_ERR);
    }

    PlascanArchive archive(projectPath);
    if (!archive.isValid())
    {
        fatalQt(QStringLiteral("无法打开 .plascan 项目归档: %1").arg(projectPath), cli::EXIT_IO_ERR);
    }

    QString archiveError;
    QByteArray coreData = archive.readEntry(ProjectFilesManager::kArchiveCoreFile, &archiveError);
    if (coreData.isEmpty())
    {
        coreData = archive.readEntry(QStringLiteral("project.json"), &archiveError);
    }
    if (coreData.isEmpty())
    {
        fatalQt(QStringLiteral("项目归档缺少 project_files.json/project.json: %1").arg(archiveError),
                cli::EXIT_IO_ERR);
    }

    QJsonObject meta = readJsonObject(coreData, QStringLiteral("project_files.json"));

    QByteArray resultsData = archive.readEntry(ProjectFilesManager::kArchiveResultsFile, &archiveError);
    if (!resultsData.isEmpty())
    {
        const QJsonObject results = readJsonObject(resultsData, QStringLiteral("project_results.json"));
        for (auto it = results.constBegin(); it != results.constEnd(); ++it)
        {
            if (it.key() != QLatin1String("images"))
            {
                meta.insert(it.key(), it.value());
            }
        }
    }

    return meta;
}

QStringList resolveSelectedImages(const QJsonObject &meta, const std::vector<std::string> &tokens)
{
    QStringList selected;
    if (tokens.empty())
    {
        selected = xjw::gui::project::projectImagePaths(meta);
    }
    else
    {
        for (const std::string &rawToken : tokens)
        {
            const QString token = toQString(rawToken);
            const QString path = xjw::gui::project::resolveProjectImagePathFromToken(token, meta);
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
    const QString stamp = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"));
    return QDir(QFileInfo(projectPath).absolutePath()).filePath(QStringLiteral("bundle_adjust_cli_%1").arg(stamp));
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
        jsonDouble(laserObj, QStringLiteral("mean_rms_before")) - jsonDouble(laserObj, QStringLiteral("mean_rms_after"));
    compare[QStringLiteral("laser_after_minus_baseline_after")] =
        jsonDouble(laserObj, QStringLiteral("mean_rms_after")) - jsonDouble(baseObj, QStringLiteral("mean_rms_after"));
    compare[QStringLiteral("laser_constraints_summary")] =
        laserObj.value(QStringLiteral("laser_constraints_summary")).toObject();
    return compare;
}

void writeJsonFile(const QString &path, const QJsonObject &object)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        fatalQt(QStringLiteral("无法写入 JSON: %1").arg(path), cli::EXIT_IO_ERR);
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    file.close();
}

void printRunSummary(const QString &label, const xjw::gui::BaServiceResult &result)
{
    const QJsonObject obj = result.resultJson;
    printUtf8(stdout,
              QStringLiteral("%1: tracks=%2 optimized=%3 rms_before=%4 rms_after=%5")
                  .arg(label)
                  .arg(obj.value(QStringLiteral("track_count")).toInt())
                  .arg(obj.value(QStringLiteral("optimized_count")).toInt())
                  .arg(obj.value(QStringLiteral("mean_rms_before")).toDouble(), 0, 'f', 6)
                  .arg(obj.value(QStringLiteral("mean_rms_after")).toDouble(), 0, 'f', 6));
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication qtApp(argc, argv);
    Q_UNUSED(qtApp);

    CLI::App app{"PlaScan 光束法平差 CLI — 支持 LiDAR 点到面约束与 A/B 对比"};

    std::string projectPathRaw;
    std::string outputDirRaw;
    std::vector<std::string> imageTokens;
    std::string laserCloudRaw;

    int minMatches = 15;
    int threads = 4;
    int maxIterations = 20;
    int maxPointIterations = 20;
    int maxCameraIterations = 5;
    int chunkSize = 20000;
    double huberDelta = 3.0;
    double damping = 1e-3;
    double finiteDiffEps = 1e-6;
    double stepTolerance = 1e-6;
    bool refinePose = true;
    bool dryRun = false;
    bool force = false;
    bool abCompare = false;
    bool exportEvalPlot = false;

    double laserMaxDistance = 1.0;
    double laserVoxelSize = 0.0;
    double laserMaxCurvature = 0.2;
    int laserMaxSamples = 500000;
    double laserWeight = 1.0;
    double laserHuberDelta = 0.2;

    app.add_option("project", projectPathRaw, ".plascan 项目文件")->required();
    app.add_option("-o,--output-dir", outputDirRaw, "输出目录；未指定时在项目旁创建 bundle_adjust_cli_<timestamp>");
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
    app.add_flag("--refine-pose{true},--no-refine-pose{false}", refinePose, "是否优化相机姿态");
    app.add_flag("--dry-run", dryRun, "仅检查输入并统计 tracks，不执行优化");
    app.add_flag("--force", force, "允许使用非空输出目录");
    app.add_flag("--export-eval-plot", exportEvalPlot, "导出 BA 评估 PNG；无头命令行默认关闭");

    app.add_option("--laser-cloud", laserCloudRaw, "带 normal_x/normal_y/normal_z 的 LiDAR PLY 点云");
    app.add_option("--laser-max-distance", laserMaxDistance, "track 到 LiDAR 平面最大关联距离（米）");
    app.add_option("--laser-voxel-size", laserVoxelSize, "LiDAR 点云体素降采样尺寸（米），0 表示关闭");
    app.add_option("--laser-max-curvature", laserMaxCurvature, "允许参与约束的最大曲率");
    app.add_option("--laser-max-samples", laserMaxSamples, "LiDAR map 最大采样点数");
    app.add_option("--laser-weight", laserWeight, "LiDAR 点到面残差权重");
    app.add_option("--laser-huber-delta", laserHuberDelta, "LiDAR 残差 Huber 阈值（米）");
    app.add_flag("--ab-compare", abCompare, "一次性运行 baseline 与 LiDAR BA，并写 ba_ab_compare.json");

    CLI11_PARSE(app, argc, argv);

    const QString projectPath = QDir::cleanPath(QFileInfo(toQString(projectPathRaw)).absoluteFilePath());
    const QString outputDir = outputDirRaw.empty()
        ? defaultOutputDir(projectPath)
        : QDir::cleanPath(QFileInfo(toQString(outputDirRaw)).absoluteFilePath());
    const QString laserCloud = toQString(laserCloudRaw);
    const bool enableLaser = !laserCloud.isEmpty();

    if (abCompare && !enableLaser)
    {
        fatalQt(QStringLiteral("--ab-compare 需要同时指定 --laser-cloud"), cli::EXIT_ARG_ERR);
    }
    if (enableLaser && !QFileInfo::exists(laserCloud))
    {
        fatalQt(QStringLiteral("LiDAR 点云不存在: %1").arg(laserCloud), cli::EXIT_IO_ERR);
    }

    const QJsonObject meta = loadProjectMeta(projectPath);
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

    printUtf8(stdout,
              QStringLiteral("BA 输入: cameras=%1 tracks=%2 selected_images=%3 sidecar_v2_pairs=%4 multiview_tracks=%5")
                  .arg(static_cast<int>(baInput.cameras.size()))
                  .arg(static_cast<int>(baInput.tracks.size()))
                  .arg(selectedImages.size())
                  .arg(baInput.sidecarV2PairCount)
                  .arg(baInput.multiViewTrackCount));

    if (abCompare)
    {
        const QString baselineDir = QDir(outputDir).filePath(QStringLiteral("baseline"));
        const QString laserDir = QDir(outputDir).filePath(QStringLiteral("laser"));
        ensureOutputDirAllowed(baselineDir, force);
        ensureOutputDirAllowed(laserDir, force);

        xjw::gui::BaServiceOptions baselineOptions = makeServiceOptions(
            selectedImages, baselineDir, threads, dryRun, baOptions,
            false, QString(), laserMaxDistance, laserVoxelSize, laserMaxCurvature,
            laserMaxSamples, laserWeight, laserHuberDelta, exportEvalPlot);
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
            laserMaxSamples, laserWeight, laserHuberDelta, exportEvalPlot);
        const xjw::gui::BaServiceResult laser = runOneBa(
            baInput.cameras, baInput.tracks, baInput, laserOptions);
        if (!laser.success)
        {
            fatalQt(QStringLiteral("LiDAR BA 失败: %1").arg(laser.errorMessage), cli::EXIT_ALGO_ERR);
        }
        printRunSummary(QStringLiteral("laser"), laser);

        const QString comparePath = QDir(outputDir).filePath(QStringLiteral("ba_ab_compare.json"));
        writeJsonFile(comparePath, buildCompareJson(outputDir, baseline, laser));
        printUtf8(stdout, QStringLiteral("A/B 对比已写入: %1").arg(comparePath));
        return cli::EXIT_OK;
    }

    xjw::gui::BaServiceOptions options = makeServiceOptions(
        selectedImages, outputDir, threads, dryRun, baOptions,
        enableLaser, laserCloud, laserMaxDistance, laserVoxelSize, laserMaxCurvature,
        laserMaxSamples, laserWeight, laserHuberDelta, exportEvalPlot);
    const xjw::gui::BaServiceResult result = runOneBa(baInput.cameras, baInput.tracks, baInput, options);
    if (!result.success)
    {
        fatalQt(QStringLiteral("BA 失败: %1").arg(result.errorMessage), cli::EXIT_ALGO_ERR);
    }

    printRunSummary(enableLaser ? QStringLiteral("laser") : QStringLiteral("baseline"), result);
    printUtf8(stdout, QStringLiteral("输出目录: %1").arg(outputDir));
    return cli::EXIT_OK;
}
