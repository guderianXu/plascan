#include "cli_common.h"
#include "cli_photogrammetry_common.h"

#include "AerialTriangulationService.h"
#include "AerialTriangulationWorkflow.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{

QString fromStdPath(const std::string &value)
{
    return QString::fromStdString(value).trimmed();
}

QString normalizedToken(QString value, const QString &fallback = QString())
{
    value = value.trimmed().toLower();
    value.replace(QStringLiteral("-"), QStringLiteral("_"));
    return value.isEmpty() ? fallback : value;
}

QString referenceModeFromToken(const QString &value)
{
    const QString token = normalizedToken(value, QStringLiteral("source_code"));
    if (token == QStringLiteral("source") || token == QStringLiteral("sourcecode"))
    {
        return QStringLiteral("source_code");
    }
    if (token == QStringLiteral("estimated") || token == QStringLiteral("estimate"))
    {
        return QStringLiteral("estimated");
    }
    if (token == QStringLiteral("sequence") || token == QStringLiteral("sequential"))
    {
        return QStringLiteral("sequence");
    }
    return QStringLiteral("source_code");
}

QJsonArray featureFilesToJson(const QVector<xjw::gui::SpFileRecord> &records)
{
    QJsonArray array;
    for (const xjw::gui::SpFileRecord &record : records)
    {
        QJsonObject object;
        object[QStringLiteral("image_path")] = record.imagePath;
        object[QStringLiteral("feature_path")] = record.featurePath;
        array.append(object);
    }
    return array;
}

QJsonArray matchFilesToJson(const QVector<xjw::gui::MatchFileRecord> &records)
{
    QJsonArray array;
    for (const xjw::gui::MatchFileRecord &record : records)
    {
        QJsonObject object;
        object[QStringLiteral("pair_name")] = record.pairName;
        object[QStringLiteral("match_path")] = record.matchPath;
        object[QStringLiteral("sidecar_path")] = record.sidecarPath;
        object[QStringLiteral("settings")] = record.settings;
        array.append(object);
    }
    return array;
}

QJsonArray failedPairsToJson(const QVector<xjw::gui::FailedPairRecord> &records)
{
    QJsonArray array;
    for (const xjw::gui::FailedPairRecord &record : records)
    {
        QJsonObject object;
        object[QStringLiteral("image0_path")] = record.imagePath0;
        object[QStringLiteral("image1_path")] = record.imagePath1;
        array.append(object);
    }
    return array;
}

QJsonObject serviceResultToJson(const xjw::gui::AerialTriangulationServiceResult &service)
{
    QJsonObject object;
    object[QStringLiteral("success")] = service.success;
    object[QStringLiteral("error_message")] = service.errorMessage;
    object[QStringLiteral("summary")] = service.summary;
    object[QStringLiteral("feature_matches_ready")] = service.featureMatchesReady;
    object[QStringLiteral("registered_images")] = service.numRegisteredImages;
    object[QStringLiteral("points3d")] = service.numPoints3D;
    object[QStringLiteral("mean_reproj_error")] = service.meanReprojError;
    object[QStringLiteral("sparse_cloud_path")] = service.sparseCloudPath;
    object[QStringLiteral("quality_metadata")] = service.qualityMetadata;
    object[QStringLiteral("result_record_extra")] = service.resultRecordExtra;
    object[QStringLiteral("sfm_diagnostics")] = service.sfmDiagnostics;
    object[QStringLiteral("ba_rms_before")] = service.baRmsBefore;
    object[QStringLiteral("ba_rms_after")] = service.baRmsAfter;
    object[QStringLiteral("ba_tracks_total")] = service.baTracksTotal;
    object[QStringLiteral("ba_tracks_optimized")] = service.baTracksOptimized;
    object[QStringLiteral("ba_tracks_filtered")] = service.baTracksFiltered;
    object[QStringLiteral("duration_seconds")] = service.durationSeconds;
    object[QStringLiteral("per_camera_residuals")] = service.perCameraResiduals;
    object[QStringLiteral("new_feature_files")] = featureFilesToJson(service.newFeatureFiles);
    object[QStringLiteral("new_match_files")] = matchFilesToJson(service.newMatchFiles);
    object[QStringLiteral("failed_pairs")] = failedPairsToJson(service.failedPairs);

    QJsonObject cameraUpdates;
    for (auto it = service.pendingCamUpdates.constBegin(); it != service.pendingCamUpdates.constEnd(); ++it)
    {
        cameraUpdates.insert(it.key(), it.value());
    }
    object[QStringLiteral("pending_camera_updates")] = cameraUpdates;
    return object;
}

QJsonObject serviceOptionsToJson(const xjw::gui::AerialTriangulationServiceOptions &service)
{
    QJsonObject object;
    object[QStringLiteral("output_dir")] = service.outputDir;
    object[QStringLiteral("plascan_path")] = service.plascanPath;
    object[QStringLiteral("image_count")] = service.images.size();
    object[QStringLiteral("camera_path_count")] = service.cameraPaths.size();
    object[QStringLiteral("quality")] = service.quality;
    object[QStringLiteral("threads")] = service.threads;
    object[QStringLiteral("device")] = service.device;
    object[QStringLiteral("feature_algorithm")] = service.featureAlgorithm;
    object[QStringLiteral("match_algorithm")] = service.matchAlgorithm;
    object[QStringLiteral("feature_max_image_dim")] = service.featureMaxImageDim;
    object[QStringLiteral("auto_generate_missing_matches")] = service.autoGenerateMissingMatches;
    object[QStringLiteral("restrict_pairs")] = service.restrictPairs;
    object[QStringLiteral("adaptive_camera_model_fitting")] = service.adaptiveCameraModelFitting;
    object[QStringLiteral("allowed_pair_count")] = service.allowedPairs.size();
    object[QStringLiteral("enable_guided_rematching")] = service.enableGuidedRematching;
    object[QStringLiteral("enable_two_stage_matching")] = service.enableTwoStageMatching;
    object[QStringLiteral("skeleton_feature_max_keypoints")] = service.skeletonFeatureMaxKeypoints;
    object[QStringLiteral("tie_point_feature_max_keypoints")] = service.tiePointFeatureMaxKeypoints;
    object[QStringLiteral("tie_point_keypoint_limit_per_megapixel")] =
        service.tiePointKeypointLimitPerMegapixel;
    object[QStringLiteral("use_tie_point_dense_sift")] = service.useTiePointDenseSift;
    object[QStringLiteral("max_tiepoints_per_image")] = service.maxTiePointsPerImage;
    object[QStringLiteral("max_tiepoints_per_grid_cell")] = service.maxTiePointsPerGridCell;
    object[QStringLiteral("known_camera_pair_window")] = service.knownCameraPairWindow;
    object[QStringLiteral("known_camera_sequence_loop_closure")] = service.knownCameraSequenceLoopClosure;
    object[QStringLiteral("use_known_camera_overlap_pairs")] = service.useKnownCameraOverlapPairs;
    return object;
}

QJsonObject makeDryRunReport(const xjw::gui::AerialTriangulationResolvedConfig &config,
                             const QString &outputDir,
                             int imageCount,
                             int cameraPathCount)
{
    QJsonObject report;
    report[QStringLiteral("success")] = true;
    report[QStringLiteral("dry_run")] = true;
    report[QStringLiteral("image_count")] = imageCount;
    report[QStringLiteral("camera_path_count")] = cameraPathCount;
    report[QStringLiteral("output_dir")] = outputDir;
    report[QStringLiteral("generated_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    report[QStringLiteral("resolved_settings")] = config.resolvedSettings;
    report[QStringLiteral("service_options")] = serviceOptionsToJson(config.serviceOptions);
    return report;
}

QJsonObject makeRunReport(const xjw::gui::AerialTriangulationWorkflowResult &result,
                          const QString &outputDir,
                          int imageCount,
                          int cameraPathCount,
                          double elapsedMs)
{
    QJsonObject report;
    report[QStringLiteral("success")] = result.serviceResult.success;
    report[QStringLiteral("dry_run")] = false;
    report[QStringLiteral("image_count")] = imageCount;
    report[QStringLiteral("camera_path_count")] = cameraPathCount;
    report[QStringLiteral("output_dir")] = outputDir;
    report[QStringLiteral("elapsed_ms")] = elapsedMs;
    report[QStringLiteral("generated_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    report[QStringLiteral("resolved_settings")] = result.config.resolvedSettings;
    report[QStringLiteral("service_options")] = serviceOptionsToJson(result.config.serviceOptions);
    report[QStringLiteral("service_result")] = serviceResultToJson(result.serviceResult);
    report[QStringLiteral("registered_images")] = result.serviceResult.numRegisteredImages;
    report[QStringLiteral("points3d")] = result.serviceResult.numPoints3D;
    report[QStringLiteral("mean_reproj_error")] = result.serviceResult.meanReprojError;
    report[QStringLiteral("sparse_cloud_path")] = result.serviceResult.sparseCloudPath;
    report[QStringLiteral("summary")] = result.serviceResult.summary;
    report[QStringLiteral("error_message")] = result.serviceResult.errorMessage;
    return report;
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication qtApplication(argc, argv);
    CLI::App app{"PlaScan 空中三角测量 CLI — 对齐照片式特征/匹配/SfM/BA 流程"};

    std::string inputPath;
    std::string outputDirArg;
    std::string projectPathArg;
    std::string qualityArg = "high";
    std::string deviceArg = "auto";
    std::string referenceModeArg = "source_code";
    std::string featureAlgorithmArg = "sift";
    std::string matchAlgorithmArg = "lightglue";
    std::string matchPipelineArg;
    std::string maskApplyModeArg = "none";
    int keypointLimit = 40000;
    int tiepointLimit = 4000;
    int threads = std::max(1u, std::thread::hardware_concurrency());
    bool genericPreselection = true;
    bool noGenericPreselection = false;
    bool referencePreselection = false;
    bool resetAlignment = true;
    bool saveAfterEachStep = false;
    bool guidedImageMatching = false;
    bool adaptiveCameraModelFitting = true;
    bool noAdaptiveCameraModelFitting = false;
    bool excludeFixedTiePoints = true;
    bool includeFixedTiePoints = false;
    bool autoGenerateMissingMatches = true;
    bool noAutoGenerateMissingMatches = false;
    bool dryRunConfig = false;
    bool force = false;

    app.add_option("-i,--input", inputPath, "影像列表；每行支持 '<image>' 或 '<image> <camera.tsai>'")->required();
    app.add_option("-o,--output-dir", outputDirArg, "输出目录")->required();
    app.add_option("--project", projectPathArg, "无头项目路径，默认写到输出目录 headless.plascan");
    app.add_option("--quality", qualityArg, "精度: lowest, low, medium, high, highest");
    app.add_option("--device", deviceArg, "计算设备: auto, cpu, cuda");
    app.add_option("--reference-mode", referenceModeArg, "参考预选模式: source-code/source, estimated, sequence");
    app.add_option("--feature-algorithm", featureAlgorithmArg, "特征算法，默认 sift");
    app.add_option("--match-algorithm", matchAlgorithmArg, "匹配算法，默认 lightglue");
    app.add_option("--match-pipeline", matchPipelineArg, "匹配链路，如 sift-lightglue/sift-bf-l2");
    app.add_option("--keypoint-limit", keypointLimit, "关键点限制");
    app.add_option("--tiepoint-limit", tiepointLimit, "连接点限制");
    app.add_option("--mask-apply-mode", maskApplyModeArg, "蒙版应用阶段: none, keypoints, tiepoints");
    app.add_flag("--generic-preselection", genericPreselection, "启用通用预选");
    app.add_flag("--no-generic-preselection", noGenericPreselection, "禁用通用预选");
    app.add_flag("--reference-preselection", referencePreselection, "启用参考预选；没有相机文件时不可用");
    app.add_flag("--reset-alignment", resetAlignment, "重置当前对齐并重新准备连接点");
    app.add_flag("--save-after-each-step", saveAfterEachStep, "每个步骤完成后保存项目");
    app.add_flag("--guided-image-matching", guidedImageMatching, "启用指导图像匹配");
    app.add_flag("--adaptive-camera-model-fitting", adaptiveCameraModelFitting, "启用自适应相机模型拟合");
    app.add_flag("--no-adaptive-camera-model-fitting", noAdaptiveCameraModelFitting, "禁用自适应相机模型拟合");
    app.add_flag("--include-fixed-tie-points", includeFixedTiePoints, "包含固定/近静止连接点");
    app.add_flag("--exclude-fixed-tie-points", excludeFixedTiePoints, "排除固定/近静止连接点");
    app.add_flag("--auto-generate-missing-matches", autoGenerateMissingMatches, "缺少连接点时自动创建连接点");
    app.add_flag("--no-auto-generate-missing-matches", noAutoGenerateMissingMatches, "只复用已有特征/匹配缓存");
    app.add_option("--threads", threads, "CPU 工作线程数");
    app.add_flag("--dry-run-config", dryRunConfig, "只输出解析后的空三配置，不执行 SfM");
    app.add_flag("--force", force, "允许输出目录非空");

    CLI11_PARSE(app, argc, argv);

    const QString inputList = xjw::cli::cleanAbsolutePath(fromStdPath(inputPath));
    const QString outputDir = xjw::cli::cleanAbsolutePath(fromStdPath(outputDirArg));
    const QString projectPath = projectPathArg.empty()
        ? QDir(outputDir).filePath(QStringLiteral("headless.plascan"))
        : xjw::cli::cleanAbsolutePath(fromStdPath(projectPathArg));

    QString errorMessage;
    if (!xjw::cli::validateOutputDirectory(outputDir, force, &errorMessage) ||
        !xjw::cli::ensureDirectory(outputDir, &errorMessage))
    {
        std::fprintf(stderr, "错误: %s\n", qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }

    xjw::cli::PhotogrammetryListOptions listOptions;
    listOptions.allowImageOnlyRows = true;
    listOptions.requireExistingCameras = true;
    listOptions.loadCameras = false;
    std::vector<xjw::cli::PhotogrammetryInputItem> items;
    if (!xjw::cli::readPhotogrammetryImageList(inputList, listOptions, &items, &errorMessage))
    {
        std::fprintf(stderr, "列表读取失败: %s\n", qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }

    QStringList cameras = xjw::cli::cameraPathsForService(items);
    if (noGenericPreselection)
    {
        genericPreselection = false;
    }
    if (includeFixedTiePoints)
    {
        excludeFixedTiePoints = false;
    }
    if (noAutoGenerateMissingMatches)
    {
        autoGenerateMissingMatches = false;
    }
    if (noAdaptiveCameraModelFitting)
    {
        adaptiveCameraModelFitting = false;
    }
    if (referencePreselection && cameras.isEmpty())
    {
        std::fprintf(stderr, "错误: 参考预选需要完整相机文件；无相机场景请使用通用预选或序列策略。\n");
        return cli::EXIT_ARG_ERR;
    }

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    xjw::gui::AerialTriangulationWorkflowOptions options;
    options.images = xjw::cli::imagePaths(items);
    options.cameraPaths = cameras;
    options.projectPath = projectPath;
    options.outputDir = QDir(outputDir).filePath(QStringLiteral("assets/aerial_triangulation"));
    options.projectMeta = xjw::cli::projectMetaFromInputItems(items);
    options.quality = normalizedToken(fromStdPath(qualityArg), QStringLiteral("high"));
    options.genericPreselection = genericPreselection;
    options.referencePreselection = referencePreselection;
    options.referenceMode = referenceModeFromToken(fromStdPath(referenceModeArg));
    options.resetAlignment = resetAlignment;
    options.saveAfterEachStep = saveAfterEachStep;
    options.keypointLimit = std::max(0, keypointLimit);
    options.tiepointLimit = std::max(0, tiepointLimit);
    options.maskApplyMode = normalizedToken(fromStdPath(maskApplyModeArg), QStringLiteral("none"));
    options.excludeFixedTiePoints = excludeFixedTiePoints;
    options.guidedImageMatching = guidedImageMatching;
    options.adaptiveCameraModelFitting = adaptiveCameraModelFitting;
    options.featureAlgorithm = normalizedToken(fromStdPath(featureAlgorithmArg), QStringLiteral("sift"));
    options.matchAlgorithm = normalizedToken(fromStdPath(matchAlgorithmArg), QStringLiteral("lightglue"));
    options.matchPipeline = normalizedToken(fromStdPath(matchPipelineArg), QString());
    options.device = normalizedToken(fromStdPath(deviceArg), QStringLiteral("auto"));
    options.threads = std::max(1, threads);
    options.autoGenerateMissingMatches = autoGenerateMissingMatches;
    options.cancelFlag = cancelFlag;
    options.progressFn = [](const QString &stage, int percent)
    {
        std::fprintf(stdout, "[%3d%%] %s\n", percent, qUtf8Printable(stage));
        std::fflush(stdout);
    };
    options.pairMatchedFn = [](const QString &img0, const QString &img1, const QString &matchPath, int numMatches)
    {
        std::fprintf(stdout,
                     "[pair] %s <-> %s matches=%d path=%s\n",
                     qUtf8Printable(QFileInfo(img0).fileName()),
                     qUtf8Printable(QFileInfo(img1).fileName()),
                     numMatches,
                     qUtf8Printable(matchPath));
        std::fflush(stdout);
    };

    const QString reportPath = QDir(outputDir).filePath(QStringLiteral("aerial_triangulation_cli_report.json"));
    if (dryRunConfig)
    {
        const xjw::gui::AerialTriangulationResolvedConfig config =
            xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);
        const QJsonObject report = makeDryRunReport(config,
                                                    outputDir,
                                                    options.images.size(),
                                                    options.cameraPaths.size());
        if (!xjw::cli::writeJsonFile(reportPath, report, &errorMessage))
        {
            std::fprintf(stderr, "报告写入失败: %s\n", qUtf8Printable(errorMessage));
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stdout, "status=ok\n");
        std::fprintf(stdout, "dry_run=true\n");
        std::fprintf(stdout, "images=%d\n", static_cast<int>(options.images.size()));
        std::fprintf(stdout, "aerial_triangulation_cli_report.json=%s\n", qUtf8Printable(reportPath));
        return cli::EXIT_OK;
    }

    QElapsedTimer timer;
    timer.start();
    const xjw::gui::AerialTriangulationWorkflowResult result =
        xjw::gui::AerialTriangulationWorkflow::run(
            options,
            [](const xjw::gui::AerialTriangulationServiceOptions &serviceOptions)
            {
                return xjw::gui::AerialTriangulationService::run(serviceOptions);
            });

    const QJsonObject report = makeRunReport(result,
                                             outputDir,
                                             options.images.size(),
                                             options.cameraPaths.size(),
                                             static_cast<double>(timer.elapsed()));
    if (!xjw::cli::writeJsonFile(reportPath, report, &errorMessage))
    {
        std::fprintf(stderr, "报告写入失败: %s\n", qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }

    std::fprintf(stdout, "status=%s\n", result.serviceResult.success ? "ok" : "failed");
    std::fprintf(stdout, "registered_images=%d\n", result.serviceResult.numRegisteredImages);
    std::fprintf(stdout, "points3d=%d\n", result.serviceResult.numPoints3D);
    std::fprintf(stdout, "mean_reproj_error=%.6f\n", result.serviceResult.meanReprojError);
    if (!result.serviceResult.sparseCloudPath.isEmpty())
    {
        std::fprintf(stdout, "sparse_cloud=%s\n", qUtf8Printable(result.serviceResult.sparseCloudPath));
    }
    std::fprintf(stdout, "aerial_triangulation_cli_report.json=%s\n", qUtf8Printable(reportPath));

    if (!result.serviceResult.success)
    {
        std::fprintf(stderr, "空中三角测量失败: %s\n", qUtf8Printable(result.serviceResult.errorMessage));
        return cli::EXIT_ALGO_ERR;
    }
    return cli::EXIT_OK;
}
