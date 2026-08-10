/**
 * @file cli_aerial_triangulation.cpp
 * @brief GUI“工作流程 > 空中三角测量”的无界面入口。
 *
 * 本文件负责 CLI 层的流程编排：解析参数、打开或创建 PlaScan 项目、按需准备匹配/连接点，
 * 调用 AerialTriangulationWorkflow 完成增量 SfM 与光束法平差（BA），最后把稀疏点云、相机和报告写回项目。
 *
 * 该流程在稀疏重建结束后停止，不执行 MVS 深度估计、稠密点云、网格、纹理或 DEM/DOM。
 */
#include "cli_common.h"
#include "cli_photogrammetry_common.h"
#include "CliTokenUtils.h"
#include "FinalBaCameraExporter.h"

#include "workflow/AerialTriangulationWorkflow.h"
#include "project/ProjectSession.h"

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

// 接受 CLI 中便于输入的别名，并转换为 workflow 使用的稳定枚举字符串。
QString referenceModeFromToken(const QString &value)
{
    const QString token = xjw::cli::normalizedToken(value, QStringLiteral("source_code"));
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

// 以下序列化函数把 workflow 的配置和结果转换为稳定的 JSON 字段，
// 供 CLI 报告、自动化脚本以及项目结果记录共同使用。
QJsonArray pairDiagnosticsToJson(
    const std::vector<xjw::matchphotos::MatchPhotosMatchRecord> &records)
{
    QJsonArray array;
    for (const xjw::matchphotos::MatchPhotosMatchRecord &record : records)
    {
        QJsonObject object;
        object[QStringLiteral("image0_path")] = record.image0Path;
        object[QStringLiteral("image1_path")] = record.image1Path;
        object[QStringLiteral("image0_match_file")] = record.image0MatchFilePath;
        object[QStringLiteral("image1_match_file")] = record.image1MatchFilePath;
        object[QStringLiteral("algorithm_id")] = record.algorithmId;
        object[QStringLiteral("algorithm_version")] =
            static_cast<qint64>(record.algorithmVersion);
        object[QStringLiteral("match_count")] = record.matchCount;
        object[QStringLiteral("geometric_inlier_count")] = record.geometricInlierCount;
        object[QStringLiteral("passed_geometry")] = record.passedGeometry;
        object[QStringLiteral("settings")] = record.settings;
        array.append(object);
    }
    return array;
}

QJsonArray imageMatchFilesToJson(
    const std::vector<xjw::matchphotos::MatchPhotosImageMatchRecord> &records)
{
    QJsonArray array;
    for (const xjw::matchphotos::MatchPhotosImageMatchRecord &record : records)
    {
        QJsonObject object;
        object[QStringLiteral("image")] = record.imagePath;
        object[QStringLiteral("output")] = record.matchFilePath;
        object[QStringLiteral("neighbors")] =
            QJsonArray::fromStringList(record.neighborImagePaths);
        object[QStringLiteral("neighbor_variant_count")] =
            record.neighborVariantCount;
        object[QStringLiteral("settings")] = record.settings;
        array.append(object);
    }
    return array;
}

QJsonObject tiePointResultToJson(const xjw::matchphotos::MatchPhotosResult &tiePoints)
{
    QJsonObject object;
    object[QStringLiteral("success")] = tiePoints.success;
    object[QStringLiteral("error_message")] = tiePoints.errorMessage;
    object[QStringLiteral("tie_point_path")] = tiePoints.tiePointPath;
    object[QStringLiteral("track_count")] = tiePoints.trackCount;
    object[QStringLiteral("accepted_track_components")] = tiePoints.acceptedTrackComponents;
    object[QStringLiteral("rejected_track_conflict_components")] =
        tiePoints.rejectedTrackConflictComponents;
    object[QStringLiteral("candidate_pair_count")] =
        static_cast<int>(tiePoints.pairSelection.candidates.size());
    object[QStringLiteral("pair_diagnostics")] =
        pairDiagnosticsToJson(tiePoints.matches);
    object[QStringLiteral("image_match_files")] =
        imageMatchFilesToJson(tiePoints.imageMatchFiles);
    object[QStringLiteral("track_summary")] = tiePoints.trackSummary;
    return object;
}

QJsonObject reconstructionResultToJson(
    const xjw::aerial_triangulation::AerialTriangulationReconstructionResult &result)
{
    QJsonObject object;
    object[QStringLiteral("success")] = result.success;
    object[QStringLiteral("error_message")] = result.errorMessage;
    object[QStringLiteral("summary")] = result.summary;
    object[QStringLiteral("registered_images")] = result.numRegisteredImages;
    object[QStringLiteral("points3d")] = result.numPoints3D;
    object[QStringLiteral("mean_reproj_error")] = result.meanReprojError;
    object[QStringLiteral("sparse_cloud_path")] = result.sparseCloudPath;
    object[QStringLiteral("quality_metadata")] = result.qualityMetadata;
    object[QStringLiteral("result_record_extra")] = result.resultRecordExtra;
    object[QStringLiteral("sfm_diagnostics")] = result.sfmDiagnostics;
    object[QStringLiteral("ba_rms_before")] = result.baRmsBefore;
    object[QStringLiteral("ba_rms_after")] = result.baRmsAfter;
    object[QStringLiteral("ba_tracks_total")] = result.baTracksTotal;
    object[QStringLiteral("ba_tracks_optimized")] = result.baTracksOptimized;
    object[QStringLiteral("ba_tracks_filtered")] = result.baTracksFiltered;
    object[QStringLiteral("duration_seconds")] = result.durationSeconds;
    object[QStringLiteral("per_camera_residuals")] = result.perCameraResiduals;

    QJsonObject cameraUpdates;
    for (auto it = result.pendingCamUpdates.constBegin(); it != result.pendingCamUpdates.constEnd(); ++it)
    {
        cameraUpdates.insert(it.key(), it.value());
    }
    object[QStringLiteral("pending_camera_updates")] = cameraUpdates;
    return object;
}

QJsonObject pipelineInputToJson(
    const xjw::aerial_triangulation::PreparedAerialTriangulationInput &input)
{
    QJsonObject object;
    object[QStringLiteral("output_dir")] = input.outputDir;
    object[QStringLiteral("project_path")] = input.projectPath;
    object[QStringLiteral("marker_set_path")] = input.markerSetPath;
    object[QStringLiteral("tie_point_path")] = input.tiePointPath;
    object[QStringLiteral("image_count")] = input.images.size();
    object[QStringLiteral("camera_path_count")] = input.cameraPaths.size();
    object[QStringLiteral("quality")] = input.quality;
    object[QStringLiteral("threads")] = input.threads;
    object[QStringLiteral("device")] = input.device;
    object[QStringLiteral("use_project_camera_intrinsics")] = input.useProjectCameraIntrinsics;
    object[QStringLiteral("use_project_camera_poses")] = input.useProjectCameraPoses;
    object[QStringLiteral("adaptive_camera_model_fitting")] = input.adaptiveCameraModelFitting;
    object[QStringLiteral("use_sequence_pose_recovery")] = input.useSequencePoseRecovery;
    object[QStringLiteral("enforce_sequence_pose_consistency")] = input.enforceSequencePoseConsistency;
    object[QStringLiteral("sequence_loop_closure")] = input.sequenceLoopClosure;
    object[QStringLiteral("use_initial_pair_hint")] = input.useInitialPairHint;
    object[QStringLiteral("initial_image_id_1")] = static_cast<qint64>(input.initialImageId1);
    object[QStringLiteral("initial_image_id_2")] = static_cast<qint64>(input.initialImageId2);
    object[QStringLiteral("estimated_focal_scale")] = input.estimatedFocalScale;
    return object;
}

QJsonObject tiePointOptionsToJson(const xjw::matchphotos::MatchPhotosOptions &options)
{
    QJsonObject object;
    object[QStringLiteral("algorithm_id")] = options.algorithmId;
    object[QStringLiteral("mask_apply_mode")] = options.maskApplyMode;
    object[QStringLiteral("max_image_dimension")] = options.maxImageDim;
    object[QStringLiteral("max_keypoints")] = options.maxKeypoints;
    object[QStringLiteral("keypoint_limit_per_megapixel")] = options.keypointLimitPerMegapixel;
    object[QStringLiteral("max_tie_points_per_image")] = options.maxTiePointsPerImage;
    object[QStringLiteral("guided_image_matching")] = options.enableGuidedMatching;
    object[QStringLiteral("generic_preselection")] = options.useGenericPreselection;
    object[QStringLiteral("reference_preselection")] = options.useReferencePreselection;
    object[QStringLiteral("exclude_stationary_tie_points")] = options.excludeStationaryTiePoints;
    object[QStringLiteral("reuse_existing_matches")] = options.reuseExistingMatches;
    return object;
}

QJsonObject tiePointContextToJson(const xjw::aerial_triangulation::AerialTriangulationResolvedConfig &config)
{
    QJsonObject object;
    object[QStringLiteral("preparation_required")] = config.prepareTiePoints;
    object[QStringLiteral("force_rebuild")] = config.forceRebuildTiePoints;
    object[QStringLiteral("working_dir")] = config.tiePointContext.workingDirectory;
    object[QStringLiteral("match_dir")] = config.tiePointContext.matchDirectory;
    object[QStringLiteral("mask_count")] = config.tiePointContext.maskPaths.size();
    object[QStringLiteral("reference_camera_count")] = config.tiePointContext.referenceCameras.size();
    return object;
}

// dry-run 只报告最终解析出的配置，不运行连接点准备、SfM 或 BA。
QJsonObject makeDryRunReport(const xjw::aerial_triangulation::AerialTriangulationResolvedConfig &config,
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
    report[QStringLiteral("pipeline_input")] = pipelineInputToJson(config.pipelineInput);
    report[QStringLiteral("tie_point_options")] = tiePointOptionsToJson(config.tiePointOptions);
    report[QStringLiteral("tie_point_context")] = tiePointContextToJson(config);
    report[QStringLiteral("tie_point_preparation_executed")] = false;
    return report;
}

// 正式运行报告同时保留连接点前端和稀疏重建后端的诊断信息。
QJsonObject makeRunReport(const xjw::aerial_triangulation::AerialTriangulationResult &result,
                          const QString &outputDir,
                          int imageCount,
                          int cameraPathCount,
                          double elapsedMs)
{
    QJsonObject report;
    report[QStringLiteral("success")] = result.reconstructionResult.success;
    report[QStringLiteral("dry_run")] = false;
    report[QStringLiteral("image_count")] = imageCount;
    report[QStringLiteral("camera_path_count")] = cameraPathCount;
    report[QStringLiteral("output_dir")] = outputDir;
    report[QStringLiteral("elapsed_ms")] = elapsedMs;
    report[QStringLiteral("generated_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    report[QStringLiteral("resolved_settings")] = result.config.resolvedSettings;
    report[QStringLiteral("pipeline_input")] = pipelineInputToJson(result.config.pipelineInput);
    report[QStringLiteral("tie_point_options")] = tiePointOptionsToJson(result.config.tiePointOptions);
    report[QStringLiteral("tie_point_context")] = tiePointContextToJson(result.config);
    report[QStringLiteral("tie_point_preparation_executed")] = result.tiePointPreparationExecuted;
    report[QStringLiteral("tie_point_success")] = result.tiePointResult.success;
    report[QStringLiteral("tie_point_path")] = result.tiePointResult.tiePointPath;
    report[QStringLiteral("tie_point_track_count")] = result.tiePointResult.trackCount;
    report[QStringLiteral("tie_point_match_count")] =
        static_cast<int>(result.tiePointResult.matches.size());
    report[QStringLiteral("tie_point_candidate_pair_count")] =
        static_cast<int>(result.tiePointResult.pairSelection.candidates.size());
    report[QStringLiteral("tie_point_result")] = tiePointResultToJson(result.tiePointResult);
    report[QStringLiteral("reconstruction_result")] =
        reconstructionResultToJson(result.reconstructionResult);
    report[QStringLiteral("registered_images")] = result.reconstructionResult.numRegisteredImages;
    report[QStringLiteral("points3d")] = result.reconstructionResult.numPoints3D;
    report[QStringLiteral("mean_reproj_error")] = result.reconstructionResult.meanReprojError;
    report[QStringLiteral("sparse_cloud_path")] = result.reconstructionResult.sparseCloudPath;
    report[QStringLiteral("summary")] = result.reconstructionResult.summary;
    report[QStringLiteral("error_message")] = result.reconstructionResult.errorMessage;
    return report;
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication qtApplication(argc, argv);
    CLI::App app{"PlaScan 空中三角测量 CLI — 对齐照片式特征/匹配/SfM/BA 流程"};

    // 阶段 1：声明 CLI 参数。默认值尽量与 GUI 的空中三角测量工作流保持一致。
    std::string inputPath;
    std::string outputDirArg;
    std::string exportCameraDirArg;
    std::string projectPathArg;
    std::string chunkIdArg;
    std::string chunkNameArg;
    std::string assetsDirArg;
    std::string matchDirArg;
    std::string qualityArg = "high";
    std::string deviceArg = "auto";
    std::string referenceModeArg = "source_code";
    std::string algorithmIdArg = "sift_lightglue";
    std::string lightGlueEngineArg;
    std::string lomaRPackageArg;
    std::string maskApplyModeArg = "none";
    std::string maskDirArg;
    int keypointLimit = 40000;
    int tiepointLimit = 4000;
    int initialImageId1 = -1;
    int initialImageId2 = -1;
    int threads = std::max(1u, std::thread::hardware_concurrency());
    bool genericPreselection = true;
    bool noGenericPreselection = false;
    bool referencePreselection = false;
    bool resetAlignment = true;
    bool noResetAlignment = false;
    bool noReuseExistingMatches = false;
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
    app.add_option("--export-camera-dir", exportCameraDirArg,
                   "正式 SfM/BA 成功后导出 cameras/*.tsai 和 image_camera.lis；目录必须不存在");
    app.add_option("--project", projectPathArg, "无头项目路径，默认写到输出目录 headless.plascan");
    app.add_option("--chunk-id", chunkIdArg, "使用指定 UUID 的 Chunk");
    app.add_option("--chunk-name", chunkNameArg, "使用指定名称的 Chunk");
    app.add_option("--assets-dir", assetsDirArg, "复用特征与匹配的 assets 目录，默认使用输出目录下的 assets");
    app.add_option("--match-dir", matchDirArg,
                   "逐影像匹配目录，默认使用 assets/image_matches");
    app.add_option("--quality", qualityArg, "精度: lowest, low, medium, high, highest");
    app.add_option("--device", deviceArg, "计算设备: auto, cpu, cuda");
    app.add_option("--reference-mode", referenceModeArg, "参考预选模式: source-code/source, estimated, sequence");
    app.add_option("--algorithm-id", algorithmIdArg,
                   "统一影像匹配算法 ID: sift_lightglue, cuda_sift, loma_r")
        ->check(CLI::IsMember({"sift_lightglue", "cuda_sift", "loma_r"}));
    app.add_option("--lightglue-engine", lightGlueEngineArg,
                   "TensorRT LightGlue .engine；留空时按模型目录自动查找");
    app.add_option("--loma-r-package", lomaRPackageArg,
                   "LoMa-R TensorRT JSON 清单；留空时按模型目录自动查找");
    app.add_option("--keypoint-limit", keypointLimit, "关键点限制");
    app.add_option("--tiepoint-limit", tiepointLimit, "连接点限制");
    app.add_option("--initial-image-id-1", initialImageId1, "指定初始像对的第一个 0 基影像索引");
    app.add_option("--initial-image-id-2", initialImageId2, "指定初始像对的第二个 0 基影像索引");
    app.add_option("--mask-apply-mode", maskApplyModeArg, "蒙版应用阶段: none, keypoints, tiepoints");
    app.add_option("--mask-dir", maskDirArg, "蒙版目录，按影像文件名或 *_mask 文件名匹配");
    app.add_flag("--generic-preselection", genericPreselection, "启用通用预选");
    app.add_flag("--no-generic-preselection", noGenericPreselection, "禁用通用预选");
    app.add_flag("--reference-preselection", referencePreselection, "启用参考预选；source/estimated 需要相机文件，sequence 可无相机");
    app.add_flag("--reset-alignment", resetAlignment, "忽略当前相机外方位并重新执行 SfM/BA");
    app.add_flag("--no-reset-alignment", noResetAlignment, "以当前相机位姿作为 SfM/BA 初值");
    app.add_flag("--no-reuse-existing-matches",
                 noReuseExistingMatches,
                 "删除当前匹配和连接点缓存，重新提取、匹配并整理连接点");
    app.add_flag("--save-after-each-step", saveAfterEachStep, "每个步骤完成后保存项目");
    app.add_flag("--guided-image-matching", guidedImageMatching, "启用指导图像匹配");
    app.add_flag("--adaptive-camera-model-fitting", adaptiveCameraModelFitting, "启用自适应相机模型拟合");
    app.add_flag("--no-adaptive-camera-model-fitting", noAdaptiveCameraModelFitting, "禁用自适应相机模型拟合");
    app.add_flag("--include-fixed-tie-points", includeFixedTiePoints, "包含固定/近静止连接点");
    app.add_flag("--exclude-fixed-tie-points", excludeFixedTiePoints, "排除固定/近静止连接点");
    app.add_flag("--auto-generate-missing-matches", autoGenerateMissingMatches, "缺少连接点时自动创建连接点");
    app.add_flag("--no-auto-generate-missing-matches", noAutoGenerateMissingMatches,
                 "只复用已有逐影像匹配分片和连接点");
    app.add_option("--threads", threads, "CPU 工作线程数");
    app.add_flag("--dry-run-config", dryRunConfig, "只输出解析后的空三配置，不执行 SfM");
    app.add_flag("--force", force, "允许输出目录非空");

    CLI11_PARSE(app, argc, argv);

    // 阶段 2：规范化路径并读取影像清单。清单可仅含影像，也可为每幅影像提供初始相机文件。
    const QString inputList = xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(inputPath));
    const QString outputDir = xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(outputDirArg));
    const QString requestedCameraExportDir = exportCameraDirArg.empty()
        ? QString()
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(exportCameraDirArg));
    const QString projectPath = projectPathArg.empty()
        ? QDir(outputDir).filePath(QStringLiteral("headless.plascan"))
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(projectPathArg));
    const QString requestedAssetsDir = assetsDirArg.empty()
        ? QString()
        : xjw::cli::cleanAbsolutePath(
              xjw::cli::fromStdString(assetsDirArg));

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
    listOptions.loadCameras = referencePreselection;
    std::vector<xjw::cli::PhotogrammetryInputItem> items;
    if (!xjw::cli::readPhotogrammetryImageList(inputList, listOptions, &items, &errorMessage))
    {
        std::fprintf(stderr, "列表读取失败: %s\n", qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }

    const bool hasInitialImage1 = initialImageId1 >= 0;
    const bool hasInitialImage2 = initialImageId2 >= 0;
    if (hasInitialImage1 != hasInitialImage2)
    {
        std::fprintf(stderr, "错误: --initial-image-id-1 与 --initial-image-id-2 必须同时指定。\n");
        return cli::EXIT_ARG_ERR;
    }
    if (hasInitialImage1 &&
        (initialImageId1 == initialImageId2 ||
         initialImageId1 >= static_cast<int>(items.size()) ||
         initialImageId2 >= static_cast<int>(items.size())))
    {
        std::fprintf(stderr,
                     "错误: 初始像对索引必须互不相同，且位于 [0, %d]。\n",
                     static_cast<int>(items.size()) - 1);
        return cli::EXIT_ARG_ERR;
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
    if (noResetAlignment)
    {
        resetAlignment = false;
    }
    if (noAdaptiveCameraModelFitting)
    {
        adaptiveCameraModelFitting = false;
    }
    const QString requestedReferenceMode = referenceModeFromToken(xjw::cli::fromStdString(referenceModeArg));
    if (referencePreselection && cameras.isEmpty() && requestedReferenceMode != QStringLiteral("sequence"))
    {
        std::fprintf(stderr, "错误: 参考预选需要完整相机文件；无相机场景请使用通用预选或序列策略。\n");
        return cli::EXIT_ARG_ERR;
    }

    // 阶段 3：打开或创建项目，选择目标 Chunk，并把本次影像合并进项目元数据。
    xjw::common::project::ProjectSession projectSession;
    if (!projectSession.openOrCreate(
            projectPath,
            QFileInfo(projectPath).completeBaseName(),
            &errorMessage))
    {
        std::fprintf(stderr,
                     "工程打开/创建失败: %s\n",
                     qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }
    if (!chunkIdArg.empty() && !chunkNameArg.empty())
    {
        std::fprintf(stderr, "错误: --chunk-id 与 --chunk-name 不能同时使用\n");
        return cli::EXIT_ARG_ERR;
    }
    if (!projectSession.selectChunk(
            xjw::cli::fromStdString(chunkIdArg),
            xjw::cli::fromStdString(chunkNameArg),
            &errorMessage))
    {
        std::fprintf(stderr,
                     "Chunk 选择失败: %s\n",
                     qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }
    const QString assetsDir = requestedAssetsDir.isEmpty()
        ? QDir(projectSession.activeChunkRoot())
              .filePath(QStringLiteral("assets"))
        : requestedAssetsDir;
    const QString reconstructionDir =
        QDir(projectSession.activeChunkRoot())
            .filePath(QStringLiteral("reconstruction/sparse"));

    // 阶段 4：把 CLI 参数映射到 GUI/CLI 共用的 workflow 配置，避免在入口层重复实现算法逻辑。
    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    xjw::aerial_triangulation::AerialTriangulationOptions options;
    options.images = xjw::cli::imagePaths(items);
    options.cameraPaths = cameras;
    options.projectPath = projectPath;
    options.assetsDir = assetsDir;
    options.matchDir = matchDirArg.empty()
        ? QDir(options.assetsDir).filePath(QStringLiteral("image_matches"))
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(matchDirArg));
    options.outputDir = reconstructionDir;
    options.projectMeta = projectSession.mergedMetadata();
    options.quality = xjw::cli::normalizedToken(qualityArg, QStringLiteral("high"));
    options.genericPreselection = genericPreselection;
    options.referencePreselection = referencePreselection;
    options.referenceMode = requestedReferenceMode;
    options.resetAlignment = resetAlignment;
    options.reuseExistingMatches = !noReuseExistingMatches;
    options.saveAfterEachStep = saveAfterEachStep;
    options.keypointLimit = std::max(0, keypointLimit);
    options.tiepointLimit = std::max(0, tiepointLimit);
    options.maskApplyMode = xjw::cli::normalizedToken(maskApplyModeArg, QStringLiteral("none"));
    options.excludeFixedTiePoints = excludeFixedTiePoints;
    options.guidedImageMatching = guidedImageMatching;
    options.adaptiveCameraModelFitting = adaptiveCameraModelFitting;
    options.useInitialPairHint = hasInitialImage1;
    if (hasInitialImage1)
    {
        options.initialImageId1 = static_cast<xjw::ImageId>(initialImageId1);
        options.initialImageId2 = static_cast<xjw::ImageId>(initialImageId2);
    }
    options.matchingAlgorithmId = xjw::cli::normalizedToken(
        algorithmIdArg, QStringLiteral("sift_lightglue"));
    options.lightGlueTensorRtEnginePath = lightGlueEngineArg.empty()
        ? QString()
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(lightGlueEngineArg));
    options.lomaRTensorRtPackagePath = lomaRPackageArg.empty()
        ? QString()
        : xjw::cli::cleanAbsolutePath(xjw::cli::fromStdString(lomaRPackageArg));
    options.device = xjw::cli::normalizedToken(deviceArg, QStringLiteral("auto"));
    options.threads = std::max(1, threads);
    options.autoGenerateMissingMatches = autoGenerateMissingMatches;
    options.referenceCameras = xjw::cli::referenceCameraMap(items);
    options.maskPaths = xjw::cli::maskPathsFromDirectory(
        xjw::cli::fromStdString(maskDirArg), options.images);
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

    const QString reportPath =
        QDir(projectSession.activeChunkRoot()).filePath(
            QStringLiteral(
                "reports/aerial_triangulation_cli_report.json"));

    // dry-run 只解析配置并写出轻量报告。必须位于 mergeImages 之前，否则检查
    // 444 张影像配置也会把数 GiB 原图复制进无头工程，既慢又违背 dry-run 语义。
    if (dryRunConfig)
    {
        const xjw::aerial_triangulation::AerialTriangulationResolvedConfig config =
            xjw::aerial_triangulation::AerialTriangulationWorkflow::resolveConfig(options);
        QJsonObject report = makeDryRunReport(config,
                                              outputDir,
                                              options.images.size(),
                                              options.cameraPaths.size());
        report[QStringLiteral("camera_export_requested")] =
            !requestedCameraExportDir.isEmpty();
        report[QStringLiteral("camera_export_performed")] = false;
        report[QStringLiteral("camera_export_dir")] = requestedCameraExportDir;
        if (!xjw::cli::writeJsonFile(reportPath, report, &errorMessage))
        {
            std::fprintf(stderr, "报告写入失败: %s\n", qUtf8Printable(errorMessage));
            return cli::EXIT_IO_ERR;
        }
        QJsonObject reportRecord = report;
        reportRecord[QStringLiteral("kind")] =
            QStringLiteral("aerial_triangulation_cli_dry_run");
        reportRecord[QStringLiteral("path")] = reportPath;
        projectSession.upsertResultByPath(
            QStringLiteral("report_results"),
            QStringLiteral("path"),
            reportRecord);
        if (!projectSession.save(&errorMessage))
        {
            std::fprintf(stderr,
                         "配置报告已生成，但 Chunk 写回失败: %s\n",
                         qUtf8Printable(errorMessage));
            return cli::EXIT_IO_ERR;
        }
        std::fprintf(stdout, "status=ok\n");
        std::fprintf(stdout, "dry_run=true\n");
        std::fprintf(stdout, "images=%d\n", static_cast<int>(options.images.size()));
        std::fprintf(stdout, "aerial_triangulation_cli_report.json=%s\n", qUtf8Printable(reportPath));
        return cli::EXIT_OK;
    }

    // 正式执行才把影像导入 Chunk。导入会管理工程内共享影像资源，因此不能提前到
    // dry-run 分支之前。导入后刷新 projectMeta，供相机内参和已有外参解析使用。
    if (!projectSession.mergeImages(
            xjw::cli::inputItemsToJson(items), &errorMessage)
        || !projectSession.save(&errorMessage))
    {
        std::fprintf(stderr,
                     "工程影像初始化失败: %s\n",
                     qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }
    options.projectMeta = projectSession.mergedMetadata();

    // 阶段 5：共享 workflow 先按需生成/复用连接点，再执行初始像对、增量注册、三角化和 BA。
    QElapsedTimer timer;
    timer.start();
    const xjw::aerial_triangulation::AerialTriangulationResult result =
        xjw::aerial_triangulation::AerialTriangulationWorkflow::run(options);

    xjw::cli::FinalBaCameraExportResult cameraExport;
    QString cameraExportError;
    bool cameraExportPerformed = false;
    if (result.reconstructionResult.success && !requestedCameraExportDir.isEmpty())
    {
        cameraExportPerformed = xjw::cli::exportFinalBaCameras(
            options.images,
            result.reconstructionResult.pendingCamUpdates,
            requestedCameraExportDir,
            &cameraExport,
            &cameraExportError);
    }

    QJsonObject report = makeRunReport(result,
                                       outputDir,
                                       options.images.size(),
                                       options.cameraPaths.size(),
                                       static_cast<double>(timer.elapsed()));
    report[QStringLiteral("camera_export_requested")] =
        !requestedCameraExportDir.isEmpty();
    report[QStringLiteral("camera_export_performed")] = cameraExportPerformed;
    report[QStringLiteral("camera_export_dir")] = requestedCameraExportDir;
    report[QStringLiteral("image_camera_list_path")] = cameraExport.imageCameraList;
    report[QStringLiteral("exported_camera_count")] = cameraExport.cameraPaths.size();
    report[QStringLiteral("camera_export_error")] = cameraExportError;
    if (!xjw::cli::writeJsonFile(reportPath, report, &errorMessage))
    {
        std::fprintf(stderr, "报告写入失败: %s\n", qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }
    // 阶段 6：工程仅登记逐影像最终分片。特征描述子是任务内临时数据，
    // 像对信息和残差已封装在 `.pimatch` 中，不再产生旧缓存数组。
    for (const xjw::matchphotos::MatchPhotosImageMatchRecord &image :
         result.tiePointResult.imageMatchFiles)
    {
        QJsonObject record{
            {QStringLiteral("image"), image.imagePath},
            {QStringLiteral("output"), image.matchFilePath},
            {QStringLiteral("neighbors"),
             QJsonArray::fromStringList(image.neighborImagePaths)},
            {QStringLiteral("neighbor_variant_count"), image.neighborVariantCount},
            {QStringLiteral("settings"), image.settings},
            {QStringLiteral("created_at"),
             QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)}
        };
        projectSession.upsertResultByPath(
            QStringLiteral("image_match_results"),
            QStringLiteral("output"),
            record);
    }

    // 阶段 7：登记稀疏点云并写回优化后的相机。这里是空三终点，不进入 MVS 或地形产品生产。
    const auto &reconstruction = result.reconstructionResult;
    if (!reconstruction.sparseCloudPath.isEmpty())
    {
        QJsonObject files{
            {QStringLiteral("sparse_cloud_xyz"),
             reconstruction.sparseCloudPath}
        };
        const QJsonObject extraFiles =
            reconstruction.resultRecordExtra
                .value(QStringLiteral("files")).toObject();
        for (auto it = extraFiles.constBegin();
             it != extraFiles.constEnd();
             ++it)
        {
            files.insert(it.key(), it.value());
        }
        QJsonObject record = reconstruction.resultRecordExtra;
        record.remove(QStringLiteral("files"));
        record[QStringLiteral("created_at")] =
            QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
        record[QStringLiteral("output_dir")] = reconstructionDir;
        record[QStringLiteral("sparse_point_count")] =
            reconstruction.numPoints3D;
        record[QStringLiteral("selected_images")] =
            QJsonArray::fromStringList(options.images);
        record[QStringLiteral("files")] = files;
        record[QStringLiteral("quality_metadata")] =
            reconstruction.qualityMetadata;
        projectSession.appendResult(
            QStringLiteral("aerial_triangulation_results"), record);
    }
    int updatedCameraCount = 0;
    if (reconstruction.success
        && !projectSession.updateImageCameras(
            reconstruction.pendingCamUpdates,
            &updatedCameraCount,
            &errorMessage))
    {
        std::fprintf(stderr,
                     "空三已完成，但相机写回失败: %s\n",
                     qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }
    QJsonObject reportRecord = report;
    reportRecord[QStringLiteral("kind")] =
        QStringLiteral("aerial_triangulation_cli");
    reportRecord[QStringLiteral("path")] = reportPath;
    projectSession.upsertResultByPath(
        QStringLiteral("report_results"),
        QStringLiteral("path"),
        reportRecord);
    if (!projectSession.save(&errorMessage))
    {
        std::fprintf(stderr,
                     "空三结果已生成，但 Chunk 写回失败: %s\n",
                     qUtf8Printable(errorMessage));
        return cli::EXIT_IO_ERR;
    }
    if (result.reconstructionResult.success
        && !requestedCameraExportDir.isEmpty()
        && !cameraExportPerformed)
    {
        std::fprintf(stderr,
                     "空三已完成并写回项目，但最终 BA 相机导出失败: %s\n",
                     qUtf8Printable(cameraExportError));
        return cli::EXIT_IO_ERR;
    }

    // 输出稳定的 key=value 摘要，便于批处理脚本读取；完整诊断信息保存在 JSON 报告中。
    std::fprintf(stdout, "status=%s\n", result.reconstructionResult.success ? "ok" : "failed");
    std::fprintf(stdout, "registered_images=%d\n", result.reconstructionResult.numRegisteredImages);
    std::fprintf(stdout, "points3d=%d\n", result.reconstructionResult.numPoints3D);
    std::fprintf(stdout, "mean_reproj_error=%.6f\n", result.reconstructionResult.meanReprojError);
    if (!result.reconstructionResult.sparseCloudPath.isEmpty())
    {
        std::fprintf(stdout, "sparse_cloud=%s\n", qUtf8Printable(result.reconstructionResult.sparseCloudPath));
    }
    if (cameraExportPerformed)
    {
        std::fprintf(stdout, "camera_export_dir=%s\n", qUtf8Printable(cameraExport.outputDir));
        std::fprintf(stdout,
                     "image_camera.lis=%s\n",
                     qUtf8Printable(cameraExport.imageCameraList));
        std::fprintf(stdout,
                     "exported_cameras=%d\n",
                     static_cast<int>(cameraExport.cameraPaths.size()));
    }
    std::fprintf(stdout, "aerial_triangulation_cli_report.json=%s\n", qUtf8Printable(reportPath));
    std::fprintf(stdout,
                 "project=%s\nchunk=%d\nupdated_cameras=%d\n",
                 qUtf8Printable(projectSession.projectPath()),
                 projectSession.activeChunk().directory,
                 updatedCameraCount);

    if (!result.reconstructionResult.success)
    {
        std::fprintf(stderr, "空中三角测量失败: %s\n", qUtf8Printable(result.reconstructionResult.errorMessage));
        return cli::EXIT_ALGO_ERR;
    }
    return cli::EXIT_OK;
}
