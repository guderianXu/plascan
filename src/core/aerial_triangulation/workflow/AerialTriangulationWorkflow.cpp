/**
 * @file AerialTriangulationWorkflow.cpp
 * @brief “对齐照片”外部参数到连接点任务和 SfM 管线的统一编排实现。
 *
 * 本文件只处理工作流语义：质量预设、路径、蒙版、预选、缓存复用和进度映射。
 * 特征/匹配算法由 MatchPhotosTask 实现，位姿恢复和 BA 由 AerialTriangulationPipeline
 * 实现。保持这条边界可以保证 GUI 与 CLI 使用同一套实际参数。
 */

#include "workflow/AerialTriangulationWorkflow.h"

#include "ImageMatchRepository.h"
#include "preparation/TiePointPreparation.h"
#include "project/ProjectIO.h"
#include "search/SfmSearchPolicy.h"
#include "workflow/AerialTriangulationPipeline.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSet>

#include <algorithm>
#include <cmath>
#include <utility>

namespace xjw::aerial_triangulation
{
namespace
{

struct QualityPreset
{
    int sfmQuality = 2; ///< 传给增量 SfM 的离散质量等级 [0, 3]。
    int maxImageDimension = 3072; ///< 特征提取阶段默认长边限制。
};

/// 将自由文本参数收敛为稳定小写 token。
QString normalizedToken(QString value, const QString &fallback = QString())
{
    value = value.trimmed().toLower();
    return value.isEmpty() ? fallback : value;
}

/// 把一个总体质量选项映射到 SfM 和特征阶段的联合预设。
QualityPreset presetForQuality(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest")) return {3, 4096};
    if (token == QStringLiteral("medium") || token == QStringLiteral("standard")) return {1, 2048};
    if (token == QStringLiteral("low")) return {0, 1600};
    if (token == QStringLiteral("lowest") || token == QStringLiteral("fast")) return {0, 1200};
    return {2, 3072};
}

/// 序列预选窗口随质量提高而扩大，以增加局部冗余。
int sequenceWindowForQuality(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest")) return 8;
    if (token == QStringLiteral("high")) return 6;
    if (token == QStringLiteral("medium") || token == QStringLiteral("standard")) return 4;
    if (token == QStringLiteral("low")) return 3;
    return 2;
}

matchphotos::MatchPhotosProfile matchProfile(const QString &quality)
{
    const QString token = normalizedToken(quality, QStringLiteral("high"));
    if (token == QStringLiteral("highest") || token == QStringLiteral("high"))
    {
        return matchphotos::MatchPhotosProfile::HighAccuracy;
    }
    if (token == QStringLiteral("low") || token == QStringLiteral("lowest") ||
        token == QStringLiteral("fast"))
    {
        return matchphotos::MatchPhotosProfile::Fast;
    }
    return matchphotos::MatchPhotosProfile::Auto;
}

matchphotos::PairSelectionPreset pairPreset(const QString &quality)
{
    switch (matchProfile(quality))
    {
    case matchphotos::MatchPhotosProfile::HighAccuracy:
        return matchphotos::PairSelectionPreset::HighAccuracy;
    case matchphotos::MatchPhotosProfile::Fast:
        return matchphotos::PairSelectionPreset::Fast;
    default:
        return matchphotos::PairSelectionPreset::Auto;
    }
}

/// 将 GUI/CLI 的设备文本映射为 MatchPhotosTask 设备枚举。
matchphotos::ComputeDevice computeDevice(const QString &device)
{
    const QString token = normalizedToken(device, QStringLiteral("auto"));
    if (token == QStringLiteral("cpu"))
    {
        return matchphotos::ComputeDevice::Cpu;
    }
    if (token == QStringLiteral("cuda") || token == QStringLiteral("gpu"))
    {
        return matchphotos::ComputeDevice::Cuda;
    }
    return matchphotos::ComputeDevice::Auto;
}

/**
 * @brief 把 MatchPhotosTask 的分阶段进度映射到空三总进度的前 35%。
 *
 * 每个阶段使用固定区间，避免影像数或 pair 数变化造成总体进度倒退。
 */
int tiePointProgress(const QString &stageId, int current, int maximum)
{
    struct Range { const char *id; int first; int last; };
    static constexpr Range ranges[] = {
        {"algorithm_selection", 1, 3}, {"feature", 3, 16},
        {"generic_preselection", 16, 21}, {"reference_preselection", 21, 25},
        {"pair_selection", 25, 28}, {"matching", 28, 33},
        {"geometry", 33, 34}, {"track_build", 34, 35}, {"guided_matching", 34, 35}};
    Range selected{"unknown", 1, 35};
    const QString token = normalizedToken(stageId, QStringLiteral("unknown"));
    for (const Range &range : ranges)
    {
        if (token == QLatin1String(range.id))
        {
            selected = range;
            break;
        }
    }
    const double fraction = std::clamp(static_cast<double>(std::max(0, current)) /
                                           std::max(1, maximum),
                                       0.0,
                                       1.0);
    return selected.first + static_cast<int>(std::round(
        (selected.last - selected.first) * fraction));
}

/**
 * @brief 清理当前影像集合对应的匹配变体和最终连接点文件。
 *
 * 匹配目录属于当前工程块，一个 `.pimatch` 文件对应一幅影像。重建连接点时
 * 通过唯一仓库入口清理这些分片；不存在独立特征文件或 JSON sidecar。
 */
bool clearTiePointCache(const AerialTriangulationResolvedConfig &config, QString *errorMessage)
{
    QStringList failed;
    image_matching::ImageMatchRepository repository(
        config.tiePointContext.matchDirectory);
    QString matchClearError;
    if (!repository.clear(&matchClearError))
    {
        failed.append(matchClearError);
    }
    const QString tiePointPath = config.pipelineInput.tiePointPath;
    if (QFileInfo::exists(tiePointPath) && !QFile::remove(tiePointPath))
    {
        failed.append(tiePointPath);
    }
    if (failed.isEmpty())
    {
        return true;
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("无法清理旧连接点缓存：\n%1")
            .arg(failed.join(QLatin1Char('\n')));
    }
    return false;
}

} // namespace

AerialTriangulationResolvedConfig AerialTriangulationWorkflow::resolveConfig(
    const AerialTriangulationOptions &options)
{
    AerialTriangulationResolvedConfig resolved;

    // 第一阶段：收敛统一算法标识和质量预设。算法是否存在、版本和设备要求
    // 由 MatchPhotosAlgorithmSelector 通过注册表验证，空三层不拆解算法名称。
    const QualityPreset quality = presetForQuality(options.quality);
    const QString algorithmId = normalizedToken(
        options.matchingAlgorithmId,
        QStringLiteral("sift_lightglue"));

    // 第二阶段：所有缓存和正式结果都从工程根目录推导，避免 GUI/CLI 路径分叉。
    const QString projectRoot =
        xjw::common::project::ProjectIO::projectRootFromPlascan(
            options.projectPath);
    const QString assetsDirectory = options.assetsDir.isEmpty()
        ? QDir(projectRoot).filePath(QStringLiteral("assets"))
        : QDir::cleanPath(options.assetsDir);
    const QString canonicalTiePointPath = QDir(assetsDirectory)
        .filePath(QStringLiteral("tie_points/latest_tie_points.json"));

    // 第三阶段：装配纯 SfM/BA 输入。resetAlignment 只影响外参复用，
    // 不隐式改变连接点缓存策略。
    PreparedAerialTriangulationInput &pipeline = resolved.pipelineInput;
    pipeline.images = options.images;
    pipeline.cameraPaths = options.cameraPaths;
    pipeline.projectPath = options.projectPath;
    pipeline.markerSetPath = options.projectPath.isEmpty()
        ? QString()
        : xjw::common::project::ProjectIO::markerSetPath(options.projectPath);
    pipeline.tiePointPath = canonicalTiePointPath;
    pipeline.outputDir = QDir(QDir::cleanPath(options.outputDir))
        .filePath(QStringLiteral("sfm_sparse"));
    pipeline.projectMeta = options.projectMeta;
    pipeline.quality = quality.sfmQuality;
    pipeline.threads = resolveSfmThreadBudget(options.threads);
    pipeline.device = normalizedToken(options.device, QStringLiteral("auto"));
    pipeline.useProjectCameraIntrinsics = true;
    pipeline.useProjectCameraPoses = !options.resetAlignment;
    pipeline.adaptiveCameraModelFitting = options.adaptiveCameraModelFitting;
    pipeline.lockInputCameraPoses = options.lockInputCameraPoses;
    // “照片序列”是像对候选和初始化的弱先验，不代表相机中心必须满足等距轨迹。
    // 对环拍、变焦或物体旋转序列施加硬距离门限，会把几何验证通过的 PnP 位姿误判为异常。
    pipeline.enforceSequencePoseConsistency = false;
    pipeline.sequenceLoopClosure = options.referencePreselection &&
        normalizedToken(options.referenceMode, QStringLiteral("source_code")) == QStringLiteral("sequence");
    pipeline.useInitialPairHint = options.useInitialPairHint;
    pipeline.initialImageId1 = options.initialImageId1;
    pipeline.initialImageId2 = options.initialImageId2;
    pipeline.cancelFlag = options.cancelFlag;
    if (options.progressFn)
    {
        pipeline.progressFn = [progress = options.progressFn](const QString &stage, int percent)
        {
            progress(stage, 35 + std::clamp(percent, 0, 100) * 65 / 100);
        };
    }

    // 第四阶段：装配 MatchPhotosTask 参数。连接点上限在多视轨迹形成后按影像和
    // 网格分配；关键点上限则约束每张影像的检测结果。
    matchphotos::MatchPhotosOptions &tieOptions = resolved.tiePointOptions;
    tieOptions.planOnly = false;
    tieOptions.profile = matchProfile(options.quality);
    tieOptions.device = computeDevice(options.device);
    tieOptions.pairPolicy = matchphotos::makePairSelectionPolicy(pairPreset(options.quality));
    tieOptions.algorithmId = algorithmId;
    tieOptions.lightGlueTensorRtEnginePath =
        QDir::cleanPath(options.lightGlueTensorRtEnginePath.trimmed());
    if (options.lightGlueTensorRtEnginePath.trimmed().isEmpty())
    {
        tieOptions.lightGlueTensorRtEnginePath.clear();
    }
    tieOptions.lomaRTensorRtPackagePath =
        QDir::cleanPath(options.lomaRTensorRtPackagePath.trimmed());
    if (options.lomaRTensorRtPackagePath.trimmed().isEmpty())
    {
        tieOptions.lomaRTensorRtPackagePath.clear();
    }
    tieOptions.lomaRKeypointBudget = options.lomaRKeypointBudget;
    tieOptions.maskApplyMode = normalizedToken(options.maskApplyMode, QStringLiteral("none"));
    tieOptions.cudaDevice = std::max(0, options.cudaDevice);
    tieOptions.cudaParallelPairs = std::max(0, options.cudaParallelPairs);
    tieOptions.featurePrefetchDepth = std::clamp(options.featurePrefetchDepth, 1, 4);
    tieOptions.maxImageDim = options.featureMaxImageDim <= 0
        ? quality.maxImageDimension : options.featureMaxImageDim;
    tieOptions.matchThreshold = std::clamp(options.matchThreshold, 0.0f, 1.0f);
    tieOptions.geometryReprojThreshold = std::max(0.1, options.geometryReprojThreshold);
    tieOptions.geometryMinInliers = std::max(8, options.geometryMinInliers);
    tieOptions.geometryMaxIterations = std::max(100, options.geometryMaxIterations);
    tieOptions.enableGuidedMatching = options.guidedImageMatching;
    tieOptions.useExplicitKeypointLimit = true;
    tieOptions.maxKeypoints = std::max(0, options.keypointLimit);
    tieOptions.keypointLimitPerMegapixel = 0;
    tieOptions.maxTiePointsPerImage = std::max(0, options.tiepointLimit);
    tieOptions.tiePointGridColumns = std::clamp(options.tiePointGridColumns, 1, 64);
    tieOptions.tiePointGridRows = std::clamp(options.tiePointGridRows, 1, 64);
    const int tiePointGridCellCount = tieOptions.tiePointGridColumns * tieOptions.tiePointGridRows;
    tieOptions.maxTiePointsPerGridCell = options.maxTiePointsPerGridCell > 0
        ? options.maxTiePointsPerGridCell
        : (options.tiepointLimit > 0
               ? std::max(1, options.tiepointLimit / std::max(1, tiePointGridCellCount))
               : 0);
    tieOptions.useGenericPreselection = options.genericPreselection;
    tieOptions.useReferencePreselection = options.referencePreselection;
    tieOptions.excludeStationaryTiePoints = options.excludeFixedTiePoints;
    tieOptions.stationaryTiePointMaxPixelMotion = std::max(
        0.0f, options.stationaryTiePointMaxPixelMotion);
    tieOptions.reuseExistingMatches = options.reuseExistingMatches &&
        tieOptions.maskApplyMode != QStringLiteral("keypoints");

    // 第五阶段：解析 pair 预选。无参考相机时不得启用位姿参考预选；照片序列
    // 使用索引窗口和首尾闭环，不伪造参考相机文件。
    const QString referenceMode = normalizedToken(options.referenceMode,
                                                   QStringLiteral("source_code"));
    const bool hasReference = !options.referenceCameras.isEmpty() ||
        (!options.cameraPaths.isEmpty() && options.cameraPaths.size() == options.images.size());
    QString pairPlanningMode = options.genericPreselection ? QStringLiteral("generic")
                                                           : QStringLiteral("all_pairs");
    if (options.referencePreselection && referenceMode == QStringLiteral("sequence"))
    {
        tieOptions.pairPolicy.mode = matchphotos::PairSelectionMode::Sequence;
        tieOptions.pairPolicy.sequenceWindow = sequenceWindowForQuality(options.quality);
        tieOptions.pairPolicy.closeSequenceLoop = true;
        tieOptions.useReferencePreselection = false;
        pairPlanningMode = QStringLiteral("sequence");
    }
    else if (options.referencePreselection && hasReference)
    {
        pairPlanningMode = referenceMode;
    }
    else if (options.referencePreselection)
    {
        tieOptions.useReferencePreselection = false;
        pairPlanningMode = QStringLiteral("generic_reference_unavailable");
    }
    if (!options.allowedPairs.isEmpty())
    {
        tieOptions.pairPolicy.mode = matchphotos::PairSelectionMode::ManualOnly;
        pairPlanningMode = QStringLiteral("manual");
    }

    // 第六阶段：提供项目路径、缓存目录、蒙版及进度回调等运行时上下文。
    matchphotos::MatchPhotosContext &tieContext = resolved.tiePointContext;
    tieContext.projectPath = options.projectPath;
    tieContext.workingDirectory = assetsDirectory;
    tieContext.matchDirectory = options.matchDir.isEmpty()
        ? QDir(assetsDirectory).filePath(QStringLiteral("image_matches"))
        : QDir::cleanPath(options.matchDir);
    tieContext.pairInput.images = options.images;
    tieContext.pairInput.manualPairKeys = options.allowedPairs;
    tieContext.referenceCameras = options.referenceCameras;
    tieContext.maskPaths = options.maskPaths;
    tieContext.cancelFlag = options.cancelFlag.get();
    if (options.progressFn)
    {
        tieContext.progressCallback = [progress = options.progressFn](const QString &stageId,
                                                                      const QString &message,
                                                                      int current,
                                                                      int maximum)
        {
            progress(message.isEmpty() ? stageId : message,
                     tiePointProgress(stageId, current, maximum));
        };
    }

    // 最后独立决定“是否执行连接点任务”和“是否先清缓存”。重置对齐并不等于
    // 重匹配；用户可以重置相机后继续使用同一套匹配/连接点观测。
    resolved.prepareTiePoints = !options.reuseExistingMatches ||
        options.autoGenerateMissingMatches ||
        !QFileInfo::exists(canonicalTiePointPath);
    resolved.forceRebuildTiePoints = !options.reuseExistingMatches;
    QJsonObject &settings = resolved.resolvedSettings;
    settings.insert(QStringLiteral("quality"), normalizedToken(options.quality, QStringLiteral("high")));
    // 特征提取和匹配现在属于同一个可版本化算法实现，配置中只记录稳定算法 ID。
    settings.insert(QStringLiteral("matching_algorithm_id"), algorithmId);
    settings.insert(QStringLiteral("lightglue_tensorrt_engine"),
                    tieOptions.lightGlueTensorRtEnginePath);
    settings.insert(QStringLiteral("loma_r_tensorrt_package"),
                    tieOptions.lomaRTensorRtPackagePath);
    settings.insert(QStringLiteral("loma_r_keypoint_budget"),
                    tieOptions.lomaRKeypointBudget);
    settings.insert(QStringLiteral("pair_planning_mode"), pairPlanningMode);
    settings.insert(QStringLiteral("sequence_pair_window"), tieOptions.pairPolicy.sequenceWindow);
    settings.insert(QStringLiteral("sequence_loop_closure"), pipeline.sequenceLoopClosure);
    settings.insert(QStringLiteral("keypoint_limit"), options.keypointLimit);
    settings.insert(QStringLiteral("tiepoint_limit"), options.tiepointLimit);
    settings.insert(QStringLiteral("mask_apply_mode"), tieOptions.maskApplyMode);
    settings.insert(QStringLiteral("reset_current_alignment"), options.resetAlignment);
    settings.insert(QStringLiteral("reuse_existing_matches"), options.reuseExistingMatches);
    settings.insert(QStringLiteral("use_project_camera_intrinsics"),
                    pipeline.useProjectCameraIntrinsics);
    settings.insert(QStringLiteral("use_project_camera_poses"), pipeline.useProjectCameraPoses);
    settings.insert(QStringLiteral("adaptive_camera_model_fitting"),
                    options.adaptiveCameraModelFitting);
    settings.insert(QStringLiteral("lock_input_camera_poses"),
                    pipeline.lockInputCameraPoses);
    settings.insert(QStringLiteral("cuda_parallel_pairs_requested"), options.cudaParallelPairs);
    settings.insert(QStringLiteral("cuda_parallel_pairs_effective"), 0);
    settings.insert(QStringLiteral("cuda_device"), tieOptions.cudaDevice);
    settings.insert(QStringLiteral("threads"), pipeline.threads);
    settings.insert(QStringLiteral("feature_prefetch_depth"), tieOptions.featurePrefetchDepth);
    settings.insert(QStringLiteral("feature_max_image_dim"), tieOptions.maxImageDim);
    settings.insert(QStringLiteral("match_threshold"), tieOptions.matchThreshold);
    settings.insert(QStringLiteral("geometry_reprojection_threshold_px"),
                    tieOptions.geometryReprojThreshold);
    settings.insert(QStringLiteral("geometry_min_inliers"), tieOptions.geometryMinInliers);
    settings.insert(QStringLiteral("geometry_max_iterations"), tieOptions.geometryMaxIterations);
    settings.insert(QStringLiteral("tie_point_grid_columns"), tieOptions.tiePointGridColumns);
    settings.insert(QStringLiteral("tie_point_grid_rows"), tieOptions.tiePointGridRows);
    settings.insert(QStringLiteral("tie_point_grid_cell_limit"),
                    tieOptions.maxTiePointsPerGridCell);
    settings.insert(QStringLiteral("stationary_tie_point_max_pixel_motion"),
                    tieOptions.stationaryTiePointMaxPixelMotion);
    settings.insert(QStringLiteral("reference_preselection_available"),
                    referenceMode == QStringLiteral("sequence") || hasReference);
    settings.insert(QStringLiteral("tie_point_preparation"),
                    resolved.forceRebuildTiePoints
                        ? QStringLiteral("force_rebuild")
                        : (resolved.prepareTiePoints ? QStringLiteral("fill_missing")
                                                     : QStringLiteral("reuse")));
    return resolved;
}

AerialTriangulationResult AerialTriangulationWorkflow::run(
    const AerialTriangulationOptions &options)
{
    return run(options, [](const PreparedAerialTriangulationInput &input)
    {
        return AerialTriangulationPipeline().run(input);
    });
}

AerialTriangulationResult AerialTriangulationWorkflow::run(
    const AerialTriangulationOptions &options,
    const PipelineRunner &pipelineRunner,
    const TiePointRunner &tiePointRunner)
{
    AerialTriangulationResult result;
    result.config = resolveConfig(options);
    if (!pipelineRunner)
    {
        result.reconstructionResult.errorMessage =
            QStringLiteral("空中三角测量 workflow 缺少 SfM 管线执行器");
        result.reconstructionResult.summary = result.reconstructionResult.errorMessage;
        return result;
    }

    // 连接点准备是可选前置阶段：复用完整缓存时直接进入 SfM。
    if (result.config.prepareTiePoints)
    {
        result.tiePointPreparationExecuted = true;
        if (result.config.forceRebuildTiePoints)
        {
            QString cleanupError;
            if (!clearTiePointCache(result.config, &cleanupError))
            {
                result.reconstructionResult.errorMessage = cleanupError;
                result.reconstructionResult.summary = cleanupError;
                return result;
            }
        }
        // runner 注入只改变执行实现，不改变 resolved options，供单元测试验证边界。
        result.tiePointResult = TiePointPreparation::run(result.config.tiePointOptions,
                                                         result.config.tiePointContext,
                                                         tiePointRunner);
        if (!result.tiePointResult.success)
        {
            result.reconstructionResult.errorMessage = QStringLiteral("连接点准备失败: %1")
                .arg(result.tiePointResult.errorMessage);
            result.reconstructionResult.summary = result.reconstructionResult.errorMessage;
            return result;
        }
        if (!result.tiePointResult.tiePointPath.trimmed().isEmpty())
        {
            result.config.pipelineInput.tiePointPath = result.tiePointResult.tiePointPath;
        }
        for (const matchphotos::MatchPhotosMatchRecord &match :
             result.tiePointResult.matches)
        {
            const int effectiveWorkers = match.settings.value(
                QStringLiteral("cuda_parallel_pairs_effective")).toInt();
            if (effectiveWorkers > 0)
            {
                result.config.resolvedSettings.insert(
                    QStringLiteral("cuda_parallel_pairs_effective"),
                    effectiveWorkers);
                break;
            }
        }
        if (options.pairMatchedFn)
        {
            for (const matchphotos::MatchPhotosMatchRecord &match : result.tiePointResult.matches)
            {
                options.pairMatchedFn(match.image0Path,
                                      match.image1Path,
                                      match.image0MatchFilePath,
                                      match.matchCount);
            }
        }
    }

    // 连接点成功或已确认可复用后，才允许创建正式 SfM 候选。
    result.reconstructionResult = pipelineRunner(result.config.pipelineInput);
    // 将解析后的真实设置与输入连接点路径固化到结果记录，保证工程重开后可追溯。
    QJsonObject extra = result.reconstructionResult.resultRecordExtra;
    extra.insert(QStringLiteral("workflow_kind"),
                 QStringLiteral("aerial_triangulation_align_photos"));
    extra.insert(QStringLiteral("resolved_settings"), result.config.resolvedSettings);
    extra.insert(QStringLiteral("tie_point_path"), result.config.pipelineInput.tiePointPath);
    if (result.tiePointPreparationExecuted)
    {
        extra.insert(QStringLiteral("tie_point_track_count"), result.tiePointResult.trackCount);
        extra.insert(QStringLiteral("tie_point_match_file_count"),
                     static_cast<int>(result.tiePointResult.matches.size()));
    }
    result.reconstructionResult.resultRecordExtra = extra;
    return result;
}

} // namespace xjw::aerial_triangulation
