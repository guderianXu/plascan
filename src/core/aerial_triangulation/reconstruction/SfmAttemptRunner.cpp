/**
 * @file SfmAttemptRunner.cpp
 * @brief 将连接点 sidecar 和工程先验装配为一次隔离的 IncrementalSfm 试算。
 *
 * 一次 attempt 的职责包括：
 * 1. 严格读取当前影像集合对应的多视连接点图；
 * 2. 选择相机文件、可信工程内参或影像尺寸估计内参；
 * 3. 装载人工标记、控制点和比例尺先验；
 * 4. 配置增量 SfM/BA 并运行；
 * 5. 返回内存重建、诊断和待提交相机更新。
 *
 * 本类不写稀疏点云、不直接修改工程。焦距搜索因此可以并发运行多个互不污染的
 * attempt，并由上层只提交胜出结果。
 */

#include "reconstruction/SfmAttemptRunner.h"
#include "reconstruction/CameraIntrinsicPriorSanitizer.h"
#include "reconstruction/MarkerPriorLoader.h"
#include "search/SfmSearchPolicy.h"

#include "io/PathIO.h"
#include "ProjectCameraIO.h"
#include "project/ProjectCommonUtils.h"
#include "project/ProjectMetadata.h"
#include "pipeline/IncrementalSfm.h"

#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSize>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace xjw::aerial_triangulation
{
namespace
{

/// 将无向影像 ID 对压缩为 64 位键，用于连接点展开时去重。
quint64 pairKey(ImageId imageA, ImageId imageB)
{
    const quint64 first = std::min(imageA, imageB);
    const quint64 second = std::max(imageA, imageB);
    return (first << 32U) | second;
}

bool fail(const QString &message, QString *errorMessage)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
    return false;
}

/// 用项目的路径 token 规则将持久化影像路径映射到本次输入索引。
int selectedImageIndex(const QString &path, const QStringList &selectedImages)
{
    for (int index = 0; index < selectedImages.size(); ++index)
    {
        if (xjw::common::project::pathTokenMatchesImage(path, selectedImages.at(index)))
        {
            return index;
        }
    }
    return -1;
}

struct ParsedObservation
{
    ImageId imageId = kInvalidImageId; ///< 本次 SfM 连续影像 ID。
    FeatureIdx featureIndex = kInvalidFeatureIdx; ///< 本影像内压缩后的关键点索引。
};

/// 质量等级对初始化强度和 BA 调度频率的映射。
struct SfmQualityPreset
{
    int initMinMatches = 25;
    int initMinInliers = 5;
    int localBaInterval = 3;
    int globalBaInterval = 10;
};

SfmQualityPreset presetForQuality(int quality)
{
    switch (std::clamp(quality, 0, 3))
    {
    case 0:
        return {15, 5, 5, 15};
    case 1:
        return {20, 5, 4, 12};
    case 2:
        return {25, 5, 3, 10};
    case 3:
    default:
        return {30, 5, 3, 10};
    }
}

bool cameraMetadataHasUsablePose(const QJsonObject &cameraObject)
{
    return !cameraObject.isEmpty() &&
           !cameraObject.value(QStringLiteral("pose_initialized_as_identity")).toBool(false);
}

Camera cameraWithIdentityPose(Camera camera)
{
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {0.0, 0.0, 0.0});
    return camera;
}

/**
 * @brief 根据工作流输入统一配置 IncrementalSfm 和 BA。
 *
 * GPU/CPU 仅影响 BA 后端选择；特征与匹配已经在上游完成。自适应相机模型拟合
 * 会开启共享焦距联合优化，但已知完整相机位姿时保持输入内参稳定。
 */
void configureSfmOptions(const PreparedAerialTriangulationInput &input,
                         const std::shared_ptr<std::atomic<int>> &registeredProgress,
                         IncrementalSfmOptions *options)
{
    const SfmQualityPreset preset = presetForQuality(input.quality);
    options->initMinNumMatches = preset.initMinMatches;
    options->initMinNumInliers = preset.initMinInliers;
    options->localBAInterval = preset.localBaInterval;
    options->globalBAInterval = preset.globalBaInterval;
    options->executionProfile = input.coarseFocalEvaluation
        ? SfmExecutionProfile::CoarseEvaluation
        : SfmExecutionProfile::FullRefinement;
    options->maxRegisteredImages = input.maxRegisteredImages;

    // 固定每 3/10 张执行局部/全局 BA 只适合小工程。数百张影像时，后半程每次
    // 全局 BA 都会重新装配几十万条观测，重复次数远大于求解本身需要。大工程仍
    // 保留局部稳姿和周期全局消漂，但让间隔随总影像数增长，最终全局 BA 不受影响。
    const SfmBaSchedule baSchedule = resolveSfmBaSchedule(
        input.images.size(),
        options->localBAInterval,
        options->localBANumImages,
        options->globalBAInterval);
    options->localBAInterval = baSchedule.localInterval;
    options->localBANumImages = baSchedule.localWindowImages;
    options->globalBAInterval = baSchedule.globalInterval;
    options->baOptions.cancelFlag = input.cancelFlag;
    options->baOptions.numThreads = resolveSfmThreadBudget(input.threads);
    options->baOptions.progressCallback =
        [progress = input.progressFn,
         cancelFlag = input.cancelFlag,
         registeredProgress](int currentIteration,
                             int maxIterations,
                             double avgRms,
                             int validPoints)
        {
            if (cancelFlag && cancelFlag->load())
            {
                return false;
            }
            if (progress)
            {
                QString stage = QStringLiteral("光束法平差：迭代 %1/%2，RMS %3")
                                    .arg(currentIteration)
                                    .arg(maxIterations)
                                    .arg(avgRms, 0, 'f', 4);
                if (validPoints > 0)
                {
                    stage += QStringLiteral("，有效点 %1").arg(validPoints);
                }
                // BA 会在多次局部/全局阶段重复进入，百分比沿用当前已注册影像进度，
                // 只更新真实迭代信息，避免整体工作流进度条来回跳动。
                progress(stage, registeredProgress ? registeredProgress->load() : 0);
            }
            return true;
        };

    if (input.useInitialPairHint)
    {
        options->autoSelectInitPair = false;
        options->initImageId1 = input.initialImageId1;
        options->initImageId2 = input.initialImageId2;
    }

    const bool cpuOnly =
        input.device.trimmed().compare(QStringLiteral("cpu"), Qt::CaseInsensitive) == 0;
    if (!cpuOnly)
    {
        options->baOptions.backend = BABackend::Auto;
        options->baOptions.nativeCudaMaxPointStepNorm = 1.0;
        options->baOptions.minCeresCudaObservations = 500000;
        options->baOptions.minCeresCpuObservations = 50000;
        options->baOptions.enableBackendQualityGate = true;
        options->baOptions.maxAcceptedRmsGrowth = 1.25;
        options->baOptions.minAcceptedValidTrackRatio = 0.60;
        options->baOptions.compareAutoBackendWithLegacy = false;
        options->baOptions.allowBackendFallback = true;
    }

    if (input.quality >= 2)
    {
        options->filterMaxReprojError = 1.5;
        options->filterMinTriAngle = 2.0;
        options->iterativeBARounds = 4;
    }
    options->baOptions.filterMaxReprojError = options->filterMaxReprojError;

    options->useSequencePoseRecovery = input.useSequencePoseRecovery;
    options->enforceSequencePoseConsistency = input.enforceSequencePoseConsistency;
    options->sequenceLoopClosure = input.sequenceLoopClosure;
    if (input.adaptiveCameraModelFitting)
    {
        options->baOptions.refineSharedFocalLength = true;
        // The coarse focal search already resolves the dominant calibration
        // ambiguity. Releasing focal, pixel aspect and principal point at the
        // same time is poorly observable for a single-height orbital ring:
        // pose error can be absorbed into an artificial fy/fx ratio or a large
        // vertical principal-point drift, which then misaligns masks and depth
        // maps during dense fusion. Keep the seed aspect and principal point;
        // trusted project calibrations already carry their measured values,
        // while an uncalibrated input uses the image-centre prior.
        options->baOptions.refineSharedFocalAspectRatio = false;
        options->baOptions.refineSharedPrincipalPoint = false;
        if (input.hasTrustedFocalPrior)
        {
            // EXIF/固定镜头目录给出的焦距允许小幅吸收对焦和制造公差，但不能像
            // 无标定搜索一样漂移 10%。近垂直航测中大范围焦距漂移会与高程弯曲强耦合。
            options->baOptions.minSharedFocalScale = 0.98;
            options->baOptions.maxSharedFocalScale = 1.02;
            options->baOptions.maxSharedFocalStepScale = 1.02;
            options->baOptions.maxSharedFocalIterations = 4;
            options->baOptions.refineSharedRadialDistortion = true;
            options->baOptions.maxSharedRadialK1Abs = 0.35;
            options->baOptions.maxSharedRadialK2Abs = 0.20;
            options->baOptions.sharedRadialK1PriorSigma = 0.15;
            options->baOptions.sharedRadialK2PriorSigma = 0.08;
        }
        else
        {
            options->baOptions.minSharedFocalScale = 0.90;
            options->baOptions.maxSharedFocalScale = 1.10;
            options->baOptions.maxSharedFocalStepScale = 1.12;
            options->baOptions.maxSharedFocalIterations = 6;
        }
        if (cpuOnly && BundleAdjust::isBackendAvailable(BABackend::CeresCpu))
        {
            // 保持注册阶段与粗焦距候选相同的自动后端，避免仅因求解器不同造成
            // 点数/RMS 波动。最终完整全局 BA 释放共享内参时，BundleAdjust 会因
            // joint_shared_intrinsics_requires_ceres 自动切换到 Ceres 联合求解。
            options->baOptions.backend = BABackend::Auto;
        }
        else if (!cpuOnly)
        {
            // GPU 规模不足时回落到 Ceres CPU，而不是回落到非联合的 Legacy 自标定。
            options->baOptions.minCeresCpuObservations = 1;
        }
    }
}

} // namespace

QSize SfmAttemptRunner::resolveInputImageSize(const QString &imagePath)
{
    QImageReader reader(imagePath);
    const QSize headerSize = reader.size();
    if (headerSize.isValid())
    {
        return headerSize;
    }

    // Qt 的 TIFF 插件或 Unicode 路径可能不可用，统一 IO 会按字节解码并兼容本地路径。
    const cv::Mat decoded = xjw::common::io::readImage(imagePath, cv::IMREAD_UNCHANGED);
    if (!decoded.empty() && decoded.cols > 0 && decoded.rows > 0)
    {
        return QSize(decoded.cols, decoded.rows);
    }
    return {};
}

SfmAttemptExecutionResult SfmAttemptRunner::run(
    const PreparedAerialTriangulationInput &input) const
{
    SfmAttemptExecutionResult execution;
    // 阶段 1：把持久化多视轨迹转换为 SfM 需要的每影像关键点和 pairwise matches。
    if (!readTiePointGraph(input.tiePointPath,
                           input.images,
                           &execution.graph,
                           &execution.result.errorMessage))
    {
        execution.result.summary = execution.result.errorMessage;
        return execution;
    }

    if (input.cancelFlag && input.cancelFlag->load())
    {
        execution.result.errorMessage = QStringLiteral("用户取消");
        execution.result.summary = execution.result.errorMessage;
        return execution;
    }

    IncrementalSfmOptions sfmOptions;
    const auto registeredProgress = std::make_shared<std::atomic<int>>(0);
    configureSfmOptions(input, registeredProgress, &sfmOptions);

    // 阶段 2：相机来源优先级为完整相机文件、可信工程相机、影像尺寸估算。
    const bool hasCompleteCameraFiles = input.cameraPaths.size() == input.images.size() &&
        std::all_of(input.cameraPaths.cbegin(), input.cameraPaths.cend(), [](const QString &path)
        {
            return !path.trimmed().isEmpty() && QFileInfo::exists(path);
        });

    QMap<QString, QJsonObject> projectCameraByPath;
    QSet<QString> projectPosePaths;
    int rejectedProjectIntrinsicCount = 0;
    if ((input.useProjectCameraIntrinsics || input.useProjectCameraPoses) &&
        !input.projectMeta.isEmpty())
    {
        const QMap<QString, QJsonObject> imageMetadata =
            xjw::common::project::projectImageMetaByPath(input.projectMeta, true);
        for (auto it = imageMetadata.cbegin(); it != imageMetadata.cend(); ++it)
        {
            const QJsonObject cameraObject = it.value().value(QStringLiteral("camera")).toObject();
            Camera camera;
            if (!cameraObject.isEmpty() &&
                xjw::common::project::cameraFromJson(cameraObject, &camera) &&
                camera.isValid())
            {
                const bool trustedIntrinsic =
                    isTrustedProjectCameraIntrinsic(cameraObject);
                if (input.useProjectCameraPoses || trustedIntrinsic)
                {
                    projectCameraByPath.insert(it.key(), cameraObject);
                }
                else
                {
                    ++rejectedProjectIntrinsicCount;
                }
                if (input.useProjectCameraPoses && cameraMetadataHasUsablePose(cameraObject))
                {
                    projectPosePaths.insert(it.key());
                }
            }
        }
    }

    // 只有所有影像都具备真实外参时才进入 known-pose 路径，禁止混合已知/未知位姿。
    bool hasCompleteProjectPoseCameras = !input.images.isEmpty();
    for (const QString &imagePath : input.images)
    {
        if (!projectPosePaths.contains(xjw::common::project::normalizePath(imagePath)))
        {
            hasCompleteProjectPoseCameras = false;
            break;
        }
    }

    // 重置当前对齐时只应复用可信内参。工程文件可能保存过上次失败 SfM 的自标定结果，
    // 少数严重错误焦距会把单张相机中心吸附到模型附近，且不会明显拉高全局 RMS。
    CameraIntrinsicPriorSanitizationResult intrinsicSanitization;
    if (!hasCompleteCameraFiles && input.useProjectCameraIntrinsics &&
        !input.useProjectCameraPoses)
    {
        intrinsicSanitization = sanitizeProjectCameraIntrinsicPriors(
            input.images, &projectCameraByPath);
    }
    sfmOptions.useKnownCameraPoses = hasCompleteCameraFiles || hasCompleteProjectPoseCameras;
    if (sfmOptions.useKnownCameraPoses)
    {
        sfmOptions.baOptions.refineSharedFocalLength = false;
        sfmOptions.baOptions.refineSharedFocalAspectRatio = false;
        sfmOptions.baOptions.refineSharedPrincipalPoint = false;
        sfmOptions.refineKnownCameraPoseWithSoftPrior =
            !input.lockInputCameraPoses;
    }
    else
    {
        sfmOptions.pnpOptions.allowRelaxedInlierRatio = true;
        sfmOptions.pnpOptions.minInlierRatio =
            std::max(sfmOptions.pnpOptions.minInlierRatio, 0.25);
        sfmOptions.pnpOptions.minNumInliers =
            std::max(sfmOptions.pnpOptions.minNumInliers, 12);
        sfmOptions.pnpOptions.relaxedMinInlierRatio =
            std::min(sfmOptions.pnpOptions.relaxedMinInlierRatio, 0.05);
        sfmOptions.pnpOptions.relaxedMinNumInliers =
            std::max(sfmOptions.pnpOptions.relaxedMinNumInliers, 24);
    }

    // 阶段 3：创建影像节点。重置对齐时工程 Camera 只保留内参并将外参置为单位位姿。
    IncrementalSfm sfm(sfmOptions);
    QMap<QString, ImageId> imageIdByPath;
    for (int index = 0; index < input.images.size(); ++index)
    {
        const ImageId imageId = static_cast<ImageId>(index);
        const QString &imagePath = input.images.at(index);
        imageIdByPath.insert(xjw::common::project::normalizePath(imagePath), imageId);
        const std::vector<FeatureKeypoint> keypoints =
            execution.graph.keypointsByImage.value(imageId);

        if (hasCompleteCameraFiles)
        {
            sfm.addImage(imageId,
                         xjw::common::io::toUtf8Path(imagePath),
                         xjw::common::io::toUtf8Path(input.cameraPaths.at(index)),
                         keypoints);
            continue;
        }

        const QString normalizedPath = xjw::common::project::normalizePath(imagePath);
        const auto projectCamera = projectCameraByPath.constFind(normalizedPath);
        if (input.useProjectCameraIntrinsics && projectCamera != projectCameraByPath.cend())
        {
            Camera camera;
            if (xjw::common::project::cameraFromJson(projectCamera.value(), &camera) &&
                camera.isValid())
            {
                if (!input.useProjectCameraPoses)
                {
                    // 重置对齐时只复用内参，外参重新由相对定向/增量注册估计。
                    camera = cameraWithIdentityPose(camera);
                }
                sfm.addImageWithCamera(imageId,
                                       xjw::common::io::toUtf8Path(imagePath),
                                       camera,
                                       keypoints);
                continue;
            }
        }

        const QSize imageSize = resolveInputImageSize(imagePath);
        if (!imageSize.isValid())
        {
            execution.result.errorMessage =
                QStringLiteral("无法读取影像尺寸，不能初始化相机内参: %1").arg(imagePath);
            execution.result.summary = execution.result.errorMessage;
            return execution;
        }
        const double focal = std::max(imageSize.width(), imageSize.height()) *
            std::max(0.1, input.estimatedFocalScale);
        Camera camera;
        camera.setIntrinsics(focal,
                             focal,
                             imageSize.width() * 0.5,
                             imageSize.height() * 0.5);
        sfm.addImageWithCamera(imageId,
                               xjw::common::io::toUtf8Path(imagePath),
                               camera,
                               keypoints);
    }

    // 阶段 4：人工标记和比例尺作为 prior track/control constraint 注入，
    // 不伪装成普通自动连接点。
    const MarkerPriorLoadResult markerPriors = MarkerPriorLoader::load(
        input.markerSetPath, input.projectMeta, imageIdByPath);
    if (!markerPriors.ok)
    {
        execution.result.errorMessage = markerPriors.errorMessage;
        execution.result.summary = markerPriors.errorMessage;
        return execution;
    }
    for (const control_points::PriorTrack &track : markerPriors.tracks)
    {
        sfm.addPriorTrack(track);
    }
    for (const control_points::PriorScaleBar &scaleBar : markerPriors.scaleBars)
    {
        sfm.addPriorScaleBar(scaleBar);
    }

    // 阶段 5：连接点轨迹已展开为去重 pairwise 对应，交给 IncrementalSfm 建观测图。
    for (const PreparedTiePointMatchPair &pair : execution.graph.matchPairs)
    {
        sfm.addMatches(pair.imageA, pair.imageB, pair.matches);
    }

    // 阶段 6：运行初始对、增量 PnP/三角化、局部/全局 BA 和质量过滤。
    const IncrementalSfmResult sfmResult = sfm.run(
        [&input, registeredProgress](int registered, int total, const std::string &message)
        {
            if (input.cancelFlag && input.cancelFlag->load())
            {
                return false;
            }
            const int percent = total > 0
                ? std::clamp(static_cast<int>(100.0 * registered / total), 0, 100)
                : 0;
            registeredProgress->store(percent);
            if (input.progressFn)
            {
                input.progressFn(QString::fromStdString(message), percent);
            }
            return true;
        });

    execution.reconstruction = sfmResult.reconstruction;
    execution.result.success = sfmResult.success;
    execution.result.numRegisteredImages = sfmResult.numRegisteredImages;
    execution.result.numPoints3D = sfmResult.numPoints3D;
    execution.result.meanReprojError = sfmResult.meanReprojError;
    execution.result.baRmsBefore = sfmResult.baRmsBefore;
    execution.result.baRmsAfter = sfmResult.baRmsAfter;
    execution.result.baTracksTotal = sfmResult.baTracksTotal;
    execution.result.baTracksOptimized = sfmResult.baTracksOptimized;
    execution.result.baTracksFiltered = sfmResult.baTracksFiltered;
    execution.result.summary = QString::fromStdString(sfmResult.summary);

    // 数值后端、先验接纳和初始化选择全部进入稳定诊断字段，便于 GUI/CLI 对比。
    QJsonObject diagnostics;
    diagnostics.insert(QStringLiteral("selected_initial_pair"),
                       QJsonArray{static_cast<int>(sfmResult.selectedInitialImageId1),
                                  static_cast<int>(sfmResult.selectedInitialImageId2)});
    diagnostics.insert(QStringLiteral("prior_tracks_accepted"), sfmResult.priorTracksAccepted);
    diagnostics.insert(QStringLiteral("prior_tracks_rejected"), sfmResult.priorTracksRejected);
    diagnostics.insert(QStringLiteral("marker_prior_tracks_loaded"),
                       static_cast<int>(markerPriors.tracks.size()));
    diagnostics.insert(QStringLiteral("marker_prior_scale_bars_loaded"),
                       static_cast<int>(markerPriors.scaleBars.size()));
    diagnostics.insert(QStringLiteral("control_network_applied"), sfmResult.controlNetworkApplied);
    diagnostics.insert(QStringLiteral("control_point_constraints"),
                       sfmResult.controlPointConstraintCount);
    diagnostics.insert(QStringLiteral("ba_requested_backend"),
                       QString::fromLatin1(BundleAdjust::backendName(sfmResult.baRequestedBackend)));
    diagnostics.insert(QStringLiteral("ba_used_backend"),
                       QString::fromLatin1(BundleAdjust::backendName(sfmResult.baUsedBackend)));
    diagnostics.insert(QStringLiteral("ba_solve_status"),
                       QString::fromLatin1(BundleAdjust::solveStatusName(sfmResult.baSolveStatus)));
    diagnostics.insert(QStringLiteral("ba_solution_usable"), sfmResult.baSolutionUsable);
    diagnostics.insert(QStringLiteral("ba_result_applied"), sfmResult.baResultApplied);
    diagnostics.insert(QStringLiteral("ba_backend_fallback"), sfmResult.baBackendFallback);
    diagnostics.insert(QStringLiteral("ba_observations"), sfmResult.baObservationCount);
    diagnostics.insert(QStringLiteral("ba_total_seconds"), sfmResult.baTotalSeconds);
    diagnostics.insert(QStringLiteral("ba_refined_intrinsic_count"),
                       sfmResult.baRefinedIntrinsicCount);
    diagnostics.insert(QStringLiteral("ba_shared_focal_scale"), sfmResult.baSharedFocalScale);
    diagnostics.insert(QStringLiteral("ba_shared_focal_aspect_scale"),
                       sfmResult.baSharedFocalAspectScale);
    diagnostics.insert(QStringLiteral("ba_shared_principal_offset_x_px"),
                       sfmResult.baSharedPrincipalOffsetX);
    diagnostics.insert(QStringLiteral("ba_shared_principal_offset_y_px"),
                       sfmResult.baSharedPrincipalOffsetY);
    diagnostics.insert(QStringLiteral("ba_shared_radial_k1"), sfmResult.baSharedRadialK1);
    diagnostics.insert(QStringLiteral("ba_shared_radial_k2"), sfmResult.baSharedRadialK2);
    diagnostics.insert(QStringLiteral("ba_backend_message"),
                       QString::fromStdString(sfmResult.baBackendMessage));
    diagnostics.insert(QStringLiteral("project_intrinsic_prior_inspected"),
                       intrinsicSanitization.inspectedCameraCount);
    diagnostics.insert(QStringLiteral("project_intrinsic_prior_dominant_group"),
                       intrinsicSanitization.dominantGroupCount);
    diagnostics.insert(QStringLiteral("project_intrinsic_prior_median_focal_px"),
                       intrinsicSanitization.dominantMedianFocalPixels);
    diagnostics.insert(QStringLiteral("project_intrinsic_prior_normalized"),
                       intrinsicSanitization.normalizedCameraCount);
    diagnostics.insert(QStringLiteral("project_intrinsic_prior_normalized_images"),
                       QJsonArray::fromStringList(intrinsicSanitization.normalizedImagePaths));
    diagnostics.insert(QStringLiteral("project_intrinsic_prior_rejected"),
                       rejectedProjectIntrinsicCount);

    std::vector<double> final_camera_focals;
    if (execution.reconstruction)
    {
        for (const ImageId image_id : execution.reconstruction->registeredImageIds())
        {
            if (!execution.reconstruction->hasCamera(image_id))
            {
                continue;
            }
            const Camera &camera = execution.reconstruction->camera(image_id);
            const double focal_x = camera.focalX();
            const double focal_y = camera.focalY();
            if (std::isfinite(focal_x) && focal_x > 0.0 &&
                std::isfinite(focal_y) && focal_y > 0.0)
            {
                final_camera_focals.push_back(std::sqrt(focal_x * focal_y));
            }
        }
    }
    std::sort(final_camera_focals.begin(), final_camera_focals.end());
    diagnostics.insert(QStringLiteral("final_camera_focal_count"),
                       static_cast<int>(final_camera_focals.size()));
    if (!final_camera_focals.empty())
    {
        const std::size_t middle = final_camera_focals.size() / 2;
        const double median_focal = final_camera_focals.size() % 2 == 0
            ? 0.5 * (final_camera_focals[middle - 1] + final_camera_focals[middle])
            : final_camera_focals[middle];
        diagnostics.insert(QStringLiteral("final_camera_focal_median_px"), median_focal);
        diagnostics.insert(QStringLiteral("final_camera_focal_min_px"),
                           final_camera_focals.front());
        diagnostics.insert(QStringLiteral("final_camera_focal_max_px"),
                           final_camera_focals.back());
    }
    execution.result.sfmDiagnostics = diagnostics;

    if (!sfmResult.success)
    {
        execution.result.errorMessage = execution.result.summary.isEmpty()
            ? QStringLiteral("SfM 重建失败")
            : execution.result.summary;
        return execution;
    }

    // 仅生成待回写对象；工程服务必须在上层正式写出成功后统一提交。
    if (execution.reconstruction)
    {
        for (int index = 0; index < input.images.size(); ++index)
        {
            const ImageId imageId = static_cast<ImageId>(index);
            if (execution.reconstruction->isRegistered(imageId))
            {
                QJsonObject cameraObject = xjw::common::project::cameraToJson(
                    execution.reconstruction->camera(imageId));
                cameraObject.insert(QStringLiteral("intrinsic_source"),
                                    QStringLiteral("sfm_estimated"));
                cameraObject.insert(QStringLiteral("pose_source"),
                                    QStringLiteral("sfm_estimated"));
                execution.result.pendingCamUpdates.insert(
                    xjw::common::project::normalizePath(input.images.at(index)),
                    cameraObject);
            }
        }
    }
    return execution;
}

bool SfmAttemptRunner::readTiePointGraph(const QString &tiePointPath,
                                         const QStringList &selectedImages,
                                         PreparedTiePointGraph *graph,
                                         QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!graph)
    {
        return fail(QStringLiteral("连接点图输出参数为空"), errorMessage);
    }
    *graph = {};

    QFile file(tiePointPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        return fail(QStringLiteral("无法读取连接点文件: %1").arg(tiePointPath), errorMessage);
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject())
    {
        return fail(QStringLiteral("连接点文件不是有效 JSON 对象: %1").arg(tiePointPath), errorMessage);
    }

    const QJsonObject root = document.object();
    if (root.value(QStringLiteral("format")).toString() != QLatin1String("plascan_tie_points") ||
        root.value(QStringLiteral("format_version")).toInt() != 1)
    {
        return fail(QStringLiteral("不支持的连接点文件格式或版本"), errorMessage);
    }
    if (selectedImages.size() < 2)
    {
        return fail(QStringLiteral("SfM 至少需要两张影像"), errorMessage);
    }

    // 持久化 image_id 可能与本次选择顺序不同，必须先按路径建立显式重映射。
    QMap<int, ImageId> selectedIdByPersistedId;
    QSet<int> coveredSelectedIds;
    for (const QJsonValue &value : root.value(QStringLiteral("images")).toArray())
    {
        const QJsonObject imageObject = value.toObject();
        const int persistedId = imageObject.value(QStringLiteral("image_id")).toInt(-1);
        const int selectedId = selectedImageIndex(
            imageObject.value(QStringLiteral("path")).toString(), selectedImages);
        if (persistedId >= 0 && selectedId >= 0)
        {
            selectedIdByPersistedId.insert(persistedId, static_cast<ImageId>(selectedId));
            coveredSelectedIds.insert(selectedId);
        }
    }
    if (coveredSelectedIds.size() != selectedImages.size())
    {
        return fail(QStringLiteral("连接点文件的影像集合与当前空三影像集合不一致"), errorMessage);
    }

    graph->imagePaths.reserve(selectedImages.size());
    for (const QString &path : selectedImages)
    {
        graph->imagePaths.append(xjw::common::project::normalizePath(path));
    }

    // 连接点文件保留原始特征索引；SfM 只需要轨迹实际引用的稀疏子集。
    // 每张影像独立压缩索引可显著降低关键点内存，同时保持同一原始索引一致。
    QMap<ImageId, QMap<qulonglong, FeatureIdx>> compactIndexByOriginal;
    std::map<quint64, std::size_t> pairPosition;
    std::map<quint64, std::set<std::pair<FeatureIdx, FeatureIdx>>> pairObservations;

    for (const QJsonValue &trackValue : root.value(QStringLiteral("tracks")).toArray())
    {
        const QJsonObject trackObject = trackValue.toObject();
        const float confidence = static_cast<float>(
            std::clamp(trackObject.value(QStringLiteral("confidence")).toDouble(1.0), 0.0, 1.0));
        std::vector<ParsedObservation> observations;

        for (const QJsonValue &observationValue :
             trackObject.value(QStringLiteral("observations")).toArray())
        {
            const QJsonObject observationObject = observationValue.toObject();
            const int persistedImageId = observationObject.value(QStringLiteral("image_id")).toInt(-1);
            const qint64 originalFeatureIndex =
                observationObject.value(QStringLiteral("feature_idx")).toInteger(-1);
            const QJsonArray xy = observationObject.value(QStringLiteral("xy")).toArray();
            if (!selectedIdByPersistedId.contains(persistedImageId) ||
                originalFeatureIndex < 0 || xy.size() < 2)
            {
                continue;
            }

            const double x = xy.at(0).toDouble(std::numeric_limits<double>::quiet_NaN());
            const double y = xy.at(1).toDouble(std::numeric_limits<double>::quiet_NaN());
            if (!std::isfinite(x) || !std::isfinite(y))
            {
                continue;
            }

            const ImageId imageId = selectedIdByPersistedId.value(persistedImageId);
            QMap<qulonglong, FeatureIdx> &indexMap = compactIndexByOriginal[imageId];
            const qulonglong originalKey = static_cast<qulonglong>(originalFeatureIndex);
            FeatureIdx compactIndex = indexMap.value(originalKey, kInvalidFeatureIdx);
            if (compactIndex == kInvalidFeatureIdx)
            {
                compactIndex = static_cast<FeatureIdx>(graph->keypointsByImage[imageId].size());
                indexMap.insert(originalKey, compactIndex);
                graph->keypointsByImage[imageId].push_back(
                    {static_cast<float>(x), static_cast<float>(y)});
            }
            observations.push_back({imageId, compactIndex});
        }

        // 一条多视轨迹在同一影像最多保留一个观测，避免生成自相矛盾 pair。
        std::sort(observations.begin(), observations.end(), [](const ParsedObservation &left,
                                                               const ParsedObservation &right)
        {
            return left.imageId < right.imageId;
        });
        observations.erase(std::unique(observations.begin(), observations.end(),
                                       [](const ParsedObservation &left,
                                          const ParsedObservation &right)
        {
            return left.imageId == right.imageId;
        }), observations.end());
        if (observations.size() < 2)
        {
            continue;
        }

        // 将 N 视轨迹展开为 N(N-1)/2 条 pairwise 对应；pairObservations 防止不同
        // 输入轨迹重复引用同一特征对，最终多视身份仍由 SfM 观测图重新合并。
        ++graph->trackCount;
        for (std::size_t first = 0; first + 1 < observations.size(); ++first)
        {
            for (std::size_t second = first + 1; second < observations.size(); ++second)
            {
                const ParsedObservation &observationA = observations[first];
                const ParsedObservation &observationB = observations[second];
                const quint64 key = pairKey(observationA.imageId, observationB.imageId);
                const std::pair<FeatureIdx, FeatureIdx> featurePair{
                    observationA.featureIndex, observationB.featureIndex};
                if (!pairObservations[key].insert(featurePair).second)
                {
                    continue;
                }

                auto position = pairPosition.find(key);
                if (position == pairPosition.end())
                {
                    PreparedTiePointMatchPair pair;
                    pair.imageA = observationA.imageId;
                    pair.imageB = observationB.imageId;
                    graph->matchPairs.push_back(std::move(pair));
                    position = pairPosition.emplace(key, graph->matchPairs.size() - 1).first;
                }
                graph->matchPairs[position->second].matches.push_back(
                    {observationA.featureIndex, observationB.featureIndex, confidence});
            }
        }
    }

    if (graph->trackCount <= 0 || graph->matchPairs.empty())
    {
        return fail(QStringLiteral("连接点文件中没有可用于 SfM 的多视图轨迹"), errorMessage);
    }
    return true;
}

} // namespace xjw::aerial_triangulation
