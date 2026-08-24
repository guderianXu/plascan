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
#include "sift/SiftComputeBackend.h"
#include "workflow/AerialTriangulationPipeline.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSet>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

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
    // highest 面向完整生产重建。16 帧局部邻域可形成更多三视及以上长轨迹，
    // 同时仍远小于大型工程的全量 O(N^2) 配对。
    if (token == QStringLiteral("highest")) return 16;
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
    if (token == QStringLiteral("opencl"))
    {
        return matchphotos::ComputeDevice::OpenCl;
    }
    if (token == QStringLiteral("metal"))
    {
        return matchphotos::ComputeDevice::Metal;
    }
    return matchphotos::ComputeDevice::Auto;
}

struct ClosedSequenceEvidence
{
    bool detected = false;
    double opticalAxisConcentration = 1.0;
    double inwardAxisMedian = -1.0;
    double adjacentDistanceMadRatio = 1.0;
    double adjacentDistanceMaximumRatio = std::numeric_limits<double>::infinity();
};

double median(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return 0.5 * (values[(values.size() - 1) / 2] + values[values.size() / 2]);
}

double vectorNorm(const std::array<double, 3> &value)
{
    return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}

const FramePinholeCamera *referenceCameraForImage(const QMap<QString, FramePinholeCamera> &cameras,
                                      const QString &image)
{
    const auto direct = cameras.constFind(image);
    if (direct != cameras.constEnd())
    {
        return &direct.value();
    }
    const QString normalized = QDir::fromNativeSeparators(QDir::cleanPath(image));
    for (auto it = cameras.constBegin(); it != cameras.constEnd(); ++it)
    {
        if (QDir::fromNativeSeparators(QDir::cleanPath(it.key()))
                .compare(normalized, Qt::CaseInsensitive) == 0)
        {
            return &it.value();
        }
    }
    return nullptr;
}

ClosedSequenceEvidence detectEstimatedClosedSequence(
    const QStringList &images,
    const QMap<QString, FramePinholeCamera> &referenceCameras)
{
    ClosedSequenceEvidence evidence;
    if (images.size() < 6 || referenceCameras.size() < images.size())
    {
        return evidence;
    }

    std::vector<std::array<double, 3>> centers;
    std::vector<std::array<double, 3>> axes;
    centers.reserve(static_cast<std::size_t>(images.size()));
    axes.reserve(static_cast<std::size_t>(images.size()));
    std::array<double, 3> meanCenter{};
    std::array<double, 3> meanAxis{};
    for (const QString &image : images)
    {
        const FramePinholeCamera *referenceCamera = referenceCameraForImage(referenceCameras, image);
        if (!referenceCamera || !referenceCamera->isValid())
        {
            return evidence;
        }
        const FramePinholeCamera camera = referenceCamera->normalizedForPositiveDepth();
        const auto center = camera.cameraCenter();
        const auto rotation = camera.cameraToWorldRotation();
        const std::array<double, 3> axis{{rotation[2], rotation[5], rotation[8]}};
        centers.push_back(center);
        axes.push_back(axis);
        for (int dimension = 0; dimension < 3; ++dimension)
        {
            meanCenter[dimension] += center[dimension];
            meanAxis[dimension] += axis[dimension];
        }
    }
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        meanCenter[dimension] /= images.size();
        meanAxis[dimension] /= images.size();
    }
    evidence.opticalAxisConcentration = vectorNorm(meanAxis);

    std::vector<double> inwardDots;
    std::vector<double> adjacentDistances;
    inwardDots.reserve(centers.size());
    adjacentDistances.reserve(centers.size());
    for (std::size_t index = 0; index < centers.size(); ++index)
    {
        std::array<double, 3> towardMean{{
            meanCenter[0] - centers[index][0],
            meanCenter[1] - centers[index][1],
            meanCenter[2] - centers[index][2]}};
        const double towardNorm = vectorNorm(towardMean);
        const double axisNorm = vectorNorm(axes[index]);
        if (towardNorm <= 1.0e-12 || axisNorm <= 1.0e-12)
        {
            return evidence;
        }
        inwardDots.push_back((towardMean[0] * axes[index][0] +
                              towardMean[1] * axes[index][1] +
                              towardMean[2] * axes[index][2]) /
                             (towardNorm * axisNorm));

        const auto &next = centers[(index + 1) % centers.size()];
        const std::array<double, 3> delta{{
            centers[index][0] - next[0],
            centers[index][1] - next[1],
            centers[index][2] - next[2]}};
        const double distance = vectorNorm(delta);
        if (!std::isfinite(distance) || distance <= 1.0e-12)
        {
            return evidence;
        }
        adjacentDistances.push_back(distance);
    }

    evidence.inwardAxisMedian = median(std::move(inwardDots));
    const double adjacentMedian = median(adjacentDistances);
    std::vector<double> deviations;
    deviations.reserve(adjacentDistances.size());
    double maximumDistance = 0.0;
    for (double distance : adjacentDistances)
    {
        deviations.push_back(std::abs(distance - adjacentMedian));
        maximumDistance = std::max(maximumDistance, distance);
    }
    if (adjacentMedian <= 1.0e-12)
    {
        return evidence;
    }
    evidence.adjacentDistanceMadRatio = median(std::move(deviations)) / adjacentMedian;
    evidence.adjacentDistanceMaximumRatio = maximumDistance / adjacentMedian;
    evidence.detected = evidence.opticalAxisConcentration < 0.85 &&
        evidence.inwardAxisMedian >= 0.5 &&
        evidence.adjacentDistanceMadRatio <= 0.35 &&
        evidence.adjacentDistanceMaximumRatio <= 2.5;
    return evidence;
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
        {"algorithm_selection", 1, 3}, {"model_prepare", 3, 8},
        {"feature", 8, 18}, {"generic_preselection", 18, 22},
        {"reference_preselection", 22, 25},
        {"pair_selection", 25, 28}, {"matching", 28, 33},
        {"geometry", 33, 34}, {"guided_match", 34, 35}, {"track_build", 35, 35}};
    Range selected{"unknown", 1, 35};
    const QString token = normalizedToken(stageId, QStringLiteral("unknown"));
    if (token == QStringLiteral("model_prepare") && maximum <= 0)
    {
        return -1;
    }
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
        QStringLiteral("auto_sift"));

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
    const QString normalizedReferenceMode = normalizedToken(
        options.referenceMode, QStringLiteral("source_code"));
    const bool usesPhotoSequence = options.referencePreselection &&
        normalizedReferenceMode == QStringLiteral("sequence");
    const ClosedSequenceEvidence estimatedSequence =
        options.referencePreselection && normalizedReferenceMode == QStringLiteral("estimated")
        ? detectEstimatedClosedSequence(options.images, options.referenceCameras)
        : ClosedSequenceEvidence{};
    const bool usesSequenceGeometry = usesPhotoSequence || estimatedSequence.detected;
    const bool usesClosedSequenceGeometry = estimatedSequence.detected;
    // “照片序列”提供稳定的相邻位姿插值/外推初值，但不代表相机中心必须满足等距轨迹。
    // 位姿恢复与硬距离门控必须解耦：前者帮助连续航带跨过弱纹理帧，后者在环拍、变焦
    // 或物体旋转序列上反而可能误拒绝正确 PnP，因此保持默认关闭。
    pipeline.useSequencePoseRecovery = usesSequenceGeometry;
    pipeline.enforceSequencePoseConsistency = false;
    // “照片序列”只说明输入顺序可靠，不说明最后一张与第一张相邻。把普通航带
    // 强制闭环会在序列两端制造高优先级伪边，使首尾几张照片形成低残差但错误的
    // 刚性分支。只有已有位姿明确检测到环拍时才启用首尾闭环。
    pipeline.sequenceLoopClosure = usesClosedSequenceGeometry;
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
    tieOptions.siftMaximumRatio = std::clamp(options.siftMaximumRatio, 0.0f, 1.0f);
    tieOptions.siftMinimumAdaptiveRatio = std::clamp(
        options.siftMinimumAdaptiveRatio, 0.0f, tieOptions.siftMaximumRatio);
    tieOptions.adaptiveSiftRatio = options.adaptiveSiftRatio;
    tieOptions.geometryReprojThreshold = std::max(0.1, options.geometryReprojThreshold);
    tieOptions.geometryMinInliers = std::max(8, options.geometryMinInliers);
    tieOptions.geometryMinInlierRatio = std::clamp(options.geometryMinInlierRatio, 0.01, 0.95);
    tieOptions.geometryMinGridCoverage = std::clamp(options.geometryMinGridCoverage, 0.01, 1.0);
    tieOptions.geometryMaxIterations = std::max(100, options.geometryMaxIterations);
    tieOptions.guidedMatchingMode = options.guidedImageMatching
        ? matchphotos::GuidedMatchingMode::Automatic
        : matchphotos::GuidedMatchingMode::Disabled;
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
               ? std::max(1,
                          (options.tiepointLimit + std::max(1, tiePointGridCellCount) - 1) /
                              std::max(1, tiePointGridCellCount))
               : 0);
    // MatchPhotosTask 已在完整多视轨迹形成后执行一次按影像/网格的质量优先稀疏化。
    // SfM 仍保留自己的输入保护，但必须使用同一上限；否则扩大前端配额后会被内部
    // 默认 6000/图再次静默截断，破坏 GUI/CLI 所显示参数的真实性。
    pipeline.maxTracksPerImage = tieOptions.maxTiePointsPerImage;
    pipeline.maxTracksPerGridCell = tieOptions.maxTiePointsPerGridCell;
    pipeline.trackThinningGridColumns = tieOptions.tiePointGridColumns;
    pipeline.trackThinningGridRows = tieOptions.tiePointGridRows;
    tieOptions.useGenericPreselection = options.genericPreselection;
    tieOptions.useReferencePreselection = options.referencePreselection;
    tieOptions.excludeStationaryTiePoints = options.excludeFixedTiePoints;
    tieOptions.stationaryTiePointMaxPixelMotion = std::max(
        0.0f, options.stationaryTiePointMaxPixelMotion);
    // 原始匹配缓存键包含蒙版应用阶段及两侧蒙版文件指纹。keypoints 模式下
    // 只有完全相同的蒙版输入才能命中，因此无需再无条件禁用安全缓存复用。
    tieOptions.reuseExistingMatches = options.reuseExistingMatches;

    // 第五阶段：解析 pair 预选。无参考相机时不得启用位姿参考预选；普通照片
    // 序列只使用线性索引窗口，不把首尾伪装成相邻影像。
    const QString referenceMode = normalizedToken(options.referenceMode,
                                                   QStringLiteral("source_code"));
    tieOptions.referencePreselectionGeometry = referenceMode == QStringLiteral("estimated")
        ? matchphotos::ReferencePreselectionGeometry::SparseScene
        : matchphotos::ReferencePreselectionGeometry::GroundFootprint;
    const bool hasReference = !options.referenceCameras.isEmpty() ||
        (!options.cameraPaths.isEmpty() && options.cameraPaths.size() == options.images.size());
    tieOptions.guidedUseReferenceCameraPoses = options.guidedImageMatching &&
        options.referencePreselection && hasReference && referenceMode != QStringLiteral("estimated");
    QString pairPlanningMode = options.genericPreselection ? QStringLiteral("generic")
                                                           : QStringLiteral("all_pairs");
    if (options.referencePreselection && referenceMode == QStringLiteral("sequence"))
    {
        tieOptions.pairPolicy.mode = matchphotos::PairSelectionMode::Sequence;
        tieOptions.pairPolicy.sequenceWindow = sequenceWindowForQuality(options.quality);
        tieOptions.pairPolicy.closeSequenceLoop = false;
        tieOptions.useReferencePreselection = false;
        pairPlanningMode = QStringLiteral("sequence");
    }
    else if (options.referencePreselection && hasReference)
    {
        // 已有相机位姿是候选规划的主先验。即使旧 SfM 已产生弯曲，或参考地面
        // 估计过于宽松，也不能让重叠图退化为 O(N^2) 的近全量匹配。
        const int referenceWindow = sequenceWindowForQuality(options.quality);
        tieOptions.pairPolicy.cameraOverlapTopKPerImage = std::max(8, referenceWindow * 3);
        // 通用预选只补充少量外观闭环；其余候选由位姿重叠负责。
        tieOptions.pairPolicy.vocabularyTopKPerImage = options.genericPreselection
            ? std::max(2, referenceWindow / 2)
            : 0;
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
    tieContext.referenceSparsePointsPath = QDir(QDir::cleanPath(options.outputDir))
        .filePath(QStringLiteral("sfm_sparse/sfm_sparse_points.json"));
    tieContext.maskPaths = options.maskPaths;
    tieContext.cancelFlag = options.cancelFlag.get();
    tieContext.computeDeviceCallback = options.computeDeviceFn;
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
    resolved.prepareTiePoints = options.guidedImageMatching ||
        !options.reuseExistingMatches ||
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
    settings.insert(QStringLiteral("camera_overlap_top_k_per_image"),
                    tieOptions.pairPolicy.cameraOverlapTopKPerImage);
    settings.insert(QStringLiteral("vocabulary_top_k_per_image"),
                    tieOptions.pairPolicy.vocabularyTopKPerImage);
    settings.insert(QStringLiteral("sequence_pair_window"), tieOptions.pairPolicy.sequenceWindow);
    settings.insert(QStringLiteral("sequence_loop_closure"), pipeline.sequenceLoopClosure);
    settings.insert(QStringLiteral("sequence_geometry_source"),
                    usesPhotoSequence
                        ? QStringLiteral("explicit_sequence")
                        : (estimatedSequence.detected
                               ? QStringLiteral("estimated_pose_detected")
                               : QStringLiteral("none")));
    settings.insert(QStringLiteral("estimated_sequence_optical_axis_concentration"),
                    estimatedSequence.opticalAxisConcentration);
    settings.insert(QStringLiteral("estimated_sequence_inward_axis_median"),
                    estimatedSequence.inwardAxisMedian);
    settings.insert(QStringLiteral("estimated_sequence_adjacent_mad_ratio"),
                    estimatedSequence.adjacentDistanceMadRatio);
    settings.insert(QStringLiteral("estimated_sequence_adjacent_maximum_ratio"),
                    std::isfinite(estimatedSequence.adjacentDistanceMaximumRatio)
                        ? estimatedSequence.adjacentDistanceMaximumRatio
                        : 0.0);
    settings.insert(QStringLiteral("reference_overlap_geometry"),
                    tieOptions.referencePreselectionGeometry ==
                            matchphotos::ReferencePreselectionGeometry::SparseScene
                        ? QStringLiteral("sparse_scene_covisibility_frustum")
                        : QStringLiteral("ground_reference_sphere"));
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
    settings.insert(QStringLiteral("sift_maximum_ratio"), tieOptions.siftMaximumRatio);
    settings.insert(QStringLiteral("sift_minimum_adaptive_ratio"),
                    tieOptions.siftMinimumAdaptiveRatio);
    settings.insert(QStringLiteral("adaptive_sift_ratio"), tieOptions.adaptiveSiftRatio);
    settings.insert(QStringLiteral("geometry_reprojection_threshold_px"),
                    tieOptions.geometryReprojThreshold);
    settings.insert(QStringLiteral("geometry_min_inliers"), tieOptions.geometryMinInliers);
    settings.insert(QStringLiteral("geometry_min_inlier_ratio"), tieOptions.geometryMinInlierRatio);
    settings.insert(QStringLiteral("geometry_min_grid_coverage"),
                    tieOptions.geometryMinGridCoverage);
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
                        : (options.guidedImageMatching
                               ? QStringLiteral("guided_refresh")
                               : (resolved.prepareTiePoints ? QStringLiteral("fill_missing")
                                                            : QStringLiteral("reuse"))));
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
        const matchphotos::MatchPhotosAlgorithmPlan &algorithmPlan =
            result.tiePointResult.algorithmPlan;
        if (!algorithmPlan.computeDeviceDisplayName.trimmed().isEmpty())
        {
            result.config.resolvedSettings.insert(
                QStringLiteral("matching_compute_backend"),
                QString::fromLatin1(image_matching::siftBackendName(
                    algorithmPlan.executionBackend)));
            result.config.resolvedSettings.insert(
                QStringLiteral("matching_compute_device_name"),
                algorithmPlan.computeDeviceName);
            result.config.resolvedSettings.insert(
                QStringLiteral("matching_compute_device_display"),
                algorithmPlan.computeDeviceDisplayName);
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
