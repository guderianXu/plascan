/**
 * @file SfmSearchPolicy.cpp
 * @brief 多焦距/多初始化 SfM 候选的资源分配和确定性排序实现。
 *
 * 排序不是单纯最小化 RMS：优先保证重建成功和注册覆盖，再评价交会角、多视轨迹、
 * 影像覆盖及照片序列连续性。低 RMS 但塌缩或弱基线的模型因此不会胜出。
 */

#include "search/SfmSearchPolicy.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace xjw::aerial_triangulation
{

namespace
{

// 网络质量是由多个有噪声的汇总量组合而成。候选之间只有很小的分数差时，
// 不应让单项（尤其两视轨迹比例）压过更直接的重投影误差和有效三维点数量。
constexpr double kMaterialPhotogrammetricQualityDelta = 0.02;

/// 将交会角、多视率、观测覆盖和 RMS 压缩成 [0, 1] 网络质量分数。
double networkQualityScore(const SfmCandidateSummary &candidate)
{
    const double angleQuality = std::clamp(candidate.medianTriangulationAngleDeg / 10.0, 0.0, 1.0);
    const double multiViewQuality = std::clamp(1.0 - candidate.twoViewTrackRatio, 0.0, 1.0);
    const double coverageQuality = std::clamp(candidate.observationGridCoverage / 0.25, 0.0, 1.0);
    const double reprojectionQuality = std::isfinite(candidate.meanReprojError)
        ? 1.0 / (1.0 + std::max(0.0, candidate.meanReprojError))
        : 0.0;

    // 三角交会角和多视轨迹决定摄影测量网刚性；空间覆盖和 RMS 用于细化排序。
    return 0.35 * angleQuality +
           0.30 * multiViewQuality +
           0.20 * coverageQuality +
           0.15 * reprojectionQuality;
}

/// 根据闭环相邻相机中心距离的鲁棒离散程度评估局部连续性。
double closedSequenceQualityScore(const SfmCandidateSummary &candidate)
{
    if (!candidate.hasClosedSequenceGeometry ||
        !std::isfinite(candidate.sequenceAdjacentDistanceMaximumRatio) ||
        !std::isfinite(candidate.sequenceAdjacentDistanceMadRatio))
    {
        return 0.0;
    }

    const double maximumRatio = std::max(1.0, candidate.sequenceAdjacentDistanceMaximumRatio);
    const double madRatio = std::max(0.0, candidate.sequenceAdjacentDistanceMadRatio);

    // MAD 对单个错误相机更稳健；最大值仍保留较低权重，用于惩罚闭环中的局部断裂。
    // 两项都映射到 [0, 1]，避免任一原始极值直接支配候选排序。
    const double robustContinuity = std::exp(-4.0 * madRatio);
    const double worstEdgeContinuity = std::exp(-0.75 * (maximumRatio - 1.0));
    return 0.65 * robustContinuity + 0.35 * worstEdgeContinuity;
}

double photogrammetricQualityScore(const SfmCandidateSummary &candidate)
{
    const double aerialBlockQuality = candidate.hasAerialBlockGeometry
        ? 1.0 / (1.0 + 100.0 * std::max(0.0, candidate.cameraCenterPlanarityRatio))
        : 0.0;

    if (candidate.hasNetworkQuality && candidate.hasClosedSequenceGeometry)
    {
        return 0.50 * networkQualityScore(candidate) +
               0.50 * closedSequenceQualityScore(candidate);
    }
    if (candidate.hasNetworkQuality && candidate.hasAerialBlockGeometry)
    {
        // 近垂直摄影中，焦距与高程存在强相关。仅按交会角和点数会偏向
        // 人工穹顶解，因此把相机中心平面性作为同等级的几何刚性指标。
        return 0.65 * networkQualityScore(candidate) +
               0.35 * aerialBlockQuality;
    }
    if (candidate.hasNetworkQuality)
    {
        return networkQualityScore(candidate);
    }
    if (candidate.hasClosedSequenceGeometry)
    {
        return closedSequenceQualityScore(candidate);
    }
    return 0.0;
}

} // namespace

int resolveSfmThreadBudget(int requestedThreads)
{
    const int hardwareThreads =
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    return requestedThreads > 0
        ? std::clamp(requestedThreads, 1, hardwareThreads)
        : hardwareThreads;
}

SfmWorkerBudget allocateWorkers(int candidateCount, int totalThreads)
{
    if (candidateCount <= 0)
    {
        return {};
    }

    const int safeThreads = resolveSfmThreadBudget(totalThreads);
    // 粗焦距候选的大部分时间花在串行/弱并行的 PnP、三角化和图更新上。给单个
    // 候选固定预留 8 线程会造成大量核心空闲；让每个候选先占一个 worker，再把
    // 不能整除的逻辑线程作为内部 BA 余数分配，恰好消费全部硬件线程预算。
    const int workerCount = std::min(candidateCount, safeThreads);
    const int threadsPerWorker = safeThreads / workerCount;
    const int workersWithExtraThread = safeThreads % workerCount;
    return {workerCount, threadsPerWorker, workersWithExtraThread};
}

int focalProbeRegistrationLimit(int totalImages)
{
    // 小型环拍数据需要完整闭环才能可靠区分焦距。大工程先用 24 台相机形成
    // 轻量局部网估计焦距区间，胜出尺度再全量重放；失败时仍有全尺度兜底。
    // 这避免无先验时默认把每个焦距候选都扩展到 64 台相机。
    return totalImages > 96 ? std::min(totalImages, 24) : 0;
}

SfmBaSchedule resolveSfmBaSchedule(int totalImages,
                                   int baseLocalInterval,
                                   int baseLocalWindowImages,
                                   int baseGlobalInterval)
{
    SfmBaSchedule schedule{
        std::max(1, baseLocalInterval),
        std::max(2, baseLocalWindowImages),
        std::max(1, baseGlobalInterval)};
    if (totalImages <= 96)
    {
        return schedule;
    }

    schedule.localInterval = std::max(schedule.localInterval,
                                      totalImages > 256 ? 8 : 6);
    schedule.localWindowImages = std::max(schedule.localWindowImages,
                                          totalImages > 256 ? 12 : 10);
    schedule.globalInterval = std::max(schedule.globalInterval,
                                       std::max(24, totalImages / 8));
    return schedule;
}

bool isBetterCandidate(const SfmCandidateSummary &candidate,
                       const SfmCandidateSummary &reference)
{
    // 以下比较顺序也是稳定的产品语义，修改时必须同步候选策略测试。
    if (candidate.success != reference.success)
    {
        return candidate.success;
    }
    if (candidate.registeredImages != reference.registeredImages)
    {
        return candidate.registeredImages > reference.registeredImages;
    }
    if (candidate.hasClosedSequenceGeometry != reference.hasClosedSequenceGeometry)
    {
        return candidate.hasClosedSequenceGeometry;
    }
    if (candidate.hasNetworkQuality != reference.hasNetworkQuality)
    {
        return candidate.hasNetworkQuality;
    }
    if (candidate.hasNetworkQuality || candidate.hasClosedSequenceGeometry)
    {
        const double candidateQuality = photogrammetricQualityScore(candidate);
        const double referenceQuality = photogrammetricQualityScore(reference);
        const double materialDelta =
            candidate.hasClosedSequenceGeometry || reference.hasClosedSequenceGeometry
            ? 0.0
            : kMaterialPhotogrammetricQualityDelta;
        if (std::abs(candidateQuality - referenceQuality) > materialDelta)
        {
            return candidateQuality > referenceQuality;
        }
    }
    const bool candidateRmsFinite = std::isfinite(candidate.meanReprojError);
    const bool referenceRmsFinite = std::isfinite(reference.meanReprojError);
    if (candidateRmsFinite != referenceRmsFinite)
    {
        return candidateRmsFinite;
    }
    if (candidateRmsFinite && candidate.meanReprojError != reference.meanReprojError)
    {
        return candidate.meanReprojError < reference.meanReprojError;
    }
    if (candidate.points3D != reference.points3D)
    {
        return candidate.points3D > reference.points3D;
    }
    return candidate.candidateIndex < reference.candidateIndex;
}

bool isAcceptableCalibrationRefinement(const SfmCandidateSummary &candidate,
                                       const SfmCandidateSummary &reference)
{
    if (!candidate.success || !reference.success ||
        candidate.registeredImages < reference.registeredImages)
    {
        return false;
    }

    const int toleratedPointLoss = std::max(
        25,
        static_cast<int>(std::ceil(0.02 * static_cast<double>(reference.points3D))));
    if (candidate.points3D + toleratedPointLoss < reference.points3D)
    {
        return false;
    }

    const bool candidateRmsFinite = std::isfinite(candidate.meanReprojError);
    const bool referenceRmsFinite = std::isfinite(reference.meanReprojError);
    if (!candidateRmsFinite)
    {
        return false;
    }
    if (referenceRmsFinite &&
        candidate.meanReprojError > reference.meanReprojError * 1.02 + 1.0e-6)
    {
        return false;
    }

    if ((candidate.hasNetworkQuality || candidate.hasClosedSequenceGeometry) &&
        (reference.hasNetworkQuality || reference.hasClosedSequenceGeometry) &&
        photogrammetricQualityScore(candidate) + kMaterialPhotogrammetricQualityDelta <
            photogrammetricQualityScore(reference))
    {
        return false;
    }
    return true;
}

std::vector<SfmCandidateSummary> rankCandidates(
    const std::vector<SfmCandidateSummary> &candidates)
{
    std::vector<SfmCandidateSummary> ranked = candidates;
    std::stable_sort(ranked.begin(), ranked.end(), isBetterCandidate);
    return ranked;
}

std::vector<int> replayCandidateIndices(
    const std::vector<SfmCandidateSummary> &rankedCandidates,
    int maxReplayCount)
{
    const int count = std::min(
        std::max(0, maxReplayCount),
        static_cast<int>(rankedCandidates.size()));
    std::vector<int> indices;
    indices.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        indices.push_back(rankedCandidates[static_cast<std::size_t>(i)].candidateIndex);
    }
    return indices;
}

std::vector<double> adaptiveFocalScaleCandidates()
{
    // 0.95~1.05 覆盖全画幅 35 mm 等常见航测相机。该区间不能只用 0.85/1.2
    // 两端代替，否则近垂直航测块会通过弯曲相机轨迹吸收焦距误差。
    // 同时保留转台和 Hayabusa2 ONC-T（约 9 倍影像宽度）的长焦范围。
    return {0.55, 0.70, 0.85, 0.95, 1.0, 1.05, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 4.0, 5.2, 6.4,
            8.0, 9.0, 10.0};
}

std::vector<double> adaptiveFocalCoarseScaleCandidates()
{
    // 五个几何锚点用于轻量估计；仍覆盖广角、普通摄影、远摄与约 9 倍影像宽度
    // 的行星长焦。成功后只细化最佳邻域，失败才进入完整 18 尺度兜底。
    return {0.55, 1.0, 2.4, 5.2, 9.0};
}

std::vector<double> adaptiveFocalRefinementScaleCandidates(
    const std::vector<SfmCandidateSummary> &rankedCandidates,
    const std::vector<double> &evaluatedScales,
    int maxSeedCount)
{
    if (maxSeedCount <= 0 || rankedCandidates.empty())
    {
        return {};
    }

    constexpr double kScaleTolerance = 1.0e-9;
    const std::vector<double> allScales = adaptiveFocalScaleCandidates();
    std::vector<double> refinementScales;
    const auto alreadyEvaluated = [&](double scale)
    {
        return std::any_of(evaluatedScales.cbegin(), evaluatedScales.cend(), [scale](double value)
        {
            return std::abs(value - scale) <= kScaleTolerance;
        }) || std::any_of(refinementScales.cbegin(), refinementScales.cend(), [scale](double value)
        {
            return std::abs(value - scale) <= kScaleTolerance;
        });
    };

    const int seedCount = std::min(maxSeedCount, static_cast<int>(rankedCandidates.size()));
    for (int seedIndex = 0; seedIndex < seedCount; ++seedIndex)
    {
        const double seedScale = rankedCandidates[static_cast<std::size_t>(seedIndex)].focalScale;
        const auto upper = std::lower_bound(allScales.cbegin(), allScales.cend(), seedScale);
        if (upper != allScales.cbegin())
        {
            const double lowerScale = *std::prev(upper);
            if (!alreadyEvaluated(lowerScale))
            {
                refinementScales.push_back(lowerScale);
            }
        }

        auto upperNeighbor = upper;
        if (upperNeighbor != allScales.cend() &&
            std::abs(*upperNeighbor - seedScale) <= kScaleTolerance)
        {
            ++upperNeighbor;
        }
        if (upperNeighbor != allScales.cend() && !alreadyEvaluated(*upperNeighbor))
        {
            refinementScales.push_back(*upperNeighbor);
        }
    }
    return refinementScales;
}

bool shouldStopAdaptiveFocalReplay(int totalImages,
                                   int registeredImages,
                                   bool hasProductionSparseCloud)
{
    // 只有全量注册且已生成可发布稀疏云时才提前结束，不能仅凭低 RMS 停止。
    return totalImages > 0 &&
           registeredImages >= totalImages &&
           hasProductionSparseCloud;
}

bool shouldRunAdaptiveCameraModelRefinement(bool requested,
                                            bool hasCompleteIntrinsicPrior,
                                            bool hasMetadataFocalPrior)
{
    // EXIF 只提供焦距弱先验，不包含 Brown-Conrady 镜头畸变，不能据此跳过
    // 自标定。完整工程/外部相机标定才保持固定；元数据焦距只负责缩小焦距边界。
    (void)hasMetadataFocalPrior;
    return requested && !hasCompleteIntrinsicPrior;
}

} // namespace xjw::aerial_triangulation
