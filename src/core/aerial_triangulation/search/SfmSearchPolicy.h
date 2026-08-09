#pragma once

/**
 * @file SfmSearchPolicy.h
 * @brief 焦距/初始模型候选的并行预算、摄影测量排序和重放策略。
 *
 * 候选排序首先比较成功与注册覆盖，再比较闭环连续性、交会角、多视轨迹比例、
 * 空间覆盖和重投影误差。不能只按点数或 RMS 选择，否则退化的扁平模型可能胜出。
 */

#include <cstdint>
#include <vector>

namespace xjw::aerial_triangulation
{

struct SfmWorkerBudget
{
    int workerCount = 0; ///< 并行候选 worker 数。
    int threadsPerWorker = 0; ///< 每个 worker 的基础线程预算。
    int workersWithExtraThread = 0; ///< 前若干 worker 各多分配一个余数线程。

    /// 返回指定 worker 的内部 SfM/BA 线程预算。
    int threadsForWorker(int workerIndex) const
    {
        if (workerIndex < 0 || workerIndex >= workerCount)
        {
            return 0;
        }
        return threadsPerWorker + (workerIndex < workersWithExtraThread ? 1 : 0);
    }

    /// 返回全部候选 worker 占用的总线程预算。
    int totalThreads() const
    {
        return workerCount * threadsPerWorker + workersWithExtraThread;
    }
};

struct SfmBaSchedule
{
    int localInterval = 1; ///< 每注册多少台相机执行一次局部 BA。
    int localWindowImages = 1; ///< 局部 BA 邻域相机数。
    int globalInterval = 1; ///< 每注册多少台相机执行一次中间全局 BA。
};

/// 将 0/负数线程请求解析为本机逻辑核心数，并把显式请求限制在硬件预算内。
int resolveSfmThreadBudget(int requestedThreads);

/// 一个完整 SfM 候选的排序摘要。
struct SfmCandidateSummary
{
    int candidateIndex = -1; ///< 原候选数组下标，用作最终稳定 tie-break。
    double focalScale = 0.0; ///< 焦距像素/最长边。
    std::uint32_t initialImageId1 = 0; ///< 实际选中初始对第一 ID。
    std::uint32_t initialImageId2 = 0; ///< 实际选中初始对第二 ID。
    int registeredImages = 0; ///< 最终注册影像数。
    int points3D = 0; ///< 最终稀疏点数。
    double meanReprojError = 0.0; ///< 像素重投影误差。
    bool success = false; ///< 单次尝试是否通过自身质量门控。
    bool hasNetworkQuality = false; ///< 以下稀疏网指标是否有效。
    double medianTriangulationAngleDeg = 0.0; ///< 点交会角中位数。
    double twoViewTrackRatio = 1.0; ///< 两视点/全部点。
    double observationGridCoverage = 0.0; ///< 平均影像网格覆盖。
    bool hasAerialBlockGeometry = false; ///< 相机光轴近似平行，可按航测块评价中心平面性。
    double opticalAxisConcentration = 0.0; ///< 单位光轴平均向量长度，1 表示完全平行。
    double cameraCenterPlanarityRatio = 1.0; ///< 相机中心协方差最小特征值/迹，越小越接近平面。
    bool hasClosedSequenceGeometry = false; ///< 全部序列影像注册且首尾可评估。
    double sequenceAdjacentDistanceMedian = 0.0; ///< 含闭环边的相邻中心距离中位数。
    double sequenceAdjacentDistanceMaximumRatio = 0.0; ///< 最大相邻距离/中位数。
    double sequenceAdjacentDistanceMadRatio = 0.0; ///< 相邻距离 MAD/中位数。
};

/// 按候选数拆分全部逻辑线程；空闲线程作为余数分配给前几个 worker。
SfmWorkerBudget allocateWorkers(int candidateCount, int totalThreads);

/// 大规模无先验焦距搜索的注册上限；0 表示数据规模较小，应完整评估候选。
int focalProbeRegistrationLimit(int totalImages);

/// 根据工程规模放宽中间 BA 调度；最终全局 BA 不由该策略裁剪。
SfmBaSchedule resolveSfmBaSchedule(int totalImages,
                                  int baseLocalInterval,
                                  int baseLocalWindowImages,
                                  int baseGlobalInterval);

/// 按生产摄影测量质量执行严格弱序比较。
bool isBetterCandidate(const SfmCandidateSummary &candidate,
                       const SfmCandidateSummary &reference);

/// 自标定候选在注册覆盖不降、RMS/点数/网络质量仅有统计波动时可以替代粗焦距解。
bool isAcceptableCalibrationRefinement(const SfmCandidateSummary &candidate,
                                       const SfmCandidateSummary &reference);

/// 稳定排序并保留原 candidateIndex。
std::vector<SfmCandidateSummary> rankCandidates(
    const std::vector<SfmCandidateSummary> &candidates);

/// 从已排序候选选择最多 maxReplayCount 个原始下标进行正式重放。
std::vector<int> replayCandidateIndices(
    const std::vector<SfmCandidateSummary> &rankedCandidates,
    int maxReplayCount);

/// 返回覆盖广角、普通相机、长焦/行星相机的无量纲焦距初始化集合。
std::vector<double> adaptiveFocalScaleCandidates();

/// 返回第一阶段代表尺度；保持广角、常用航测焦距和约 9 倍影像宽度的长焦锚点。
std::vector<double> adaptiveFocalCoarseScaleCandidates();

/**
 * @brief 从粗搜排序前列候选生成第二阶段相邻尺度。
 *
 * 每个 seed 在完整尺度表中最多扩展左右各一个相邻尺度，已评估尺度会被跳过。
 * 返回顺序只由 rankedCandidates 和固定尺度表决定，保证跨线程数确定性。
 */
std::vector<double> adaptiveFocalRefinementScaleCandidates(
    const std::vector<SfmCandidateSummary> &rankedCandidates,
    const std::vector<double> &evaluatedScales,
    int maxSeedCount = 2);

/// 已全量注册且存在生产稀疏云时可提前停止更多焦距重放。
bool shouldStopAdaptiveFocalReplay(int totalImages,
                                   int registeredImages,
                                   bool hasProductionSparseCloud);

/// 完整相机标定保持内参固定；EXIF 焦距仅作为弱先验，仍允许联合估计镜头畸变。
bool shouldRunAdaptiveCameraModelRefinement(bool requested,
                                            bool hasCompleteIntrinsicPrior,
                                            bool hasMetadataFocalPrior);

} // namespace xjw::aerial_triangulation
