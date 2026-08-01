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
    int threadsPerWorker = 0; ///< 每个 IncrementalSfm 的 CPU 线程预算。
};

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
    bool hasClosedSequenceGeometry = false; ///< 全部序列影像注册且首尾可评估。
    double sequenceAdjacentDistanceMedian = 0.0; ///< 含闭环边的相邻中心距离中位数。
    double sequenceAdjacentDistanceMaximumRatio = 0.0; ///< 最大相邻距离/中位数。
    double sequenceAdjacentDistanceMadRatio = 0.0; ///< 相邻距离 MAD/中位数。
};

/// 在总线程预算内分配最多 4 个候选 worker，并为每个保留足够内部并行度。
SfmWorkerBudget allocateWorkers(int candidateCount, int totalThreads);

/// 按生产摄影测量质量执行严格弱序比较。
bool isBetterCandidate(const SfmCandidateSummary &candidate,
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

/// 已全量注册且存在生产稀疏云时可提前停止更多焦距重放。
bool shouldStopAdaptiveFocalReplay(int totalImages,
                                   int registeredImages,
                                   bool hasProductionSparseCloud);

} // namespace xjw::aerial_triangulation
