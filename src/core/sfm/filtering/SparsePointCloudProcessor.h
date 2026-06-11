#pragma once

#include <plapoint/filters/preprocessing.h>

#include <vector>

namespace xjw 
{

struct SparsePointCloudPoint 
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double rmsReprojPx = 0.0;
    double minTriAngleDeg = 0.0;
    int trackLen = 0;
};

struct SparsePointCloudFilterOptions 
{
    bool filterByReprojError = true;
    double maxReprojError = 2.5;

    bool filterByTrackLen = true;
    int minTrackLen = 3;

    bool filterByTriAngle = true;
    double minTriAngleDeg = 2.0;

    bool filterByStatistical = true;
    int statK = 16;
    double statStdDevMul = 2.5;

    bool filterByNormalConsistency = false;
    double normalConsistencyMinMeanAbsDot = 0.45;

    /// 是否启用半径密度过滤
    bool filterByDensity = false;
    /// 密度过滤球半径（场景单位，如米）
    double densityRadius = 0.5;
    /// 半径内最少邓居数（少于此则删除）
    int densityMinNeighbors = 5;

    /// 通用点云过滤在 plapoint 中使用的处理设备
    plapoint::ProcessingDevice processingDevice = plapoint::ProcessingDevice::Auto;
};

struct SparsePointCloudFilterStats 
{
    int inputPoints = 0;
    int outputPoints = 0;
    int removedByReprojError = 0;
    int removedByTrackLen = 0;
    int removedByTriAngle = 0;
    int removedByStatistical = 0;
    int removedByNormalConsistency = 0;
    int removedByDensity = 0;
};

struct SparsePointCloudRefineRound 
{
    int round = 0;
    int inputPoints = 0;
    double maxReprojError = 0.0;
    double minTriAngleDeg = 0.0;
    int minTrackLen = 0;
    double statStdDevMul = 0.0;
    SparsePointCloudFilterStats stats;
};

struct SparsePointCloudOptimizeRound
{
    int round = 0;
    int inputPoints = 0;
    int outputPoints = 0;
    SparsePointCloudFilterOptions filterOptions;
    SparsePointCloudFilterStats filterStats;
};

struct SparsePointCloudRefineOptions 
{
    int knnNeighbors = 20;
    double stdDevMultiplier = 2.0;
    double maxReprojError = 4.0;
    double minTriAngleDeg = 2.0;
    int minTrackLen = 3;
    int iterRounds = 3;
    bool retriangulate = true;
    bool normalConsistency = false;
    double normalConsistencyMinMeanAbsDot = 0.45;
    plapoint::ProcessingDevice processingDevice = plapoint::ProcessingDevice::Auto;
};

/**
 * @brief 稀疏点云空间清理选项。
 *
 * 基于体素网格对点云做空间局部质量控制，包括:
 *   1. 孤岛体素副除（少于 minVoxelPoints 点的体素整体副除）
 *   2. 体素内局部重投影误差过滤
 *   3. 几何等价点去重
 */
struct SparsePointCloudSpatialCleanupOptions 
{
    /// 体素边长；0 表示自动估算（中位最近1邻距离 * 3）
    double voxelSize           = 0.0;
    /// 孤岛体素最少点数：体素内点数 < 此就将整体副除
    int    minVoxelPoints      = 2;
    /// 是否在每个体素内做局部重投影误差过滤
    bool   localReprojFilter   = true;
    /// 局部重投影误差过滤标准差倍数
    double localReprojStdMul   = 2.5;
    /// 点去重半径（< 0 表示禁用；范围内保留 trackLen 最大者）
    double deduplicationRadius = -1.0;
};

/**
 * @brief 稀疏点云空间清理结果。
 */
struct SparsePointCloudSpatialCleanupResult 
{
    int inputPoints             = 0;   ///< 输入点数
    int outputPoints            = 0;   ///< 输出点数
    int removedByVoxelIsolation = 0;   ///< 孤岛体素副除
    int removedByLocalReproj    = 0;   ///< 局部重投影副除
    int removedByDeduplication  = 0;   ///< 去重副除
};

using SparseCloudLocalOptimOptions = SparsePointCloudSpatialCleanupOptions;
using SparseCloudLocalOptimResult = SparsePointCloudSpatialCleanupResult;

struct SparsePointCloudOptimizeOptions
{
    SparsePointCloudFilterOptions filterOptions;
    bool iterative = false;
    int iterRounds = 1;
    bool restartFromInputEachRound = false;
    bool tightenThresholds = false;
    bool enableSpatialCleanup = false;
    SparsePointCloudSpatialCleanupOptions spatialCleanupOptions;
};

struct SparsePointCloudOptimizeResult
{
    std::vector<SparsePointCloudPoint> points;
    int inputPoints = 0;
    int outputPoints = 0;
    int removedTotal = 0;
    std::vector<SparsePointCloudOptimizeRound> rounds;
};

struct SparsePointCloudRefineResult 
{
    std::vector<SparsePointCloudPoint> points;
    int inputPoints = 0;
    int outputPoints = 0;
    int removedTotal = 0;
    std::vector<SparsePointCloudRefineRound> rounds;
};

class SparsePointCloudProcessor 
{
public:
    static SparsePointCloudFilterStats filter(std::vector<SparsePointCloudPoint> *points,
                                              const SparsePointCloudFilterOptions &options);

    static SparsePointCloudRefineResult refine(const std::vector<SparsePointCloudPoint> &basePoints,
                                               const SparsePointCloudRefineOptions &options);

    static SparsePointCloudOptimizeResult optimize(const std::vector<SparsePointCloudPoint> &basePoints,
                                                   const SparsePointCloudOptimizeOptions &options);

    /**
     * @brief 基于空间结构清理稀疏点云。
     *
     * 基于体素网格对点云做空间局部质量控制：
     *   1. 删除孤岛体素（内部点数 < minVoxelPoints）
     *   2. 体素内局部重投影误差高的点
     *   3. 可选：删除几何等价的冠余点（保留轨迹最长者）
     */
    static SparsePointCloudSpatialCleanupResult spatialCleanup(
        std::vector<SparsePointCloudPoint> *points,
        const SparsePointCloudSpatialCleanupOptions &options);

    static SparseCloudLocalOptimResult localOptim(std::vector<SparsePointCloudPoint> *points,
                                                  const SparseCloudLocalOptimOptions &options);
};

} // namespace xjw
