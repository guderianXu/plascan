#pragma once

// ============================================================
// 文件：InitialSparsePointCloudTriangulator.h
// 功能：基于已有相机与匹配轨迹过滤初始稀疏点云。
//
// 说明：
//   - 本模块只负责算法筛选与三维点质量评估；
//   - 不依赖 Qt，也不负责文件写出；
//   - 供 GUI 层和后续无头流程统一复用。
// ============================================================

#include "BundleAdjust.h"
#include "Camera.h"

#include <array>
#include <string>
#include <vector>

namespace xjw
{

/**
 * @brief 单个初始稀疏点的导出数据。
 */
struct InitialSparsePoint
{
    std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
    int sourceTrackIndex = -1;  ///< 该三维点对应的原始轨迹下标（可用于颜色采样）
    int trackLength = 0;
    double minTriAngleDeg = 0.0;
    double rmsReprojPx = 0.0;
};

/**
 * @brief 初始稀疏点云三角化选项。
 */
struct InitialSparseTriangulationOptions
{
    double minTriAngleDeg = 2.0;
    double maxReprojErrorPx = 2.0;
    int minObservations = 2;
    bool ignoreTwoViewTracks = false;
    int minTrackLength = 2;
};

/**
 * @brief 初始稀疏点云三角化结果。
 */
struct InitialSparseTriangulationResult
{
    bool success = false;
    int candidateTrackCount = 0;
    int exportedPointCount = 0;
    int rejectedByObservationCount = 0;
    int rejectedByTriAngleCount = 0;
    int rejectedByReprojCount = 0;
    std::string errorMessage;
    std::vector<InitialSparsePoint> points;
};

/**
 * @brief 初始稀疏点云过滤器。
 *
 * 输入为已有相机和匹配轨迹，输出经过基础几何约束过滤后的初始三维点。
 */
class InitialSparsePointCloudFilter
{
public:
    /**
     * @brief 生成初始稀疏点云。
     *
     * @param cameras  相机列表
     * @param tracks   匹配轨迹列表
     * @param options  三角化过滤选项
     * @return 过滤后的初始稀疏点结果
     */
    static InitialSparseTriangulationResult triangulate(
        const std::vector<Camera> &cameras,
        const std::vector<BATrack> &tracks,
        const InitialSparseTriangulationOptions &options = InitialSparseTriangulationOptions());
};

using InitialSparsePointCloudTriangulator = InitialSparsePointCloudFilter;

} // namespace xjw