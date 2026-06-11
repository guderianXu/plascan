#pragma once

// ============================================================
// 文件：SfmPointCloudFilter.h
// 功能：SfM 稀疏点云批量过滤器。
//
// 参考 COLMAP 的 point_cloud_filter 工具，提供多种过滤策略：
//   1. 最大重投影误差过滤
//   2. 最小轨迹长度过滤
//   3. 最小三角化角度过滤
//   4. 统计离群点过滤（基于 KNN 距离）
// ============================================================

#include "reconstruction/SfmReconstruction.h"

#include <plapoint/filters/preprocessing.h>

#include <functional>
#include <string>

namespace xjw {

/**
 * @brief 稀疏点云过滤选项。
 */
struct SfmPointCloudFilterOptions {
    /// 是否启用重投影误差过滤
    bool filterByReprojError = true;
    /// 最大允许重投影误差（像素）
    double maxReprojError = 2.0;

    /// 是否启用最小轨迹长度过滤
    bool filterByTrackLen = true;
    /// 最小轨迹长度（观测数 < 此值的点将被剔除）
    int minTrackLen = 3;

    /// 是否启用三角化角度过滤
    bool filterByTriAngle = true;
    /// 最小三角化角（度）
    double minTriAngleDeg = 2.0;

    /// 是否启用统计离群点过滤
    bool filterByStatistical = true;
    /// KNN 邻居数
    int statK = 16;
    /// 标准差倍数阈值（平均距离超过 mean + stdDevMul * stdDev 的点被剔除）
    double statStdDevMul = 2.5;

    /// 通用点云过滤在 plapoint 中使用的处理设备
    plapoint::ProcessingDevice processingDevice = plapoint::ProcessingDevice::Auto;
};

/**
 * @brief 稀疏点云过滤结果。
 */
struct SfmPointCloudFilterResult {
    int pointsBefore = 0;           ///< 过滤前点数
    int pointsAfter = 0;            ///< 过滤后点数

    int removedByReprojError = 0;   ///< 因重投影误差过大被移除的点数
    int removedByTrackLen = 0;      ///< 因轨迹过短被移除的点数
    int removedByTriAngle = 0;      ///< 因三角化角过小被移除的点数
    int removedByStatistical = 0;   ///< 因统计离群被移除的点数

    /// 可读结果描述
    std::string summary() const;
};

/**
 * @brief 进度回调：(当前步骤描述, 进度百分比 0-100)，返回 false 中止。
 */
using FilterProgressCallback = std::function<bool(const std::string &step, int percent)>;

/**
 * @brief SfM 稀疏点云批量过滤器。
 *
 * 对 SfmReconstruction 中的 3D 点依次应用各过滤策略，
 * 直接修改重建对象中的点云数据。
 *
 * 典型用法：
 * @code
 *   SfmPointCloudFilter filter(reconstruction);
 *   auto result = filter.run(options, progressCallback);
 * @endcode
 */
class SfmPointCloudFilter {
public:
    explicit SfmPointCloudFilter(SfmReconstruction &reconstruction);

    /**
     * @brief 执行过滤。
     * @param options     过滤选项
     * @param progressCb  可选进度回调
     * @return 过滤结果统计
     */
    SfmPointCloudFilterResult run(
        const SfmPointCloudFilterOptions &options = SfmPointCloudFilterOptions(),
        FilterProgressCallback progressCb = nullptr);

private:
    SfmReconstruction &reconstruction_;

    /// 按重投影误差过滤
    int filterByReprojError(double maxError);

    /// 按最小轨迹长度过滤
    int filterByTrackLen(int minLen);

    /// 按三角化角度过滤
    int filterByTriAngle(double minAngleDeg);

    /// 统计离群点过滤（基于 KNN 空间距离）
    int filterByStatistical(int k,
                            double stdDevMul,
                            plapoint::ProcessingDevice processingDevice);
};

} // namespace xjw
