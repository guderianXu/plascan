#pragma once

/**
 * @file SfmBundleAdjustCoordinator.h
 * @brief SfM 状态与独立 bundle_adjust 模块之间的适配/调度层。
 *
 * 协调器从当前 SfmReconstruction 构造 Camera/BATrack，选择局部或全局相机集合，
 * 调用 BundleAdjust 公共入口，再将通过质量门控的相机和点原子写回重建。
 */

#include "IncrementalSfm.h"

namespace xjw
{

/// 负责 BA 调度，不拥有 IncrementalSfm。
class SfmBundleAdjustCoordinator
{
  public:
    explicit SfmBundleAdjustCoordinator(IncrementalSfm &owner);

    /**
     * @brief 执行一次局部或全局 BA。
     * @param localOnly true 时只释放当前局部窗口相机，窗口外相机作为 gauge 锚。
     * @param anchorIds 调用方额外要求固定的影像 ID。
     */
    void run(bool localOnly = false, const std::vector<ImageId> &anchorIds = {});

    /// 按“BA -> 重三角化 -> 过滤”的稳定化循环执行配置的全局优化轮次。
    void iterative();

    /// 删除在任一有效轨迹观测中落到相机后方的三维点，并清理影像关联。
    int filterNegativeDepthPoints();

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw
