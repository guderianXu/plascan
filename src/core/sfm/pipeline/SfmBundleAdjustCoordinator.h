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

    /// 按“BA -> 重三角化 -> 过滤”的稳定化循环执行全局优化。
    /// 周期稳定化最多两轮；最终精化使用完整配置轮数并允许提前收敛。
    void iterative(bool finalRefinement = false);

    /// 删除在任一有效轨迹观测中落到相机后方的三维点，并清理影像关联。
    int filterNegativeDepthPoints();

    /// 共享镜头内参只在全部影像注册后的全局 BA 中更新，避免改变待注册影像的 PnP 几何。
    static bool shouldRefineSharedIntrinsics(bool localOnly,
                                             int activeCameraCount,
                                             int registeredImageCount,
                                             int totalImageCount);

    /// 同时检查点网和共享镜头参数，避免点数稳定但自标定仍在漂移时提前结束。
    static bool hasIterativeGlobalBaConverged(int completedRoundCount,
                                              double pointChangeRate,
                                              bool sharedIntrinsicsRefined,
                                              double focalScaleChange,
                                              double radialCoefficientChange);

    /// 接近注册终点时跳过周期全局 BA，避免数张影像后立刻重复最终全局 BA。
    static bool shouldRunPeriodicGlobalBa(int registeredImageCount,
                                          int registrationTarget,
                                          int iterationsSinceGlobalBa,
                                          int globalBaInterval);

    /// 大型弱连接网的全局 BA 只使用三视图以上轨迹；两视图点留给局部稳姿和后续重三角化。
    static bool shouldUseMultiViewOnlyGlobalBa(bool localOnly,
                                               int activeCameraCount,
                                               int totalTrackCount,
                                               int twoViewTrackCount,
                                               int multiViewTrackCount);

    /// 仅在最终共享内参自标定时保留当前相机层，不按影像数量或场景形状推断穹顶。
    static bool shouldPreserveCameraLayer(bool localOnly,
                                          bool hasAbsoluteConstraint,
                                          bool completeRegistration,
                                          bool refiningSharedIntrinsics);

    /// 周期全局 BA 限制为两轮；最终精化保留用户配置轮数。
    static int iterativeGlobalBaRoundLimit(int configuredRounds,
                                           bool finalRefinement);

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw
