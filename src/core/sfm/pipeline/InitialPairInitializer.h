#pragma once

/**
 * @file InitialPairInitializer.h
 * @brief 增量 SfM 的初始影像对选择与双视模型初始化。
 *
 * 初始对决定整个无先验重建的坐标系和基线尺度。本类依据几何内点、分布覆盖、
 * 视差和退化检测排序候选，并可在多个候选模型之间试算后选择更稳定的种子。
 */

#include "IncrementalSfm.h"

namespace xjw
{

/// 仅操作 owner 的试算状态；失败候选必须通过 resetTrial 完整回滚。
class InitialPairInitializer
{
  public:
    explicit InitialPairInitializer(IncrementalSfm &owner);

    /// 返回按几何质量降序排列的候选对；显式 initial pair hint 会优先但仍需验证。
    std::vector<std::pair<ImageId, ImageId>> selectCandidates(int maxCandidates) const;

    /**
     * @brief 从一对影像恢复相对位姿、注册两台相机并三角化种子点。
     * @return 只有通过内点、正深度、视差和初始点数门控才返回 true。
     */
    bool initialize(ImageId id1, ImageId id2);

    /// 在尝试下一个候选前恢复重建快照并重建依赖该状态的三角化器。
    void resetTrial(const SfmReconstruction &baseReconstruction);

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw
