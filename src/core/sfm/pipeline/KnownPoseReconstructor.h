#pragma once

/**
 * @file KnownPoseReconstructor.h
 * @brief 已知相机位姿路径的稀疏重建阶段。
 *
 * 与无相机增量 SfM 不同，本路径不估计初始对和 PnP 位姿；它注册可信相机，
 * 在固定/软先验位姿上构建多视轨迹、三角化并执行受配置控制的 BA。
 */

#include "IncrementalSfm.h"

namespace xjw
{

/**
 * @brief 已知位姿重建器。
 *
 * 持有 IncrementalSfm 仅作为共享状态和配置的非拥有引用，生命周期必须短于 owner。
 */
class KnownPoseReconstructor
{
  public:
    explicit KnownPoseReconstructor(IncrementalSfm &owner);

    /// 执行已知位姿注册、轨迹三角化、质量过滤和 BA，并持续上报阶段进度。
    IncrementalSfmResult run(SfmProgressCallback progressCb);

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw
