#pragma once

/**
 * @file ImageRegistrationEngine.h
 * @brief 无外部位姿时的增量影像注册循环。
 *
 * 每轮按可见已三角化点数选择候选影像，构造 2D-3D 对应并执行 PnP RANSAC；
 * 注册成功后立即扩展轨迹、三角化新点，并按触发条件协调局部/全局 BA。
 */

#include "IncrementalSfm.h"

namespace xjw
{

/// 增量注册执行器，操作 owner 中的重建、对应图和三角化器。
class ImageRegistrationEngine
{
  public:
    explicit ImageRegistrationEngine(IncrementalSfm &owner);

    /**
     * @brief 从已完成初始对的 owner 状态继续注册剩余影像。
     * @param totalImages 用于进度和覆盖率统计，不改变 owner 的影像集合。
     */
    IncrementalSfmResult run(int totalImages, SfmProgressCallback progressCb);

  private:
    IncrementalSfm &_owner;
};

} // namespace xjw
