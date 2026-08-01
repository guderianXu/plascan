#pragma once

/**
 * @file BaTrackBuilder.h
 * @brief 将工程匹配输入合并为 BA 可消费的多视轨迹。
 *
 * 有特征索引的 sidecar v2 匹配会先经过 MultiViewTrackBuilder 合并，保证同一轨迹
 * 每幅影像最多一个观测；没有稳定索引的旧格式只能退化为独立双视轨迹。
 */

#include "project/BaInputBuilder.h"
#include "project/ProjectMatchInputReader.h"

namespace xjw::core::project
{

/**
 * @brief 把匹配对追加为 BA track，并更新多视轨迹统计。
 *
 * 每条轨迹会用首个可用观测对三角化初值；若相机深度轴元数据不一致，会尝试
 * 方向回退。所有观测权重继承匹配置信度，输出追加到 result 而非覆盖已有控制轨迹。
 */
void appendBaTracks(const ProjectMatchInput &input, BaInputBuildResult *result);

} // namespace xjw::core::project
