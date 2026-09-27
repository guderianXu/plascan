#pragma once

/**
 * @file BaTrackBuilder.h
 * @brief 将工程匹配输入合并为 BA 可消费的多视轨迹。
 *
 * 有特征索引的 sidecar v2 匹配会按“对齐照片”参考策略合并，并删除同一轨迹内
 * 来自重复影像的冲突观测；没有稳定索引的旧格式只能退化为独立双视轨迹。
 */

#include "project/BaInputBuilder.h"
#include "project/ProjectMatchInputReader.h"

namespace xjw::core::project
{

    /**
     * @brief 把匹配对追加为 BA track，并更新多视轨迹统计。
     *
     * 每条轨迹用 PlaCamera 的首个可交会观测对生成初值；若全部交会失败，
     * 使用前两台相机的光心中点。所有观测权重继承匹配置信度，输出追加到 result。
     */
    /// Returns false if a selected camera has no matching PlaCamera instance.
    bool appendBaTracks(const ProjectMatchInput& input, BaInputBuildResult* result);

} // namespace xjw::core::project
