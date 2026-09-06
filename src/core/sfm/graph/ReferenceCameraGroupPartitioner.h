#pragma once

#include "common/SfmTypes.h"

#include <cstddef>
#include <vector>

namespace xjw
{

    using ReferenceCameraGroup = std::vector<ImageId>;

    struct ReferenceCameraGroupPartitionOptions
    {
        std::size_t maximumGroupSize = 50;
        std::size_t minimumLargeSide = 50;
    };

    /**
     * @brief 按“对齐照片”的最大生成森林收缩策略生成互斥相机核心块。
     *
     * 轨迹共视数作为边权；每轮收缩时一条组间边只累计最强的 11 条原始边。
     * 小于约 100 张影像的连通网络通常保持单块，大型网络形成约 50 张以上的核心块。
     */
    std::vector<ReferenceCameraGroup>
    partitionReferenceCameraGroups(const std::vector<ImageId>& imageIds,
                                   const std::vector<Track>& tracks,
                                   const ReferenceCameraGroupPartitionOptions& options = {});

} // namespace xjw
