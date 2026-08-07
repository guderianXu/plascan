#pragma once

#include "CorrespondenceGraph.h"

#include <cstddef>
#include <vector>

namespace xjw
{

struct CovisibilityPartitionOptions
{
    std::size_t targetCoreSize = 64;
    std::size_t overlapSize = 12;
};

struct CovisibilityBlock
{
    std::vector<ImageId> coreImageIds;
    std::vector<ImageId> overlapImageIds;

    std::vector<ImageId> activeImageIds() const;
};

/**
 * @brief 按已验证匹配数加权的共视图划分重叠块。
 *
 * 每台相机只属于一个 core；overlap 从相邻 core 中选取与本块连接最强的相机。
 * 算法不使用影像编号连续性、相机轨迹或场景平面假设，适用于航测和任意三维重建。
 */
class CovisibilityPartitioner
{
  public:
    static std::vector<CovisibilityBlock> partition(
        const std::vector<ImageId> &registeredImageIds,
        const CorrespondenceGraph &graph,
        const CovisibilityPartitionOptions &options = {});
};

} // namespace xjw
