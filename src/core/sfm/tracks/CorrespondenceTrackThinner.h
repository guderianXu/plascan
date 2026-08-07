#pragma once

/**
 * @file CorrespondenceTrackThinner.h
 * @brief 在进入 SfM 前按影像与网格配额稀疏化多视轨迹。
 *
 * 稀疏化的对象是完整 track，而不是单条 pairwise match。保留集合确定后，
 * CorrespondenceGraph 只删除不属于保留轨迹的原始边，不合成新的影像对关系。
 */

#include "common/SfmTypes.h"

#include <cstddef>
#include <vector>

namespace xjw
{

class CorrespondenceGraph;
class SfmReconstruction;

struct CorrespondenceTrackThinningOptions
{
    int maxTracksPerImage = 0; ///< 每幅影像最多保留的轨迹数；<=0 表示不限。
    int maxTracksPerGridCell = 0; ///< 每个影像网格单元最多保留数；<=0 表示不限。
    int gridColumns = 8; ///< 空间均匀化网格列数。
    int gridRows = 8; ///< 空间均匀化网格行数。
};

struct CorrespondenceTrackThinningResult
{
    int inputTrackCount = 0; ///< 冲突消解后、质量稀疏化前的轨迹数。
    int retainedTrackCount = 0; ///< 最终保留轨迹数。
    int prunedTrackCount = 0; ///< 因配额被删除的轨迹数。
    std::size_t removedMatchCount = 0; ///< 从 CorrespondenceGraph 删除的原始匹配边数。
    std::size_t retainedMatchCount = 0; ///< 图中剩余原始匹配边数。
    std::vector<Track> retainedTracks; ///< 与裁剪后对应图一致的完整多视轨迹。
};

/**
 * @brief 构建多视轨迹、执行质量稀疏化并原位裁剪对应图。
 *
 * 函数会修改 graph；reconstruction 仅用于提供图像 ID 和关键点坐标。
 * 图像宽高在此兼容接口中由关键点最大坐标估计，生产空三应优先在上游使用真实尺寸。
 */
CorrespondenceTrackThinningResult thinCorrespondenceTracks(
    const SfmReconstruction &reconstruction,
    CorrespondenceGraph *graph,
    const CorrespondenceTrackThinningOptions &options);

} // namespace xjw
