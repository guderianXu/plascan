#pragma once

#include "BundleAdjust.h"

#include <cstddef>
#include <vector>

namespace xjw::sfm_calibration_preview
{

/**
 * @brief 为共享内参候选粗筛选取确定性的相机/像面均衡轨迹。
 */
std::vector<std::size_t> selectTrackIndices(
    const std::vector<FramePinholeCamera> &cameras,
    const std::vector<BATrack> &tracks,
    std::size_t maximumTrackCount);

} // namespace xjw::sfm_calibration_preview
