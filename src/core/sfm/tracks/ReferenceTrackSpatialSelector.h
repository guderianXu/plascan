#pragma once

#include "common/SfmTypes.h"

#include <map>
#include <vector>

namespace xjw::detail
{

    struct ReferenceTrackImageFeatures
    {
        std::vector<FeatureKeypoint> keypoints;
        float width = 0.0f;
        float height = 0.0f;
    };

    /**
     * @brief 按“对齐照片”的逐影像水位策略选择空间均衡的完整多视轨迹。
     *
     * tiePointLimit 是单幅影像的连接点目标上限。最终结果是各影像所选轨迹的并集，
     * 因而某幅影像最终参与的轨迹数可能因其它影像的选择而超过该目标值。
     */
    std::vector<Track> selectReferenceSpatialTracks(const std::map<ImageId, ReferenceTrackImageFeatures>& images,
                                                    std::vector<Track> tracks,
                                                    std::size_t tiePointLimit);

} // namespace xjw::detail
