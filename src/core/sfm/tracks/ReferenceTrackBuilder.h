#pragma once

/**
 * @file ReferenceTrackBuilder.h
 * @brief 复现“对齐照片”匹配后多视轨迹构建、冲突清理和空间选择语义。
 */

#include "ReferenceTrackSpatialSelector.h"

#include <cstddef>
#include <map>
#include <vector>

namespace xjw
{

    struct ReferenceTrackBuildOptions
    {
        std::size_t tiePointLimit = 0;        ///< 每幅影像目标连接点上限；0 表示不执行空间选择。
        bool excludeStationaryTracks = false; ///< 按特征尺度删除像方近似静止轨迹。
    };

    struct ReferenceTrackBuildResult
    {
        std::vector<Track> tracks;
        int generatedTrackCount = 0;
        int invalidMatchCount = 0;
        int removedDuplicateObservations = 0;
        int removedShortTracks = 0;
        int prunedStationaryTracks = 0;
        int tracksBeforeSpatialSelection = 0;
        int prunedBySpatialSelection = 0;
        std::map<int, int> trackLengthHistogram;
    };

    class ReferenceTrackBuilder
    {
    public:
        struct MatchIndexPair
        {
            FeatureIdx first = kInvalidFeatureIdx;
            FeatureIdx second = kInvalidFeatureIdx;
        };

        void addMatchPair(ImageId imageA, ImageId imageB, const std::vector<MatchIndexPair>& matches);

        void setImageKeypoints(ImageId imageId,
                               const std::vector<FeatureKeypoint>& keypoints,
                               float imageWidth = 0.0f,
                               float imageHeight = 0.0f);

        ReferenceTrackBuildResult build(const ReferenceTrackBuildOptions& options = {}) const;

    private:
        struct Edge
        {
            ImageId imageA = kInvalidImageId;
            ImageId imageB = kInvalidImageId;
            FeatureIdx featureA = kInvalidFeatureIdx;
            FeatureIdx featureB = kInvalidFeatureIdx;
        };

        std::vector<Edge> _edges;
        std::map<ImageId, detail::ReferenceTrackImageFeatures> _images;
    };

} // namespace xjw
