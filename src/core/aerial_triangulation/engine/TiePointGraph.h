#pragma once

#include "common/SfmTypes.h"

#include <cstddef>
#include <filesystem>
#include <map>
#include <vector>

namespace xjw::aerial_triangulation::engine
{

    struct TiePointPair
    {
        ImageId imageA = kInvalidImageId;
        ImageId imageB = kInvalidImageId;
        std::vector<FeatureMatch> matches;
    };

    struct TiePointGraph
    {
        std::vector<std::filesystem::path> imagePaths;
        std::map<ImageId, std::vector<FeatureKeypoint>> keypointsByImage;
        std::vector<TiePointPair> matchPairs;
        std::vector<Track> tracks;
        int trackCount = 0;
        std::size_t directEdgeCount = 0;
        std::size_t synthesizedClosureEdgeCount = 0;
        bool usesRawDirectEdges = false;
    };

} // namespace xjw::aerial_triangulation::engine
