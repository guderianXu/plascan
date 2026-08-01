#pragma once

#include "DepthMapFusion.h"

#include <cstddef>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace xjw::mvs
{

struct StreamingDepthFusionConfig
{
    int minConsistentViews = 2;
    float depthConsistency = 2.0f;
    int workerCount = 1;
    int neighborCount = 4;
    int cacheFrameLimit = 32;
    std::size_t preReduceThreshold = 2000000;
    bool useColor = true;
    std::shared_ptr<std::atomic_bool> cancelFlag;
};

struct StreamingDepthFusionResult
{
    std::vector<FusedPoint> points;
    std::vector<DepthPostProcessStats> depthPostprocessStats;
};

using FusionFrameLoader = std::function<bool(int, FusionFrameInput *, std::string *)>;
using FusionProgress = std::function<void(const std::string &, int)>;
using FusedCloudReducer = std::function<void(std::vector<FusedPoint> *)>;

std::vector<int> streamingFusionWindowIndices(int referenceIndex,
                                              int frameCount,
                                              int neighborCount);

bool fuseDepthMapsStreaming(int frameCount,
                            const StreamingDepthFusionConfig &config,
                            const FusionFrameLoader &frameLoader,
                            StreamingDepthFusionResult *result,
                            std::string *errorMessage,
                            const FusionProgress &progress = {},
                            const FusedCloudReducer &cloudReducer = {});

} // namespace xjw::mvs
