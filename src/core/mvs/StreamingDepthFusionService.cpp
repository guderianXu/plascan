#include "StreamingDepthFusionService.h"

#include <algorithm>
#include <chrono>
#include <iterator>

namespace xjw::mvs
{

namespace
{

bool cancellationRequested(const StreamingDepthFusionConfig &config)
{
    return config.cancelFlag &&
           config.cancelFlag->load(std::memory_order_relaxed);
}

bool reportCancellation(std::string *errorMessage)
{
    if (errorMessage)
    {
        *errorMessage = "MVS depth fusion cancelled";
    }
    return false;
}

} // namespace

std::vector<int> streamingFusionWindowIndices(int referenceIndex,
                                              int frameCount,
                                              int neighborCount)
{
    std::vector<int> indices;
    if (referenceIndex < 0 || referenceIndex >= frameCount || frameCount <= 0)
    {
        return indices;
    }

    indices.push_back(referenceIndex);
    const int maxNeighbors = std::min(std::max(1, neighborCount), frameCount - 1);
    for (int offset = 1; static_cast<int>(indices.size()) < maxNeighbors + 1; ++offset)
    {
        const int left = referenceIndex - offset;
        if (left >= 0)
        {
            indices.push_back(left);
            if (static_cast<int>(indices.size()) >= maxNeighbors + 1)
            {
                break;
            }
        }

        const int right = referenceIndex + offset;
        if (right < frameCount)
        {
            indices.push_back(right);
        }
        if (left < 0 && right >= frameCount)
        {
            break;
        }
    }
    return indices;
}

bool fuseDepthMapsStreaming(int frameCount,
                            const StreamingDepthFusionConfig &config,
                            const FusionFrameLoader &frameLoader,
                            StreamingDepthFusionResult *result,
                            std::string *errorMessage,
                            const FusionProgress &progress,
                            const FusedCloudReducer &cloudReducer)
{
    if (!result || !frameLoader)
    {
        if (errorMessage)
        {
            *errorMessage = "Streaming depth fusion output or frame loader is empty";
        }
        return false;
    }
    if (frameCount < 2)
    {
        if (errorMessage)
        {
            *errorMessage = "MVS 深度图融合至少需要 2 帧";
        }
        return false;
    }
    if (cancellationRequested(config))
    {
        return reportCancellation(errorMessage);
    }

    result->points.clear();
    result->depthPostprocessStats.clear();
    const int neighborCount = std::min(frameCount - 1, std::max(1, config.neighborCount));
    const bool cacheFrames = frameCount <= std::max(0, config.cacheFrameLimit);
    std::vector<FusionFrameInput> cachedFrames;
    if (cacheFrames)
    {
        cachedFrames.reserve(static_cast<std::size_t>(frameCount));
        for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex)
        {
            if (cancellationRequested(config))
            {
                return reportCancellation(errorMessage);
            }
            FusionFrameInput frame;
            if (!frameLoader(frameIndex, &frame, errorMessage))
            {
                return false;
            }
            result->depthPostprocessStats.push_back(frame.depthPostprocess);
            cachedFrames.push_back(std::move(frame));
        }
    }

    for (int referenceIndex = 0; referenceIndex < frameCount; ++referenceIndex)
    {
        if (cancellationRequested(config))
        {
            return reportCancellation(errorMessage);
        }
        const auto startedAt = std::chrono::steady_clock::now();
        const std::vector<int> indices = streamingFusionWindowIndices(
            referenceIndex, frameCount, neighborCount);
        std::vector<FusionFrameInput> frames;
        frames.reserve(indices.size());
        for (const int frameIndex : indices)
        {
            if (cancellationRequested(config))
            {
                return reportCancellation(errorMessage);
            }
            if (cacheFrames)
            {
                frames.push_back(cachedFrames[static_cast<std::size_t>(frameIndex)]);
            }
            else
            {
                FusionFrameInput frame;
                if (!frameLoader(frameIndex, &frame, errorMessage))
                {
                    return false;
                }
                frames.push_back(std::move(frame));
            }
        }
        if (frames.size() < 2)
        {
            continue;
        }
        if (!cacheFrames)
        {
            result->depthPostprocessStats.push_back(frames.front().depthPostprocess);
        }

        StereoFusionConfig fusionConfig;
        fusionConfig.minNumPixels = std::max(1, config.minConsistentViews);
        float configured_reprojection_pixels = config.depthConsistency;
        fusionConfig.maxDepthError = 0.05f;
        fusionConfig.checkNumImages = std::min(neighborCount, static_cast<int>(frames.size()) - 1);
        fusionConfig.workerCount = std::max(1, config.workerCount);
        fusionConfig.useColor = config.useColor;
        fusionConfig.colorCacheCapacity = config.useColor ? 2 : 0;
        fusionConfig.fuseOnlyFirstFrame = true;
        fusionConfig.cancelFlag = config.cancelFlag;
        fusionConfig.minGeometryObservationCount = std::min(
            std::max(1, config.minConsistentViews),
            static_cast<int>(frames.size()));
        if (frameCount <= 32)
        {
            fusionConfig.minNumPixels = std::min(fusionConfig.minNumPixels, 2);
        }
        if (frames.size() <= 2)
        {
            fusionConfig.minNumPixels = 1;
            fusionConfig.maxDepthError = std::max(fusionConfig.maxDepthError, 0.08f);
            configured_reprojection_pixels = std::max(
                configured_reprojection_pixels, 3.0f);
        }

        fusionConfig.maxReprojError = configured_reprojection_pixels;
        fusionConfig.localDepthGradientRadiusPixels = 1;
        fusionConfig.pixelParametersUsePreparedRaster = true;

        DepthMapFusion fusion(fusionConfig);
        std::vector<FusedPoint> batchPoints;
        std::string fusionError;
        const bool ok = fusion.fuse(
            frames,
            batchPoints,
            [referenceIndex, frameCount, &progress](const std::string &stage, float ratio) {
                if (progress)
                {
                    const int percent = 70 + static_cast<int>(
                        ((static_cast<float>(referenceIndex) + ratio)
                         / static_cast<float>(std::max(1, frameCount))) * 20.0f);
                    progress(stage, percent);
                }
            },
            &fusionError);
        if (!ok)
        {
            if (cancellationRequested(config))
            {
                return reportCancellation(errorMessage);
            }
            if (errorMessage)
            {
                *errorMessage = fusionError;
            }
            return false;
        }

        result->points.insert(result->points.end(),
                              std::make_move_iterator(batchPoints.begin()),
                              std::make_move_iterator(batchPoints.end()));
        if (cloudReducer && result->points.size() > config.preReduceThreshold)
        {
            cloudReducer(&result->points);
        }
        if (progress)
        {
            const double elapsedMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - startedAt).count();
            progress("batch=" + std::to_string(referenceIndex + 1) + "/"
                         + std::to_string(frameCount) + " window=" + std::to_string(frames.size())
                         + " points=" + std::to_string(result->points.size()) + " elapsed_ms="
                         + std::to_string(elapsedMs),
                     70 + ((referenceIndex + 1) * 20) / std::max(1, frameCount));
        }
    }

    if (result->points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "MVS 流式融合没有生成有效稠密点";
        }
        return false;
    }
    return true;
}

} // namespace xjw::mvs
