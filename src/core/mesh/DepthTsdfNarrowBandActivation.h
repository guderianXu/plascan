#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace cv
{
class Mat;
}

namespace xjw
{
class FramePinholeCamera;
}

namespace xjw::mesh
{

struct DepthTsdfLayout;

/**
 * @brief Non-owning input view used to activate TSDF blocks near measured depth.
 *
 * The referenced camera and matrices must outlive build(). depth must be
 * CV_32FC1. Optional masks must be CV_8UC1 and have the same size as depth.
 * A zero mask value means unknown and never contributes free-space evidence.
 */
struct DepthTsdfNarrowBandFrameView
{
    const FramePinholeCamera *camera = nullptr;
    const cv::Mat *depth = nullptr;
    const cv::Mat *depthValidMask = nullptr;
    const cv::Mat *supportMask = nullptr;
};

struct DepthTsdfNarrowBandActivationOptions
{
    int blockSizeSamples = 8;
    int depthStride = 1;
    float truncationDistance = 0.0f;
    float rayStepVoxels = 1.0f;
    int haloBlocks = 0;
    std::function<bool()> isCancelled;
};

struct DepthTsdfNarrowBandActivationStatistics
{
    std::uint64_t totalBlocks = 0;
    std::uint64_t activeBlocks = 0;
    std::uint64_t validSourceSamples = 0;
    std::uint64_t markedRaySamples = 0;
};

/**
 * @brief Builds a fixed-size block mask around observed depth truncation bands.
 *
 * This is a clean-room implementation of depth-driven sparse block activation.
 * It does not integrate TSDF values and does not classify unobserved samples as
 * free space. A failed or cancelled build leaves no partially active result.
 */
class DepthTsdfNarrowBandActivation
{
public:
    bool build(
        const DepthTsdfLayout &layout,
        const std::vector<DepthTsdfNarrowBandFrameView> &frames,
        const DepthTsdfNarrowBandActivationOptions &options);

    bool isSampleActive(int x, int y, int z) const;

    bool isValid() const { return _valid; }
    bool wasCancelled() const { return _cancelled; }

    const DepthTsdfNarrowBandActivationStatistics &statistics() const
    {
        return _statistics;
    }

private:
    bool _valid = false;
    bool _cancelled = false;
    int _blockSizeSamples = 0;
    int _sampleCountX = 0;
    int _sampleCountY = 0;
    int _sampleCountZ = 0;
    int _blockCountX = 0;
    int _blockCountY = 0;
    int _blockCountZ = 0;
    std::vector<std::uint8_t> _activeBlocks;
    DepthTsdfNarrowBandActivationStatistics _statistics;
};

} // namespace xjw::mesh
