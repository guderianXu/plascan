#pragma once

#include <array>

#include <opencv2/core.hpp>

#include "MvsTypes.h"

namespace xjw
{
namespace mvs
{

/// Builds the 16-float source-camera block consumed by every PatchMatch backend.
/// Relative translation is formed from double-precision camera centres before
/// the local result is converted to float, preserving small baselines at large
/// world-coordinate origins.
std::array<float, 16> buildPatchMatchSourceCameraData(
    const FramePinholeCamera &reference,
    const FramePinholeCamera &source,
    int downsampleFactor);

/// Applies the shared host-side PatchMatch depth filters.
/// Invalid pixels never participate in a neighbourhood and remain invalid.
/// Bilateral range weights use log-depth differences, making the result
/// invariant to a uniform change of world units.
void postprocessPatchMatchDepth(cv::Mat &depth,
                                const PatchMatchConfig &config);

} // namespace mvs
} // namespace xjw
