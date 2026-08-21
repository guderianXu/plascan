#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace xjw::mesh::texture_v4
{

struct TextureSeamConstraint
{
    int firstChart = -1;
    int secondChart = -1;
    /// Linear-sRGB target offset satisfying firstOffset-secondOffset=difference.
    cv::Vec3f differenceLinearBgr{};
    float weight = 0.0f;
};

struct TextureSeamLevelingStats
{
    int constraintCount = 0;
    int connectedChartCount = 0;
    int adjustedChartCount = 0;
    int adjustedPixelCount = 0;
    float maximumAbsoluteLinearCorrection = 0.0f;
};

/// Solve one robust, symmetric chart-offset graph from true shared-edge
/// observations, then apply the global solution only inside a bounded chart
/// border band.  Interiors remain untouched and disconnected charts fail
/// closed with zero correction.
TextureSeamLevelingStats applyTextureSeamLeveling(
    cv::Mat *atlasBgr,
    const cv::Mat &chartIndexMap,
    const std::vector<TextureSeamConstraint> &constraints,
    int chartCount,
    int borderBlendRadiusPixels,
    float maximumAbsoluteLinearCorrection);

} // namespace xjw::mesh::texture_v4
