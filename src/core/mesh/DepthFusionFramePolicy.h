#pragma once

#include <array>
#include <vector>

namespace xjw::mesh
{

struct DepthFusionView
{
    int frameIndex = -1;
    int refIndex = -1;
    std::array<double, 3> cameraCenter{};
};

struct OrbitalCoverageStatistics
{
    bool valid = false;
    int activeViewCount = 0;
    double medianAngularSpacingDegrees = 0.0;
    double maximumAngularGapDegrees = 0.0;
    double maximumAngularGapRatio = 0.0;
};

class DepthFusionFramePolicy
{
public:
    static OrbitalCoverageStatistics evaluateOrbitalCoverage(
        const std::vector<DepthFusionView> &views,
        const std::vector<float> &weights);

    static bool canRejectWithoutCoverageGap(
        const std::vector<DepthFusionView> &views,
        const std::vector<float> &weights,
        int candidateFrameIndex,
        double maximumGapRatio,
        int minimumRetainedFrames,
        OrbitalCoverageStatistics *trialCoverage = nullptr);
};

} // namespace xjw::mesh
