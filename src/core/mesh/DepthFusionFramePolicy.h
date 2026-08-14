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

enum class OrbitalFrameRole
{
    NormalSector,
    GapBoundary,
    GapOpposite
};

struct OrbitalFrameRoleAssignment
{
    int frameIndex = -1;
    int refIndex = -1;
    double azimuthDegrees = 0.0;
    OrbitalFrameRole role = OrbitalFrameRole::NormalSector;
};

struct OrbitalCoverageStatistics
{
    bool valid = false;
    bool significantGap = false;
    int activeViewCount = 0;
    double medianAngularSpacingDegrees = 0.0;
    double maximumAngularGapDegrees = 0.0;
    double maximumAngularGapRatio = 0.0;
    int gapStartFrameIndex = -1;
    int gapStartRefIndex = -1;
    int gapEndFrameIndex = -1;
    int gapEndRefIndex = -1;
    int gapOppositeFrameIndex = -1;
    int gapOppositeRefIndex = -1;
    std::vector<double> angularGapDegreesDescending;
    std::vector<OrbitalFrameRoleAssignment> frameRoles;
};

const char *orbitalFrameRoleId(OrbitalFrameRole role);

class DepthFusionFramePolicy
{
public:
    static OrbitalCoverageStatistics evaluateOrbitalCoverage(
        const std::vector<DepthFusionView> &views,
        const std::vector<float> &weights,
        double significantGapRatio = 1.5);

    static bool canRejectWithoutCoverageGap(
        const std::vector<DepthFusionView> &views,
        const std::vector<float> &weights,
        int candidateFrameIndex,
        double maximumGapRatio,
        int minimumRetainedFrames,
        OrbitalCoverageStatistics *trialCoverage = nullptr);

    static std::vector<int> selectCoverageComplementaryCandidates(
        const std::vector<DepthFusionView> &fixedViews,
        const std::vector<DepthFusionView> &candidateViews,
        int maximumSelectedCount);
};

} // namespace xjw::mesh
