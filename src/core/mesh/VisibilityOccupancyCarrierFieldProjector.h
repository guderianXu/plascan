#pragma once

#include "MeshTypes.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct VisibilityOccupancyCarrierFieldProjectionOptions
{
    int iterations = 3;
    int maximumBacktrackingSteps = 6;
    double relaxation = 0.40;
    double maximumStepSpacingRatio = 0.15;
    double maximumCumulativeDisplacementSpacingRatio = 0.30;

    bool smoothNarrowBand = true;
    double narrowBandWidthSpacingRatio = 2.0;
    double scalarSmoothingRelaxation = 0.12;

    double minimumNormalDot = 0.50;
    double minimumFaceAreaRatio = 0.25;
    double minimumSurfaceAreaRatio = 0.90;
    double maximumSurfaceAreaRatio = 1.05;
    double minimumAbsoluteVolumeRatio = 0.92;
    double maximumAbsoluteVolumeRatio = 1.05;

    std::function<bool()> isCancelled;
};

struct VisibilityOccupancyCarrierFieldProjectionStatistics
{
    std::uint64_t inputVertexCount = 0;
    std::uint64_t inputFaceCount = 0;
    int requestedIterationCount = 0;
    int completedIterationCount = 0;
    int attemptedBlendCount = 0;
    int acceptedFullStepCount = 0;
    int acceptedHalfStepCount = 0;
    int backtrackingAttemptCount = 0;
    int rejectedBlendCount = 0;
    double minimumAcceptedBlend = 1.0;
    bool fieldSmoothingApplied = false;
    std::uint64_t smoothedSampleCount = 0;
    std::uint64_t projectedVertexCount = 0;
    std::uint64_t stepClampedVertexCount = 0;
    std::uint64_t cumulativeClampedVertexCount = 0;
    std::uint64_t locallyRejectedFaceCount = 0;
    std::uint64_t locallyFrozenVertexCount = 0;
    std::uint64_t rollbackCount = 0;
    double minimumSpacing = 0.0;
    double resolvedMaximumStep = 0.0;
    double resolvedMaximumCumulativeDisplacement = 0.0;
    double meanAbsoluteFieldResidualBefore = 0.0;
    double meanAbsoluteFieldResidualAfter = 0.0;
    double p90AbsoluteFieldResidualBefore = 0.0;
    double p90AbsoluteFieldResidualAfter = 0.0;
    double maximumAbsoluteFieldResidualBefore = 0.0;
    double maximumAbsoluteFieldResidualAfter = 0.0;
    double initialSurfaceArea = 0.0;
    double finalSurfaceArea = 0.0;
    double initialAbsoluteVolume = 0.0;
    double finalAbsoluteVolume = 0.0;
    double finalSurfaceAreaRatio = 1.0;
    double finalAbsoluteVolumeRatio = 1.0;
};

struct VisibilityOccupancyCarrierFieldProjectionResult
{
    bool ok = false;
    bool cancelled = false;
    bool rolledBack = false;
    std::string errorMessage;
    TriMesh mesh;
    VisibilityOccupancyCarrierFieldProjectionStatistics statistics;
};

/**
 * @brief Projects a fixed-topology carrier towards a signed grid field.
 *
 * The optional scalar prefilter performs one sign-preserving, six-neighbour
 * narrow-band smoothing pass. Vertex updates are synchronous weak Newton
 * steps with local face and global area/volume guards. Face indices and all
 * non-position vertex attributes are preserved. Cancellation and failed
 * global guards return the complete input mesh.
 */
class VisibilityOccupancyCarrierFieldProjector
{
public:
    static VisibilityOccupancyCarrierFieldProjectionResult project(
        const TriMesh &mesh,
        const std::array<int, 3> &sampleDimensions,
        const std::array<float, 3> &boundsMin,
        const std::array<float, 3> &boundsMax,
        const std::vector<float> &signedWorldDistance,
        const VisibilityOccupancyCarrierFieldProjectionOptions &options = {});
};

} // namespace xjw::mesh
