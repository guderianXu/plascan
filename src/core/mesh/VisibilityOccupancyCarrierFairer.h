#pragma once

#include "MeshTypes.h"

#include <cstdint>
#include <functional>
#include <string>

namespace xjw::mesh
{

struct VisibilityOccupancyCarrierFairingOptions
{
    int iterations = 4;
    double lambda = 0.33;
    double mu = -0.34;

    // A positive absolute limit takes precedence. Otherwise the limit is
    // derived from maximumDisplacementMeanEdgeRatio and the input mean edge.
    double absoluteMaximumDisplacement = 0.0;
    double maximumDisplacementMeanEdgeRatio = 0.20;

    // Local guards are evaluated against the original carrier after every
    // lambda or mu half-step.
    double minimumNormalDot = 0.0;
    double minimumFaceAreaRatio = 0.10;

    // Global guards are also relative to the original carrier.
    double minimumSurfaceAreaRatio = 0.75;
    double maximumSurfaceAreaRatio = 1.25;
    double minimumAbsoluteVolumeRatio = 0.75;
    double maximumAbsoluteVolumeRatio = 1.25;

    std::function<bool()> isCancelled;
};

struct VisibilityOccupancyCarrierFairingStatistics
{
    std::uint64_t inputVertexCount = 0;
    std::uint64_t inputFaceCount = 0;
    std::uint64_t uniqueEdgeCount = 0;
    int requestedIterationCount = 0;
    int completedIterationCount = 0;
    int attemptedHalfStepCount = 0;
    int acceptedHalfStepCount = 0;
    std::uint64_t displacementClampedVertexCount = 0;
    std::uint64_t locallyRejectedFaceCount = 0;
    std::uint64_t locallyFrozenVertexCount = 0;
    std::uint64_t rollbackCount = 0;
    double meanEdgeLength = 0.0;
    double resolvedMaximumDisplacement = 0.0;
    double maximumAppliedDisplacement = 0.0;
    double initialSurfaceArea = 0.0;
    double finalSurfaceArea = 0.0;
    double initialAbsoluteVolume = 0.0;
    double finalAbsoluteVolume = 0.0;
    double finalSurfaceAreaRatio = 1.0;
    double finalAbsoluteVolumeRatio = 1.0;
};

struct VisibilityOccupancyCarrierFairingResult
{
    bool ok = false;
    bool cancelled = false;
    bool rolledBack = false;
    std::string errorMessage;
    TriMesh mesh;
    VisibilityOccupancyCarrierFairingStatistics statistics;
};

/**
 * @brief Deterministic inverse-distance Taubin fairing for a fixed carrier.
 *
 * Vertex positions are updated synchronously. Vertex/face counts, face
 * indices, colors, normals and all other vertex attributes are preserved.
 * Cancellation or a failed global guard returns the complete input mesh.
 */
class VisibilityOccupancyCarrierFairer
{
public:
    static VisibilityOccupancyCarrierFairingResult fair(
        const TriMesh &mesh,
        const VisibilityOccupancyCarrierFairingOptions &options = {});
};

} // namespace xjw::mesh
