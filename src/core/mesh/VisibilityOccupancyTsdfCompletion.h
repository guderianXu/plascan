#pragma once

#include "VisibilityOccupancySurfaceBuilder.h"

#include <cstdint>
#include <vector>

namespace xjw::mesh
{

struct DepthTsdfLayout;

struct VisibilityOccupancyTsdfCompletionOptions
{
    bool enableTopologyLockedResidualBlend = true;
    float truncationDistanceWorld = 0.0f;
    float observedBand = 0.60f;
    float carrierBand = 0.85f;
    float maximumResidual = 0.30f;
    float detailBlend = 0.70f;
    bool preserveAllObservedSamples = false;
    bool preserveObservedNearSurface = true;
    bool requireOccupancySignAgreement = true;
    float maximumPreservedAbsoluteTsdf = 0.45f;
    float signedDistanceNormalizationSamples = 3.0f;
};

struct VisibilityOccupancyTsdfCompletionStatistics
{
    std::uint64_t recoveredUnsupportedSampleCount = 0;
    std::uint64_t preservedObservedSampleCount = 0;
    std::uint64_t overriddenObservedSampleCount = 0;
    std::uint64_t forcedExteriorBoundarySampleCount = 0;
    std::uint64_t adjustedExactIsoValueSampleCount = 0;
    std::uint64_t trustedObservationSampleCount = 0;
    std::uint64_t ignoredSignConflictObservationCount = 0;
    std::uint64_t blendedSampleCount = 0;
    std::uint64_t clippedResidualSampleCount = 0;
    std::uint64_t carrierSignMismatchSampleCount = 0;
    float maximumAppliedResidual = 0.0f;
};

class VisibilityOccupancyTsdfCompletion
{
public:
    static VisibilityOccupancyTsdfCompletionStatistics apply(
        const DepthTsdfLayout &layout,
        const VisibilityOccupancyResult &occupancy,
        const VisibilityOccupancyTsdfCompletionOptions &options,
        std::vector<float> *tsdf,
        std::vector<std::uint8_t> *supported);
};

} // namespace xjw::mesh
