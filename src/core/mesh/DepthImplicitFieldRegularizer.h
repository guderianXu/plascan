#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace xjw::mesh
{

struct DepthImplicitFieldRegularizationOptions
{
    int coarseToFineLevels = 2;
    int passesPerLevel = 1;
    float smoothness = 0.30f;
    float dataFidelity = 1.0f;
    float maximumUpdate = 0.10f;
    float edgeThreshold = 0.35f;
    bool preserveFieldSign = true;
    bool recoverAxialGaps = false;
    int minimumBridgeAxes = 1;
    float maximumBridgePredictionDelta = 0.20f;
};

struct DepthImplicitFieldRegularizationStatistics
{
    bool executed = false;
    bool cancelled = false;
    std::uint64_t bridgeCandidateCount = 0;
    std::uint64_t recoveredSampleCount = 0;
    std::uint64_t updateOperationCount = 0;
    double meanAbsoluteUpdate = 0.0;
    float maximumAbsoluteUpdate = 0.0f;
    std::int64_t elapsedMs = 0;
};

class DepthImplicitFieldRegularizer
{
public:
    static DepthImplicitFieldRegularizationStatistics regularize(
        const std::array<int, 3> &sampleDimensions,
        const std::vector<float> &surfaceEvidenceField,
        const std::vector<float> &observationWeight,
        const std::vector<std::uint16_t> &geometrySourceMask,
        const std::vector<std::uint8_t> &eligible,
        const DepthImplicitFieldRegularizationOptions &options,
        std::vector<float> *field,
        std::vector<std::uint8_t> *supported,
        const std::function<bool()> &isCancelled = {});
};

} // namespace xjw::mesh
