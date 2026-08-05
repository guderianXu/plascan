#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xjw::mesh
{

struct DepthMeasuredSupportConnectivityOptions
{
    float minimumObservationWeight = 0.60f;
    int minimumSourceCount = 2;
    int minimumGeometrySupport = 2;
    float maximumInverseDepthSpread = 0.015f;
    float minimumSurfaceWeightRatio = 0.10f;
    float maximumAbsoluteTsdf = 0.45f;
    int minimumSupportedCellCorners = 2;
    int minimumComponentCells = 3;
    int minimumAnchorCells = 2;
    float maximumSingleVoteAbsoluteTsdf = 0.20f;
};

struct DepthMeasuredSupportConnectivityStatistics
{
    std::uint64_t consideredSampleCount = 0;
    std::uint64_t rejectedObservationWeightCount = 0;
    std::uint64_t rejectedSourceCount = 0;
    std::uint64_t rejectedGeometrySupportCount = 0;
    std::uint64_t rejectedDepthSpreadCount = 0;
    std::uint64_t rejectedSurfaceWeightCount = 0;
    std::uint64_t rejectedAbsoluteTsdfCount = 0;
    std::uint64_t eligibleSampleCount = 0;
    std::uint64_t rejectedZeroCrossingCount = 0;
    std::uint64_t recoveredSampleCount = 0;
    std::uint64_t unlockedCellCount = 0;
    std::uint64_t candidateCellCount = 0;
    std::uint64_t acceptedCellCount = 0;
    int componentCount = 0;
    int acceptedComponentCount = 0;
    int rejectedSmallComponentCount = 0;
    int rejectedAnchorComponentCount = 0;
    int rejectedBoundaryComponentCount = 0;
};

struct DepthMeasuredSupportConnectivityInput
{
    std::array<int, 3> sampleDimensions{};
    const std::vector<float> *tsdf = nullptr;
    const std::vector<float> *weight = nullptr;
    const std::vector<float> *surfaceObservationWeight = nullptr;
    const std::vector<float> *maximumEvidenceObservationWeight = nullptr;
    const std::vector<std::uint16_t> *geometrySourceMask = nullptr;
    const std::vector<std::uint16_t> *minimumInverseDepthSpread = nullptr;
    const std::vector<std::uint16_t> *maximumGeometrySupportCount = nullptr;
};

class DepthMeasuredSupportConnectivity
{
public:
    static DepthMeasuredSupportConnectivityStatistics recover(
        const DepthMeasuredSupportConnectivityInput &input,
        const DepthMeasuredSupportConnectivityOptions &options,
        std::vector<std::uint8_t> *supported);
};

} // namespace xjw::mesh
