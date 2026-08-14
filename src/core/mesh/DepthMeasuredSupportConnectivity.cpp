#include "DepthMeasuredSupportConnectivity.h"

#include "DepthTsdfCellSheetRecovery.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace xjw::mesh
{
namespace
{

std::size_t sampleCount(const std::array<int, 3> &dimensions)
{
    if (dimensions[0] < 2 || dimensions[1] < 2 || dimensions[2] < 2)
    {
        return 0;
    }
    return static_cast<std::size_t>(dimensions[0]) *
        static_cast<std::size_t>(dimensions[1]) *
        static_cast<std::size_t>(dimensions[2]);
}

std::size_t sampleIndex(const std::array<int, 3> &dimensions,
                        int x,
                        int y,
                        int z)
{
    return (static_cast<std::size_t>(z) * static_cast<std::size_t>(dimensions[1]) +
            static_cast<std::size_t>(y)) *
            static_cast<std::size_t>(dimensions[0]) +
        static_cast<std::size_t>(x);
}

bool validInput(const DepthMeasuredSupportConnectivityInput &input,
                const std::vector<std::uint8_t> *supported)
{
    const std::size_t expected_size = sampleCount(input.sampleDimensions);
    return supported && expected_size > 0 && input.tsdf && input.weight &&
        input.surfaceObservationWeight && input.maximumEvidenceObservationWeight &&
        input.geometrySourceMask && input.minimumInverseDepthSpread &&
        input.maximumGeometrySupportCount && supported->size() == expected_size &&
        input.tsdf->size() == expected_size && input.weight->size() == expected_size &&
        input.surfaceObservationWeight->size() == expected_size &&
        input.maximumEvidenceObservationWeight->size() == expected_size &&
        input.geometrySourceMask->size() == expected_size &&
        input.minimumInverseDepthSpread->size() == expected_size &&
        input.maximumGeometrySupportCount->size() == expected_size;
}

struct CellState
{
    bool observedPositive = false;
    bool observedNegative = false;
    bool supportedPositive = false;
    bool supportedNegative = false;
    int supportedCount = 0;
};

CellState cellState(const DepthMeasuredSupportConnectivityInput &input,
                    const std::vector<std::uint8_t> &supported,
                    int x,
                    int y,
                    int z)
{
    CellState state;
    for (int dz = 0; dz <= 1; ++dz)
    {
        for (int dy = 0; dy <= 1; ++dy)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                const std::size_t index = sampleIndex(
                    input.sampleDimensions, x + dx, y + dy, z + dz);
                const float value = (*input.tsdf)[index];
                if ((*input.weight)[index] > 0.0f)
                {
                    state.observedPositive = state.observedPositive || value >= 0.0f;
                    state.observedNegative = state.observedNegative || value < 0.0f;
                }
                if (supported[index] != 0)
                {
                    ++state.supportedCount;
                    state.supportedPositive = state.supportedPositive || value >= 0.0f;
                    state.supportedNegative = state.supportedNegative || value < 0.0f;
                }
            }
        }
    }
    return state;
}

} // namespace

DepthMeasuredSupportConnectivityStatistics
DepthMeasuredSupportConnectivity::recover(
    const DepthMeasuredSupportConnectivityInput &input,
    const DepthMeasuredSupportConnectivityOptions &options,
    std::vector<std::uint8_t> *supported)
{
    DepthMeasuredSupportConnectivityStatistics statistics;
    if (!validInput(input, supported))
    {
        return statistics;
    }

    const std::vector<std::uint8_t> frozen_support = *supported;
    std::vector<std::uint8_t> eligible(frozen_support.size(), 0);
    const int minimum_source_count = std::clamp(
        options.minimumSourceCount,
        2,
        static_cast<int>(kDepthGeometrySourceSlotCount));
    const int minimum_geometry_support = std::clamp(options.minimumGeometrySupport, 2, 16);
    const float maximum_spread = std::clamp(options.maximumInverseDepthSpread, 0.001f, 0.05f);
    const float minimum_surface_ratio = std::clamp(options.minimumSurfaceWeightRatio, 0.01f, 1.0f);
    const float maximum_absolute_tsdf = std::clamp(options.maximumAbsoluteTsdf, 0.05f, 0.95f);
    const float minimum_observation_weight = std::clamp(options.minimumObservationWeight, 0.01f, 1.0f);
    const int minimum_supported_corners = std::clamp(options.minimumSupportedCellCorners, 1, 7);

    for (std::size_t index = 0; index < frozen_support.size(); ++index)
    {
        if (frozen_support[index] != 0 || (*input.weight)[index] <= 0.0f ||
            !std::isfinite((*input.tsdf)[index]))
        {
            continue;
        }
        ++statistics.consideredSampleCount;
        if ((*input.maximumEvidenceObservationWeight)[index] < minimum_observation_weight)
        {
            ++statistics.rejectedObservationWeightCount;
            continue;
        }
        if (static_cast<int>((*input.geometrySourceMask)[index].count()) <
            minimum_source_count)
        {
            ++statistics.rejectedSourceCount;
            continue;
        }
        if ((*input.maximumGeometrySupportCount)[index] < minimum_geometry_support)
        {
            ++statistics.rejectedGeometrySupportCount;
            continue;
        }
        const std::uint16_t spread = (*input.minimumInverseDepthSpread)[index];
        if (spread == std::numeric_limits<std::uint16_t>::max() ||
            static_cast<float>(spread) / 100000.0f > maximum_spread)
        {
            ++statistics.rejectedDepthSpreadCount;
            continue;
        }
        if ((*input.surfaceObservationWeight)[index] <= 0.0f ||
            (*input.surfaceObservationWeight)[index] / (*input.weight)[index] <
                minimum_surface_ratio)
        {
            ++statistics.rejectedSurfaceWeightCount;
            continue;
        }
        if (std::fabs((*input.tsdf)[index]) > maximum_absolute_tsdf)
        {
            ++statistics.rejectedAbsoluteTsdfCount;
            continue;
        }
        eligible[index] = 1;
        ++statistics.eligibleSampleCount;
    }

    const int size_x = input.sampleDimensions[0];
    const int size_y = input.sampleDimensions[1];
    const int size_z = input.sampleDimensions[2];
    DepthTsdfLayout layout;
    layout.ok = true;
    layout.cells = {size_x - 1, size_y - 1, size_z - 1};
    layout.sampleCount = frozen_support.size();
    const DepthTsdfZeroCrossingRecoveryStatistics recovery =
        recoverGeometryVerifiedZeroCrossingCellSheets(
            layout,
            *input.tsdf,
            *input.weight,
            *input.geometrySourceMask,
            eligible,
            minimum_supported_corners,
            std::max(1, options.minimumComponentCells),
            std::max(1, options.minimumAnchorCells),
            std::clamp(options.maximumSingleVoteAbsoluteTsdf, 0.0f, 1.0f),
            supported,
            true);
    statistics.recoveredSampleCount = recovery.recoveredSampleCount;
    statistics.candidateCellCount = recovery.candidateCellCount;
    statistics.acceptedCellCount = recovery.acceptedCellCount;
    statistics.componentCount = recovery.componentCount;
    statistics.acceptedComponentCount = recovery.acceptedComponentCount;
    statistics.rejectedSmallComponentCount = recovery.rejectedSmallComponentCount;
    statistics.rejectedAnchorComponentCount = recovery.rejectedAnchorComponentCount;
    statistics.rejectedBoundaryComponentCount =
        recovery.rejectedBoundaryComponentCount;
    statistics.rejectedZeroCrossingCount =
        statistics.eligibleSampleCount > recovery.candidateSampleCount
        ? statistics.eligibleSampleCount - recovery.candidateSampleCount
        : 0;
    for (int z = 0; z < size_z - 1; ++z)
    {
        for (int y = 0; y < size_y - 1; ++y)
        {
            for (int x = 0; x < size_x - 1; ++x)
            {
                const CellState before = cellState(input, frozen_support, x, y, z);
                if (!before.observedPositive || !before.observedNegative ||
                    (before.supportedPositive && before.supportedNegative))
                {
                    continue;
                }
                const CellState after = cellState(input, *supported, x, y, z);
                statistics.unlockedCellCount +=
                    after.supportedPositive && after.supportedNegative;
            }
        }
    }
    return statistics;
}

} // namespace xjw::mesh
