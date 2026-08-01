#include "VisibilityOccupancyTsdfCompletion.h"

#include "DepthTsdfSurfaceBuilder.h"

#include <algorithm>
#include <cmath>

namespace xjw::mesh
{
namespace
{

constexpr float kMinimumAbsoluteIsoDistance = 1.0e-4f;

std::size_t fullIndex(
    const DepthTsdfLayout &layout,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) *
                static_cast<std::size_t>(layout.cells[1] + 1) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(layout.cells[0] + 1) +
           static_cast<std::size_t>(x);
}

float trilinearSignedDistance(
    const VisibilityOccupancyResult &occupancy,
    float x,
    float y,
    float z)
{
    const auto axis = [](float coordinate, int size, int *low, int *high, float *fraction)
    {
        const float clamped = std::clamp(
            coordinate, 0.0f, static_cast<float>(size - 1));
        *low = static_cast<int>(std::floor(clamped));
        *high = std::min(*low + 1, size - 1);
        *fraction = clamped - static_cast<float>(*low);
    };

    int x0 = 0;
    int x1 = 0;
    int y0 = 0;
    int y1 = 0;
    int z0 = 0;
    int z1 = 0;
    float tx = 0.0f;
    float ty = 0.0f;
    float tz = 0.0f;
    axis(x, occupancy.sampleDimensions[0], &x0, &x1, &tx);
    axis(y, occupancy.sampleDimensions[1], &y0, &y1, &ty);
    axis(z, occupancy.sampleDimensions[2], &z0, &z1, &tz);
    const auto value = [&](int sx, int sy, int sz)
    {
        return occupancy.signedDistanceSamples[
            occupancy.index(sx, sy, sz)];
    };
    const float c00 = value(x0, y0, z0) * (1.0f - tx) +
                      value(x1, y0, z0) * tx;
    const float c10 = value(x0, y1, z0) * (1.0f - tx) +
                      value(x1, y1, z0) * tx;
    const float c01 = value(x0, y0, z1) * (1.0f - tx) +
                      value(x1, y0, z1) * tx;
    const float c11 = value(x0, y1, z1) * (1.0f - tx) +
                      value(x1, y1, z1) * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

bool nearestSampleIsOccupied(
    const VisibilityOccupancyResult &occupancy,
    float x,
    float y,
    float z)
{
    const auto nearest = [](float coordinate, int size)
    {
        return std::clamp(
            static_cast<int>(std::lround(coordinate)),
            0,
            size - 1);
    };
    const std::size_t index = occupancy.index(
        nearest(x, occupancy.sampleDimensions[0]),
        nearest(y, occupancy.sampleDimensions[1]),
        nearest(z, occupancy.sampleDimensions[2]));
    if (index < occupancy.occupied.size())
    {
        return occupancy.occupied[index] != 0;
    }
    return occupancy.signedDistanceSamples[index] < 0.0f;
}

float avoidExactIsoValue(
    float value,
    bool occupied,
    std::uint64_t *adjusted_count)
{
    if (std::fabs(value) >= kMinimumAbsoluteIsoDistance)
    {
        return value;
    }
    if (adjusted_count != nullptr)
    {
        ++(*adjusted_count);
    }
    if (value < 0.0f || (value == 0.0f && occupied))
    {
        return -kMinimumAbsoluteIsoDistance;
    }
    return kMinimumAbsoluteIsoDistance;
}

float smoothBandWeight(float absolute_value, float band)
{
    if (!(band > kMinimumAbsoluteIsoDistance) ||
        absolute_value >= band)
    {
        return 0.0f;
    }
    const float normalized = std::clamp(
        absolute_value / band, 0.0f, 1.0f);
    const float smooth =
        normalized * normalized * (3.0f - 2.0f * normalized);
    return 1.0f - smooth;
}

float carrierWorldStep(
    const VisibilityOccupancyResult &occupancy)
{
    std::array<float, 3> steps{};
    for (int axis = 0; axis < 3; ++axis)
    {
        steps[axis] =
            (occupancy.boundsMax[axis] - occupancy.boundsMin[axis]) /
            static_cast<float>(std::max(
                1, occupancy.sampleDimensions[axis] - 1));
    }
    std::sort(steps.begin(), steps.end());
    return steps[1];
}

} // namespace

VisibilityOccupancyTsdfCompletionStatistics
VisibilityOccupancyTsdfCompletion::apply(
    const DepthTsdfLayout &layout,
    const VisibilityOccupancyResult &occupancy,
    const VisibilityOccupancyTsdfCompletionOptions &options,
    std::vector<float> *tsdf,
    std::vector<std::uint8_t> *supported)
{
    VisibilityOccupancyTsdfCompletionStatistics statistics;
    if (!layout.ok || !occupancy.ok || tsdf == nullptr ||
        supported == nullptr ||
        tsdf->size() != layout.sampleCount ||
        supported->size() != layout.sampleCount ||
        occupancy.signedDistanceSamples.empty())
    {
        return statistics;
    }
    const float normalization = std::max(
        0.5f, options.signedDistanceNormalizationSamples);
    const float maximum_preserved = std::clamp(
        options.maximumPreservedAbsoluteTsdf, 0.0f, 1.0f);
    const float carrier_scale =
        options.enableTopologyLockedResidualBlend &&
            options.truncationDistanceWorld > 1.0e-8f
        ? (occupancy.signedDistanceSamplesAreWorldUnits
               ? 1.0f
               : carrierWorldStep(occupancy)) /
              options.truncationDistanceWorld
        : 1.0f / normalization;
    const float observed_band = std::clamp(
        options.observedBand, 0.01f, 1.0f);
    const float carrier_band = std::clamp(
        options.carrierBand, 0.01f, 1.0f);
    const float maximum_residual = std::clamp(
        options.maximumResidual, 0.0f, 1.0f);
    const float detail_blend = std::clamp(
        options.detailBlend, 0.0f, 1.0f);
    for (int z = 0; z <= layout.cells[2]; ++z)
    {
        const float coarse_z =
            static_cast<float>(z) *
            static_cast<float>(occupancy.sampleDimensions[2] - 1) /
            static_cast<float>(std::max(1, layout.cells[2]));
        for (int y = 0; y <= layout.cells[1]; ++y)
        {
            const float coarse_y =
                static_cast<float>(y) *
                static_cast<float>(occupancy.sampleDimensions[1] - 1) /
                static_cast<float>(std::max(1, layout.cells[1]));
            for (int x = 0; x <= layout.cells[0]; ++x)
            {
                const std::size_t index = fullIndex(layout, x, y, z);
                const float coarse_x =
                    static_cast<float>(x) *
                    static_cast<float>(occupancy.sampleDimensions[0] - 1) /
                    static_cast<float>(std::max(1, layout.cells[0]));
                const bool occupancy_inside = nearestSampleIsOccupied(
                    occupancy, coarse_x, coarse_y, coarse_z);
                const float occupancy_tsdf = avoidExactIsoValue(
                    std::clamp(
                        trilinearSignedDistance(
                            occupancy, coarse_x, coarse_y, coarse_z) *
                            carrier_scale,
                        -1.0f,
                        1.0f),
                    occupancy_inside,
                    &statistics.adjustedExactIsoValueSampleCount);
                const bool boundary = x == 0 || y == 0 || z == 0 ||
                    x == layout.cells[0] ||
                    y == layout.cells[1] ||
                    z == layout.cells[2];
                if (boundary)
                {
                    (*tsdf)[index] = std::max(0.01f, occupancy_tsdf);
                    (*supported)[index] = 1;
                    ++statistics.forcedExteriorBoundarySampleCount;
                    continue;
                }

                const bool was_supported = (*supported)[index] != 0;
                if (options.enableTopologyLockedResidualBlend)
                {
                    float blended_tsdf = occupancy_tsdf;
                    if (was_supported)
                    {
                        const float observed_tsdf = std::clamp(
                            (*tsdf)[index], -1.0f, 1.0f);
                        const bool sign_agrees =
                            (observed_tsdf < 0.0f) ==
                            (occupancy_tsdf < 0.0f);
                        if (!sign_agrees)
                        {
                            ++statistics
                                  .ignoredSignConflictObservationCount;
                        }
                        else
                        {
                            const float confidence =
                                smoothBandWeight(
                                    std::fabs(observed_tsdf),
                                    observed_band) *
                                smoothBandWeight(
                                    std::fabs(occupancy_tsdf),
                                    carrier_band);
                            if (confidence > 0.0f)
                            {
                                ++statistics.trustedObservationSampleCount;
                                const float raw_residual =
                                    observed_tsdf - occupancy_tsdf;
                                const float clipped_residual = std::clamp(
                                    raw_residual,
                                    -maximum_residual,
                                    maximum_residual);
                                statistics.clippedResidualSampleCount +=
                                    clipped_residual != raw_residual;
                                const float applied_residual =
                                    detail_blend *
                                    confidence *
                                    clipped_residual;
                                blended_tsdf += applied_residual;
                                statistics.maximumAppliedResidual = std::max(
                                    statistics.maximumAppliedResidual,
                                    std::fabs(applied_residual));
                                statistics.blendedSampleCount +=
                                    std::fabs(applied_residual) > 1.0e-8f;
                            }
                        }
                        ++statistics.overriddenObservedSampleCount;
                    }
                    else
                    {
                        ++statistics.recoveredUnsupportedSampleCount;
                    }

                    if (occupancy_tsdf < 0.0f)
                    {
                        blended_tsdf = -std::clamp(
                            -blended_tsdf,
                            kMinimumAbsoluteIsoDistance,
                            1.0f);
                    }
                    else
                    {
                        blended_tsdf = std::clamp(
                            blended_tsdf,
                            kMinimumAbsoluteIsoDistance,
                            1.0f);
                    }
                    (*tsdf)[index] = avoidExactIsoValue(
                        blended_tsdf,
                        occupancy_inside,
                        &statistics.adjustedExactIsoValueSampleCount);
                    statistics.carrierSignMismatchSampleCount +=
                        ((*tsdf)[index] < 0.0f) !=
                        (occupancy_tsdf < 0.0f);
                    (*supported)[index] = 1;
                    continue;
                }

                const bool sign_agrees =
                    ((*tsdf)[index] < 0.0f) ==
                    (occupancy_tsdf < 0.0f);
                const bool preserve_near_surface = was_supported &&
                    options.preserveObservedNearSurface &&
                    std::fabs((*tsdf)[index]) <= maximum_preserved &&
                    (!options.requireOccupancySignAgreement || sign_agrees);
                const bool preserve = was_supported &&
                    (options.preserveAllObservedSamples ||
                     preserve_near_surface);
                if (preserve)
                {
                    ++statistics.preservedObservedSampleCount;
                }
                else
                {
                    (*tsdf)[index] = occupancy_tsdf;
                    statistics.recoveredUnsupportedSampleCount +=
                        !was_supported;
                    statistics.overriddenObservedSampleCount +=
                        was_supported;
                }
                (*tsdf)[index] = avoidExactIsoValue(
                    (*tsdf)[index],
                    occupancy_inside,
                    &statistics.adjustedExactIsoValueSampleCount);
                (*supported)[index] = 1;
            }
        }
    }
    return statistics;
}

} // namespace xjw::mesh
