#include "DepthTsdfInternals.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;

    DepthTsdfLayout DepthTsdfSurfaceBuilder::makeLayout(const std::array<float, 3>& boundsMin,
                                                        const std::array<float, 3>& boundsMax,
                                                        int resolution,
                                                        bool includeColor,
                                                        int nestedResolution)
    {
        DepthTsdfLayout layout;
        layout.boundsMin = boundsMin;
        layout.boundsMax = boundsMax;
        if (resolution < 8)
        {
            return layout;
        }

        std::array<float, 3> extents{};
        float longestExtent = 0.0f;
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(boundsMin[axis]) || !std::isfinite(boundsMax[axis]) ||
                boundsMax[axis] <= boundsMin[axis])
            {
                return layout;
            }
            extents[axis] = boundsMax[axis] - boundsMin[axis];
            longestExtent = std::max(longestExtent, extents[axis]);
        }

        const bool use_nested_resolution =
            nestedResolution >= 4 && resolution >= nestedResolution && resolution % nestedResolution == 0;
        const int nested_scale = use_nested_resolution ? resolution / nestedResolution : 1;
        for (int axis = 0; axis < 3; ++axis)
        {
            const int unaligned_cells = std::max(
                1,
                static_cast<int>(std::lround(static_cast<double>(resolution) * static_cast<double>(extents[axis]) /
                                             static_cast<double>(longestExtent))));
            if (use_nested_resolution)
            {
                const int nested_cells = std::max(1,
                                                  static_cast<int>(std::lround(static_cast<double>(nestedResolution) *
                                                                               static_cast<double>(extents[axis]) /
                                                                               static_cast<double>(longestExtent))));
                layout.cells[axis] = nested_cells * nested_scale;
            }
            else
            {
                layout.cells[axis] = unaligned_cells;
            }
            layout.voxelSize[axis] = extents[axis] / static_cast<float>(layout.cells[axis]);
        }

        std::uint64_t sampleCount = 1;
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!checkedMultiply(sampleCount, static_cast<std::uint64_t>(layout.cells[axis]) + 1u, &sampleCount))
            {
                return layout;
            }
        }

        const std::uint64_t bytesPerSample = kBaseBytesPerSample + (includeColor ? kColorBytesPerSample : 0u);
        std::uint64_t requiredBytes = 0;
        if (!checkedMultiply(sampleCount, bytesPerSample, &requiredBytes))
        {
            return layout;
        }

        layout.sampleCount = sampleCount;
        layout.requiredBytes = requiredBytes;
        layout.ok = true;
        return layout;
    }

    DepthTsdfResult DepthTsdfSurfaceBuilder::validateAllocation(const std::array<float, 3>& boundsMin,
                                                                const std::array<float, 3>& boundsMax,
                                                                const DepthTsdfOptions& options)
    {
        DepthTsdfResult result;
        int nested_occupancy_resolution = 0;
        const int occupancy_resolution = std::clamp(options.visibilityOccupancyResolution, 24, 128);
        if (options.enableVisibilityOccupancyCompletion && options.visibilityOccupancyAlignCarrierGrid &&
            options.resolution >= occupancy_resolution && options.resolution % occupancy_resolution == 0)
        {
            nested_occupancy_resolution = occupancy_resolution;
        }
        result.layout = makeLayout(
            boundsMin, boundsMax, options.resolution, options.calculateVertexColors, nested_occupancy_resolution);
        if (!result.layout.ok)
        {
            result.errorMessage = QStringLiteral("Invalid TSDF bounds or resolution=%1").arg(options.resolution);
            return result;
        }
        if (options.enableSurfacePatchSupport || options.enableContourBandZeroCrossingSupport ||
            options.enableMeasuredSupportConnectivity || options.collectZeroCrossingDiagnostics ||
            options.collectAcquisitionGapReport || options.enableOrbitalGapBoundaryRecovery ||
            options.enableCrossViewAnchoredSurfaceRecovery || options.enableGlobalImplicitRegularization ||
            options.enableAdaptiveTgvRegularization)
        {
            std::uint64_t evidence_bytes = 0;
            const std::size_t float_field_count = options.enableContourBandZeroCrossingSupport ? 4u : 3u;
            if (!checkedMultiply(result.layout.sampleCount,
                                 sizeof(DepthGeometrySourceMask) + sizeof(std::uint16_t) +
                                     sizeof(float) * float_field_count +
                                     (options.enableCrossViewAnchoredSurfaceRecovery ? sizeof(std::uint8_t) : 0u),
                                 &evidence_bytes) ||
                result.layout.requiredBytes > std::numeric_limits<std::uint64_t>::max() - evidence_bytes)
            {
                result.layout.ok = false;
                result.errorMessage = QStringLiteral("TSDF surface-patch evidence allocation overflow");
                return result;
            }
            result.layout.requiredBytes += evidence_bytes;
        }
        if (options.enableAdaptiveTgvRegularization)
        {
            constexpr std::uint64_t kSparseTgvBytesPerActiveSample =
                sizeof(AdaptiveTsdfOctreeNode) + sizeof(float) * 23u + 32u;
            std::uint64_t adaptive_bytes = 0;
            if (!checkedMultiply(result.layout.sampleCount,
                                 sizeof(DepthVisibilityHistogram) + kSparseTgvBytesPerActiveSample,
                                 &adaptive_bytes) ||
                result.layout.requiredBytes > std::numeric_limits<std::uint64_t>::max() - adaptive_bytes)
            {
                result.layout.ok = false;
                result.errorMessage = QStringLiteral("TSDF adaptive TGV allocation overflow");
                return result;
            }
            result.layout.requiredBytes += adaptive_bytes;
        }

        const std::uint64_t available =
            options.availableMemoryBytes > 0 ? options.availableMemoryBytes : availablePhysicalMemoryBytes();
        const std::uint64_t budget = available > 0 ? available * 3u / 4u : 0u;
        if (budget > 0 && result.layout.requiredBytes > budget)
        {
            result.errorMessage =
                QStringLiteral(
                    "TSDF allocation rejected: resolution=%1 cells=%2x%3x%4 required=%5 bytes available=%6 bytes")
                    .arg(options.resolution)
                    .arg(result.layout.cells[0])
                    .arg(result.layout.cells[1])
                    .arg(result.layout.cells[2])
                    .arg(result.layout.requiredBytes)
                    .arg(available);
            return result;
        }

        result.ok = true;
        return result;
    }
} // namespace xjw::mesh
