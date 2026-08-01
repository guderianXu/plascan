#pragma once

#include "MeshTypes.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct VisibilityOccupancyBoundaryOptions
{
    std::function<bool()> isCancelled;
};

struct VisibilityOccupancyBoundaryStatistics
{
    std::uint64_t occupiedCellCount = 0;
    std::uint64_t exposedQuadCount = 0;
    std::uint64_t outputVertexCount = 0;
    std::uint64_t outputFaceCount = 0;
    std::uint64_t uniqueEdgeCount = 0;
    std::uint64_t boundaryEdgeCount = 0;
    std::uint64_t nonManifoldEdgeCount = 0;
    std::uint64_t nonManifoldVertexCount = 0;
    int eulerCharacteristic = 0;
    bool closedTwoManifold = false;
};

struct VisibilityOccupancyBoundaryResult
{
    bool ok = false;
    bool cancelled = false;
    std::string errorMessage;
    TriMesh mesh;
    VisibilityOccupancyBoundaryStatistics statistics;
};

/**
 * @brief Extracts the exact boundary of occupied axis-aligned grid cells.
 *
 * Each occupancy sample is interpreted as a closed cubical cell centered at
 * the corresponding sample coordinate. This is the same discrete complex
 * used by the occupancy Euler calculation, so no scalar-field ambiguity or
 * marching-cubes topology decision is introduced here.
 */
class VisibilityOccupancyBoundaryExtractor
{
public:
    static VisibilityOccupancyBoundaryResult extract(
        const std::array<float, 3> &boundsMin,
        const std::array<float, 3> &boundsMax,
        const std::array<int, 3> &cellDimensions,
        const std::vector<std::uint8_t> &occupied,
        const VisibilityOccupancyBoundaryOptions &options = {});
};

} // namespace xjw::mesh
