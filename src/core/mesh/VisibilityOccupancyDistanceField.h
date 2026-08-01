#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct VisibilityOccupancyDistanceFieldResult
{
    bool ok = false;
    std::string error;
    std::vector<float> signedWorldDistance;
};

/**
 * @brief Computes a signed physical distance field for a coarse occupancy grid.
 *
 * Occupied samples are negative and empty samples are positive. Grid-boundary
 * samples are always treated as exterior, which guarantees a positive outer
 * shell. Distances are propagated from every label-boundary sample with a
 * 26-neighbour multi-source Dijkstra search using anisotropic world spacing.
 */
class VisibilityOccupancyDistanceField
{
public:
    static VisibilityOccupancyDistanceFieldResult build(
        const std::array<int, 3> &sampleDimensions,
        const std::array<float, 3> &boundsMin,
        const std::array<float, 3> &boundsMax,
        const std::vector<std::uint8_t> &occupied);
};

} // namespace xjw::mesh
