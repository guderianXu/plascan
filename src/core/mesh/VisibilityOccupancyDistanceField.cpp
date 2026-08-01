#include "VisibilityOccupancyDistanceField.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>

namespace xjw::mesh
{
namespace
{

struct PendingSample
{
    float distance = 0.0f;
    std::size_t index = 0;
};

struct GreaterDistance
{
    bool operator()(const PendingSample &lhs, const PendingSample &rhs) const
    {
        return lhs.distance > rhs.distance;
    }
};

bool checkedSampleCount(
    const std::array<int, 3> &dimensions,
    std::size_t *sampleCount)
{
    if (sampleCount == nullptr)
    {
        return false;
    }
    std::size_t count = 1;
    for (const int dimension : dimensions)
    {
        if (dimension < 2)
        {
            return false;
        }
        const std::size_t extent = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / extent)
        {
            return false;
        }
        count *= extent;
    }
    *sampleCount = count;
    return true;
}

std::size_t gridIndex(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) *
                static_cast<std::size_t>(dimensions[1]) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(dimensions[0]) +
           static_cast<std::size_t>(x);
}

std::array<int, 3> gridCoordinates(
    const std::array<int, 3> &dimensions,
    std::size_t index)
{
    const int x = static_cast<int>(
        index % static_cast<std::size_t>(dimensions[0]));
    index /= static_cast<std::size_t>(dimensions[0]);
    const int y = static_cast<int>(
        index % static_cast<std::size_t>(dimensions[1]));
    const int z = static_cast<int>(
        index / static_cast<std::size_t>(dimensions[1]));
    return {x, y, z};
}

bool isGridBoundary(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return x == 0 || y == 0 || z == 0 ||
        x + 1 == dimensions[0] ||
        y + 1 == dimensions[1] ||
        z + 1 == dimensions[2];
}

float neighborCost(
    const std::array<float, 3> &spacing,
    int dx,
    int dy,
    int dz)
{
    const float world_x = spacing[0] * static_cast<float>(dx);
    const float world_y = spacing[1] * static_cast<float>(dy);
    const float world_z = spacing[2] * static_cast<float>(dz);
    return std::sqrt(
        world_x * world_x + world_y * world_y + world_z * world_z);
}

template <typename Visitor>
void visitTwentySixNeighbors(
    const std::array<int, 3> &dimensions,
    const std::array<float, 3> &spacing,
    std::size_t index,
    Visitor visitor)
{
    const std::array<int, 3> coordinate =
        gridCoordinates(dimensions, index);
    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                if (dx == 0 && dy == 0 && dz == 0)
                {
                    continue;
                }
                const int nx = coordinate[0] + dx;
                const int ny = coordinate[1] + dy;
                const int nz = coordinate[2] + dz;
                if (nx < 0 || nx >= dimensions[0] ||
                    ny < 0 || ny >= dimensions[1] ||
                    nz < 0 || nz >= dimensions[2])
                {
                    continue;
                }
                visitor(
                    gridIndex(dimensions, nx, ny, nz),
                    neighborCost(spacing, dx, dy, dz));
            }
        }
    }
}

bool validateBounds(
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::array<int, 3> &dimensions,
    std::array<float, 3> *spacing)
{
    if (spacing == nullptr)
    {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(boundsMin[axis]) ||
            !std::isfinite(boundsMax[axis]) ||
            !(boundsMax[axis] > boundsMin[axis]))
        {
            return false;
        }
        (*spacing)[axis] =
            (boundsMax[axis] - boundsMin[axis]) /
            static_cast<float>(dimensions[axis] - 1);
    }
    return true;
}

} // namespace

VisibilityOccupancyDistanceFieldResult
VisibilityOccupancyDistanceField::build(
    const std::array<int, 3> &sampleDimensions,
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::vector<std::uint8_t> &occupied)
{
    VisibilityOccupancyDistanceFieldResult result;
    std::size_t sample_count = 0;
    if (!checkedSampleCount(sampleDimensions, &sample_count))
    {
        result.error = "sample dimensions must be at least two and must not overflow";
        return result;
    }
    if (occupied.size() != sample_count)
    {
        result.error = "occupancy sample count does not match sample dimensions";
        return result;
    }
    std::array<float, 3> spacing{};
    if (!validateBounds(boundsMin, boundsMax, sampleDimensions, &spacing))
    {
        result.error = "occupancy bounds must be finite and strictly increasing";
        return result;
    }

    std::vector<std::uint8_t> labels(sample_count, 0);
    for (std::size_t index = 0; index < sample_count; ++index)
    {
        const std::array<int, 3> coordinate =
            gridCoordinates(sampleDimensions, index);
        if (!isGridBoundary(
                sampleDimensions,
                coordinate[0],
                coordinate[1],
                coordinate[2]))
        {
            labels[index] = occupied[index] != 0 ? 1 : 0;
        }
    }

    const float infinity = std::numeric_limits<float>::infinity();
    std::vector<float> distances(sample_count, infinity);
    std::priority_queue<
        PendingSample,
        std::vector<PendingSample>,
        GreaterDistance> pending;

    for (std::size_t index = 0; index < sample_count; ++index)
    {
        float seed_distance = infinity;
        visitTwentySixNeighbors(
            sampleDimensions,
            spacing,
            index,
            [&](std::size_t neighbor, float cost)
            {
                if (labels[index] != labels[neighbor])
                {
                    seed_distance = std::min(seed_distance, 0.5f * cost);
                }
            });
        if (std::isfinite(seed_distance))
        {
            distances[index] = seed_distance;
            pending.push({seed_distance, index});
        }
    }

    while (!pending.empty())
    {
        const PendingSample current = pending.top();
        pending.pop();
        if (current.distance != distances[current.index])
        {
            continue;
        }
        visitTwentySixNeighbors(
            sampleDimensions,
            spacing,
            current.index,
            [&](std::size_t neighbor, float cost)
            {
                if (labels[current.index] != labels[neighbor])
                {
                    return;
                }
                const float candidate = current.distance + cost;
                if (candidate < distances[neighbor])
                {
                    distances[neighbor] = candidate;
                    pending.push({candidate, neighbor});
                }
            });
    }

    const float extent_x = boundsMax[0] - boundsMin[0];
    const float extent_y = boundsMax[1] - boundsMin[1];
    const float extent_z = boundsMax[2] - boundsMin[2];
    const float fallback_distance = std::max(
        std::sqrt(
            extent_x * extent_x +
            extent_y * extent_y +
            extent_z * extent_z),
        0.5f * std::min({spacing[0], spacing[1], spacing[2]}));

    result.signedWorldDistance.resize(sample_count);
    for (std::size_t index = 0; index < sample_count; ++index)
    {
        const float magnitude = std::isfinite(distances[index])
            ? distances[index]
            : fallback_distance;
        result.signedWorldDistance[index] =
            labels[index] != 0 ? -magnitude : magnitude;
    }
    result.ok = true;
    return result;
}

} // namespace xjw::mesh
