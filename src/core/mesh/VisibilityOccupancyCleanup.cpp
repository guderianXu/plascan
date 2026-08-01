#include "VisibilityOccupancyCleanup.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>

namespace xjw::mesh::detail
{
namespace
{

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

template <typename Visitor>
void visitSixNeighbors(
    const std::array<int, 3> &dimensions,
    std::size_t index,
    Visitor visitor)
{
    const int x = static_cast<int>(index % dimensions[0]);
    const int y = static_cast<int>(
        (index / dimensions[0]) % dimensions[1]);
    const int z = static_cast<int>(
        index / (dimensions[0] * dimensions[1]));
    const int offsets[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
    for (const auto &offset : offsets)
    {
        const int nx = x + offset[0];
        const int ny = y + offset[1];
        const int nz = z + offset[2];
        if (nx >= 0 && nx < dimensions[0] &&
            ny >= 0 && ny < dimensions[1] &&
            nz >= 0 && nz < dimensions[2])
        {
            visitor(gridIndex(dimensions, nx, ny, nz));
        }
    }
}

} // namespace

std::uint64_t retainLargestVisibilityFullComponent(
    const std::array<int, 3> &dimensions,
    std::vector<std::uint8_t> *occupied)
{
    if (occupied == nullptr || occupied->empty())
    {
        return 0;
    }
    std::vector<int> component(occupied->size(), -1);
    std::vector<std::size_t> component_sizes;
    std::queue<std::size_t> pending;
    int component_id = 0;
    for (std::size_t seed = 0; seed < occupied->size(); ++seed)
    {
        if ((*occupied)[seed] == 0 || component[seed] >= 0)
        {
            continue;
        }
        component[seed] = component_id;
        pending.push(seed);
        std::size_t count = 0;
        while (!pending.empty())
        {
            const std::size_t current = pending.front();
            pending.pop();
            ++count;
            visitSixNeighbors(
                dimensions,
                current,
                [&](std::size_t neighbor)
                {
                    if ((*occupied)[neighbor] != 0 &&
                        component[neighbor] < 0)
                    {
                        component[neighbor] = component_id;
                        pending.push(neighbor);
                    }
                });
        }
        component_sizes.push_back(count);
        ++component_id;
    }
    if (component_sizes.empty())
    {
        return 0;
    }
    const int largest = static_cast<int>(
        std::distance(
            component_sizes.begin(),
            std::max_element(component_sizes.begin(), component_sizes.end())));
    std::uint64_t removed = 0;
    for (std::size_t index = 0; index < occupied->size(); ++index)
    {
        if ((*occupied)[index] != 0 && component[index] != largest)
        {
            (*occupied)[index] = 0;
            ++removed;
        }
    }
    return removed;
}

std::uint64_t fillInteriorVisibilityEmptyBubbles(
    const std::array<int, 3> &dimensions,
    std::vector<std::uint8_t> *occupied,
    const std::vector<std::uint8_t> *protectedEmpty)
{
    if (occupied == nullptr || occupied->empty())
    {
        return 0;
    }
    std::vector<std::uint8_t> exterior(occupied->size(), 0);
    std::queue<std::size_t> pending;
    for (int z = 0; z < dimensions[2]; ++z)
    {
        for (int y = 0; y < dimensions[1]; ++y)
        {
            for (int x = 0; x < dimensions[0]; ++x)
            {
                const bool boundary = x == 0 || y == 0 || z == 0 ||
                    x + 1 == dimensions[0] ||
                    y + 1 == dimensions[1] ||
                    z + 1 == dimensions[2];
                const std::size_t index = gridIndex(dimensions, x, y, z);
                if (boundary && (*occupied)[index] == 0 &&
                    exterior[index] == 0)
                {
                    exterior[index] = 1;
                    pending.push(index);
                }
            }
        }
    }
    while (!pending.empty())
    {
        const std::size_t current = pending.front();
        pending.pop();
        visitSixNeighbors(
            dimensions,
            current,
            [&](std::size_t neighbor)
            {
                if ((*occupied)[neighbor] == 0 && exterior[neighbor] == 0)
                {
                    exterior[neighbor] = 1;
                    pending.push(neighbor);
                }
            });
    }
    const bool has_protected_empty =
        protectedEmpty != nullptr &&
        protectedEmpty->size() == occupied->size();
    std::vector<std::uint8_t> visited(occupied->size(), 0);
    std::uint64_t filled = 0;
    for (std::size_t seed = 0; seed < occupied->size(); ++seed)
    {
        if ((*occupied)[seed] != 0 || exterior[seed] != 0 ||
            visited[seed] != 0)
        {
            continue;
        }
        std::vector<std::size_t> bubble;
        bool contains_protected = false;
        visited[seed] = 1;
        pending.push(seed);
        while (!pending.empty())
        {
            const std::size_t current = pending.front();
            pending.pop();
            bubble.push_back(current);
            contains_protected = contains_protected ||
                (has_protected_empty && (*protectedEmpty)[current] != 0);
            visitSixNeighbors(
                dimensions,
                current,
                [&](std::size_t neighbor)
                {
                    if ((*occupied)[neighbor] == 0 &&
                        exterior[neighbor] == 0 && visited[neighbor] == 0)
                    {
                        visited[neighbor] = 1;
                        pending.push(neighbor);
                    }
                });
        }
        if (contains_protected)
        {
            continue;
        }
        for (const std::size_t index : bubble)
        {
            (*occupied)[index] = 1;
        }
        filled += bubble.size();
    }
    return filled;
}

std::uint64_t closeVisibilityOccupancySixConnected(
    const std::array<int, 3> &dimensions,
    int iterations,
    std::vector<std::uint8_t> *occupied,
    const std::vector<std::uint8_t> *protectedEmpty)
{
    if (occupied == nullptr || occupied->empty() || iterations <= 0)
    {
        return 0;
    }
    const bool has_protected_empty =
        protectedEmpty != nullptr &&
        protectedEmpty->size() == occupied->size();
    const std::vector<std::uint8_t> original = *occupied;
    std::vector<std::uint8_t> dilated = *occupied;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        std::vector<std::uint8_t> next = dilated;
        for (std::size_t index = 0; index < dilated.size(); ++index)
        {
            if (dilated[index] != 0)
            {
                continue;
            }
            if (has_protected_empty && (*protectedEmpty)[index] != 0)
            {
                continue;
            }
            visitSixNeighbors(
                dimensions,
                index,
                [&](std::size_t neighbor)
                {
                    next[index] =
                        next[index] != 0 || dilated[neighbor] != 0;
                });
        }
        dilated = std::move(next);
    }

    std::vector<std::uint8_t> closed = std::move(dilated);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        std::vector<std::uint8_t> next = closed;
        for (int z = 0; z < dimensions[2]; ++z)
        {
            for (int y = 0; y < dimensions[1]; ++y)
            {
                for (int x = 0; x < dimensions[0]; ++x)
                {
                    const std::size_t index =
                        gridIndex(dimensions, x, y, z);
                    const bool boundary =
                        x == 0 || y == 0 || z == 0 ||
                        x + 1 == dimensions[0] ||
                        y + 1 == dimensions[1] ||
                        z + 1 == dimensions[2];
                    if (boundary)
                    {
                        next[index] = 0;
                        continue;
                    }
                    if (closed[index] == 0)
                    {
                        continue;
                    }
                    if (original[index] != 0)
                    {
                        next[index] = 1;
                        continue;
                    }
                    visitSixNeighbors(
                        dimensions,
                        index,
                        [&](std::size_t neighbor)
                        {
                            next[index] =
                                next[index] != 0 && closed[neighbor] != 0;
                        });
                }
            }
        }
        closed = std::move(next);
    }

    std::uint64_t changed = 0;
    for (std::size_t index = 0; index < original.size(); ++index)
    {
        changed += original[index] != closed[index];
    }
    *occupied = std::move(closed);
    return changed;
}

std::vector<float> visibilitySignedDistanceSamples(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    int maximumDistance)
{
    const int maximum_distance = std::max(1, maximumDistance);
    const int unknown = std::numeric_limits<int>::max();
    std::vector<int> distance(occupied.size(), unknown);
    std::queue<std::size_t> pending;
    for (std::size_t index = 0; index < occupied.size(); ++index)
    {
        visitSixNeighbors(
            dimensions,
            index,
            [&](std::size_t neighbor)
            {
                if (occupied[index] != occupied[neighbor] &&
                    distance[index] == unknown)
                {
                    distance[index] = 0;
                    pending.push(index);
                }
            });
    }
    while (!pending.empty())
    {
        const std::size_t current = pending.front();
        pending.pop();
        if (distance[current] >= maximum_distance)
        {
            continue;
        }
        visitSixNeighbors(
            dimensions,
            current,
            [&](std::size_t neighbor)
            {
                if (occupied[current] == occupied[neighbor] &&
                    distance[neighbor] == unknown)
                {
                    distance[neighbor] = distance[current] + 1;
                    pending.push(neighbor);
                }
            });
    }
    std::vector<float> result(occupied.size(), 0.0f);
    for (std::size_t index = 0; index < occupied.size(); ++index)
    {
        const int value = distance[index] == unknown
            ? maximum_distance
            : std::min(maximum_distance, distance[index]);
        const float magnitude = static_cast<float>(value) + 0.5f;
        result[index] = occupied[index] != 0 ? -magnitude : magnitude;
    }
    return result;
}

} // namespace xjw::mesh::detail
