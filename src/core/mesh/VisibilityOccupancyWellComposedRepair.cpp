#include "VisibilityOccupancyWellComposedRepair.h"

#include "VisibilityOccupancyHandleRepair.h"

#include <algorithm>
#include <array>
#include <limits>
#include <queue>

namespace xjw::mesh
{
namespace
{

struct DefectCounts
{
    std::uint64_t edgeCheckerboards = 0;
    std::uint64_t vertexOccupiedComponents = 0;
    std::uint64_t vertexEmptyComponents = 0;

    std::uint64_t total() const
    {
        return edgeCheckerboards + vertexOccupiedComponents +
            vertexEmptyComponents;
    }
};

bool sampleCount(const std::array<int, 3> &dimensions, std::size_t *count)
{
    std::size_t value = 1;
    for (const int dimension : dimensions)
    {
        if (dimension <= 0 ||
            value > std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(dimension))
        {
            return false;
        }
        value *= static_cast<std::size_t>(dimension);
    }
    *count = value;
    return true;
}

std::size_t gridIndex(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) * dimensions[1] + y) *
        dimensions[0] + x;
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
        index / (static_cast<std::size_t>(dimensions[0]) * dimensions[1]));
    constexpr int offsets[6][3] = {
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

std::vector<std::uint8_t> exteriorReachableEmpty(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied)
{
    std::vector<std::uint8_t> exterior(occupied.size(), 0);
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
                if (boundary && occupied[index] == 0)
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
                if (occupied[neighbor] == 0 && exterior[neighbor] == 0)
                {
                    exterior[neighbor] = 1;
                    pending.push(neighbor);
                }
            });
    }
    return exterior;
}

std::array<std::size_t, 8> blockIndices(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    std::array<std::size_t, 8> indices{};
    for (int local = 0; local < 8; ++local)
    {
        indices[local] = gridIndex(
            dimensions,
            x + (local & 1),
            y + ((local >> 1) & 1),
            z + ((local >> 2) & 1));
    }
    return indices;
}

int localComponents(
    const std::array<std::size_t, 8> &indices,
    const std::vector<std::uint8_t> &occupied,
    bool targetOccupied,
    std::array<int, 8> *labels)
{
    labels->fill(-1);
    int component_count = 0;
    std::array<int, 8> pending{};
    for (int seed = 0; seed < 8; ++seed)
    {
        if (((occupied[indices[seed]] != 0) != targetOccupied) ||
            (*labels)[seed] >= 0)
        {
            continue;
        }
        int begin = 0;
        int end = 0;
        pending[end++] = seed;
        (*labels)[seed] = component_count;
        while (begin < end)
        {
            const int current = pending[begin++];
            constexpr int flips[3] = {1, 2, 4};
            for (const int flip : flips)
            {
                const int neighbor = current ^ flip;
                if ((*labels)[neighbor] < 0 &&
                    ((occupied[indices[neighbor]] != 0) == targetOccupied))
                {
                    (*labels)[neighbor] = component_count;
                    pending[end++] = neighbor;
                }
            }
        }
        ++component_count;
    }
    return component_count;
}

bool isCheckerboard(
    const std::array<std::size_t, 4> &indices,
    const std::vector<std::uint8_t> &occupied)
{
    const bool a = occupied[indices[0]] != 0;
    const bool b = occupied[indices[1]] != 0;
    const bool c = occupied[indices[2]] != 0;
    const bool d = occupied[indices[3]] != 0;
    return a == d && b == c && a != b;
}

template <typename Visitor>
bool visitEdgeSquares(
    const std::array<int, 3> &dimensions,
    const VisibilityOccupancyWellComposedRepairOptions &options,
    Visitor visitor)
{
    const auto cancelled = [&options]()
    {
        return options.isCancelled && options.isCancelled();
    };
    for (int z = 0; z < dimensions[2]; ++z)
    {
        if ((z & 3) == 0 && cancelled())
        {
            return false;
        }
        for (int y = 0; y + 1 < dimensions[1]; ++y)
        {
            for (int x = 0; x + 1 < dimensions[0]; ++x)
            {
                visitor(std::array<std::size_t, 4>{
                    gridIndex(dimensions, x, y, z),
                    gridIndex(dimensions, x + 1, y, z),
                    gridIndex(dimensions, x, y + 1, z),
                    gridIndex(dimensions, x + 1, y + 1, z)});
            }
        }
    }
    for (int y = 0; y < dimensions[1]; ++y)
    {
        if ((y & 3) == 0 && cancelled())
        {
            return false;
        }
        for (int z = 0; z + 1 < dimensions[2]; ++z)
        {
            for (int x = 0; x + 1 < dimensions[0]; ++x)
            {
                visitor(std::array<std::size_t, 4>{
                    gridIndex(dimensions, x, y, z),
                    gridIndex(dimensions, x + 1, y, z),
                    gridIndex(dimensions, x, y, z + 1),
                    gridIndex(dimensions, x + 1, y, z + 1)});
            }
        }
    }
    for (int x = 0; x < dimensions[0]; ++x)
    {
        if ((x & 3) == 0 && cancelled())
        {
            return false;
        }
        for (int z = 0; z + 1 < dimensions[2]; ++z)
        {
            for (int y = 0; y + 1 < dimensions[1]; ++y)
            {
                visitor(std::array<std::size_t, 4>{
                    gridIndex(dimensions, x, y, z),
                    gridIndex(dimensions, x, y + 1, z),
                    gridIndex(dimensions, x, y, z + 1),
                    gridIndex(dimensions, x, y + 1, z + 1)});
            }
        }
    }
    return true;
}

bool countDefects(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    const VisibilityOccupancyWellComposedRepairOptions &options,
    DefectCounts *counts)
{
    *counts = {};
    if (!visitEdgeSquares(
            dimensions,
            options,
            [&](const std::array<std::size_t, 4> &indices)
            {
                counts->edgeCheckerboards +=
                    isCheckerboard(indices, occupied);
            }))
    {
        return false;
    }
    for (int z = 0; z + 1 < dimensions[2]; ++z)
    {
        if ((z & 3) == 0 && options.isCancelled && options.isCancelled())
        {
            return false;
        }
        for (int y = 0; y + 1 < dimensions[1]; ++y)
        {
            for (int x = 0; x + 1 < dimensions[0]; ++x)
            {
                const auto indices = blockIndices(dimensions, x, y, z);
                std::array<int, 8> labels{};
                counts->vertexOccupiedComponents +=
                    localComponents(indices, occupied, true, &labels) > 1;
                counts->vertexEmptyComponents +=
                    localComponents(indices, occupied, false, &labels) > 1;
            }
        }
    }
    return true;
}

struct ProposalBuilder
{
    std::vector<std::uint8_t> *working = nullptr;
    const std::vector<std::uint8_t> *protectedEmpty = nullptr;
    std::size_t maximumFilled = 0;
    std::size_t filled = 0;
    VisibilityOccupancyWellComposedRepairStatistics *statistics = nullptr;

    bool fill(std::size_t index)
    {
        if ((*working)[index] != 0)
        {
            return false;
        }
        if ((*protectedEmpty)[index] != 0)
        {
            ++statistics->protectedFillRefusalCount;
            return false;
        }
        if (filled >= maximumFilled)
        {
            ++statistics->maximumFillRefusalCount;
            return false;
        }
        (*working)[index] = 1;
        ++filled;
        return true;
    }
};

void repairVertexBlock(
    const std::array<std::size_t, 8> &indices,
    ProposalBuilder *builder)
{
    std::array<int, 8> occupied_labels{};
    int occupied_components = localComponents(
        indices, *builder->working, true, &occupied_labels);
    while (occupied_components > 1)
    {
        int selected = -1;
        int selected_adjacent = -1;
        int selected_distance = 4;
        for (int local = 0; local < 8; ++local)
        {
            const std::size_t index = indices[local];
            if ((*builder->working)[index] != 0 ||
                (*builder->protectedEmpty)[index] != 0)
            {
                continue;
            }
            std::array<std::uint8_t, 8> adjacent_labels{};
            for (const int flip : {1, 2, 4})
            {
                const int neighbor = local ^ flip;
                if (occupied_labels[neighbor] >= 0)
                {
                    adjacent_labels[occupied_labels[neighbor]] = 1;
                }
            }
            const int adjacent = static_cast<int>(std::count(
                adjacent_labels.cbegin(), adjacent_labels.cend(), 1));
            if (adjacent == 0)
            {
                continue;
            }
            int distance = 0;
            if (adjacent == 1)
            {
                const int source = static_cast<int>(std::distance(
                    adjacent_labels.cbegin(),
                    std::find(adjacent_labels.cbegin(), adjacent_labels.cend(), 1)));
                distance = 4;
                for (int other = 0; other < 8; ++other)
                {
                    if (occupied_labels[other] >= 0 &&
                        occupied_labels[other] != source)
                    {
                        distance = std::min(
                            distance,
                            ((local ^ other) & 1 ? 1 : 0) +
                                ((local ^ other) & 2 ? 1 : 0) +
                                ((local ^ other) & 4 ? 1 : 0));
                    }
                }
            }
            if (adjacent > selected_adjacent ||
                (adjacent == selected_adjacent && distance < selected_distance) ||
                (adjacent == selected_adjacent && distance == selected_distance &&
                 (selected < 0 || index < indices[selected])))
            {
                selected = local;
                selected_adjacent = adjacent;
                selected_distance = distance;
            }
        }
        if (selected < 0 || !builder->fill(indices[selected]))
        {
            break;
        }
        occupied_components = localComponents(
            indices, *builder->working, true, &occupied_labels);
    }

    std::array<int, 8> empty_labels{};
    const int empty_components = localComponents(
        indices, *builder->working, false, &empty_labels);
    if (empty_components <= 1)
    {
        return;
    }
    std::array<int, 8> sizes{};
    std::array<std::uint8_t, 8> contains_protected{};
    std::array<std::size_t, 8> minimum_index{};
    minimum_index.fill(std::numeric_limits<std::size_t>::max());
    for (int local = 0; local < 8; ++local)
    {
        const int component = empty_labels[local];
        if (component < 0)
        {
            continue;
        }
        ++sizes[component];
        contains_protected[component] = contains_protected[component] ||
            (*builder->protectedEmpty)[indices[local]] != 0;
        minimum_index[component] = std::min(
            minimum_index[component], indices[local]);
    }
    int kept_component = -1;
    if (std::none_of(
            contains_protected.cbegin(),
            contains_protected.cbegin() + empty_components,
            [](std::uint8_t value)
            {
                return value != 0;
            }))
    {
        for (int component = 0; component < empty_components; ++component)
        {
            if (kept_component < 0 || sizes[component] > sizes[kept_component] ||
                (sizes[component] == sizes[kept_component] &&
                 minimum_index[component] < minimum_index[kept_component]))
            {
                kept_component = component;
            }
        }
    }
    for (int local = 0; local < 8; ++local)
    {
        const int component = empty_labels[local];
        if (component >= 0 && component != kept_component &&
            contains_protected[component] == 0)
        {
            builder->fill(indices[local]);
        }
    }
}

bool buildProposal(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &protected_empty,
    const VisibilityOccupancyWellComposedRepairOptions &options,
    std::size_t maximum_filled,
    VisibilityOccupancyWellComposedRepairStatistics *statistics,
    std::vector<std::uint8_t> *working,
    std::size_t *filled)
{
    ProposalBuilder builder{
        working, &protected_empty, maximum_filled, 0, statistics};
    if (!visitEdgeSquares(
            dimensions,
            options,
            [&](const std::array<std::size_t, 4> &indices)
            {
                if (!isCheckerboard(indices, *working))
                {
                    return;
                }
                std::array<std::size_t, 2> empty{};
                int empty_count = 0;
                for (const std::size_t index : indices)
                {
                    if ((*working)[index] == 0)
                    {
                        empty[empty_count++] = index;
                    }
                }
                std::sort(empty.begin(), empty.begin() + empty_count);
                for (int candidate = 0; candidate < empty_count; ++candidate)
                {
                    if (builder.fill(empty[candidate]))
                    {
                        break;
                    }
                }
            }))
    {
        return false;
    }
    for (int z = 0; z + 1 < dimensions[2]; ++z)
    {
        if ((z & 3) == 0 && options.isCancelled && options.isCancelled())
        {
            return false;
        }
        for (int y = 0; y + 1 < dimensions[1]; ++y)
        {
            for (int x = 0; x + 1 < dimensions[0]; ++x)
            {
                repairVertexBlock(
                    blockIndices(dimensions, x, y, z), &builder);
            }
        }
    }
    *filled = builder.filled;
    return true;
}

} // namespace

VisibilityOccupancyWellComposedRepairResult
VisibilityOccupancyWellComposedRepair::repair(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    const std::vector<std::uint8_t> &protectedEmpty,
    const VisibilityOccupancyWellComposedRepairOptions &options)
{
    VisibilityOccupancyWellComposedRepairResult result;
    result.occupied = occupied;
    std::size_t count = 0;
    if (!sampleCount(dimensions, &count) || count != occupied.size() ||
        count != protectedEmpty.size())
    {
        result.error = "visibility occupancy well-composed repair input is invalid";
        return result;
    }
    const auto cancellationRequested = [&options]()
    {
        return options.isCancelled && options.isCancelled();
    };
    if (cancellationRequested())
    {
        result.cancelled = true;
        return result;
    }

    result.statistics.sampleCount = count;
    const std::vector<std::uint8_t> initial_exterior =
        exteriorReachableEmpty(dimensions, occupied);
    std::vector<std::uint8_t> required_protected(count, 0);
    for (std::size_t index = 0; index < count; ++index)
    {
        required_protected[index] = occupied[index] == 0 &&
            protectedEmpty[index] != 0 && initial_exterior[index] != 0;
        result.statistics.protectedExteriorSampleCountBefore +=
            required_protected[index] != 0;
    }

    int current_euler =
        VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
            dimensions, result.occupied);
    result.statistics.bodyEulerBefore = current_euler;
    DefectCounts defects;
    if (!countDefects(dimensions, result.occupied, options, &defects))
    {
        result.cancelled = true;
        result.occupied = occupied;
        return result;
    }
    result.statistics.edgeCheckerboardCountBefore = defects.edgeCheckerboards;
    result.statistics.vertexOccupiedComponentDefectCountBefore =
        defects.vertexOccupiedComponents;
    result.statistics.vertexEmptyComponentDefectCountBefore =
        defects.vertexEmptyComponents;

    const int maximum_passes = std::max(0, options.maximumPasses);
    for (int pass = 0; pass < maximum_passes && defects.total() > 0; ++pass)
    {
        if (cancellationRequested())
        {
            result.cancelled = true;
            result.occupied = occupied;
            return result;
        }
        const std::size_t remaining_fill_limit =
            options.maximumFilledSampleCount >
                    result.statistics.filledSampleCount
            ? options.maximumFilledSampleCount -
                  result.statistics.filledSampleCount
            : 0;
        std::vector<std::uint8_t> candidate = result.occupied;
        std::size_t pass_filled = 0;
        if (!buildProposal(
                dimensions,
                protectedEmpty,
                options,
                remaining_fill_limit,
                &result.statistics,
                &candidate,
                &pass_filled))
        {
            result.cancelled = true;
            result.occupied = occupied;
            return result;
        }
        if (pass_filled == 0)
        {
            break;
        }
        ++result.statistics.attemptedPassCount;
        const int candidate_euler =
            VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
                dimensions, candidate);
        const auto preservesProtectedReachability =
            [&](const std::vector<std::uint8_t> &occupancy)
        {
            const std::vector<std::uint8_t> candidate_exterior =
                exteriorReachableEmpty(dimensions, occupancy);
            for (std::size_t index = 0; index < count; ++index)
            {
                if (required_protected[index] != 0 &&
                    candidate_exterior[index] == 0)
                {
                    return false;
                }
            }
            return true;
        };
        const bool batch_euler_safe = candidate_euler >= current_euler;
        const bool batch_reachability_safe = batch_euler_safe &&
            preservesProtectedReachability(candidate);
        if (!batch_euler_safe)
        {
            ++result.statistics.rolledBackEulerPassCount;
        }
        else if (!batch_reachability_safe)
        {
            ++result.statistics.rolledBackProtectedReachabilityPassCount;
        }
        if (batch_euler_safe && batch_reachability_safe)
        {
            result.occupied = std::move(candidate);
            result.statistics.filledSampleCount += pass_filled;
            ++result.statistics.acceptedPassCount;
            current_euler = candidate_euler;
        }
        else
        {
            std::size_t individually_filled = 0;
            for (std::size_t index = 0; index < count; ++index)
            {
                if (result.occupied[index] != 0 || candidate[index] == 0)
                {
                    continue;
                }
                result.occupied[index] = 1;
                const int individual_euler =
                    VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
                        dimensions, result.occupied);
                if (individual_euler < current_euler)
                {
                    result.occupied[index] = 0;
                    ++result.statistics.rejectedEulerSampleCount;
                    continue;
                }
                if (!preservesProtectedReachability(result.occupied))
                {
                    result.occupied[index] = 0;
                    ++result.statistics
                          .rejectedProtectedReachabilitySampleCount;
                    continue;
                }
                current_euler = individual_euler;
                ++individually_filled;
            }
            if (individually_filled == 0)
            {
                break;
            }
            result.statistics.filledSampleCount += individually_filled;
            ++result.statistics.acceptedPassCount;
        }
        if (!countDefects(dimensions, result.occupied, options, &defects))
        {
            result.cancelled = true;
            result.occupied = occupied;
            return result;
        }
    }

    if (!countDefects(dimensions, result.occupied, options, &defects))
    {
        result.cancelled = true;
        result.occupied = occupied;
        return result;
    }
    result.statistics.remainingEdgeCheckerboardCount =
        defects.edgeCheckerboards;
    result.statistics.remainingVertexOccupiedComponentDefectCount =
        defects.vertexOccupiedComponents;
    result.statistics.remainingVertexEmptyComponentDefectCount =
        defects.vertexEmptyComponents;
    const std::vector<std::uint8_t> final_exterior =
        exteriorReachableEmpty(dimensions, result.occupied);
    for (std::size_t index = 0; index < count; ++index)
    {
        result.statistics.protectedExteriorSampleCountAfter +=
            required_protected[index] != 0 && final_exterior[index] != 0;
    }
    result.statistics.bodyEulerAfter = current_euler;
    result.ok = true;
    return result;
}

} // namespace xjw::mesh
