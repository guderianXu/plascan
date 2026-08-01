#include "BinaryGridMinCutSolver.h"

#include "BinaryGridMinCutSolverInternal.h"

#include <limits>
#include <queue>

namespace xjw::mesh
{
namespace
{

constexpr BinaryGridCapacity kCapacityLimit =
    std::numeric_limits<BinaryGridCapacity>::max() / 8;

bool addChecked(BinaryGridCapacity value, BinaryGridCapacity *sum)
{
    if (value < 0 || value > kCapacityLimit || *sum > kCapacityLimit - value)
    {
        return false;
    }
    *sum += value;
    return true;
}

bool validateAxis(
    const std::vector<BinaryGridCapacity> &capacities,
    const BinaryGridMinCutProblem &problem,
    int axis,
    std::string *error)
{
    if (capacities.empty())
    {
        return true;
    }
    if (capacities.size() != problem.nodeCount())
    {
        *error = "pairwise capacity array size does not match grid node count";
        return false;
    }
    for (int z = 0; z < problem.sizeZ; ++z)
    {
        for (int y = 0; y < problem.sizeY; ++y)
        {
            for (int x = 0; x < problem.sizeX; ++x)
            {
                const auto capacity = capacities[problem.index(x, y, z)];
                if (capacity < 0 || capacity > kCapacityLimit)
                {
                    *error = "pairwise capacities must be non-negative and bounded";
                    return false;
                }
                const bool boundary =
                    (axis == 0 && x + 1 == problem.sizeX) ||
                    (axis == 1 && y + 1 == problem.sizeY) ||
                    (axis == 2 && z + 1 == problem.sizeZ);
                if (boundary && capacity != 0)
                {
                    *error = "positive-axis boundary capacity must be zero";
                    return false;
                }
            }
        }
    }
    return true;
}

bool validateProblem(const BinaryGridMinCutProblem &problem, std::string *error)
{
    if (problem.sizeX <= 0 || problem.sizeY <= 0 || problem.sizeZ <= 0)
    {
        *error = "grid dimensions must be positive";
        return false;
    }
    const std::size_t node_count = problem.nodeCount();
    constexpr std::size_t kMaximumIndexedNodes =
        static_cast<std::size_t>(std::numeric_limits<int>::max() - 2) / 10U;
    if (node_count == 0 || node_count > kMaximumIndexedNodes)
    {
        *error = "grid node count exceeds solver indexing range";
        return false;
    }
    if (problem.sourceCapacities.size() != node_count ||
        problem.sinkCapacities.size() != node_count)
    {
        *error = "unary capacity array size does not match grid node count";
        return false;
    }

    BinaryGridCapacity source_sum = 0;
    BinaryGridCapacity sink_sum = 0;
    for (std::size_t index = 0; index < node_count; ++index)
    {
        if (!addChecked(problem.sourceCapacities[index], &source_sum) ||
            !addChecked(problem.sinkCapacities[index], &sink_sum))
        {
            *error = "unary capacities are negative or exceed solver range";
            return false;
        }
    }
    return validateAxis(problem.positiveXCapacities, problem, 0, error) &&
           validateAxis(problem.positiveYCapacities, problem, 1, error) &&
           validateAxis(problem.positiveZCapacities, problem, 2, error);
}

BinaryGridCapacity axisCapacity(
    const std::vector<BinaryGridCapacity> &capacities,
    std::size_t index)
{
    return capacities.empty() ? 0 : capacities[index];
}

bool cancellationRequested(const std::function<bool()> &is_cancelled)
{
    return is_cancelled && is_cancelled();
}

bool computeEnergy(
    const BinaryGridMinCutProblem &problem,
    const std::vector<BinaryGridLabel> &labels,
    BinaryGridCapacity *energy)
{
    *energy = 0;
    const auto add_energy = [&](BinaryGridCapacity value)
    {
        return addChecked(value, energy);
    };
    for (int z = 0; z < problem.sizeZ; ++z)
    {
        for (int y = 0; y < problem.sizeY; ++y)
        {
            for (int x = 0; x < problem.sizeX; ++x)
            {
                const std::size_t index = problem.index(x, y, z);
                const bool full = labels[index] == BinaryGridLabel::Full;
                if (!add_energy(
                        full ? problem.sinkCapacities[index] : problem.sourceCapacities[index]))
                {
                    return false;
                }
                if (x + 1 < problem.sizeX &&
                    labels[index] != labels[problem.index(x + 1, y, z)] &&
                    !add_energy(axisCapacity(problem.positiveXCapacities, index)))
                {
                    return false;
                }
                if (y + 1 < problem.sizeY &&
                    labels[index] != labels[problem.index(x, y + 1, z)] &&
                    !add_energy(axisCapacity(problem.positiveYCapacities, index)))
                {
                    return false;
                }
                if (z + 1 < problem.sizeZ &&
                    labels[index] != labels[problem.index(x, y, z + 1)] &&
                    !add_energy(axisCapacity(problem.positiveZCapacities, index)))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

bool markSourceSet(
    const detail::ResidualGraph &graph,
    int source,
    const std::function<bool()> &is_cancelled,
    std::vector<std::uint8_t> *reachable)
{
    reachable->assign(graph.heads.size(), 0);
    std::queue<int> frontier;
    (*reachable)[static_cast<std::size_t>(source)] = 1;
    frontier.push(source);
    std::uint64_t work = 0;
    while (!frontier.empty())
    {
        const int node = frontier.front();
        frontier.pop();
        if ((++work & 0x0FFFU) == 0U && cancellationRequested(is_cancelled))
        {
            return false;
        }
        for (int edge_index = graph.heads[static_cast<std::size_t>(node)];
             edge_index >= 0;
             edge_index = graph.edges[static_cast<std::size_t>(edge_index)].next)
        {
            const auto &edge = graph.edges[static_cast<std::size_t>(edge_index)];
            if (edge.capacity > 0 &&
                (*reachable)[static_cast<std::size_t>(edge.destination)] == 0)
            {
                (*reachable)[static_cast<std::size_t>(edge.destination)] = 1;
                frontier.push(edge.destination);
            }
        }
    }
    return true;
}

} // namespace

std::size_t BinaryGridMinCutProblem::nodeCount() const
{
    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0)
    {
        return 0;
    }
    const std::size_t x = static_cast<std::size_t>(sizeX);
    const std::size_t y = static_cast<std::size_t>(sizeY);
    const std::size_t z = static_cast<std::size_t>(sizeZ);
    if (x > std::numeric_limits<std::size_t>::max() / y ||
        x * y > std::numeric_limits<std::size_t>::max() / z)
    {
        return 0;
    }
    return x * y * z;
}

std::size_t BinaryGridMinCutProblem::index(int x, int y, int z) const
{
    return (static_cast<std::size_t>(z) * static_cast<std::size_t>(sizeY) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(sizeX) +
           static_cast<std::size_t>(x);
}

BinaryGridMinCutResult BinaryGridMinCutSolver::solve(
    const BinaryGridMinCutProblem &problem,
    const std::function<bool()> &isCancelled)
{
    BinaryGridMinCutResult result;
    if (!validateProblem(problem, &result.error))
    {
        return result;
    }
    if (cancellationRequested(isCancelled))
    {
        result.cancelled = true;
        return result;
    }

    const int node_count = static_cast<int>(problem.nodeCount());
    const int source = node_count;
    const int sink = node_count + 1;
    detail::ResidualGraph graph(node_count + 2);
    graph.reserve(static_cast<std::size_t>(node_count) * 10U);
    for (int node = 0; node < node_count; ++node)
    {
        if ((node & 0x0FFF) == 0 && cancellationRequested(isCancelled))
        {
            result.cancelled = true;
            return result;
        }
        const auto index = static_cast<std::size_t>(node);
        if (problem.sourceCapacities[index] > 0)
        {
            graph.addDirected(source, node, problem.sourceCapacities[index]);
        }
        if (problem.sinkCapacities[index] > 0)
        {
            graph.addDirected(node, sink, problem.sinkCapacities[index]);
        }
    }
    for (int z = 0; z < problem.sizeZ; ++z)
    {
        for (int y = 0; y < problem.sizeY; ++y)
        {
            for (int x = 0; x < problem.sizeX; ++x)
            {
                const std::size_t index = problem.index(x, y, z);
                if ((index & 0x0FFFU) == 0U && cancellationRequested(isCancelled))
                {
                    result.cancelled = true;
                    return result;
                }
                const int node = static_cast<int>(index);
                const auto add_neighbor = [&](int neighbor, BinaryGridCapacity capacity)
                {
                    if (capacity > 0)
                    {
                        graph.addUndirected(node, neighbor, capacity);
                        ++result.statistics.pairwiseEdgeCount;
                    }
                };
                if (x + 1 < problem.sizeX)
                {
                    add_neighbor(node + 1, axisCapacity(problem.positiveXCapacities, index));
                }
                if (y + 1 < problem.sizeY)
                {
                    add_neighbor(
                        node + problem.sizeX,
                        axisCapacity(problem.positiveYCapacities, index));
                }
                if (z + 1 < problem.sizeZ)
                {
                    add_neighbor(
                        node + problem.sizeX * problem.sizeY,
                        axisCapacity(problem.positiveZCapacities, index));
                }
            }
        }
    }

    const auto flow = detail::solveMaximumFlow(&graph, source, sink, isCancelled);
    result.cancelled = flow.cancelled;
    result.error = flow.error;
    result.statistics.pushCount = flow.pushCount;
    result.statistics.relabelCount = flow.relabelCount;
    result.statistics.dischargeCount = flow.dischargeCount;
    result.statistics.maximumFlow = flow.maximumFlow;
    if (!flow.solved)
    {
        return result;
    }

    std::vector<std::uint8_t> reachable;
    if (!markSourceSet(graph, source, isCancelled, &reachable))
    {
        result.cancelled = true;
        return result;
    }
    result.labels.resize(static_cast<std::size_t>(node_count), BinaryGridLabel::Empty);
    for (int node = 0; node < node_count; ++node)
    {
        if (reachable[static_cast<std::size_t>(node)] != 0)
        {
            result.labels[static_cast<std::size_t>(node)] = BinaryGridLabel::Full;
            ++result.statistics.sourceSetNodeCount;
        }
    }
    result.statistics.nodeCount = static_cast<std::uint64_t>(node_count);
    if (!computeEnergy(problem, result.labels, &result.statistics.cutEnergy))
    {
        result.labels.clear();
        result.error = "cut energy exceeds solver range";
        return result;
    }
    if (result.statistics.maximumFlow != result.statistics.cutEnergy)
    {
        result.labels.clear();
        result.error = "maximum flow and cut energy disagree";
        return result;
    }
    result.solved = true;
    return result;
}

} // namespace xjw::mesh
