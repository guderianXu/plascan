#include "TextureLabelOptimizer.h"

#include "BinaryGridMinCutSolverInternal.h"

#include <algorithm>
#include <limits>
#include <queue>

namespace xjw::mesh::texture_v4
{
    namespace
    {

        TextureLabelCost saturatingAdd(TextureLabelCost first, TextureLabelCost second)
        {
            if (first >= kForbiddenTextureLabelCost - second)
            {
                return kForbiddenTextureLabelCost;
            }
            return first + second;
        }

        TextureLabelCost unaryCost(const TextureLabelOptimizationProblem& problem, int node, int label)
        {
            return std::clamp<TextureLabelCost>(problem.unaryCost(node, label), 0, kForbiddenTextureLabelCost);
        }

        TextureLabelCost labelingEnergy(const TextureLabelOptimizationProblem& problem, const std::vector<int>& labels)
        {
            TextureLabelCost energy = 0;
            for (int node = 0; node < problem.nodeCount; ++node)
            {
                energy = saturatingAdd(energy, unaryCost(problem, node, labels[static_cast<std::size_t>(node)]));
            }
            for (const TextureLabelEdge& edge : problem.edges)
            {
                if (labels[static_cast<std::size_t>(edge.first)] != labels[static_cast<std::size_t>(edge.second)])
                {
                    energy = saturatingAdd(energy, edge.cost);
                }
            }
            return energy;
        }

        void addTerminalWeights(detail::ResidualGraph* graph,
                                int source,
                                int sink,
                                int node,
                                TextureLabelCost source_capacity,
                                TextureLabelCost sink_capacity)
        {
            const TextureLabelCost common = std::min(source_capacity, sink_capacity);
            source_capacity -= common;
            sink_capacity -= common;
            if (source_capacity > 0)
            {
                graph->addDirected(source, node, source_capacity);
            }
            if (sink_capacity > 0)
            {
                graph->addDirected(node, sink, sink_capacity);
            }
        }

        std::vector<std::uint8_t>
        sourcePartition(const detail::ResidualGraph& graph, int source, const std::function<bool()>& is_cancelled)
        {
            std::vector<std::uint8_t> reached(graph.heads.size(), 0);
            std::queue<int> pending;
            reached[static_cast<std::size_t>(source)] = 1;
            pending.push(source);
            int visited_count = 0;
            while (!pending.empty())
            {
                if (++visited_count % 4096 == 0 && is_cancelled && is_cancelled())
                {
                    return {};
                }
                const int node = pending.front();
                pending.pop();
                for (int edge_index = graph.heads[static_cast<std::size_t>(node)]; edge_index >= 0;
                     edge_index = graph.edges[static_cast<std::size_t>(edge_index)].next)
                {
                    const detail::ResidualEdge& edge = graph.edges[static_cast<std::size_t>(edge_index)];
                    if (edge.capacity > 0 && reached[static_cast<std::size_t>(edge.destination)] == 0)
                    {
                        reached[static_cast<std::size_t>(edge.destination)] = 1;
                        pending.push(edge.destination);
                    }
                }
            }
            return reached;
        }

        bool validProblem(const TextureLabelOptimizationProblem& problem, std::string* error)
        {
            if (problem.nodeCount <= 0 || problem.labelCount <= 0 ||
                static_cast<int>(problem.initialLabels.size()) != problem.nodeCount || !problem.unaryCost)
            {
                *error = "invalid texture label optimization dimensions";
                return false;
            }
            if (static_cast<std::uint64_t>(problem.nodeCount) * 4U + problem.edges.size() * 10U >
                static_cast<std::uint64_t>(std::numeric_limits<int>::max() - 2))
            {
                *error = "texture graph exceeds the supported node/edge index range";
                return false;
            }
            for (const int label : problem.initialLabels)
            {
                if (label < 0 || label >= problem.labelCount)
                {
                    *error = "initial texture label is out of range";
                    return false;
                }
            }
            for (const TextureLabelEdge& edge : problem.edges)
            {
                if (edge.first < 0 || edge.second < 0 || edge.first >= problem.nodeCount ||
                    edge.second >= problem.nodeCount || edge.first == edge.second || edge.cost < 0 ||
                    edge.cost >= kForbiddenTextureLabelCost)
                {
                    *error = "invalid texture label edge";
                    return false;
                }
            }
            return true;
        }

    } // namespace

    TextureLabelOptimizationResult optimizeTextureLabels(const TextureLabelOptimizationProblem& problem)
    {
        TextureLabelOptimizationResult result;
        if (!validProblem(problem, &result.error))
        {
            return result;
        }
        result.labels = problem.initialLabels;
        result.energy = labelingEnergy(problem, result.labels);
        std::vector<TextureLabelCost> incident_costs(static_cast<std::size_t>(problem.nodeCount), 0);
        TextureLabelCost capacity_budget = result.energy;
        for (const TextureLabelEdge& edge : problem.edges)
        {
            capacity_budget = saturatingAdd(capacity_budget, edge.cost);
            incident_costs[edge.first] = saturatingAdd(incident_costs[edge.first], edge.cost);
            incident_costs[edge.second] = saturatingAdd(incident_costs[edge.second], edge.cost);
        }
        if (capacity_budget >= kForbiddenTextureLabelCost / 4)
        {
            result.error = "texture label initial state is infeasible or capacities exceed the safe integer range";
            return result;
        }

        for (int pass = 0; pass < std::max(1, problem.maximumPasses); ++pass)
        {
            bool pass_changed = false;
            for (int alpha = 0; alpha < problem.labelCount; ++alpha)
            {
                if (problem.progressFn)
                {
                    problem.progressFn(pass, alpha);
                }
                if (problem.isCancelled && problem.isCancelled())
                {
                    result.cancelled = true;
                    result.labels.clear();
                    return result;
                }
                const int source = problem.nodeCount + static_cast<int>(problem.edges.size());
                const int sink = source + 1;
                detail::ResidualGraph graph(sink + 1);
                graph.reserve(static_cast<std::size_t>(problem.nodeCount) * 4U + problem.edges.size() * 10U);
                for (int node = 0; node < problem.nodeCount; ++node)
                {
                    if (node % 4096 == 0 && problem.isCancelled && problem.isCancelled())
                    {
                        result.cancelled = true;
                        result.labels.clear();
                        return result;
                    }
                    const int current = result.labels[static_cast<std::size_t>(node)];
                    // A node-local hard bound dominates every possible incident
                    // edge saving, without N * global-infinity preflow overflow.
                    const TextureLabelCost current_cost = unaryCost(problem, node, current);
                    const TextureLabelCost hard_bound = current_cost + incident_costs[node] + 1;
                    const TextureLabelCost switch_cost = std::min(unaryCost(problem, node, alpha), hard_bound);
                    const TextureLabelCost keep_cost = current == alpha ? hard_bound : current_cost;
                    addTerminalWeights(&graph, source, sink, node, switch_cost, keep_cost);
                }

                int auxiliary = problem.nodeCount;
                int edge_count = 0;
                for (const TextureLabelEdge& edge : problem.edges)
                {
                    if (++edge_count % 4096 == 0 && problem.isCancelled && problem.isCancelled())
                    {
                        result.cancelled = true;
                        result.labels.clear();
                        return result;
                    }
                    const int first_label = result.labels[static_cast<std::size_t>(edge.first)];
                    const int second_label = result.labels[static_cast<std::size_t>(edge.second)];
                    if (first_label == alpha && second_label == alpha)
                    {
                        continue;
                    }
                    if (first_label == second_label)
                    {
                        graph.addUndirected(edge.first, edge.second, edge.cost);
                    }
                    else if (first_label == alpha)
                    {
                        graph.addDirected(edge.second, edge.first, edge.cost);
                    }
                    else if (second_label == alpha)
                    {
                        graph.addDirected(edge.first, edge.second, edge.cost);
                    }
                    else
                    {
                        const TextureLabelCost infinity = edge.cost * 2 + 1;
                        addTerminalWeights(&graph, source, sink, auxiliary, 0, edge.cost);
                        graph.addDirected(edge.first, auxiliary, edge.cost);
                        graph.addDirected(auxiliary, edge.first, infinity);
                        graph.addDirected(edge.second, auxiliary, edge.cost);
                        graph.addDirected(auxiliary, edge.second, infinity);
                        ++auxiliary;
                    }
                }

                const detail::MaximumFlowOutcome flow =
                    detail::solveMaximumFlow(&graph, source, sink, problem.isCancelled);
                if (flow.cancelled)
                {
                    result.cancelled = true;
                    result.labels.clear();
                    return result;
                }
                if (!flow.solved)
                {
                    result.error = flow.error.empty() ? "texture alpha-expansion maximum flow failed" : flow.error;
                    result.labels.clear();
                    return result;
                }

                const std::vector<std::uint8_t> source_side = sourcePartition(graph, source, problem.isCancelled);
                if (source_side.empty())
                {
                    result.cancelled = true;
                    result.labels.clear();
                    return result;
                }
                std::vector<int> proposal = result.labels;
                for (int node = 0; node < problem.nodeCount; ++node)
                {
                    if (source_side[static_cast<std::size_t>(node)] == 0)
                    {
                        proposal[static_cast<std::size_t>(node)] = alpha;
                    }
                }
                const TextureLabelCost proposal_energy = labelingEnergy(problem, proposal);
                if (proposal_energy < result.energy)
                {
                    result.labels.swap(proposal);
                    result.energy = proposal_energy;
                    pass_changed = true;
                }
                ++result.completedExpansionCount;
            }
            if (!pass_changed)
            {
                break;
            }
        }

        for (int node = 0; node < problem.nodeCount; ++node)
        {
            if (result.labels[static_cast<std::size_t>(node)] != problem.initialLabels[static_cast<std::size_t>(node)])
            {
                ++result.changedNodeCount;
            }
        }
        result.solved = true;
        return result;
    }

} // namespace xjw::mesh::texture_v4
