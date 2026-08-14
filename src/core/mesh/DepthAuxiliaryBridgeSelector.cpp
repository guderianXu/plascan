#include "DepthAuxiliaryBridgeSelector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xjw::mesh
{
namespace
{

constexpr double kDistanceEpsilon = 1.0e-12;

class DisjointSet
{
public:
    explicit DisjointSet(int size) : _parent(static_cast<std::size_t>(size))
    {
        std::iota(_parent.begin(), _parent.end(), 0);
    }

    int find(int value)
    {
        if (_parent[static_cast<std::size_t>(value)] != value)
        {
            _parent[static_cast<std::size_t>(value)] = find(
                _parent[static_cast<std::size_t>(value)]);
        }
        return _parent[static_cast<std::size_t>(value)];
    }

    bool unite(int left, int right)
    {
        left = find(left);
        right = find(right);
        if (left == right)
        {
            return false;
        }
        _parent[static_cast<std::size_t>(right)] = left;
        return true;
    }

private:
    std::vector<int> _parent;
};

bool isFiniteAtLeast(double value, double minimum)
{
    return std::isfinite(value) && value >= minimum;
}

bool isEligibleAuxiliary(
    const DepthAuxiliaryBridgeNode &node,
    const DepthAuxiliaryBridgeSelectionOptions &options)
{
    return !node.primary
        && node.frameIndex >= 0
        && node.refIndex >= 0
        && std::isfinite(node.sparseAbsoluteDepthMedianLogError)
        && node.sparseAbsoluteDepthMedianLogError >= 0.0
        && node.sparseAbsoluteDepthMedianLogError
            < options.maximumSparseAbsoluteDepthMedianLogError
        && isFiniteAtLeast(
            node.validWithinMaskRatio,
            options.minimumValidWithinMaskRatio)
        && isFiniteAtLeast(
            node.consistencyRetentionRatio,
            options.minimumConsistencyRetentionRatio)
        && isFiniteAtLeast(
            node.largestComponentRatio,
            options.minimumLargestComponentRatio)
        && isFiniteAtLeast(
            node.meanConfidence,
            options.minimumMeanConfidence)
        && node.sourceViewCount >= options.minimumSourceViewCount
        && node.trustedPixelCount >= options.minimumTrustedPixelCount;
}

double auxiliaryCost(
    const DepthAuxiliaryBridgeNode &node,
    const DepthAuxiliaryBridgeSelectionOptions &options,
    double auxiliary_hop_cost)
{
    const double sparse_scale = std::max(
        options.maximumSparseAbsoluteDepthMedianLogError,
        std::numeric_limits<double>::epsilon());
    const double sparse_penalty = std::clamp(
        node.sparseAbsoluteDepthMedianLogError / sparse_scale, 0.0, 1.0);
    const double confidence_penalty = std::clamp(
        1.0 - node.meanConfidence, 0.0, 1.0);
    const double coverage_penalty = std::clamp(
        1.0 - node.validWithinMaskRatio, 0.0, 1.0);
    const double component_penalty = std::clamp(
        1.0 - node.largestComponentRatio, 0.0, 1.0);
    const double reason_penalty = std::min(0.20, 0.02 * node.qualityReasonCount);
    return auxiliary_hop_cost
        + 0.45 * sparse_penalty
        + 0.20 * confidence_penalty
        + 0.15 * coverage_penalty
        + 0.10 * component_penalty
        + reason_penalty;
}

std::vector<int> sortedAuxiliaryFrames(
    const std::vector<int> &path,
    const std::vector<DepthAuxiliaryBridgeNode> &nodes)
{
    std::vector<int> frames;
    for (const int index : path)
    {
        if (!nodes[static_cast<std::size_t>(index)].primary)
        {
            frames.push_back(nodes[static_cast<std::size_t>(index)].frameIndex);
        }
    }
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    return frames;
}

bool betterPath(
    double candidate_distance,
    const std::vector<int> &candidate_auxiliary_frames,
    double current_distance,
    const std::vector<int> &current_auxiliary_frames)
{
    if (candidate_distance + kDistanceEpsilon < current_distance)
    {
        return true;
    }
    if (std::abs(candidate_distance - current_distance) > kDistanceEpsilon)
    {
        return false;
    }
    return candidate_auxiliary_frames < current_auxiliary_frames;
}

struct ComponentPath
{
    int left = -1;
    int right = -1;
    double cost = std::numeric_limits<double>::infinity();
    std::vector<int> nodeIndices;
    std::vector<int> auxiliaryFrameIndices;
};

} // namespace

DepthAuxiliaryBridgeSelectionResult DepthAuxiliaryBridgeSelector::select(
    const std::vector<DepthAuxiliaryBridgeNode> &nodes,
    const DepthAuxiliaryBridgeSelectionOptions &options)
{
    DepthAuxiliaryBridgeSelectionResult result;
    const int node_count = static_cast<int>(nodes.size());
    if (node_count == 0)
    {
        result.failClosed = true;
        return result;
    }

    std::unordered_map<int, int> node_by_ref;
    node_by_ref.reserve(nodes.size());
    std::unordered_set<int> frame_indices;
    frame_indices.reserve(nodes.size());
    bool invalid_identity = false;
    for (int index = 0; index < node_count; ++index)
    {
        const auto &node = nodes[static_cast<std::size_t>(index)];
        if (node.frameIndex < 0 || node.refIndex < 0
            || !frame_indices.insert(node.frameIndex).second
            || !node_by_ref.emplace(node.refIndex, index).second)
        {
            invalid_identity = true;
            break;
        }
    }
    if (invalid_identity)
    {
        result.failClosed = true;
        return result;
    }

    std::vector<std::vector<int>> adjacency(nodes.size());
    for (int index = 0; index < node_count; ++index)
    {
        for (const int source_ref :
             nodes[static_cast<std::size_t>(index)].geometrySourceIndices)
        {
            const auto source = node_by_ref.find(source_ref);
            if (source == node_by_ref.end() || source->second == index)
            {
                continue;
            }
            adjacency[static_cast<std::size_t>(index)].push_back(source->second);
            adjacency[static_cast<std::size_t>(source->second)].push_back(index);
        }
    }
    for (auto &neighbors : adjacency)
    {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());
    }

    DisjointSet primary_sets(node_count);
    std::vector<int> primary_indices;
    for (int index = 0; index < node_count; ++index)
    {
        if (nodes[static_cast<std::size_t>(index)].primary)
        {
            primary_indices.push_back(index);
        }
    }
    if (primary_indices.empty())
    {
        result.failClosed = true;
        return result;
    }
    for (const int index : primary_indices)
    {
        for (const int neighbor : adjacency[static_cast<std::size_t>(index)])
        {
            if (nodes[static_cast<std::size_t>(neighbor)].primary)
            {
                primary_sets.unite(index, neighbor);
            }
        }
    }

    std::map<int, int> component_by_root;
    std::vector<int> primary_component(nodes.size(), -1);
    std::vector<std::vector<int>> component_nodes;
    for (const int index : primary_indices)
    {
        const int root = primary_sets.find(index);
        auto inserted = component_by_root.emplace(
            root, static_cast<int>(component_by_root.size()));
        const int component = inserted.first->second;
        if (inserted.second)
        {
            component_nodes.emplace_back();
        }
        primary_component[static_cast<std::size_t>(index)] = component;
        component_nodes[static_cast<std::size_t>(component)].push_back(index);
    }
    result.primaryComponentCount = static_cast<int>(component_nodes.size());
    if (result.primaryComponentCount == 1)
    {
        result.connected = true;
        return result;
    }

    std::vector<unsigned char> traversable(nodes.size(), 0);
    for (int index = 0; index < node_count; ++index)
    {
        const auto &node = nodes[static_cast<std::size_t>(index)];
        traversable[static_cast<std::size_t>(index)] = static_cast<unsigned char>(
            node.primary || isEligibleAuxiliary(node, options));
    }
    // Make path length lexicographically dominant over bounded quality
    // penalties. This keeps the selected recovery surface minimal before
    // comparing the quality of equal-hop alternatives.
    const double auxiliary_hop_cost =
        2.0 * static_cast<double>(nodes.size() + 1u);

    std::vector<ComponentPath> paths;
    for (int source_component = 0;
         source_component < result.primaryComponentCount;
         ++source_component)
    {
        std::vector<double> distance(
            nodes.size(), std::numeric_limits<double>::infinity());
        std::vector<std::vector<int>> path_nodes(nodes.size());
        using QueueEntry = std::pair<double, int>;
        std::priority_queue<
            QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;
        for (const int source_node :
             component_nodes[static_cast<std::size_t>(source_component)])
        {
            distance[static_cast<std::size_t>(source_node)] = 0.0;
            path_nodes[static_cast<std::size_t>(source_node)] = {source_node};
            queue.emplace(0.0, source_node);
        }

        while (!queue.empty())
        {
            const auto [queued_distance, index] = queue.top();
            queue.pop();
            if (queued_distance
                > distance[static_cast<std::size_t>(index)] + kDistanceEpsilon)
            {
                continue;
            }
            for (const int neighbor : adjacency[static_cast<std::size_t>(index)])
            {
                if (!traversable[static_cast<std::size_t>(neighbor)])
                {
                    continue;
                }
                const double step_cost =
                    nodes[static_cast<std::size_t>(neighbor)].primary
                    ? 0.0
                    : auxiliaryCost(
                        nodes[static_cast<std::size_t>(neighbor)],
                        options,
                        auxiliary_hop_cost);
                const double candidate_distance = queued_distance + step_cost;
                std::vector<int> candidate_path =
                    path_nodes[static_cast<std::size_t>(index)];
                candidate_path.push_back(neighbor);
                if (betterPath(
                        candidate_distance,
                        sortedAuxiliaryFrames(candidate_path, nodes),
                        distance[static_cast<std::size_t>(neighbor)],
                        sortedAuxiliaryFrames(
                            path_nodes[static_cast<std::size_t>(neighbor)],
                            nodes)))
                {
                    distance[static_cast<std::size_t>(neighbor)] =
                        candidate_distance;
                    path_nodes[static_cast<std::size_t>(neighbor)] =
                        std::move(candidate_path);
                    queue.emplace(candidate_distance, neighbor);
                }
            }
        }

        for (int target_component = source_component + 1;
             target_component < result.primaryComponentCount;
             ++target_component)
        {
            ComponentPath best;
            best.left = source_component;
            best.right = target_component;
            for (const int target_node :
                 component_nodes[static_cast<std::size_t>(target_component)])
            {
                const double candidate_distance =
                    distance[static_cast<std::size_t>(target_node)];
                if (!std::isfinite(candidate_distance))
                {
                    continue;
                }
                const std::vector<int> candidate_auxiliary =
                    sortedAuxiliaryFrames(
                        path_nodes[static_cast<std::size_t>(target_node)], nodes);
                if (betterPath(candidate_distance,
                               candidate_auxiliary,
                               best.cost,
                               best.auxiliaryFrameIndices))
                {
                    best.cost = candidate_distance;
                    best.nodeIndices =
                        path_nodes[static_cast<std::size_t>(target_node)];
                    best.auxiliaryFrameIndices = candidate_auxiliary;
                }
            }
            if (std::isfinite(best.cost))
            {
                paths.push_back(std::move(best));
            }
        }
    }

    std::sort(paths.begin(), paths.end(), [](const auto &left, const auto &right)
    {
        return std::tie(left.cost,
                        left.auxiliaryFrameIndices,
                        left.left,
                        left.right)
            < std::tie(right.cost,
                       right.auxiliaryFrameIndices,
                       right.left,
                       right.right);
    });

    DisjointSet component_sets(result.primaryComponentCount);
    std::set<int> selected;
    int accepted_edges = 0;
    for (const ComponentPath &path : paths)
    {
        if (!component_sets.unite(path.left, path.right))
        {
            continue;
        }
        selected.insert(path.auxiliaryFrameIndices.begin(),
                        path.auxiliaryFrameIndices.end());
        ++accepted_edges;
        if (accepted_edges + 1 == result.primaryComponentCount)
        {
            break;
        }
    }

    result.connected = accepted_edges + 1 == result.primaryComponentCount;
    result.failClosed = !result.connected;
    if (result.connected)
    {
        result.selectedAuxiliaryFrameIndices.assign(
            selected.begin(), selected.end());
        std::unordered_map<int, int> ref_by_frame;
        ref_by_frame.reserve(nodes.size());
        for (const DepthAuxiliaryBridgeNode &node : nodes)
        {
            ref_by_frame.emplace(node.frameIndex, node.refIndex);
        }
        result.selectedAuxiliaryRefIndices.reserve(selected.size());
        for (const int frame_index : selected)
        {
            result.selectedAuxiliaryRefIndices.push_back(
                ref_by_frame.at(frame_index));
        }
        std::sort(result.selectedAuxiliaryRefIndices.begin(),
                  result.selectedAuxiliaryRefIndices.end());
    }
    return result;
}

} // namespace xjw::mesh
