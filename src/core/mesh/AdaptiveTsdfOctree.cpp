#include "AdaptiveTsdfOctree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace xjw::mesh
{
namespace
{

std::size_t sampleIndex(const std::array<int, 3> &dimensions,
                        int x,
                        int y,
                        int z)
{
    return static_cast<std::size_t>(z) *
               static_cast<std::size_t>(dimensions[0]) *
               static_cast<std::size_t>(dimensions[1]) +
        static_cast<std::size_t>(y) *
            static_cast<std::size_t>(dimensions[0]) +
        static_cast<std::size_t>(x);
}

std::uint64_t nodeKey(int level, int x, int y, int z)
{
    return (static_cast<std::uint64_t>(level) << 58u) |
        (static_cast<std::uint64_t>(z) << 38u) |
        (static_cast<std::uint64_t>(y) << 19u) |
        static_cast<std::uint64_t>(x);
}

struct MergeGroup
{
    std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
    std::uint8_t childMask = 0;
    float minimumValue = std::numeric_limits<float>::infinity();
    float maximumValue = -std::numeric_limits<float>::infinity();
};

std::unordered_map<std::uint64_t, int> buildLeafLookup(
    const std::vector<AdaptiveTsdfOctreeNode> &leaves)
{
    std::unordered_map<std::uint64_t, int> lookup;
    lookup.reserve(leaves.size() * 2);
    for (int index = 0; index < static_cast<int>(leaves.size()); ++index)
    {
        const AdaptiveTsdfOctreeNode &leaf = leaves[index];
        lookup.emplace(
            nodeKey(leaf.level,
                    leaf.origin[0],
                    leaf.origin[1],
                    leaf.origin[2]),
            index);
    }
    return lookup;
}

int findLeaf(const AdaptiveTsdfOctreeResult &octree,
             const std::unordered_map<std::uint64_t, int> &lookup,
             int x,
             int y,
             int z)
{
    if (x < 0 || y < 0 || z < 0 ||
        x >= octree.dimensions[0] ||
        y >= octree.dimensions[1] ||
        z >= octree.dimensions[2])
    {
        return -1;
    }
    const int maximum_level = octree.statistics.maximumLevel;
    for (int level = maximum_level; level >= 0; --level)
    {
        const int size = 1 << level;
        const int origin_x = x / size * size;
        const int origin_y = y / size * size;
        const int origin_z = z / size * size;
        const auto iterator = lookup.find(
            nodeKey(level, origin_x, origin_y, origin_z));
        if (iterator != lookup.cend())
        {
            return iterator->second;
        }
    }
    return -1;
}

template <typename Visitor>
void visitFaceNeighbors(const AdaptiveTsdfOctreeResult &octree,
                        const std::unordered_map<std::uint64_t, int> &lookup,
                        int leafIndex,
                        int direction,
                        Visitor visitor)
{
    const AdaptiveTsdfOctreeNode &leaf = octree.leaves[leafIndex];
    const int axis = direction / 2;
    const bool positive = direction % 2 != 0;
    const int first_tangent = (axis + 1) % 3;
    const int second_tangent = (axis + 2) % 3;
    const std::array<int, 2> tangent_offsets = {
        std::max(0, leaf.size / 4),
        std::max(0, leaf.size * 3 / 4)};
    std::array<int, 4> visited{{-1, -1, -1, -1}};
    int visited_count = 0;
    for (const int first_offset : tangent_offsets)
    {
        for (const int second_offset : tangent_offsets)
        {
            std::array<int, 3> point = leaf.origin;
            point[axis] += positive ? leaf.size : -1;
            point[first_tangent] += std::min(
                leaf.size - 1, first_offset);
            point[second_tangent] += std::min(
                leaf.size - 1, second_offset);
            const int neighbor = findLeaf(
                octree, lookup, point[0], point[1], point[2]);
            bool already_visited = false;
            for (int index = 0; index < visited_count; ++index)
            {
                already_visited = already_visited ||
                    visited[index] == neighbor;
            }
            if (neighbor >= 0 && !already_visited)
            {
                visited[visited_count++] = neighbor;
                visitor(neighbor);
            }
        }
    }
}

void rebuildFaceNeighbors(AdaptiveTsdfOctreeResult *octree)
{
    const auto lookup = buildLeafLookup(octree->leaves);
    for (int leaf_index = 0;
         leaf_index < static_cast<int>(octree->leaves.size());
         ++leaf_index)
    {
        AdaptiveTsdfOctreeNode &leaf = octree->leaves[leaf_index];
        leaf.faceNeighbors.fill(-1);
        for (int direction = 0; direction < 6; ++direction)
        {
            int selected = -1;
            visitFaceNeighbors(
                *octree,
                lookup,
                leaf_index,
                direction,
                [&](int neighbor)
                {
                    if (selected < 0 ||
                        octree->leaves[neighbor].level <
                            octree->leaves[selected].level)
                    {
                        selected = neighbor;
                    }
                });
            leaf.faceNeighbors[direction] = selected;
        }
    }
}

AdaptiveTsdfOctreeNode mergedParent(
    const std::array<int, 8> &children,
    const std::vector<AdaptiveTsdfOctreeNode> &leaves)
{
    const AdaptiveTsdfOctreeNode &first = leaves[children[0]];
    AdaptiveTsdfOctreeNode parent;
    parent.level = first.level + 1;
    parent.size = first.size * 2;
    parent.origin = {
        first.origin[0] / parent.size * parent.size,
        first.origin[1] / parent.size * parent.size,
        first.origin[2] / parent.size * parent.size};
    float weighted_value = 0.0f;
    for (const int child_index : children)
    {
        const AdaptiveTsdfOctreeNode &child = leaves[child_index];
        const float child_weight = std::max(
            0.01f, child.observationWeight);
        weighted_value += child.value * child_weight;
        parent.observationWeight += child_weight;
        parent.geometrySourceMask = static_cast<std::uint16_t>(
            parent.geometrySourceMask | child.geometrySourceMask);
        parent.activeSampleCount += child.activeSampleCount;
        parent.supportedSampleCount += child.supportedSampleCount;
        for (std::size_t bin = 0; bin < parent.histogram.bins.size(); ++bin)
        {
            parent.histogram.bins[bin] += child.histogram.bins[bin];
        }
    }
    parent.value = weighted_value /
        std::max(0.01f, parent.observationWeight);
    return parent;
}

void splitLeaf(const AdaptiveTsdfOctreeNode &parent,
               std::vector<AdaptiveTsdfOctreeNode> *children)
{
    const int child_size = parent.size / 2;
    for (int child_z = 0; child_z < 2; ++child_z)
    {
        for (int child_y = 0; child_y < 2; ++child_y)
        {
            for (int child_x = 0; child_x < 2; ++child_x)
            {
                AdaptiveTsdfOctreeNode child = parent;
                child.level = parent.level - 1;
                child.size = child_size;
                child.origin = {
                    parent.origin[0] + child_x * child_size,
                    parent.origin[1] + child_y * child_size,
                    parent.origin[2] + child_z * child_size};
                child.observationWeight /= 8.0f;
                child.activeSampleCount =
                    std::max<std::uint32_t>(1, parent.activeSampleCount / 8);
                child.supportedSampleCount =
                    parent.supportedSampleCount / 8;
                for (float &bin : child.histogram.bins)
                {
                    bin /= 8.0f;
                }
                child.faceNeighbors.fill(-1);
                children->push_back(child);
            }
        }
    }
}

} // namespace

AdaptiveTsdfOctreeResult AdaptiveTsdfOctree::build(
    const std::array<int, 3> &dimensions,
    const std::vector<float> &field,
    const std::vector<float> &observationWeight,
    const std::vector<std::uint16_t> &geometrySourceMask,
    const std::vector<std::uint8_t> &active,
    const std::vector<std::uint8_t> &supported,
    const std::vector<DepthVisibilityHistogram> &histograms,
    const AdaptiveTsdfOctreeOptions &options)
{
    AdaptiveTsdfOctreeResult result;
    result.dimensions = dimensions;
    const std::size_t expected_size =
        static_cast<std::size_t>(dimensions[0]) *
        static_cast<std::size_t>(dimensions[1]) *
        static_cast<std::size_t>(dimensions[2]);
    if (dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[2] <= 0 ||
        field.size() != expected_size ||
        observationWeight.size() != expected_size ||
        geometrySourceMask.size() != expected_size ||
        active.size() != expected_size ||
        supported.size() != expected_size ||
        histograms.size() != expected_size)
    {
        throw std::invalid_argument(
            "AdaptiveTsdfOctree input arrays do not match dimensions");
    }

    for (int z = 0; z < dimensions[2]; ++z)
    {
        for (int y = 0; y < dimensions[1]; ++y)
        {
            for (int x = 0; x < dimensions[0]; ++x)
            {
                const std::size_t index = sampleIndex(
                    dimensions, x, y, z);
                if (active[index] == 0)
                {
                    continue;
                }
                AdaptiveTsdfOctreeNode leaf;
                leaf.origin = {x, y, z};
                leaf.value = field[index];
                leaf.observationWeight = observationWeight[index];
                leaf.geometrySourceMask = geometrySourceMask[index];
                leaf.activeSampleCount = 1;
                leaf.supportedSampleCount = supported[index] != 0 ? 1 : 0;
                histograms[index].accumulate(&leaf.histogram);
                result.leaves.push_back(leaf);
            }
        }
    }
    result.statistics.inputActiveSampleCount = result.leaves.size();

    const int maximum_merge_level = std::clamp(
        options.maximumMergeLevel, 0, 12);
    for (int child_level = 0;
         child_level < maximum_merge_level;
         ++child_level)
    {
        const int child_size = 1 << child_level;
        const int parent_size = child_size * 2;
        std::unordered_map<std::uint64_t, MergeGroup> groups;
        groups.reserve(result.leaves.size() / 4 + 1);
        for (int index = 0;
             index < static_cast<int>(result.leaves.size());
             ++index)
        {
            const AdaptiveTsdfOctreeNode &leaf = result.leaves[index];
            if (leaf.level != child_level)
            {
                continue;
            }
            const int parent_x = leaf.origin[0] / parent_size * parent_size;
            const int parent_y = leaf.origin[1] / parent_size * parent_size;
            const int parent_z = leaf.origin[2] / parent_size * parent_size;
            const int child_slot =
                ((leaf.origin[2] - parent_z) / child_size) * 4 +
                ((leaf.origin[1] - parent_y) / child_size) * 2 +
                (leaf.origin[0] - parent_x) / child_size;
            MergeGroup &group = groups[
                nodeKey(child_level + 1, parent_x, parent_y, parent_z)];
            group.children[child_slot] = index;
            group.childMask = static_cast<std::uint8_t>(
                group.childMask | (1u << child_slot));
            group.minimumValue = std::min(
                group.minimumValue, leaf.value);
            group.maximumValue = std::max(
                group.maximumValue, leaf.value);
        }

        std::vector<std::uint8_t> merged(result.leaves.size(), 0);
        std::vector<AdaptiveTsdfOctreeNode> parents;
        for (const auto &[key, group] : groups)
        {
            (void)key;
            const bool crosses_zero =
                group.minimumValue < 0.0f && group.maximumValue >= 0.0f;
            const float minimum_absolute = std::min(
                std::fabs(group.minimumValue),
                std::fabs(group.maximumValue));
            if (group.childMask != 0xffu ||
                crosses_zero ||
                minimum_absolute <
                    options.minimumMergeAbsoluteField ||
                group.maximumValue - group.minimumValue >
                    options.maximumMergeFieldRange)
            {
                continue;
            }
            parents.push_back(mergedParent(group.children, result.leaves));
            for (const int child : group.children)
            {
                merged[child] = 1;
            }
            result.statistics.mergedNodeCount += 7;
        }
        if (parents.empty())
        {
            continue;
        }
        std::vector<AdaptiveTsdfOctreeNode> next;
        next.reserve(result.leaves.size() + parents.size());
        for (int index = 0;
             index < static_cast<int>(result.leaves.size());
             ++index)
        {
            if (merged[index] == 0)
            {
                next.push_back(result.leaves[index]);
            }
        }
        next.insert(next.end(), parents.begin(), parents.end());
        result.leaves.swap(next);
    }

    for (const AdaptiveTsdfOctreeNode &leaf : result.leaves)
    {
        result.statistics.maximumLevel = std::max(
            result.statistics.maximumLevel, leaf.level);
    }
    bool final_neighbors_ready = false;
    for (int balance_pass = 0; balance_pass < 16; ++balance_pass)
    {
        const auto lookup = buildLeafLookup(result.leaves);
        std::unordered_set<int> split_indices;
        for (int leaf_index = 0;
             leaf_index < static_cast<int>(result.leaves.size());
             ++leaf_index)
        {
            result.leaves[leaf_index].faceNeighbors.fill(-1);
            for (int direction = 0; direction < 6; ++direction)
            {
                visitFaceNeighbors(
                    result,
                    lookup,
                    leaf_index,
                    direction,
                    [&](int neighbor)
                    {
                        int &selected =
                            result.leaves[leaf_index]
                                .faceNeighbors[direction];
                        if (selected < 0 ||
                            result.leaves[neighbor].level <
                                result.leaves[selected].level)
                        {
                            selected = neighbor;
                        }
                        const int level_difference =
                            result.leaves[leaf_index].level -
                            result.leaves[neighbor].level;
                        if (level_difference > 1)
                        {
                            split_indices.insert(leaf_index);
                        }
                        else if (level_difference < -1)
                        {
                            split_indices.insert(neighbor);
                        }
                    });
            }
        }
        if (split_indices.empty())
        {
            final_neighbors_ready = true;
            break;
        }
        std::vector<AdaptiveTsdfOctreeNode> balanced;
        balanced.reserve(result.leaves.size() + split_indices.size() * 7);
        for (int index = 0;
             index < static_cast<int>(result.leaves.size());
             ++index)
        {
            if (split_indices.count(index) == 0)
            {
                balanced.push_back(result.leaves[index]);
                continue;
            }
            splitLeaf(result.leaves[index], &balanced);
            ++result.statistics.balanceSplitCount;
        }
        result.leaves.swap(balanced);
    }
    if (!final_neighbors_ready)
    {
        rebuildFaceNeighbors(&result);
    }
    result.statistics.twoToOneBalanced = final_neighbors_ready ||
        isTwoToOneBalanced(result);
    return result;
}

bool AdaptiveTsdfOctree::isTwoToOneBalanced(
    const AdaptiveTsdfOctreeResult &octree)
{
    const auto lookup = buildLeafLookup(octree.leaves);
    for (int leaf_index = 0;
         leaf_index < static_cast<int>(octree.leaves.size());
         ++leaf_index)
    {
        bool balanced = true;
        for (int direction = 0; direction < 6; ++direction)
        {
            visitFaceNeighbors(
                octree,
                lookup,
                leaf_index,
                direction,
                [&](int neighbor)
                {
                    balanced = balanced &&
                        std::abs(octree.leaves[leaf_index].level -
                                 octree.leaves[neighbor].level) <= 1;
                });
        }
        if (!balanced)
        {
            return false;
        }
    }
    return true;
}

} // namespace xjw::mesh
