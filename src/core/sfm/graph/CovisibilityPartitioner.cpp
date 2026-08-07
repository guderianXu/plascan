#include "CovisibilityPartitioner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xjw
{

namespace
{

using WeightedNeighbor = std::pair<ImageId, std::size_t>;
using Adjacency = std::unordered_map<ImageId, std::vector<WeightedNeighbor>>;

std::vector<std::vector<ImageId>> connectedComponents(
    const std::vector<ImageId> &image_ids,
    const Adjacency &adjacency)
{
    std::unordered_set<ImageId> unseen(image_ids.begin(), image_ids.end());
    std::vector<std::vector<ImageId>> components;
    for (ImageId start : image_ids)
    {
        if (unseen.erase(start) == 0)
        {
            continue;
        }

        std::vector<ImageId> component;
        std::vector<ImageId> stack{start};
        while (!stack.empty())
        {
            const ImageId current = stack.back();
            stack.pop_back();
            component.push_back(current);
            const auto neighbors = adjacency.find(current);
            if (neighbors == adjacency.end())
            {
                continue;
            }
            for (const auto &[neighbor, weight] : neighbors->second)
            {
                (void)weight;
                if (unseen.erase(neighbor) > 0)
                {
                    stack.push_back(neighbor);
                }
            }
        }
        std::sort(component.begin(), component.end());
        components.push_back(std::move(component));
    }
    return components;
}

std::size_t weightedDegree(ImageId image_id, const Adjacency &adjacency)
{
    std::size_t degree = 0;
    const auto neighbors = adjacency.find(image_id);
    if (neighbors != adjacency.end())
    {
        for (const auto &[neighbor, weight] : neighbors->second)
        {
            (void)neighbor;
            degree += weight;
        }
    }
    return degree;
}

ImageId strongestCandidate(const std::unordered_set<ImageId> &unassigned,
                           const std::unordered_map<ImageId, std::size_t> &attachment,
                           const Adjacency &adjacency)
{
    ImageId best = kInvalidImageId;
    std::size_t best_attachment = 0;
    std::size_t best_degree = 0;
    for (ImageId candidate : unassigned)
    {
        const auto score = attachment.find(candidate);
        const std::size_t candidate_attachment =
            score == attachment.end() ? 0 : score->second;
        const std::size_t candidate_degree = weightedDegree(candidate, adjacency);
        if (best == kInvalidImageId || candidate_attachment > best_attachment ||
            (candidate_attachment == best_attachment && candidate_degree > best_degree) ||
            (candidate_attachment == best_attachment && candidate_degree == best_degree &&
             candidate < best))
        {
            best = candidate;
            best_attachment = candidate_attachment;
            best_degree = candidate_degree;
        }
    }
    return best;
}

void addAttachment(ImageId selected,
                   const std::unordered_set<ImageId> &unassigned,
                   const Adjacency &adjacency,
                   std::unordered_map<ImageId, std::size_t> *attachment)
{
    const auto neighbors = adjacency.find(selected);
    if (neighbors == adjacency.end())
    {
        return;
    }
    for (const auto &[neighbor, weight] : neighbors->second)
    {
        if (unassigned.count(neighbor) > 0)
        {
            (*attachment)[neighbor] += weight;
        }
    }
}

} // namespace

std::vector<ImageId> CovisibilityBlock::activeImageIds() const
{
    std::vector<ImageId> result = coreImageIds;
    result.insert(result.end(), overlapImageIds.begin(), overlapImageIds.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<CovisibilityBlock> CovisibilityPartitioner::partition(
    const std::vector<ImageId> &registeredImageIds,
    const CorrespondenceGraph &graph,
    const CovisibilityPartitionOptions &options)
{
    std::vector<ImageId> image_ids = registeredImageIds;
    std::sort(image_ids.begin(), image_ids.end());
    image_ids.erase(std::unique(image_ids.begin(), image_ids.end()), image_ids.end());
    if (image_ids.empty())
    {
        return {};
    }

    const std::size_t target_size = std::max<std::size_t>(2, options.targetCoreSize);
    const std::unordered_set<ImageId> registered(image_ids.begin(), image_ids.end());
    Adjacency adjacency;
    for (const ImagePair &pair : graph.imagePairs())
    {
        if (registered.count(pair.first) == 0 || registered.count(pair.second) == 0)
        {
            continue;
        }
        const std::size_t weight = graph.numMatchesBetween(pair.first, pair.second);
        if (weight == 0)
        {
            continue;
        }
        adjacency[pair.first].push_back({pair.second, weight});
        adjacency[pair.second].push_back({pair.first, weight});
    }

    std::vector<CovisibilityBlock> blocks;
    for (const std::vector<ImageId> &component : connectedComponents(image_ids, adjacency))
    {
        const std::size_t block_count =
            std::max<std::size_t>(1, (component.size() + target_size - 1) / target_size);
        std::unordered_set<ImageId> unassigned(component.begin(), component.end());
        for (std::size_t block_index = 0; block_index < block_count; ++block_index)
        {
            const std::size_t remaining_blocks = block_count - block_index;
            const std::size_t desired_size =
                (unassigned.size() + remaining_blocks - 1) / remaining_blocks;
            CovisibilityBlock block;
            std::unordered_map<ImageId, std::size_t> attachment;
            while (block.coreImageIds.size() < desired_size && !unassigned.empty())
            {
                const ImageId selected = strongestCandidate(unassigned, attachment, adjacency);
                if (selected == kInvalidImageId)
                {
                    break;
                }
                unassigned.erase(selected);
                block.coreImageIds.push_back(selected);
                attachment.erase(selected);
                addAttachment(selected, unassigned, adjacency, &attachment);
            }
            std::sort(block.coreImageIds.begin(), block.coreImageIds.end());
            blocks.push_back(std::move(block));
        }
    }

    const std::size_t overlap_size = options.overlapSize;
    for (CovisibilityBlock &block : blocks)
    {
        const std::unordered_set<ImageId> core(block.coreImageIds.begin(), block.coreImageIds.end());
        std::unordered_map<ImageId, std::size_t> overlap_scores;
        for (ImageId image_id : block.coreImageIds)
        {
            const auto neighbors = adjacency.find(image_id);
            if (neighbors == adjacency.end())
            {
                continue;
            }
            for (const auto &[neighbor, weight] : neighbors->second)
            {
                if (core.count(neighbor) == 0)
                {
                    overlap_scores[neighbor] += weight;
                }
            }
        }

        std::vector<std::pair<ImageId, std::size_t>> ranked(overlap_scores.begin(), overlap_scores.end());
        std::sort(ranked.begin(), ranked.end(), [](const auto &left, const auto &right)
        {
            if (left.second != right.second)
            {
                return left.second > right.second;
            }
            return left.first < right.first;
        });
        if (ranked.size() > overlap_size)
        {
            ranked.resize(overlap_size);
        }
        for (const auto &[image_id, score] : ranked)
        {
            (void)score;
            block.overlapImageIds.push_back(image_id);
        }
        std::sort(block.overlapImageIds.begin(), block.overlapImageIds.end());
    }

    return blocks;
}

} // namespace xjw
