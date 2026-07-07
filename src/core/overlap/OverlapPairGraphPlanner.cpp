#include "OverlapPairGraphPlanner.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace xjw
{
namespace
{

std::uint64_t pairKey(int indexA, int indexB)
{
    if (indexA > indexB)
    {
        std::swap(indexA, indexB);
    }
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(indexA)) << 32) |
           static_cast<std::uint32_t>(indexB);
}

bool validPair(int imageCount, int indexA, int indexB)
{
    return indexA >= 0 &&
           indexB >= 0 &&
           indexA < imageCount &&
           indexB < imageCount &&
           indexA != indexB;
}

void appendSource(OverlapPairGraphEdge *edge, OverlapPairGraphSource source)
{
    if (!edge)
    {
        return;
    }
    if (std::find(edge->sources.begin(), edge->sources.end(), source) == edge->sources.end())
    {
        edge->sources.push_back(source);
    }
}

struct Neighbor
{
    int index = -1;
    double score = 0.0;
};

struct DisjointSet
{
    explicit DisjointSet(int count)
        : parents(static_cast<std::size_t>(count))
    {
        std::iota(parents.begin(), parents.end(), 0);
    }

    int root(int index)
    {
        int result = index;
        while (parents[static_cast<std::size_t>(result)] != result)
        {
            result = parents[static_cast<std::size_t>(result)];
        }

        while (parents[static_cast<std::size_t>(index)] != index)
        {
            const int parent = parents[static_cast<std::size_t>(index)];
            parents[static_cast<std::size_t>(index)] = result;
            index = parent;
        }
        return result;
    }

    void unite(int left, int right)
    {
        const int rootLeft = root(left);
        const int rootRight = root(right);
        if (rootLeft != rootRight)
        {
            parents[static_cast<std::size_t>(rootRight)] = rootLeft;
        }
    }

    int componentCount()
    {
        std::unordered_set<int> roots;
        for (int i = 0; i < static_cast<int>(parents.size()); ++i)
        {
            roots.insert(root(i));
        }
        return static_cast<int>(roots.size());
    }

    std::vector<int> parents;
};

std::unordered_map<std::uint64_t, OverlapPairGraphEdge> uniqueInputEdges(
    const std::vector<OverlapPairGraphInputEdge> &edges,
    int imageCount)
{
    std::unordered_map<std::uint64_t, OverlapPairGraphEdge> unique;
    for (const OverlapPairGraphInputEdge &input : edges)
    {
        if (!validPair(imageCount, input.indexA, input.indexB))
        {
            continue;
        }

        const int indexA = std::min(input.indexA, input.indexB);
        const int indexB = std::max(input.indexA, input.indexB);
        const std::uint64_t key = pairKey(indexA, indexB);
        auto it = unique.find(key);
        if (it == unique.end() || input.bowScore > it->second.bowScore)
        {
            OverlapPairGraphEdge edge;
            edge.indexA = indexA;
            edge.indexB = indexB;
            edge.bowScore = input.bowScore;
            edge.sharedWordCount = input.sharedWordCount;
            edge.geometricInliers = input.geometricInliers;
            unique[key] = edge;
        }
    }
    return unique;
}

std::vector<std::vector<Neighbor>> buildTopNeighbors(
    const std::unordered_map<std::uint64_t, OverlapPairGraphEdge> &edges,
    int imageCount,
    double minSimilarity)
{
    std::vector<std::vector<Neighbor>> neighbors(static_cast<std::size_t>(imageCount));
    for (const auto &entry : edges)
    {
        const OverlapPairGraphEdge &edge = entry.second;
        if (edge.bowScore < minSimilarity)
        {
            continue;
        }
        neighbors[static_cast<std::size_t>(edge.indexA)].push_back({edge.indexB, edge.bowScore});
        neighbors[static_cast<std::size_t>(edge.indexB)].push_back({edge.indexA, edge.bowScore});
    }

    for (std::vector<Neighbor> &row : neighbors)
    {
        std::sort(row.begin(), row.end(), [](const Neighbor &left, const Neighbor &right)
        {
            if (left.score != right.score)
            {
                return left.score > right.score;
            }
            return left.index < right.index;
        });
    }
    return neighbors;
}

bool containsTopNeighbor(const std::vector<std::vector<Neighbor>> &neighbors,
                         int indexA,
                         int indexB,
                         int topK)
{
    if (indexA < 0 || indexA >= static_cast<int>(neighbors.size()))
    {
        return false;
    }

    const std::vector<Neighbor> &row = neighbors[static_cast<std::size_t>(indexA)];
    const int limit = std::min(std::max(0, topK), static_cast<int>(row.size()));
    for (int i = 0; i < limit; ++i)
    {
        if (row[static_cast<std::size_t>(i)].index == indexB)
        {
            return true;
        }
    }
    return false;
}

OverlapPairGraphEdge *addOrUpdateEdge(std::unordered_map<std::uint64_t, OverlapPairGraphEdge> *accepted,
                                      const std::unordered_map<std::uint64_t, OverlapPairGraphEdge> &inputByKey,
                                      int indexA,
                                      int indexB,
                                      OverlapPairGraphSource source)
{
    if (!accepted)
    {
        return nullptr;
    }

    const int a = std::min(indexA, indexB);
    const int b = std::max(indexA, indexB);
    const std::uint64_t key = pairKey(a, b);
    auto acceptedIt = accepted->find(key);
    if (acceptedIt == accepted->end())
    {
        auto inputIt = inputByKey.find(key);
        OverlapPairGraphEdge edge;
        if (inputIt != inputByKey.end())
        {
            edge = inputIt->second;
        }
        else
        {
            edge.indexA = a;
            edge.indexB = b;
        }
        acceptedIt = accepted->insert({key, edge}).first;
    }

    appendSource(&acceptedIt->second, source);
    return &acceptedIt->second;
}

DisjointSet buildComponents(const std::unordered_map<std::uint64_t, OverlapPairGraphEdge> &accepted, int imageCount)
{
    DisjointSet components(imageCount);
    for (const auto &entry : accepted)
    {
        const OverlapPairGraphEdge &edge = entry.second;
        components.unite(edge.indexA, edge.indexB);
    }
    return components;
}

std::vector<int> edgeDegrees(const std::unordered_map<std::uint64_t, OverlapPairGraphEdge> &accepted,
                             int imageCount)
{
    std::vector<int> degrees(static_cast<std::size_t>(imageCount), 0);
    for (const auto &entry : accepted)
    {
        const OverlapPairGraphEdge &edge = entry.second;
        ++degrees[static_cast<std::size_t>(edge.indexA)];
        ++degrees[static_cast<std::size_t>(edge.indexB)];
    }
    return degrees;
}

int addBowComponentBridges(std::unordered_map<std::uint64_t, OverlapPairGraphEdge> *accepted,
                           const std::unordered_map<std::uint64_t, OverlapPairGraphEdge> &inputByKey,
                           int imageCount,
                           int maxBridgeCount)
{
    int bridgeCount = 0;
    while (true)
    {
        DisjointSet components = buildComponents(*accepted, imageCount);
        if (components.componentCount() <= 1)
        {
            break;
        }
        if (maxBridgeCount > 0 && bridgeCount >= maxBridgeCount)
        {
            break;
        }

        const OverlapPairGraphEdge *best = nullptr;
        for (const auto &entry : inputByKey)
        {
            if (accepted->find(entry.first) != accepted->end())
            {
                continue;
            }

            const OverlapPairGraphEdge &edge = entry.second;
            if (edge.bowScore <= 0.0 || components.root(edge.indexA) == components.root(edge.indexB))
            {
                continue;
            }

            if (!best ||
                edge.bowScore > best->bowScore ||
                (edge.bowScore == best->bowScore && pairKey(edge.indexA, edge.indexB) < pairKey(best->indexA, best->indexB)))
            {
                best = &edge;
            }
        }

        if (!best)
        {
            break;
        }

        addOrUpdateEdge(accepted, inputByKey, best->indexA, best->indexB, OverlapPairGraphSource::BowComponentBridge);
        ++bridgeCount;
    }
    return bridgeCount;
}

int addSequenceWindowEdges(std::unordered_map<std::uint64_t, OverlapPairGraphEdge> *accepted,
                           const std::unordered_map<std::uint64_t, OverlapPairGraphEdge> &inputByKey,
                           int imageCount,
                           int sequenceWindow)
{
    if (!accepted || imageCount < 2)
    {
        return 0;
    }

    int addedCount = 0;
    const int window = std::max(1, sequenceWindow);
    for (int distance = 1; distance <= window; ++distance)
    {
        for (int i = 0; i + distance < imageCount; ++i)
        {
            const int j = i + distance;
            const bool existed = accepted->find(pairKey(i, j)) != accepted->end();
            addOrUpdateEdge(accepted, inputByKey, i, j, OverlapPairGraphSource::SequenceBridge);
            if (!existed)
            {
                ++addedCount;
            }
        }
    }
    return addedCount;
}

void addSequenceLoopEdges(std::unordered_map<std::uint64_t, OverlapPairGraphEdge> *accepted,
                          const std::unordered_map<std::uint64_t, OverlapPairGraphEdge> &inputByKey,
                          int imageCount,
                          int sequenceWindow)
{
    if (imageCount < 3)
    {
        return;
    }

    const int window = std::min(std::max(1, sequenceWindow), imageCount - 1);
    for (int i = 0; i < imageCount; ++i)
    {
        for (int distance = 1; distance <= window; ++distance)
        {
            const int j = (i + distance) % imageCount;
            if (j > i)
            {
                continue;
            }
            addOrUpdateEdge(accepted, inputByKey, i, j, OverlapPairGraphSource::SequenceLoop);
        }
    }
}

std::string makeDetail(const OverlapPairGraphPlan &plan)
{
    std::ostringstream stream;
    stream << "components_before=" << plan.componentCountBeforeRepair
           << " components_after=" << plan.componentCountAfterRepair
           << " bow_bridges=" << plan.bowBridgeCount
           << " sequence_bridges=" << plan.sequenceBridgeCount
           << " accepted=" << plan.edges.size();
    return stream.str();
}

} // namespace

const char *overlapPairGraphSourceId(OverlapPairGraphSource source)
{
    switch (source)
    {
    case OverlapPairGraphSource::BowMutualTopK:
        return "bow_mutual_top_k";
    case OverlapPairGraphSource::BowOneWayTopK:
        return "bow_one_way_top_k";
    case OverlapPairGraphSource::BowComponentBridge:
        return "bow_component_bridge";
    case OverlapPairGraphSource::SequenceBridge:
        return "sequence_bridge";
    case OverlapPairGraphSource::SequenceLoop:
        return "sequence_loop";
    }
    return "unknown";
}

OverlapPairGraphPlan OverlapPairGraphPlanner::plan(const std::vector<OverlapPairGraphInputEdge> &edges,
                                                   const OverlapPairGraphPlannerOptions &options)
{
    OverlapPairGraphPlan plan;
    const int imageCount = std::max(0, options.imageCount);
    if (imageCount <= 0)
    {
        plan.detail = makeDetail(plan);
        return plan;
    }

    const int topK = std::max(1, options.topK);
    const int minPairsPerImage = std::max(0, options.minPairsPerImage);
    const double minSimilarity = std::max(0.0, options.minSimilarity);
    const std::unordered_map<std::uint64_t, OverlapPairGraphEdge> inputByKey =
        uniqueInputEdges(edges, imageCount);
    const std::vector<std::vector<Neighbor>> topNeighbors =
        buildTopNeighbors(inputByKey, imageCount, minSimilarity);

    std::unordered_map<std::uint64_t, OverlapPairGraphEdge> accepted;
    for (const auto &entry : inputByKey)
    {
        const OverlapPairGraphEdge &edge = entry.second;
        const bool aHasB = containsTopNeighbor(topNeighbors, edge.indexA, edge.indexB, topK);
        const bool bHasA = containsTopNeighbor(topNeighbors, edge.indexB, edge.indexA, topK);
        if (options.mutualTopK)
        {
            if (aHasB && bHasA)
            {
                addOrUpdateEdge(&accepted, inputByKey, edge.indexA, edge.indexB,
                                OverlapPairGraphSource::BowMutualTopK);
            }
        }
        else if (aHasB || bHasA)
        {
            addOrUpdateEdge(&accepted, inputByKey, edge.indexA, edge.indexB,
                            OverlapPairGraphSource::BowOneWayTopK);
        }
    }

    if (options.keepOneWayTopK)
    {
        std::vector<int> degrees = edgeDegrees(accepted, imageCount);
        std::vector<const OverlapPairGraphEdge *> candidates;
        candidates.reserve(inputByKey.size());
        for (const auto &entry : inputByKey)
        {
            const OverlapPairGraphEdge &edge = entry.second;
            const bool aHasB = containsTopNeighbor(topNeighbors, edge.indexA, edge.indexB, topK);
            const bool bHasA = containsTopNeighbor(topNeighbors, edge.indexB, edge.indexA, topK);
            if (aHasB || bHasA)
            {
                candidates.push_back(&edge);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const OverlapPairGraphEdge *left, const OverlapPairGraphEdge *right)
        {
            if (left->bowScore != right->bowScore)
            {
                return left->bowScore > right->bowScore;
            }
            return pairKey(left->indexA, left->indexB) < pairKey(right->indexA, right->indexB);
        });

        for (const OverlapPairGraphEdge *edge : candidates)
        {
            if (minPairsPerImage > 0 &&
                degrees[static_cast<std::size_t>(edge->indexA)] >= minPairsPerImage &&
                degrees[static_cast<std::size_t>(edge->indexB)] >= minPairsPerImage)
            {
                continue;
            }
            OverlapPairGraphEdge *acceptedEdge =
                addOrUpdateEdge(&accepted, inputByKey, edge->indexA, edge->indexB,
                                OverlapPairGraphSource::BowOneWayTopK);
            if (acceptedEdge && acceptedEdge->sources.size() == 1)
            {
                ++degrees[static_cast<std::size_t>(edge->indexA)];
                ++degrees[static_cast<std::size_t>(edge->indexB)];
            }
        }
    }

    plan.componentCountBeforeRepair = buildComponents(accepted, imageCount).componentCount();
    if (options.connectComponents)
    {
        plan.bowBridgeCount = addBowComponentBridges(&accepted,
                                                     inputByKey,
                                                     imageCount,
                                                     std::max(0, options.componentBridgeMaxPairs));
        if (options.useSequenceFallback)
        {
            // BoW 的连通性只代表外观相似，不代表这些边都能通过几何验证。
            // 对按采集顺序绕目标拍摄的数据，固定保留顺序窗口能给后续匹配和 SfM
            // 足够的局部桥接机会，避免候选图“看似连通、实际有效匹配断图”。
            plan.sequenceBridgeCount = addSequenceWindowEdges(&accepted,
                                                              inputByKey,
                                                              imageCount,
                                                              options.sequenceWindow);
        }
    }

    if (options.closeSequenceLoop)
    {
        addSequenceLoopEdges(&accepted, inputByKey, imageCount, options.sequenceWindow);
    }

    plan.componentCountAfterRepair = buildComponents(accepted, imageCount).componentCount();
    plan.edges.reserve(accepted.size());
    for (const auto &entry : accepted)
    {
        plan.edges.push_back(entry.second);
    }
    std::sort(plan.edges.begin(), plan.edges.end(), [](const OverlapPairGraphEdge &left,
                                                       const OverlapPairGraphEdge &right)
    {
        if (left.bowScore != right.bowScore)
        {
            return left.bowScore > right.bowScore;
        }
        return pairKey(left.indexA, left.indexB) < pairKey(right.indexA, right.indexB);
    });
    plan.detail = makeDetail(plan);
    return plan;
}

} // namespace xjw
