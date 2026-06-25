#include "tracks/MultiViewTrackBuilder.h"

#include <algorithm>
#include <cmath>

namespace xjw
{

namespace
{

class DisjointSet
{
public:
    int add(const MultiViewTrackBuilder::ObservationKey &key)
    {
        const auto it = _indexByKey.find(key);
        if (it != _indexByKey.end())
        {
            return it->second;
        }

        const int index = static_cast<int>(_keys.size());
        _indexByKey.emplace(key, index);
        _keys.push_back(key);
        _parent.push_back(index);
        return index;
    }

    int addMergedRoot(int a, int b, int *mergedRoot)
    {
        int rootA = find(a);
        int rootB = find(b);
        if (rootA == rootB)
        {
            if (mergedRoot)
            {
                *mergedRoot = rootA;
            }
            return rootA;
        }
        if (rootB < rootA)
        {
            std::swap(rootA, rootB);
        }
        _parent[rootB] = rootA;
        if (mergedRoot)
        {
            *mergedRoot = rootB;
        }
        return rootA;
    }

    int find(int index)
    {
        int root = index;
        while (_parent[root] != root)
        {
            root = _parent[root];
        }
        while (_parent[index] != index)
        {
            const int next = _parent[index];
            _parent[index] = root;
            index = next;
        }
        return root;
    }

    const std::vector<MultiViewTrackBuilder::ObservationKey> &keys() const
    {
        return _keys;
    }

private:
    std::map<MultiViewTrackBuilder::ObservationKey, int> _indexByKey;
    std::vector<MultiViewTrackBuilder::ObservationKey> _keys;
    std::vector<int> _parent;
};

} // namespace

void MultiViewTrackBuilder::addMatchPair(ImageId imageA,
                                         ImageId imageB,
                                         const std::vector<MatchIndexPair> &matches)
{
    if (imageA == imageB || imageA == kInvalidImageId || imageB == kInvalidImageId)
    {
        return;
    }

    for (const MatchIndexPair &match : matches)
    {
        if (match.first == kInvalidFeatureIdx || match.second == kInvalidFeatureIdx)
        {
            continue;
        }

        _edges.push_back({
            ObservationKey{imageA, match.first},
            ObservationKey{imageB, match.second},
            std::isfinite(match.score) ? match.score : 0.0f,
            static_cast<int>(_edges.size())
        });
    }
}

MultiViewTrackBuildResult MultiViewTrackBuilder::build() const
{
    MultiViewTrackBuildResult result;

    DisjointSet disjointSet;
    std::vector<std::pair<int, int>> indexedEdges;
    indexedEdges.reserve(_edges.size());
    for (const auto &edge : _edges)
    {
        const int left = disjointSet.add(edge.first);
        const int right = disjointSet.add(edge.second);
        indexedEdges.emplace_back(left, right);
    }

    const std::vector<ObservationKey> &keys = disjointSet.keys();
    std::vector<std::map<ImageId, FeatureIdx>> featureByImageByRoot(keys.size());
    for (int i = 0; i < static_cast<int>(keys.size()); ++i)
    {
        featureByImageByRoot[static_cast<size_t>(i)].emplace(keys[static_cast<size_t>(i)].imageId,
                                                             keys[static_cast<size_t>(i)].featureIdx);
    }

    std::vector<int> order(_edges.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i)
    {
        order[static_cast<size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&](int left, int right)
    {
        const Edge &leftEdge = _edges[static_cast<size_t>(left)];
        const Edge &rightEdge = _edges[static_cast<size_t>(right)];
        if (leftEdge.score != rightEdge.score)
        {
            return leftEdge.score > rightEdge.score;
        }
        return leftEdge.insertionOrder < rightEdge.insertionOrder;
    });

    std::vector<double> edgeScoreSumByRoot(keys.size(), 0.0);
    std::vector<int> edgeScoreCountByRoot(keys.size(), 0);

    for (int edgeIndex : order)
    {
        const auto [leftIndex, rightIndex] = indexedEdges[static_cast<size_t>(edgeIndex)];
        int leftRoot = disjointSet.find(leftIndex);
        int rightRoot = disjointSet.find(rightIndex);
        if (leftRoot == rightRoot)
        {
            continue;
        }

        const auto &leftFeatures = featureByImageByRoot[static_cast<size_t>(leftRoot)];
        const auto &rightFeatures = featureByImageByRoot[static_cast<size_t>(rightRoot)];
        bool hasConflict = false;
        for (const auto &entry : rightFeatures)
        {
            const auto it = leftFeatures.find(entry.first);
            if (it != leftFeatures.end() && it->second != entry.second)
            {
                hasConflict = true;
                break;
            }
        }

        if (hasConflict)
        {
            ++result.rejectedConflictEdges;
            continue;
        }

        int mergedRoot = leftRoot;
        const int newRoot = disjointSet.addMergedRoot(leftRoot, rightRoot, &mergedRoot);
        if (newRoot != mergedRoot)
        {
            edgeScoreSumByRoot[static_cast<size_t>(newRoot)] +=
                edgeScoreSumByRoot[static_cast<size_t>(mergedRoot)];
            edgeScoreCountByRoot[static_cast<size_t>(newRoot)] +=
                edgeScoreCountByRoot[static_cast<size_t>(mergedRoot)];
            edgeScoreSumByRoot[static_cast<size_t>(mergedRoot)] = 0.0;
            edgeScoreCountByRoot[static_cast<size_t>(mergedRoot)] = 0;
        }
        edgeScoreSumByRoot[static_cast<size_t>(newRoot)] +=
            std::max(0.0f, _edges[static_cast<size_t>(edgeIndex)].score);
        ++edgeScoreCountByRoot[static_cast<size_t>(newRoot)];

        auto &newFeatures = featureByImageByRoot[static_cast<size_t>(newRoot)];
        auto &oldFeatures = featureByImageByRoot[static_cast<size_t>(mergedRoot)];
        if (newRoot != mergedRoot)
        {
            newFeatures.insert(oldFeatures.begin(), oldFeatures.end());
            oldFeatures.clear();
        }
    }

    std::map<int, std::vector<ObservationKey>> components;
    for (int i = 0; i < static_cast<int>(keys.size()); ++i)
    {
        components[disjointSet.find(i)].push_back(keys[static_cast<size_t>(i)]);
    }

    result.totalComponents = static_cast<int>(components.size());
    for (auto &entry : components)
    {
        std::vector<ObservationKey> &component = entry.second;
        std::sort(component.begin(), component.end());

        bool hasConflict = false;
        std::map<ImageId, FeatureIdx> featureByImage;
        for (const ObservationKey &observation : component)
        {
            const auto inserted = featureByImage.emplace(observation.imageId, observation.featureIdx);
            if (!inserted.second)
            {
                hasConflict = true;
                break;
            }
        }

        if (hasConflict)
        {
            ++result.rejectedConflictComponents;
            continue;
        }
        if (component.size() < 2)
        {
            continue;
        }

        Track track;
        track.elements.reserve(component.size());
        for (const ObservationKey &observation : component)
        {
            track.elements.push_back(TrackElement{observation.imageId, observation.featureIdx});
        }
        const int scoreCount = edgeScoreCountByRoot[static_cast<size_t>(entry.first)];
        track.confidence = scoreCount > 0
            ? edgeScoreSumByRoot[static_cast<size_t>(entry.first)] / static_cast<double>(scoreCount)
            : 1.0;

        ++result.acceptedComponents;
        ++result.trackLengthHistogram[static_cast<int>(track.length())];
        result.meanTrackConfidence += track.confidence;
        result.trackConfidenceScores.push_back(track.confidence);
        result.tracks.push_back(std::move(track));
    }
    if (!result.trackConfidenceScores.empty())
    {
        result.meanTrackConfidence /=
            static_cast<double>(result.trackConfidenceScores.size());
    }

    return result;
}

} // namespace xjw
