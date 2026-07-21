#include "tracks/MultiViewTrackBuilder.h"
#include "common/DisjointSet.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace xjw
{

namespace
{

void rebuildTrackStats(MultiViewTrackBuildResult *result)
{
    if (!result)
    {
        return;
    }

    result->acceptedComponents = static_cast<int>(result->tracks.size());
    result->trackLengthHistogram.clear();
    result->trackConfidenceScores.clear();
    result->meanTrackConfidence = 0.0;
    for (const Track &track : result->tracks)
    {
        ++result->trackLengthHistogram[static_cast<int>(track.length())];
        result->trackConfidenceScores.push_back(track.confidence);
        result->meanTrackConfidence += track.confidence;
    }
}

bool isStationaryTrack(const Track &track,
                       const std::map<ImageId, std::vector<FeatureKeypoint>> &keypointsByImage,
                       float maxPixelMotion)
{
    if (track.length() < 2)
    {
        return false;
    }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    int validObservationCount = 0;

    for (const TrackElement &element : track.elements)
    {
        const auto imageIt = keypointsByImage.find(element.imageId);
        if (imageIt == keypointsByImage.end() ||
            element.featureIdx >= static_cast<FeatureIdx>(imageIt->second.size()))
        {
            return false;
        }

        const FeatureKeypoint &keypoint = imageIt->second[static_cast<std::size_t>(element.featureIdx)];
        minX = std::min(minX, keypoint.x);
        minY = std::min(minY, keypoint.y);
        maxX = std::max(maxX, keypoint.x);
        maxY = std::max(maxY, keypoint.y);
        ++validObservationCount;
    }

    return validObservationCount >= 2 &&
        (maxX - minX) <= maxPixelMotion &&
        (maxY - minY) <= maxPixelMotion;
}

void pruneStationaryTracks(const MultiViewTrackBuilder::BuildOptions &options,
                           const std::map<ImageId, std::vector<FeatureKeypoint>> &keypointsByImage,
                           MultiViewTrackBuildResult *result)
{
    if (!result || !options.excludeStationaryTracks || result->tracks.empty())
    {
        return;
    }

    const float maxPixelMotion = std::max(0.0f, options.stationaryTrackMaxPixelMotion);
    std::vector<Track> keptTracks;
    keptTracks.reserve(result->tracks.size());
    for (const Track &track : result->tracks)
    {
        if (isStationaryTrack(track, keypointsByImage, maxPixelMotion))
        {
            ++result->prunedStationaryTracks;
            continue;
        }
        keptTracks.push_back(track);
    }

    result->tracks = std::move(keptTracks);
    rebuildTrackStats(result);
}

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

void MultiViewTrackBuilder::setImageKeypoints(ImageId imageId, const std::vector<FeatureKeypoint> &keypoints)
{
    if (imageId == kInvalidImageId)
    {
        return;
    }
    _keypointsByImage[imageId] = keypoints;
}

MultiViewTrackBuildResult MultiViewTrackBuilder::build(const BuildOptions &options) const
{
    MultiViewTrackBuildResult result;

    detail::DisjointSet disjointSet;
    std::map<ObservationKey, int> indexByKey;
    std::vector<ObservationKey> keys;
    const auto observationIndex = [&](const ObservationKey &key)
    {
        const auto existing = indexByKey.find(key);
        if (existing != indexByKey.end())
        {
            return existing->second;
        }
        const int index = disjointSet.add();
        indexByKey.emplace(key, index);
        keys.push_back(key);
        return index;
    };

    std::vector<std::pair<int, int>> indexedEdges;
    indexedEdges.reserve(_edges.size());
    for (const auto &edge : _edges)
    {
        const int left = observationIndex(edge.first);
        const int right = observationIndex(edge.second);
        indexedEdges.emplace_back(left, right);
    }

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

        const detail::DisjointSet::MergeResult mergeResult =
            disjointSet.unite(leftRoot, rightRoot);
        const int newRoot = mergeResult.root;
        const int mergedRoot = mergeResult.absorbedRoot;
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

    pruneStationaryTracks(options, _keypointsByImage, &result);

    if (options.enableQualityThinning && !result.tracks.empty())
    {
        std::vector<int> order(result.tracks.size());
        for (int i = 0; i < static_cast<int>(order.size()); ++i)
        {
            order[static_cast<std::size_t>(i)] = i;
        }
        std::sort(order.begin(), order.end(), [&](int left, int right)
        {
            const Track &leftTrack = result.tracks[static_cast<std::size_t>(left)];
            const Track &rightTrack = result.tracks[static_cast<std::size_t>(right)];
            if (leftTrack.length() != rightTrack.length())
            {
                return leftTrack.length() > rightTrack.length();
            }
            if (leftTrack.confidence != rightTrack.confidence)
            {
                return leftTrack.confidence > rightTrack.confidence;
            }
            return left < right;
        });

        std::map<ImageId, int> acceptedByImage;
        std::map<std::tuple<ImageId, int, int>, int> acceptedByGridCell;
        std::vector<Track> thinnedTracks;
        thinnedTracks.reserve(result.tracks.size());

        const int gridColumns = std::max(1, options.gridColumns);
        const int gridRows = std::max(1, options.gridRows);
        const bool useGridLimit =
            options.maxTracksPerGridCell > 0 && options.imageWidth > 0.0f && options.imageHeight > 0.0f;

        for (int trackIndex : order)
        {
            const Track &track = result.tracks[static_cast<std::size_t>(trackIndex)];
            bool keep = true;
            for (const TrackElement &element : track.elements)
            {
                if (options.maxTracksPerImage > 0 &&
                    acceptedByImage[element.imageId] >= options.maxTracksPerImage)
                {
                    keep = false;
                    break;
                }

                if (!useGridLimit)
                {
                    continue;
                }

                const auto keypointsIt = _keypointsByImage.find(element.imageId);
                if (keypointsIt == _keypointsByImage.end() ||
                    element.featureIdx >= static_cast<FeatureIdx>(keypointsIt->second.size()))
                {
                    continue;
                }

                const FeatureKeypoint &keypoint = keypointsIt->second[static_cast<std::size_t>(element.featureIdx)];
                const int col = std::max(0, std::min(gridColumns - 1,
                    static_cast<int>(std::floor(keypoint.x / options.imageWidth * gridColumns))));
                const int row = std::max(0, std::min(gridRows - 1,
                    static_cast<int>(std::floor(keypoint.y / options.imageHeight * gridRows))));
                const auto gridKey = std::make_tuple(element.imageId, col, row);
                if (acceptedByGridCell[gridKey] >= options.maxTracksPerGridCell)
                {
                    keep = false;
                    break;
                }
            }

            if (!keep)
            {
                ++result.prunedByQualityThinning;
                continue;
            }

            for (const TrackElement &element : track.elements)
            {
                ++acceptedByImage[element.imageId];
                if (!useGridLimit)
                {
                    continue;
                }

                const auto keypointsIt = _keypointsByImage.find(element.imageId);
                if (keypointsIt == _keypointsByImage.end() ||
                    element.featureIdx >= static_cast<FeatureIdx>(keypointsIt->second.size()))
                {
                    continue;
                }

                const FeatureKeypoint &keypoint = keypointsIt->second[static_cast<std::size_t>(element.featureIdx)];
                const int col = std::max(0, std::min(gridColumns - 1,
                    static_cast<int>(std::floor(keypoint.x / options.imageWidth * gridColumns))));
                const int row = std::max(0, std::min(gridRows - 1,
                    static_cast<int>(std::floor(keypoint.y / options.imageHeight * gridRows))));
                ++acceptedByGridCell[std::make_tuple(element.imageId, col, row)];
            }
            thinnedTracks.push_back(track);
        }

        result.tracks = std::move(thinnedTracks);
        rebuildTrackStats(&result);
    }

    if (!result.trackConfidenceScores.empty())
    {
        result.meanTrackConfidence /=
            static_cast<double>(result.trackConfidenceScores.size());
    }

    return result;
}

} // namespace xjw
