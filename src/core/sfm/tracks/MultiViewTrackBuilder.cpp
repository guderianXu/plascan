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

MultiViewTrackBuildResult MultiViewTrackBuilder::build() const
{
    return build(BuildOptions{});
}

MultiViewTrackBuildResult MultiViewTrackBuilder::build(const BuildOptions& options) const
{
    MultiViewTrackBuildResult result;

    // 阶段 1：为每个全局观测键分配确定性并查集索引。一个节点代表“一幅影像中的
    // 一个特征”，而不是一条匹配，因此同一节点可自然连接多个 pair。
    detail::DisjointSet disjointSet;
    std::map<ObservationKey, int> indexByKey;
    std::vector<ObservationKey> keys;
    const auto observationIndex = [&](const ObservationKey& key)
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
    for (const auto& edge : _edges)
    {
        const int left = observationIndex(edge.first);
        const int right = observationIndex(edge.second);
        indexedEdges.emplace_back(left, right);
    }

    // 预先建立无重复的观测邻接表。它表示所有通过像对几何验证的直接对应，后续
    // 合并时既可检测精确三角闭环，也可统计两个组件间是否存在第二条独立支持边。
    std::vector<std::vector<int>> neighbors(keys.size());
    for (const auto& [left, right] : indexedEdges)
    {
        neighbors[static_cast<std::size_t>(left)].push_back(right);
        neighbors[static_cast<std::size_t>(right)].push_back(left);
    }
    for (std::vector<int>& row : neighbors)
    {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }

    const auto commonNeighborCount = [&neighbors](int left, int right, int limit)
    {
        const std::vector<int>& leftNeighbors = neighbors[static_cast<std::size_t>(left)];
        const std::vector<int>& rightNeighbors = neighbors[static_cast<std::size_t>(right)];
        std::size_t leftIndex = 0;
        std::size_t rightIndex = 0;
        int count = 0;
        while (leftIndex < leftNeighbors.size() && rightIndex < rightNeighbors.size())
        {
            if (leftNeighbors[leftIndex] < rightNeighbors[rightIndex])
            {
                ++leftIndex;
            }
            else if (rightNeighbors[rightIndex] < leftNeighbors[leftIndex])
            {
                ++rightIndex;
            }
            else
            {
                ++count;
                if (count >= limit)
                {
                    break;
                }
                ++leftIndex;
                ++rightIndex;
            }
        }
        return count;
    };

    // 每个并查集根维护 imageId -> featureIdx，用于在合并前 O(component images)
    // 检查“一条物点轨迹中同一影像只能有一个观测”的硬约束。
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
    // 阶段 2：高置信匹配优先合并；同分时保留输入顺序。这样冲突环路中优先留下
    // 更可靠边，同时保证相同输入每次构建结果一致。
    std::sort(order.begin(),
              order.end(),
              [&](int left, int right)
              {
                  const Edge& leftEdge = _edges[static_cast<size_t>(left)];
                  const Edge& rightEdge = _edges[static_cast<size_t>(right)];
                  if (leftEdge.score != rightEdge.score)
                  {
                      return leftEdge.score > rightEdge.score;
                  }
                  return leftEdge.insertionOrder < rightEdge.insertionOrder;
              });

    std::vector<double> edgeScoreSumByRoot(keys.size(), 0.0);
    std::vector<int> edgeScoreCountByRoot(keys.size(), 0);
    std::vector<std::vector<int>> membersByRoot(keys.size());
    for (int index = 0; index < static_cast<int>(keys.size()); ++index)
    {
        membersByRoot[static_cast<std::size_t>(index)].push_back(index);
    }

    for (int edgeIndex : order)
    {
        const auto [leftIndex, rightIndex] = indexedEdges[static_cast<size_t>(edgeIndex)];
        int leftRoot = disjointSet.find(leftIndex);
        int rightRoot = disjointSet.find(rightIndex);
        if (leftRoot == rightRoot)
        {
            continue;
        }

        const auto& leftFeatures = featureByImageByRoot[static_cast<size_t>(leftRoot)];
        const auto& rightFeatures = featureByImageByRoot[static_cast<size_t>(rightRoot)];
        bool hasConflict = false;
        for (const auto& entry : rightFeatures)
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
            // 拒绝当前边而不是丢弃整个组件。组件中已有高分一致关系继续保留，
            // 低分冲突边计入诊断供上游判断匹配质量。
            ++result.rejectedConflictEdges;
            continue;
        }

        const int minimumComponentSize = std::max(2, options.bridgeCheckMinComponentSize);
        const std::size_t leftComponentSize = membersByRoot[static_cast<std::size_t>(leftRoot)].size();
        const std::size_t rightComponentSize = membersByRoot[static_cast<std::size_t>(rightRoot)].size();
        if (options.enableBridgeConsistencyCheck &&
            leftComponentSize >= static_cast<std::size_t>(minimumComponentSize) &&
            rightComponentSize >= static_cast<std::size_t>(minimumComponentSize))
        {
            const int minimumCycleSupport = std::max(1, options.minimumBridgeCycleSupport);
            const bool cycleSupported =
                commonNeighborCount(leftIndex, rightIndex, minimumCycleSupport) >= minimumCycleSupport;

            int crossEdgeCount = 0;
            const int minimumCrossEdges = std::max(2, options.minimumBridgeCrossEdges);
            const std::vector<int>& smallerComponent = leftComponentSize <= rightComponentSize
                                                           ? membersByRoot[static_cast<std::size_t>(leftRoot)]
                                                           : membersByRoot[static_cast<std::size_t>(rightRoot)];
            const int oppositeRoot = leftComponentSize <= rightComponentSize ? rightRoot : leftRoot;
            for (int member : smallerComponent)
            {
                for (int neighbor : neighbors[static_cast<std::size_t>(member)])
                {
                    if (disjointSet.find(neighbor) == oppositeRoot)
                    {
                        ++crossEdgeCount;
                        if (crossEdgeCount >= minimumCrossEdges)
                        {
                            break;
                        }
                    }
                }
                if (crossEdgeCount >= minimumCrossEdges)
                {
                    break;
                }
            }
            const bool crossEdgeSupported = crossEdgeCount >= minimumCrossEdges;

            const auto componentMeanScore = [&](int root)
            {
                const int count = edgeScoreCountByRoot[static_cast<std::size_t>(root)];
                return count > 0 ? edgeScoreSumByRoot[static_cast<std::size_t>(root)] / static_cast<double>(count)
                                 : 0.0;
            };
            const double internalScore = std::min(componentMeanScore(leftRoot), componentMeanScore(rightRoot));
            const double minimumScoreRatio =
                std::clamp(static_cast<double>(options.unsupportedBridgeMinScoreRatio), 0.0, 1.0);
            const bool confidenceSupported =
                internalScore <= 0.0 || static_cast<double>(_edges[static_cast<std::size_t>(edgeIndex)].score) >=
                                            internalScore * minimumScoreRatio;
            const int matureComponentSize = std::max(minimumComponentSize, options.matureBridgeComponentSize);
            const bool matureComponents = leftComponentSize >= static_cast<std::size_t>(matureComponentSize) &&
                                          rightComponentSize >= static_cast<std::size_t>(matureComponentSize);

            // 两个成熟多视组件之间的单边关系即使分数较高也不足以证明同一物点；
            // 小组件则保留高置信桥接逃生口，避免连续影像只形成链式对应时被过度拆分。
            if (!cycleSupported && !crossEdgeSupported && (matureComponents || !confidenceSupported))
            {
                ++result.rejectedInconsistentBridgeEdges;
                continue;
            }
            if (cycleSupported || crossEdgeSupported)
            {
                ++result.acceptedSupportedBridgeEdges;
            }
        }

        const detail::DisjointSet::MergeResult mergeResult = disjointSet.unite(leftRoot, rightRoot);
        const int newRoot = mergeResult.root;
        const int mergedRoot = mergeResult.absorbedRoot;
        if (newRoot != mergedRoot)
        {
            edgeScoreSumByRoot[static_cast<size_t>(newRoot)] += edgeScoreSumByRoot[static_cast<size_t>(mergedRoot)];
            edgeScoreCountByRoot[static_cast<size_t>(newRoot)] += edgeScoreCountByRoot[static_cast<size_t>(mergedRoot)];
            edgeScoreSumByRoot[static_cast<size_t>(mergedRoot)] = 0.0;
            edgeScoreCountByRoot[static_cast<size_t>(mergedRoot)] = 0;
            auto& newMembers = membersByRoot[static_cast<std::size_t>(newRoot)];
            auto& oldMembers = membersByRoot[static_cast<std::size_t>(mergedRoot)];
            newMembers.insert(newMembers.end(), oldMembers.begin(), oldMembers.end());
            oldMembers.clear();
        }
        edgeScoreSumByRoot[static_cast<size_t>(newRoot)] +=
            std::max(0.0f, _edges[static_cast<size_t>(edgeIndex)].score);
        ++edgeScoreCountByRoot[static_cast<size_t>(newRoot)];

        auto& newFeatures = featureByImageByRoot[static_cast<size_t>(newRoot)];
        auto& oldFeatures = featureByImageByRoot[static_cast<size_t>(mergedRoot)];
        if (newRoot != mergedRoot)
        {
            newFeatures.insert(oldFeatures.begin(), oldFeatures.end());
            oldFeatures.clear();
        }
    }

    // 阶段 3：物化最终组件，并把被接受边的平均分作为 track confidence。
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

    // 对转台/固定传感器数据，可选删除在所有影像中像素位置近乎不动的伪特征。
    // 缺少任一关键点坐标时保守保留，避免把“数据不完整”误判为静止。
    pruneStationaryTracks(options, _keypointsByImage, &result);

    if (options.enableQualityThinning && !result.tracks.empty())
    {
        // 阶段 4：完整长轨迹优先，其次高置信度。只有轨迹在每个参与影像的总配额
        // 和网格配额都可容纳时才整体接受，禁止留下残缺子轨迹。
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
