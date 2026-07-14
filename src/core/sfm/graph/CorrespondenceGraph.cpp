#include "CorrespondenceGraph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace xjw
{

namespace
{

std::uint64_t observationKey(ImageId imageId, FeatureIdx featureIdx)
{
    return (static_cast<std::uint64_t>(imageId) << 32)
        | static_cast<std::uint64_t>(featureIdx);
}

} // namespace

// 静态空匹配列表
const std::vector<FeatureMatch> CorrespondenceGraph::EMPTY_MATCHES;

// ---- 构建阶段 ----

void CorrespondenceGraph::addImage(ImageId imageId, size_t numFeatures)
{
    imageFeatureCounts[imageId] = numFeatures;
}

/**
 * @brief 添加图像特征计数并为后续匹配构建索引做准备。
 *
 * 该函数仅记录每幅图像的特征数量，真实对应关系在
 * buildCorrespondences() 中构建。
 */

void CorrespondenceGraph::addMatches(
    ImageId id1, ImageId id2,
    const std::vector<FeatureMatch> &matches)
{
    if (matches.empty())
    {
        return;
    }

    ImagePair pair(id1, id2);
    auto &storedMatches = pairMatches[pair];
    if (id1 <= id2)
    {
        storedMatches.insert(storedMatches.end(), matches.begin(), matches.end());
        return;
    }

    // ImagePair 始终按图像 ID 升序存储，反向输入时必须同步交换特征索引。
    storedMatches.reserve(storedMatches.size() + matches.size());
    for (const FeatureMatch &match : matches)
    {
        storedMatches.push_back({match.idx2, match.idx1, match.score});
    }
}

bool CorrespondenceGraph::addPriorTrack(const std::string &sourceId,
                                        const std::vector<TrackElement> &observations,
                                        float confidence)
{
    if (sourceId.empty() || observations.size() < 2 || !std::isfinite(confidence))
    {
        return false;
    }

    std::unordered_set<ImageId> image_ids;
    for (const TrackElement &observation : observations)
    {
        const auto image_it = imageFeatureCounts.find(observation.imageId);
        const std::uint64_t key = observationKey(observation.imageId, observation.featureIdx);
        if (image_it == imageFeatureCounts.end() || observation.featureIdx >= image_it->second
            || !image_ids.insert(observation.imageId).second
            || priorTrackByObservation.find(key) != priorTrackByObservation.end())
        {
            return false;
        }
    }

    for (const TrackElement &observation : observations)
    {
        priorTrackByObservation.emplace(observationKey(observation.imageId, observation.featureIdx),
                                        sourceId);
    }
    for (std::size_t first = 0; first < observations.size(); ++first)
    {
        for (std::size_t second = first + 1; second < observations.size(); ++second)
        {
            addMatches(observations[first].imageId,
                       observations[second].imageId,
                       {{observations[first].featureIdx,
                         observations[second].featureIdx,
                         confidence}});
        }
    }
    return true;
}

/**
 * @brief 构建特征对应关系索引。
 *
 * 根据已注册的图像和两两匹配，生成每张图像每个特征点对应到其它图像的查询表。
 * 该函数应在所有 `addImage()` 与 `addMatches()` 调用结束后执行一次。
 */
void CorrespondenceGraph::buildCorrespondences()
{
    // 为每幅图像初始化对应关系列表
    correspondences.clear();
    for (auto &[imageId, numFeatures] : imageFeatureCounts)
    {
        correspondences[imageId].resize(numFeatures);
    }

    // 遍历所有匹配，双向建立对应关系
    for (auto &[pair, matches] : pairMatches)
    {
        for (const auto &match : matches)
        {
            // pair.first → pair.second
            if (correspondences.count(pair.first))
            {
                auto &firstImageCorrespondences = correspondences[pair.first];
                if (match.idx1 < firstImageCorrespondences.size())
                {
                    firstImageCorrespondences[match.idx1].push_back({pair.second, match.idx2});
                }
            }

            // pair.second → pair.first
            if (correspondences.count(pair.second))
            {
                auto &secondImageCorrespondences = correspondences[pair.second];
                if (match.idx2 < secondImageCorrespondences.size())
                {
                    secondImageCorrespondences[match.idx2].push_back({pair.first, match.idx1});
                }
            }
        }
    }
}

std::vector<ImagePair> CorrespondenceGraph::imagePairs() const
{
    std::vector<ImagePair> result;
    result.reserve(pairMatches.size());
    for (const auto &entry : pairMatches)
    {
        if (!entry.second.empty())
        {
            result.push_back(entry.first);
        }
    }
    std::sort(result.begin(), result.end(), [](const ImagePair &left, const ImagePair &right)
    {
        if (left.first != right.first)
        {
            return left.first < right.first;
        }
        return left.second < right.second;
    });
    return result;
}

std::size_t CorrespondenceGraph::retainMatchesInTracks(const std::vector<Track> &tracks)
{
    using Observation = std::pair<ImageId, FeatureIdx>;
    constexpr std::size_t kAmbiguousTrack = std::numeric_limits<std::size_t>::max();

    std::map<Observation, std::size_t> trackByObservation;
    for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
    {
        for (const TrackElement &element : tracks[trackIndex].elements)
        {
            const Observation observation{element.imageId, element.featureIdx};
            const auto [it, inserted] = trackByObservation.emplace(observation, trackIndex);
            if (!inserted && it->second != trackIndex)
            {
                it->second = kAmbiguousTrack;
            }
        }
    }

    std::size_t removedCount = 0;
    for (auto pairIt = pairMatches.begin(); pairIt != pairMatches.end();)
    {
        const ImagePair &pair = pairIt->first;
        std::vector<FeatureMatch> &matches = pairIt->second;
        const auto newEnd = std::remove_if(matches.begin(), matches.end(), [&](const FeatureMatch &match)
        {
            const auto firstIt = trackByObservation.find({pair.first, match.idx1});
            const auto secondIt = trackByObservation.find({pair.second, match.idx2});
            const bool keep = firstIt != trackByObservation.end() &&
                secondIt != trackByObservation.end() &&
                firstIt->second != kAmbiguousTrack &&
                firstIt->second == secondIt->second;
            if (!keep)
            {
                ++removedCount;
            }
            return !keep;
        });
        matches.erase(newEnd, matches.end());

        if (matches.empty())
        {
            pairIt = pairMatches.erase(pairIt);
        }
        else
        {
            ++pairIt;
        }
    }

    correspondences.clear();
    return removedCount;
}

// ---- 查询接口 ----

size_t CorrespondenceGraph::numMatchesBetween(ImageId id1, ImageId id2) const
{
    ImagePair pair(id1, id2);
    auto it = pairMatches.find(pair);
    return (it != pairMatches.end()) ? it->second.size() : 0;
}

const std::vector<FeatureMatch> &CorrespondenceGraph::matchesBetween(
    ImageId id1, ImageId id2) const
{
    ImagePair pair(id1, id2);
    auto it = pairMatches.find(pair);
    return (it != pairMatches.end()) ? it->second : EMPTY_MATCHES;
}

std::vector<CorrespondenceGraph::Correspondence>
CorrespondenceGraph::findCorrespondences(
    ImageId imageId, FeatureIdx featureIdx) const
{
    auto imageIt = correspondences.find(imageId);
    if (imageIt == correspondences.end())
    {
        return {};
    }
    if (featureIdx >= imageIt->second.size())
    {
        return {};
    }
    return imageIt->second[featureIdx];
}

std::vector<ImageId> CorrespondenceGraph::connectedImages(
    ImageId imageId) const
{
    std::unordered_set<ImageId> neighborImageIds;
    for (auto &[pair, matches] : pairMatches)
    {
        if (matches.empty())
        {
            continue;
        }
        if (pair.first == imageId)
        {
            neighborImageIds.insert(pair.second);
        }
        if (pair.second == imageId)
        {
            neighborImageIds.insert(pair.first);
        }
    }
    return {neighborImageIds.begin(), neighborImageIds.end()};
}

std::vector<std::pair<ImageId, size_t>>
CorrespondenceGraph::topConnectedImages(
    ImageId imageId, size_t topN) const
{
    std::vector<std::pair<ImageId, size_t>> result;
    for (auto &[pair, matches] : pairMatches)
    {
        if (matches.empty())
        {
            continue;
        }
        if (pair.first == imageId)
        {
            result.emplace_back(pair.second, matches.size());
        }
        else if (pair.second == imageId)
        {
            result.emplace_back(pair.first, matches.size());
        }
    }

    // 按匹配数降序排列
    std::sort(
        result.begin(),
        result.end(),
        [](const auto &leftItem, const auto &rightItem)
        {
            return leftItem.second > rightItem.second;
        });

    if (result.size() > topN)
    {
        result.resize(topN);
    }

    return result;
}

std::string CorrespondenceGraph::priorTrackId(ImageId imageId, FeatureIdx featureIdx) const
{
    const auto it = priorTrackByObservation.find(observationKey(imageId, featureIdx));
    return it == priorTrackByObservation.end() ? std::string() : it->second;
}

} // namespace xjw
