#include "CorrespondenceGraph.h"

#include <algorithm>

namespace xjw
{

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
    storedMatches.insert(storedMatches.end(), matches.begin(), matches.end());
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

} // namespace xjw
