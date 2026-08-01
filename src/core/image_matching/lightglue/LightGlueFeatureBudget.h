#pragma once

#include "../FeatureSet.h"
#include "../MatchResult.h"

#include <QString>
#include <QVector>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <vector>

namespace xjw::image_matching
{

/// 特征限额后的内存结果，并保留回映到原始 SIFT 索引的表。
struct BudgetedFeatureData
{
    FeatureSet features;
    std::vector<int> originalIndices;
    bool limited = false;
};

/// CUDA 设备显存快照；不可查询时 available=false 并使用保守默认预算。
struct LightGlueGpuMemoryInfo
{
    bool available = false;
    std::uint64_t freeBytes = 0;
    std::uint64_t totalBytes = 0;
    int deviceIndex = -1;
};

/// 根据空闲显存选择不会轻易 OOM 的固定关键点桶上限。
inline int siftLightGlueCudaBudgetForMemory(const LightGlueGpuMemoryInfo &memory)
{
    if (!memory.available || memory.freeBytes == 0 || memory.totalBytes == 0)
    {
        return 4096;
    }

    const double gib = 1024.0 * 1024.0 * 1024.0;
    const double freeGiB = static_cast<double>(memory.freeBytes) / gib;
    if (freeGiB >= 18.0)
    {
        return 16384;
    }
    if (freeGiB >= 13.0)
    {
        return 12288;
    }
    if (freeGiB >= 9.0)
    {
        return 8192;
    }
    if (freeGiB >= 6.0)
    {
        return 6144;
    }
    if (freeGiB >= 4.0)
    {
        return 4096;
    }
    if (freeGiB >= 2.5)
    {
        return 3072;
    }
    return 2048;
}

/// 合并用户关键点上限和设备安全上限，二者取较小值。
inline int resolveSiftLightGlueKeypointBudget(
    int configuredMaxKeypoints,
    const LightGlueGpuMemoryInfo &gpuMemory)
{
    const int safeLimit = siftLightGlueCudaBudgetForMemory(gpuMemory);
    if (configuredMaxKeypoints > 0)
    {
        return std::min(configuredMaxKeypoints, safeLimit);
    }
    return safeLimit;
}

/// 无显存信息时使用保守默认值，主要供无 CUDA 上下文的配置解析使用。
inline int resolveSiftLightGlueKeypointBudget(int configuredMaxKeypoints)
{
    return resolveSiftLightGlueKeypointBudget(
        configuredMaxKeypoints,
        LightGlueGpuMemoryInfo{});
}

/**
 * @brief 把显存预算收敛到实际加载的固定 TensorRT 桶容量。
 *
 * LightGlue engine 的 `[1,K,*]` 维度在构建时已经固定。即使显存策略允许更多
 * 关键点，也不能向较小的 engine 传入超过 K 的特征，否则 TensorRT 后端会在
 * 第一个像对上失败。engineBucketKeypoints 未知时保留原预算，由加载后的 matcher
 * 再执行同样的保护。
 */
inline int clampLightGlueKeypointBudgetToEngine(int requestedBudget,
                                                int engineBucketKeypoints)
{
    if (engineBucketKeypoints <= 0)
    {
        return requestedBudget;
    }
    if (requestedBudget <= 0)
    {
        return engineBucketKeypoints;
    }
    return std::min(requestedBudget, engineBucketKeypoints);
}

/// 大桶适当降低置信度门限，避免高特征预算反而丢失弱纹理正确对应。
inline float resolveSiftLightGlueMatchThreshold(
    float configuredThreshold,
    int keypointBudget,
    const LightGlueGpuMemoryInfo &gpuMemory)
{
    Q_UNUSED(gpuMemory);
    if (configuredThreshold <= 0.0f || keypointBudget <= 0)
    {
        return configuredThreshold;
    }

    float adaptiveThreshold = configuredThreshold;
    if (keypointBudget >= 12288)
    {
        adaptiveThreshold = std::min(adaptiveThreshold, 0.08f);
    }
    else if (keypointBudget >= 8192)
    {
        adaptiveThreshold = std::min(adaptiveThreshold, 0.10f);
    }
    else if (keypointBudget >= 6144)
    {
        adaptiveThreshold = std::min(adaptiveThreshold, 0.12f);
    }
    else if (keypointBudget >= 4096)
    {
        adaptiveThreshold = std::min(adaptiveThreshold, 0.15f);
    }

    return std::clamp(adaptiveThreshold, 0.01f, 1.0f);
}

/// 生成 OOM 重试桶序列，每次减半且最低保持 1024 个关键点。
inline QVector<int> lightGlueRetryKeypointBudgets(int primaryBudget)
{
    if (primaryBudget <= 0)
    {
        return {primaryBudget};
    }

    QVector<int> budgets;
    int budget = primaryBudget;
    while (budget >= 1024)
    {
        budgets.append(budget);
        if (budget == 1024)
        {
            break;
        }
        budget = std::max(1024, budget / 2);
    }
    if (budgets.isEmpty())
    {
        budgets.append(primaryBudget);
    }
    return budgets;
}

/// 读取统一检测响应；缺失或非有限值时退回到 KeyPoint::response。
inline float featureScoreAt(const xjw::image_matching::FeatureSet &input, int index)
{
    if (index >= 0 && index < static_cast<int>(input.scores.size()))
    {
        const float score = input.scores[static_cast<std::size_t>(index)];
        if (std::isfinite(score))
        {
            return score;
        }
    }
    if (index >= 0 && index < input.size())
    {
        const float score = input.keypoints[static_cast<std::size_t>(index)].response;
        if (std::isfinite(score))
        {
            return score;
        }
    }
    return 0.0f;
}

/// 获取特征坐标所属影像尺寸；旧内存对象缺尺寸时从坐标范围推断。
inline std::pair<float, float> inferFeatureImageSize(const xjw::image_matching::FeatureSet &input)
{
    if (input.imageWidth > 0 && input.imageHeight > 0)
    {
        return {static_cast<float>(input.imageWidth), static_cast<float>(input.imageHeight)};
    }

    float maxX = 1.0f;
    float maxY = 1.0f;
    for (const cv::KeyPoint &keypoint : input.keypoints)
    {
        maxX = std::max(maxX, keypoint.pt.x);
        maxY = std::max(maxY, keypoint.pt.y);
    }
    return {maxX + 1.0f, maxY + 1.0f};
}

/**
 * @brief 选择进入固定 LightGlue 桶的原始特征索引。
 *
 * 第一轮按规则网格保留每格最高响应点，第二轮再按全局响应补满，避免只保留
 * 高纹理边缘而损失物体内部覆盖。stable_sort 和索引决胜保证结果可复现。
 */
inline std::vector<int> selectLightGlueBudgetIndices(const xjw::image_matching::FeatureSet &input,
                                                     int maxKeypoints)
{
    const int count = input.size();
    if (maxKeypoints <= 0 || count <= maxKeypoints)
    {
        std::vector<int> indices(static_cast<std::size_t>(count));
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }

    std::vector<int> ranked(static_cast<std::size_t>(count));
    std::iota(ranked.begin(), ranked.end(), 0);
    std::stable_sort(ranked.begin(), ranked.end(), [&](int lhs, int rhs)
    {
        const float leftScore = featureScoreAt(input, lhs);
        const float rightScore = featureScoreAt(input, rhs);
        if (leftScore == rightScore)
        {
            return lhs < rhs;
        }
        return leftScore > rightScore;
    });

    const auto [width, height] = inferFeatureImageSize(input);
    const double aspect = std::max(0.25, std::min(4.0, static_cast<double>(width) / std::max(1.0f, height)));
    const int gridCols = std::max(1, static_cast<int>(std::round(std::sqrt(maxKeypoints * aspect))));
    const int gridRows = std::max(1, static_cast<int>(std::ceil(static_cast<double>(maxKeypoints) / gridCols)));
    std::vector<int> cellOwner(static_cast<std::size_t>(gridCols * gridRows), -1);
    std::vector<char> selected(static_cast<std::size_t>(count), 0);
    std::vector<int> indices;
    indices.reserve(static_cast<std::size_t>(maxKeypoints));

    auto cellIndexFor = [&](const cv::KeyPoint &keypoint) -> int
    {
        const int col = std::clamp(static_cast<int>(std::floor(keypoint.pt.x / std::max(1.0f, width) * gridCols)),
                                   0,
                                   gridCols - 1);
        const int row = std::clamp(static_cast<int>(std::floor(keypoint.pt.y / std::max(1.0f, height) * gridRows)),
                                   0,
                                   gridRows - 1);
        return row * gridCols + col;
    };

    for (const int index : ranked)
    {
        const int cell = cellIndexFor(input.keypoints[static_cast<std::size_t>(index)]);
        if (cellOwner[static_cast<std::size_t>(cell)] >= 0)
        {
            continue;
        }
        cellOwner[static_cast<std::size_t>(cell)] = index;
        selected[static_cast<std::size_t>(index)] = 1;
        indices.push_back(index);
        if (static_cast<int>(indices.size()) >= maxKeypoints)
        {
            break;
        }
    }

    if (static_cast<int>(indices.size()) < maxKeypoints)
    {
        for (const int index : ranked)
        {
            if (selected[static_cast<std::size_t>(index)])
            {
                continue;
            }
            selected[static_cast<std::size_t>(index)] = 1;
            indices.push_back(index);
            if (static_cast<int>(indices.size()) >= maxKeypoints)
            {
                break;
            }
        }
    }

    return indices;
}

/// 按选中索引复制关键点、分数和描述子，并返回后处理所需原始索引映射。
inline BudgetedFeatureData budgetFeatureDataForLightGlue(
    const xjw::image_matching::FeatureSet &input,
    int maxKeypoints)
{
    BudgetedFeatureData result;
    result.originalIndices = selectLightGlueBudgetIndices(input, maxKeypoints);
    result.limited = static_cast<int>(result.originalIndices.size()) < input.size();
    if (!result.limited)
    {
        result.features = input;
        return result;
    }

    result.features.sourceAlgorithm = input.sourceAlgorithm;
    result.features.imageWidth = input.imageWidth;
    result.features.imageHeight = input.imageHeight;
    result.features.keypoints.reserve(result.originalIndices.size());
    result.features.scores.reserve(result.originalIndices.size());

    for (const int originalIndex : result.originalIndices)
    {
        result.features.keypoints.push_back(input.keypoints[static_cast<std::size_t>(originalIndex)]);
        result.features.scores.push_back(featureScoreAt(input, originalIndex));
    }

    if (!input.descriptors.empty() && input.descriptors.rows >= input.size())
    {
        result.features.descriptors =
            cv::Mat(static_cast<int>(result.originalIndices.size()),
                    input.descriptors.cols,
                    input.descriptors.type());
        for (int row = 0; row < static_cast<int>(result.originalIndices.size()); ++row)
        {
            const int originalIndex = result.originalIndices[static_cast<std::size_t>(row)];
            input.descriptors.row(originalIndex).copyTo(result.features.descriptors.row(row));
        }
    }

    return result;
}

inline xjw::image_matching::MatchResult remapLightGlueMatchResultToOriginal(
    const xjw::image_matching::MatchResult &limited,
    const BudgetedFeatureData &features0,
    int originalCount0,
    const BudgetedFeatureData &features1,
    int originalCount1)
{
    xjw::image_matching::MatchResult remapped;
    remapped.sourceAlgorithm = limited.sourceAlgorithm;
    remapped.matches0.assign(originalCount0, -1);
    remapped.matches1.assign(originalCount1, -1);
    remapped.matchingScores0.assign(originalCount0, 0.0f);
    remapped.matchingScores1.assign(originalCount1, 0.0f);

    const int limitedCount0 = std::min(static_cast<int>(limited.matches0.size()),
                                       static_cast<int>(features0.originalIndices.size()));
    const int limitedCount1 = static_cast<int>(features1.originalIndices.size());
    for (int index0 = 0; index0 < limitedCount0; ++index0)
    {
        const int index1 = limited.matches0[static_cast<std::size_t>(index0)];
        if (index1 < 0 || index1 >= limitedCount1)
        {
            continue;
        }

        const int original0 = features0.originalIndices[static_cast<std::size_t>(index0)];
        const int original1 = features1.originalIndices[static_cast<std::size_t>(index1)];
        if (original0 < 0 || original0 >= originalCount0 || original1 < 0 || original1 >= originalCount1)
        {
            continue;
        }

        const float score = index0 < static_cast<int>(limited.matchingScores0.size())
            ? limited.matchingScores0[static_cast<std::size_t>(index0)]
            : 1.0f;
        remapped.matches0[static_cast<std::size_t>(original0)] = original1;
        remapped.matches1[static_cast<std::size_t>(original1)] = original0;
        remapped.matchingScores0[static_cast<std::size_t>(original0)] = score;
        remapped.matchingScores1[static_cast<std::size_t>(original1)] = score;
    }

    remapped.buildCvMatchesFromIndices();
    return remapped;
}

} // namespace xjw::image_matching
