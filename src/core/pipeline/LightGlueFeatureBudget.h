#pragma once

#include "feature_extractors/FeatureData.h"
#include "feature_match/match.h"

#include <QString>
#include <QVector>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <vector>

namespace xjw::pipeline
{

struct BudgetedFeatureData
{
    xjw::feature_extractors::FeatureData features;
    std::vector<int> originalIndices;
    bool limited = false;
};

struct LightGlueGpuMemoryInfo
{
    bool available = false;
    std::uint64_t freeBytes = 0;
    std::uint64_t totalBytes = 0;
    int deviceIndex = -1;
};

inline bool isSiftLightGlueCuda(const QString &featureAlgorithm,
                                const QString &matchAlgorithm,
                                bool useCuda)
{
    return useCuda
        && matchAlgorithm.trimmed().toLower() == QStringLiteral("lightglue")
        && featureAlgorithm.trimmed().toLower() == QStringLiteral("sift");
}

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

inline int resolveLightGlueKeypointBudget(const QString &featureAlgorithm,
                                          const QString &matchAlgorithm,
                                          bool useCuda,
                                          int configuredMaxKeypoints,
                                          const LightGlueGpuMemoryInfo &gpuMemory)
{
    if (matchAlgorithm.trimmed().toLower() != QStringLiteral("lightglue"))
    {
        return configuredMaxKeypoints;
    }

    const QString feature = featureAlgorithm.trimmed().toLower();
    if (feature != QStringLiteral("sift"))
    {
        return configuredMaxKeypoints;
    }

    const int safeLimit = useCuda ? siftLightGlueCudaBudgetForMemory(gpuMemory) : 8192;
    if (configuredMaxKeypoints > 0)
    {
        return std::min(configuredMaxKeypoints, safeLimit);
    }
    return safeLimit;
}

inline int resolveLightGlueKeypointBudget(const QString &featureAlgorithm,
                                          const QString &matchAlgorithm,
                                          bool useCuda,
                                          int configuredMaxKeypoints)
{
    return resolveLightGlueKeypointBudget(
        featureAlgorithm,
        matchAlgorithm,
        useCuda,
        configuredMaxKeypoints,
        LightGlueGpuMemoryInfo{});
}

inline float resolveLightGlueMatchThreshold(const QString &featureAlgorithm,
                                            const QString &matchAlgorithm,
                                            bool useCuda,
                                            float configuredThreshold,
                                            int keypointBudget,
                                            const LightGlueGpuMemoryInfo &gpuMemory)
{
    Q_UNUSED(gpuMemory);
    if (!isSiftLightGlueCuda(featureAlgorithm, matchAlgorithm, useCuda)
        || configuredThreshold <= 0.0f
        || keypointBudget <= 0)
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

inline float featureScoreAt(const xjw::feature_extractors::FeatureData &input, int index)
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

inline std::pair<float, float> inferFeatureImageSize(const xjw::feature_extractors::FeatureData &input)
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

inline std::vector<int> selectLightGlueBudgetIndices(const xjw::feature_extractors::FeatureData &input,
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

inline BudgetedFeatureData budgetFeatureDataForLightGlue(
    const xjw::feature_extractors::FeatureData &input,
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

inline xjw::feature_match::MatchResult remapLightGlueMatchResultToOriginal(
    const xjw::feature_match::MatchResult &limited,
    const BudgetedFeatureData &features0,
    int originalCount0,
    const BudgetedFeatureData &features1,
    int originalCount1)
{
    xjw::feature_match::MatchResult remapped;
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

} // namespace xjw::pipeline
