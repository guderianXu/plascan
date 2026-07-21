#include <gtest/gtest.h>

#include "lightglue/LightGlueFeatureBudget.h"

#include "feature_match/match.h"

#include <opencv2/core.hpp>

namespace
{

xjw::feature_extractors::FeatureData makeFeatureData(int count, int width = 4000, int height = 3000)
{
    xjw::feature_extractors::FeatureData data;
    data.sourceAlgorithm = "sift";
    data.imageWidth = width;
    data.imageHeight = height;
    data.keypoints.reserve(static_cast<std::size_t>(count));
    data.scores.reserve(static_cast<std::size_t>(count));
    data.descriptors = cv::Mat(count, 128, CV_32F);

    for (int i = 0; i < count; ++i)
    {
        const float x = static_cast<float>((i * 37) % width);
        const float y = static_cast<float>((i * 53) % height);
        cv::KeyPoint keypoint(x, y, 1.0f);
        keypoint.response = static_cast<float>(i);
        data.keypoints.push_back(keypoint);
        data.scores.push_back(static_cast<float>(i));
        data.descriptors.row(i).setTo(static_cast<float>(i));
    }

    return data;
}

} // namespace

TEST(LightGlueFeatureBudgetTest, CapsSiftLightGlueOnCudaByDefault)
{
    EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
                  QStringLiteral("sift"), QStringLiteral("lightglue"), true, -1),
              4096);
    EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
                  QStringLiteral("sift"), QStringLiteral("lightglue"), true, 2048),
              2048);
    EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
                  QStringLiteral("sift"), QStringLiteral("lightglue"), true, 9999),
              4096);
    EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
                  QStringLiteral("disk"), QStringLiteral("lightglue"), true, -1),
              -1);
}

TEST(LightGlueFeatureBudgetTest, RaisesSiftLightGlueBudgetWhenGpuMemoryAllows)
{
    xjw::feature_match::LightGlueGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = 7ull * 1024ull * 1024ull * 1024ull;
    memory.totalBytes = 8ull * 1024ull * 1024ull * 1024ull;

    EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
                  QStringLiteral("sift"), QStringLiteral("lightglue"), true, -1, memory),
              6144);
    EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
                  QStringLiteral("sift"), QStringLiteral("lightglue"), true, 8192, memory),
              6144);
    EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
                  QStringLiteral("sift"), QStringLiteral("lightglue"), true, 2048, memory),
              2048);
}

TEST(LightGlueFeatureBudgetTest, LowGpuMemoryKeepsSiftLightGlueConservative)
{
    xjw::feature_match::LightGlueGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = 3ull * 1024ull * 1024ull * 1024ull;
    memory.totalBytes = 8ull * 1024ull * 1024ull * 1024ull;

    EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
                  QStringLiteral("sift"), QStringLiteral("lightglue"), true, -1, memory),
              3072);
}

TEST(LightGlueFeatureBudgetTest, LargeGpuMemoryUsesLargerSiftLightGlueBudget)
{
    xjw::feature_match::LightGlueGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = 14ull * 1024ull * 1024ull * 1024ull;
    memory.totalBytes = 16ull * 1024ull * 1024ull * 1024ull;

    EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
                  QStringLiteral("sift"), QStringLiteral("lightglue"), true, -1, memory),
              12288);
}

TEST(LightGlueFeatureBudgetTest, RelaxesSiftLightGlueThresholdOnlyWithinSafeBounds)
{
    xjw::feature_match::LightGlueGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = 7ull * 1024ull * 1024ull * 1024ull;
    memory.totalBytes = 8ull * 1024ull * 1024ull * 1024ull;

    EXPECT_FLOAT_EQ(xjw::feature_match::resolveLightGlueMatchThreshold(
                        QStringLiteral("sift"), QStringLiteral("lightglue"), true, 0.20f, 6144, memory),
                    0.12f);
    EXPECT_FLOAT_EQ(xjw::feature_match::resolveLightGlueMatchThreshold(
                        QStringLiteral("sift"), QStringLiteral("lightglue"), true, 0.05f, 6144, memory),
                    0.05f);
    EXPECT_FLOAT_EQ(xjw::feature_match::resolveLightGlueMatchThreshold(
                        QStringLiteral("disk"), QStringLiteral("lightglue"), true, 0.20f, 6144, memory),
                    0.20f);
}

TEST(LightGlueFeatureBudgetTest, RetryBudgetsUseMemoryAwareBudgetAsStartingPoint)
{
    const QVector<int> budgets = xjw::feature_match::lightGlueRetryKeypointBudgets(12288);

    ASSERT_GE(budgets.size(), 4);
    EXPECT_EQ(budgets.at(0), 12288);
    EXPECT_EQ(budgets.at(1), 6144);
    EXPECT_EQ(budgets.at(2), 3072);
    EXPECT_EQ(budgets.at(3), 1536);
    EXPECT_EQ(budgets.back(), 1024);
}

TEST(LightGlueFeatureBudgetTest, SelectsBudgetedFeaturesAndTracksOriginalIndices)
{
    const xjw::feature_extractors::FeatureData input = makeFeatureData(10);
    const xjw::feature_match::BudgetedFeatureData budgeted =
        xjw::feature_match::budgetFeatureDataForLightGlue(input, 4);

    ASSERT_TRUE(budgeted.limited);
    ASSERT_EQ(budgeted.features.size(), 4);
    ASSERT_EQ(budgeted.originalIndices.size(), 4);

    for (const int originalIndex : budgeted.originalIndices)
    {
        ASSERT_GE(originalIndex, 0);
        ASSERT_LT(originalIndex, input.size());
    }

    EXPECT_EQ(budgeted.features.imageWidth, input.imageWidth);
    EXPECT_EQ(budgeted.features.imageHeight, input.imageHeight);
    EXPECT_EQ(budgeted.features.descriptorDim(), input.descriptorDim());
}

TEST(LightGlueFeatureBudgetTest, RemapsBudgetedMatchResultBackToOriginalKeypointIndices)
{
    const xjw::feature_extractors::FeatureData input0 = makeFeatureData(8);
    const xjw::feature_extractors::FeatureData input1 = makeFeatureData(9);
    xjw::feature_match::BudgetedFeatureData budgeted0 =
        xjw::feature_match::budgetFeatureDataForLightGlue(input0, 3);
    xjw::feature_match::BudgetedFeatureData budgeted1 =
        xjw::feature_match::budgetFeatureDataForLightGlue(input1, 4);

    xjw::feature_match::MatchResult limited;
    limited.sourceAlgorithm = "lightglue";
    limited.matches0.assign(3, -1);
    limited.matches1.assign(4, -1);
    limited.matchingScores0.assign(3, 0.0f);
    limited.matchingScores1.assign(4, 0.0f);
    limited.matches0[1] = 2;
    limited.matches1[2] = 1;
    limited.matchingScores0[1] = 0.75f;
    limited.matchingScores1[2] = 0.75f;
    limited.buildCvMatchesFromIndices();

    const xjw::feature_match::MatchResult remapped =
        xjw::feature_match::remapLightGlueMatchResultToOriginal(
            limited,
            budgeted0,
            input0.size(),
            budgeted1,
            input1.size());

    ASSERT_EQ(remapped.matches0.size(), static_cast<std::size_t>(input0.size()));
    ASSERT_EQ(remapped.matches1.size(), static_cast<std::size_t>(input1.size()));
    ASSERT_EQ(remapped.numMatches, 1);

    const int originalQuery = budgeted0.originalIndices.at(1);
    const int originalTrain = budgeted1.originalIndices.at(2);
    EXPECT_EQ(remapped.matches0.at(static_cast<std::size_t>(originalQuery)), originalTrain);
    EXPECT_EQ(remapped.matches1.at(static_cast<std::size_t>(originalTrain)), originalQuery);
    ASSERT_EQ(remapped.cvMatches.size(), 1);
    EXPECT_EQ(remapped.cvMatches.front().queryIdx, originalQuery);
    EXPECT_EQ(remapped.cvMatches.front().trainIdx, originalTrain);
}
