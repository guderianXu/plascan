#include "lightglue/LightGlueFeatureBudget.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

namespace
{

xjw::image_matching::FeatureSet makeFeatureSet(int count,
                                                int width = 4000,
                                                int height = 3000)
{
    xjw::image_matching::FeatureSet data;
    data.imageWidth = width;
    data.imageHeight = height;
    data.keypoints.reserve(static_cast<std::size_t>(count));
    data.scores.reserve(static_cast<std::size_t>(count));
    data.descriptors = cv::Mat(count, 128, CV_32F);
    for (int index = 0; index < count; ++index)
    {
        cv::KeyPoint keypoint(static_cast<float>((index * 37) % width),
                              static_cast<float>((index * 53) % height),
                              1.0f);
        keypoint.response = static_cast<float>(index);
        data.keypoints.push_back(keypoint);
        data.scores.push_back(static_cast<float>(index));
        data.descriptors.row(index).setTo(static_cast<float>(index));
    }
    return data;
}

} // namespace

TEST(LightGlueFeatureBudgetTest, CapsConfiguredBudgetByGpuMemory)
{
    EXPECT_EQ(xjw::image_matching::resolveSiftLightGlueKeypointBudget(-1), 4096);
    EXPECT_EQ(xjw::image_matching::resolveSiftLightGlueKeypointBudget(2048), 2048);
    EXPECT_EQ(xjw::image_matching::resolveSiftLightGlueKeypointBudget(9999), 4096);

    xjw::image_matching::LightGlueGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = 7ull * 1024ull * 1024ull * 1024ull;
    memory.totalBytes = 8ull * 1024ull * 1024ull * 1024ull;
    EXPECT_EQ(xjw::image_matching::resolveSiftLightGlueKeypointBudget(-1, memory), 6144);
    EXPECT_EQ(xjw::image_matching::resolveSiftLightGlueKeypointBudget(8192, memory), 6144);
}

TEST(LightGlueFeatureBudgetTest, AdaptsThresholdWithinSafeBounds)
{
    xjw::image_matching::LightGlueGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = 7ull * 1024ull * 1024ull * 1024ull;
    memory.totalBytes = 8ull * 1024ull * 1024ull * 1024ull;

    EXPECT_FLOAT_EQ(xjw::image_matching::resolveSiftLightGlueMatchThreshold(
                        0.20f, 6144, memory),
                    0.12f);
    EXPECT_FLOAT_EQ(xjw::image_matching::resolveSiftLightGlueMatchThreshold(
                        0.05f, 6144, memory),
                    0.05f);
}

TEST(LightGlueFeatureBudgetTest, ClampsRuntimeBudgetToFixedEngineBucket)
{
    EXPECT_EQ(xjw::image_matching::clampLightGlueKeypointBudgetToEngine(12288, 4096),
              4096);
    EXPECT_EQ(xjw::image_matching::clampLightGlueKeypointBudgetToEngine(2048, 4096),
              2048);
    EXPECT_EQ(xjw::image_matching::clampLightGlueKeypointBudgetToEngine(0, 4096),
              4096);
    EXPECT_EQ(xjw::image_matching::clampLightGlueKeypointBudgetToEngine(8192, 0),
              8192);
}

TEST(LightGlueFeatureBudgetTest, SelectsFeaturesAndTracksOriginalIndices)
{
    const auto input = makeFeatureSet(10);
    const auto budgeted = xjw::image_matching::budgetFeatureDataForLightGlue(input, 4);

    ASSERT_TRUE(budgeted.limited);
    ASSERT_EQ(budgeted.features.size(), 4);
    ASSERT_EQ(budgeted.originalIndices.size(), 4u);
    EXPECT_EQ(budgeted.features.imageWidth, input.imageWidth);
    EXPECT_EQ(budgeted.features.imageHeight, input.imageHeight);
    EXPECT_EQ(budgeted.features.descriptorDimension(), input.descriptorDimension());
}

TEST(LightGlueFeatureBudgetTest, RemapsBudgetedMatchesToOriginalIndices)
{
    const auto input0 = makeFeatureSet(8);
    const auto input1 = makeFeatureSet(9);
    const auto budgeted0 = xjw::image_matching::budgetFeatureDataForLightGlue(input0, 3);
    const auto budgeted1 = xjw::image_matching::budgetFeatureDataForLightGlue(input1, 4);

    xjw::image_matching::MatchResult limited;
    limited.sourceAlgorithm = "sift_lightglue";
    limited.matches0.assign(3, -1);
    limited.matches1.assign(4, -1);
    limited.matchingScores0.assign(3, 0.0f);
    limited.matchingScores1.assign(4, 0.0f);
    limited.matches0[1] = 2;
    limited.matches1[2] = 1;
    limited.matchingScores0[1] = 0.75f;
    limited.matchingScores1[2] = 0.75f;
    limited.buildCvMatchesFromIndices();

    const auto remapped = xjw::image_matching::remapLightGlueMatchResultToOriginal(
        limited, budgeted0, input0.size(), budgeted1, input1.size());

    ASSERT_EQ(remapped.numMatches, 1);
    const int original0 = budgeted0.originalIndices.at(1);
    const int original1 = budgeted1.originalIndices.at(2);
    EXPECT_EQ(remapped.matches0.at(static_cast<std::size_t>(original0)), original1);
    EXPECT_EQ(remapped.matches1.at(static_cast<std::size_t>(original1)), original0);
}
