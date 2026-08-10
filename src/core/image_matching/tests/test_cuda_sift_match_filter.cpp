#include "cuda_sift/CudaSiftMatchFilter.h"
#include "cuda_sift/CudaSiftAlgorithm.h"

#include <gtest/gtest.h>

#include <limits>

namespace xjw::image_matching
{
namespace
{

TEST(CudaSiftMatchFilterTest, KeepsOnlyMutualMatchesAboveConfidenceThreshold)
{
    const std::vector<CudaSiftNearestMatch> forward = {
        {1, 0.90f, 0.60f},
        {0, 0.95f, 0.90f},
        {2, 0.80f, 0.70f}};
    const std::vector<CudaSiftNearestMatch> reverse = {
        {2, 0.90f, 0.60f},
        {0, 0.85f, 0.50f},
        {2, 0.90f, 0.75f}};

    const MatchResult result = filterCudaSiftMutualMatches(
        forward, reverse, 0.15f);

    ASSERT_EQ(result.matches0.size(), 3U);
    EXPECT_EQ(result.matches0[0], 1);
    EXPECT_EQ(result.matches0[1], -1);
    EXPECT_EQ(result.matches0[2], 2);
    ASSERT_EQ(result.cvMatches.size(), 2U);
    EXPECT_EQ(result.sourceAlgorithm, "cuda_sift");
}

TEST(CudaSiftMatchFilterTest, RejectsNonFiniteAndLowSimilarityMatches)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<CudaSiftNearestMatch> forward = {
        {0, nan, 0.1f},
        {1, 0.10f, 0.95f}};
    const std::vector<CudaSiftNearestMatch> reverse = {
        {0, 0.90f, 0.1f},
        {1, 0.10f, 0.95f}};

    const MatchResult result = filterCudaSiftMutualMatches(
        forward, reverse, 0.15f);

    EXPECT_TRUE(result.cvMatches.empty());
    EXPECT_EQ(result.numMatches, 0);
}

TEST(CudaSiftMatchFilterTest, KeepsAmbiguousMutualCandidatesForGeometryVerification)
{
    const std::vector<CudaSiftNearestMatch> forward = {{0, 0.90f, 0.97f}};
    const std::vector<CudaSiftNearestMatch> reverse = {{0, 0.88f, 0.98f}};

    const MatchResult result = filterCudaSiftMutualMatches(
        forward, reverse, 0.15f);

    ASSERT_EQ(result.cvMatches.size(), 1U);
    EXPECT_FLOAT_EQ(result.matchingScores0[0], 0.88f);
}

TEST(CudaSiftMatchFilterTest, CpuBackendMatchesNormalizedSiftDescriptors)
{
    FeatureSet features0;
    features0.imageWidth = 100;
    features0.imageHeight = 100;
    features0.keypoints = {cv::KeyPoint(10.0f, 10.0f, 1.0f),
                           cv::KeyPoint(20.0f, 20.0f, 1.0f)};
    features0.scores = {1.0f, 1.0f};
    features0.descriptors = cv::Mat::zeros(2, 128, CV_32F);
    features0.descriptors.at<float>(0, 0) = 512.0f;
    features0.descriptors.at<float>(1, 1) = 256.0f;

    FeatureSet features1;
    features1.imageWidth = 100;
    features1.imageHeight = 100;
    features1.keypoints = features0.keypoints;
    features1.scores = features0.scores;
    features1.descriptors = cv::Mat::zeros(2, 128, CV_32F);
    features1.descriptors.at<float>(0, 1) = 1.0f;
    features1.descriptors.at<float>(1, 0) = 1.0f;

    ImageMatchingRuntimeConfig config;
    config.forceCpuSift = true;
    config.allowCpuSiftFallback = true;
    config.matchThreshold = 0.15f;
    CudaSiftAlgorithm algorithm(config);

    const MatchResult result = algorithm.matchFeatures(features0, features1);

    ASSERT_EQ(result.cvMatches.size(), 2U);
    EXPECT_EQ(result.matches0[0], 1);
    EXPECT_EQ(result.matches0[1], 0);
}

} // namespace
} // namespace xjw::image_matching
