#include "cuda_sift/CudaSiftMatchFilter.h"
#include "cuda_sift/CudaSiftAlgorithm.h"
#include "sift/SiftFeatureExtractor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace xjw::image_matching
{
namespace
{

FeatureSet makeCudaSiftFeatures(int count)
{
    FeatureSet features;
    features.imageWidth = 1024;
    features.imageHeight = 1024;
    features.sourceAlgorithm = "sift";
    features.keypoints.reserve(static_cast<std::size_t>(count));
    features.scores.assign(static_cast<std::size_t>(count), 1.0f);
    features.descriptors = cv::Mat::zeros(count, 128, CV_32F);
    for (int index = 0; index < count; ++index)
    {
        features.keypoints.emplace_back(static_cast<float>(index % 1024),
                                        static_cast<float>(index / 1024),
                                        1.0f);
        float *descriptor = features.descriptors.ptr<float>(index);
        std::uint32_t state = static_cast<std::uint32_t>(index + 1);
        float squaredNorm = 0.0f;
        for (int dimension = 0; dimension < 128; ++dimension)
        {
            state = state * 1664525U + 1013904223U;
            descriptor[dimension] = static_cast<float>((state >> 8U) & 0xffffU) / 65535.0f;
            squaredNorm += descriptor[dimension] * descriptor[dimension];
        }
        const float inverseNorm = 1.0f / std::sqrt(squaredNorm);
        for (int dimension = 0; dimension < 128; ++dimension)
        {
            descriptor[dimension] *= inverseNorm;
        }
    }
    return features;
}

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

TEST(CudaSiftMatchFilterTest, CudaBackendHandlesPartialQueryAndCandidateTiles)
{
    if (!SiftFeatureExtractor::isCudaAvailable(0))
    {
        GTEST_SKIP() << "CUDA SIFT device is unavailable";
    }

    ImageMatchingRuntimeConfig config;
    config.cudaDevice = 0;
    config.forceCpuSift = false;
    config.allowCpuSiftFallback = false;
    config.matchThreshold = 0.15f;
    CudaSiftAlgorithm algorithm(config);

    const MatchResult result = algorithm.matchFeatures(
        makeCudaSiftFeatures(607), makeCudaSiftFeatures(531));

    EXPECT_EQ(result.matches0.size(), 607U);
    EXPECT_EQ(result.matchingScores0.size(), 607U);
    EXPECT_EQ(result.numMatches, 531);
}

} // namespace
} // namespace xjw::image_matching
