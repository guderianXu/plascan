#include "cuda_sift/CudaSiftMatchFilter.h"

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

} // namespace
} // namespace xjw::image_matching
