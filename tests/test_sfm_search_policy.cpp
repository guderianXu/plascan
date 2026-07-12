#include <gtest/gtest.h>

#include "SfmSearchPolicy.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace
{

using xjw::sfm_search::SfmCandidateSummary;

TEST(SfmSearchPolicyTest, AllocatesFourWorkersAcrossThirtyTwoThreads)
{
    const auto budget = xjw::sfm_search::allocateWorkers(6, 32);

    EXPECT_EQ(budget.workerCount, 4);
    EXPECT_EQ(budget.threadsPerWorker, 8);
}

TEST(SfmSearchPolicyTest, KeepsSmallThreadBudgetsSerial)
{
    const auto budget = xjw::sfm_search::allocateWorkers(6, 7);

    EXPECT_EQ(budget.workerCount, 1);
    EXPECT_EQ(budget.threadsPerWorker, 7);
}

TEST(SfmSearchPolicyTest, RegistrationCoverageDominatesRmsAndPointCount)
{
    const SfmCandidateSummary full{
        0, 3.2, 1, 2, 16, 4021, 0.6501, true};
    const SfmCandidateSummary partial{
        1, 1.2, 3, 4, 15, 8000, 0.20, true};

    EXPECT_TRUE(xjw::sfm_search::isBetterCandidate(full, partial));
    EXPECT_FALSE(xjw::sfm_search::isBetterCandidate(partial, full));
}

TEST(SfmSearchPolicyTest, FiniteRmsWinsAfterEqualCoverageAndPoints)
{
    const SfmCandidateSummary finite{
        0, 2.8, 1, 2, 16, 4000, 0.7, true};
    const SfmCandidateSummary invalid{
        1, 3.2, 3, 4, 16, 4000,
        std::numeric_limits<double>::quiet_NaN(), true};

    EXPECT_TRUE(xjw::sfm_search::isBetterCandidate(finite, invalid));
}

TEST(SfmSearchPolicyTest, LowerRmsDominatesPointCountAfterEqualCoverage)
{
    const SfmCandidateSummary lowerRms{
        0, 3.2, 0, 15, 10, 1600, 0.63, true};
    const SfmCandidateSummary morePoints{
        1, 2.0, 0, 15, 10, 1800, 0.70, true};

    EXPECT_TRUE(xjw::sfm_search::isBetterCandidate(lowerRms, morePoints));
}

TEST(SfmSearchPolicyTest, RankingIsDeterministicAndReplayIsLimitedToThree)
{
    const std::vector<SfmCandidateSummary> candidates{
        {4, 2.8, 1, 2, 16, 3900, 0.66, true},
        {2, 3.2, 3, 4, 16, 4100, 0.65, true},
        {1, 2.4, 5, 6, 15, 5000, 0.40, true},
        {0, 2.0, 7, 8, 14, 6000, 0.30, true}};

    const auto ranked = xjw::sfm_search::rankCandidates(candidates);
    const auto replay = xjw::sfm_search::replayCandidateIndices(ranked, 3);

    ASSERT_EQ(ranked.size(), 4u);
    EXPECT_EQ(ranked[0].candidateIndex, 2);
    EXPECT_EQ(ranked[1].candidateIndex, 4);
    ASSERT_EQ(replay.size(), 3u);
    EXPECT_EQ(replay[0], 2);
    EXPECT_EQ(replay[1], 4);
    EXPECT_EQ(replay[2], 1);
}

TEST(SfmSearchPolicyTest, AdaptiveFocalSweepCoversNarrowFieldOfViewCameras)
{
    const std::vector<double> scales = xjw::sfm_search::adaptiveFocalScaleCandidates();

    ASSERT_FALSE(scales.empty());
    EXPECT_GE(scales.back(), 5.2);
    EXPECT_NE(std::find(scales.begin(), scales.end(), 5.2), scales.end());
}

TEST(SfmSearchPolicyTest, AdaptiveReplayStopsOnlyAfterFullRegistration)
{
    EXPECT_FALSE(xjw::sfm_search::shouldStopAdaptiveFocalReplay(16, 12, true));
    EXPECT_FALSE(xjw::sfm_search::shouldStopAdaptiveFocalReplay(16, 16, false));
    EXPECT_TRUE(xjw::sfm_search::shouldStopAdaptiveFocalReplay(16, 16, true));
}

} // namespace
