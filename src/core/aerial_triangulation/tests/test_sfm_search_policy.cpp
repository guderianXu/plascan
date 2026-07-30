#include <gtest/gtest.h>

#include "search/SfmSearchPolicy.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace
{

using xjw::aerial_triangulation::SfmCandidateSummary;

TEST(SfmSearchPolicyTest, AllocatesFourWorkersAcrossThirtyTwoThreads)
{
    const auto budget = xjw::aerial_triangulation::allocateWorkers(6, 32);

    EXPECT_EQ(budget.workerCount, 4);
    EXPECT_EQ(budget.threadsPerWorker, 8);
}

TEST(SfmSearchPolicyTest, KeepsSmallThreadBudgetsSerial)
{
    const auto budget = xjw::aerial_triangulation::allocateWorkers(6, 7);

    EXPECT_EQ(budget.workerCount, 1);
    EXPECT_EQ(budget.threadsPerWorker, 7);
}

TEST(SfmSearchPolicyTest, RegistrationCoverageDominatesRmsAndPointCount)
{
    const SfmCandidateSummary full{
        0, 3.2, 1, 2, 16, 4021, 0.6501, true};
    const SfmCandidateSummary partial{
        1, 1.2, 3, 4, 15, 8000, 0.20, true};

    EXPECT_TRUE(xjw::aerial_triangulation::isBetterCandidate(full, partial));
    EXPECT_FALSE(xjw::aerial_triangulation::isBetterCandidate(partial, full));
}

TEST(SfmSearchPolicyTest, SuccessfulCandidateDominatesFailedHighCoverageAttempt)
{
    const SfmCandidateSummary failedFull{
        0, 5.2, 1, 2, 16, 9000, 0.20, false};
    const SfmCandidateSummary successfulPartial{
        1, 1.2, 3, 4, 12, 2400, 0.75, true};

    EXPECT_TRUE(xjw::aerial_triangulation::isBetterCandidate(successfulPartial, failedFull));
    EXPECT_FALSE(xjw::aerial_triangulation::isBetterCandidate(failedFull, successfulPartial));
}

TEST(SfmSearchPolicyTest, FiniteRmsWinsAfterEqualCoverageAndPoints)
{
    const SfmCandidateSummary finite{
        0, 2.8, 1, 2, 16, 4000, 0.7, true};
    const SfmCandidateSummary invalid{
        1, 3.2, 3, 4, 16, 4000,
        std::numeric_limits<double>::quiet_NaN(), true};

    EXPECT_TRUE(xjw::aerial_triangulation::isBetterCandidate(finite, invalid));
}

TEST(SfmSearchPolicyTest, LowerRmsDominatesPointCountAfterEqualCoverage)
{
    const SfmCandidateSummary lowerRms{
        0, 3.2, 0, 15, 10, 1600, 0.63, true};
    const SfmCandidateSummary morePoints{
        1, 2.0, 0, 15, 10, 1800, 0.70, true};

    EXPECT_TRUE(xjw::aerial_triangulation::isBetterCandidate(lowerRms, morePoints));
}

TEST(SfmSearchPolicyTest, StrongerPhotogrammetricNetworkDominatesMarginalRmsGain)
{
    SfmCandidateSummary strongNetwork{
        0, 1.2, 0, 1, 9, 12451, 0.554, true};
    strongNetwork.hasNetworkQuality = true;
    strongNetwork.medianTriangulationAngleDeg = 8.4;
    strongNetwork.twoViewTrackRatio = 0.70;
    strongNetwork.observationGridCoverage = 0.13;

    SfmCandidateSummary weakNetwork{
        1, 5.2, 6, 7, 9, 5088, 0.548, true};
    weakNetwork.hasNetworkQuality = true;
    weakNetwork.medianTriangulationAngleDeg = 2.8;
    weakNetwork.twoViewTrackRatio = 0.77;
    weakNetwork.observationGridCoverage = 0.13;

    EXPECT_TRUE(xjw::aerial_triangulation::isBetterCandidate(strongNetwork, weakNetwork));
    EXPECT_FALSE(xjw::aerial_triangulation::isBetterCandidate(weakNetwork, strongNetwork));
}

TEST(SfmSearchPolicyTest, ClosedSequenceContinuityRejectsMarginallyStrongerDegenerateNetwork)
{
    SfmCandidateSummary continuous{
        0, 4.0, 0, 1, 12, 2219, 0.468, true};
    continuous.hasNetworkQuality = true;
    continuous.medianTriangulationAngleDeg = 58.77;
    continuous.twoViewTrackRatio = 0.454;
    continuous.observationGridCoverage = 0.112;
    continuous.hasClosedSequenceGeometry = true;
    continuous.sequenceAdjacentDistanceMaximumRatio = 1.48;
    continuous.sequenceAdjacentDistanceMadRatio = 0.010;

    SfmCandidateSummary degenerate{
        1, 3.2, 0, 1, 12, 2332, 0.521, true};
    degenerate.hasNetworkQuality = true;
    degenerate.medianTriangulationAngleDeg = 59.32;
    degenerate.twoViewTrackRatio = 0.371;
    degenerate.observationGridCoverage = 0.115;
    degenerate.hasClosedSequenceGeometry = true;
    degenerate.sequenceAdjacentDistanceMaximumRatio = 1.64;
    degenerate.sequenceAdjacentDistanceMadRatio = 0.011;

    EXPECT_TRUE(xjw::aerial_triangulation::isBetterCandidate(continuous, degenerate));
    EXPECT_FALSE(xjw::aerial_triangulation::isBetterCandidate(degenerate, continuous));
}

TEST(SfmSearchPolicyTest, SmallClosedSequenceDifferencesFallBackToNetworkQuality)
{
    SfmCandidateSummary strongerNetwork{
        0, 3.2, 0, 1, 12, 2300, 0.52, true};
    strongerNetwork.hasNetworkQuality = true;
    strongerNetwork.medianTriangulationAngleDeg = 59.0;
    strongerNetwork.twoViewTrackRatio = 0.37;
    strongerNetwork.observationGridCoverage = 0.116;
    strongerNetwork.hasClosedSequenceGeometry = true;
    strongerNetwork.sequenceAdjacentDistanceMaximumRatio = 1.55;
    strongerNetwork.sequenceAdjacentDistanceMadRatio = 0.011;

    SfmCandidateSummary similarSequence{
        1, 4.0, 0, 1, 12, 2200, 0.47, true};
    similarSequence.hasNetworkQuality = true;
    similarSequence.medianTriangulationAngleDeg = 58.8;
    similarSequence.twoViewTrackRatio = 0.45;
    similarSequence.observationGridCoverage = 0.112;
    similarSequence.hasClosedSequenceGeometry = true;
    similarSequence.sequenceAdjacentDistanceMaximumRatio = 1.48;
    similarSequence.sequenceAdjacentDistanceMadRatio = 0.010;

    EXPECT_TRUE(xjw::aerial_triangulation::isBetterCandidate(
        strongerNetwork, similarSequence));
}

TEST(SfmSearchPolicyTest, BalancedQualityRejectsLongFocalSequenceOutlier)
{
    // temple 实测：10 倍焦距只有最坏相邻基线比略小，其余摄影测量指标均明显较差。
    // 单个极值不能压过重投影、多视轨迹和稳健闭环连续性。
    SfmCandidateSummary balanced{
        0, 2.4, 14, 15, 16, 4735, 0.3044073359, true};
    balanced.hasNetworkQuality = true;
    balanced.medianTriangulationAngleDeg = 45.320758;
    balanced.twoViewTrackRatio = 0.3759239704;
    balanced.observationGridCoverage = 0.096924;
    balanced.hasClosedSequenceGeometry = true;
    balanced.sequenceAdjacentDistanceMaximumRatio = 4.9488195611;
    balanced.sequenceAdjacentDistanceMadRatio = 0.0119897050;

    SfmCandidateSummary longFocalOutlier{
        1, 10.0, 14, 15, 16, 4509, 0.5642177339, true};
    longFocalOutlier.hasNetworkQuality = true;
    longFocalOutlier.medianTriangulationAngleDeg = 31.134545;
    longFocalOutlier.twoViewTrackRatio = 0.5504546463;
    longFocalOutlier.observationGridCoverage = 0.089862;
    longFocalOutlier.hasClosedSequenceGeometry = true;
    longFocalOutlier.sequenceAdjacentDistanceMaximumRatio = 3.8983537706;
    longFocalOutlier.sequenceAdjacentDistanceMadRatio = 0.2142429022;

    EXPECT_TRUE(xjw::aerial_triangulation::isBetterCandidate(
        balanced, longFocalOutlier));
    EXPECT_FALSE(xjw::aerial_triangulation::isBetterCandidate(
        longFocalOutlier, balanced));
}

TEST(SfmSearchPolicyTest, RankingIsDeterministicAndReplayIsLimitedToThree)
{
    const std::vector<SfmCandidateSummary> candidates{
        {4, 2.8, 1, 2, 16, 3900, 0.66, true},
        {2, 3.2, 3, 4, 16, 4100, 0.65, true},
        {1, 2.4, 5, 6, 15, 5000, 0.40, true},
        {0, 2.0, 7, 8, 14, 6000, 0.30, true}};

    const auto ranked = xjw::aerial_triangulation::rankCandidates(candidates);
    const auto replay = xjw::aerial_triangulation::replayCandidateIndices(ranked, 3);

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
    const std::vector<double> scales = xjw::aerial_triangulation::adaptiveFocalScaleCandidates();

    ASSERT_FALSE(scales.empty());
    EXPECT_GE(scales.back(), 9.0);
    EXPECT_NE(std::find(scales.begin(), scales.end(), 5.2), scales.end());
    EXPECT_NE(std::find(scales.begin(), scales.end(), 9.0), scales.end());
}

TEST(SfmSearchPolicyTest, AdaptiveReplayStopsOnlyAfterFullRegistration)
{
    EXPECT_FALSE(xjw::aerial_triangulation::shouldStopAdaptiveFocalReplay(16, 12, true));
    EXPECT_FALSE(xjw::aerial_triangulation::shouldStopAdaptiveFocalReplay(16, 16, false));
    EXPECT_TRUE(xjw::aerial_triangulation::shouldStopAdaptiveFocalReplay(16, 16, true));
}

} // namespace
