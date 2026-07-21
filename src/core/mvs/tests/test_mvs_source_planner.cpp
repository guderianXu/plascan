#include "MvsSourcePlanner.h"

#include <gtest/gtest.h>

using xjw::mvs::mvsSourcePlanEntryToJson;
using xjw::mvs::MvsSourceCandidate;
using xjw::mvs::MvsSourcePlannerOptions;
using xjw::mvs::MvsSourceRejectReason;
using xjw::mvs::MvsSourceTier;
using xjw::mvs::planMvsSourceViews;
using xjw::mvs::planMvsSourceViewsVerifiedFirst;

namespace
{

MvsSourceCandidate candidate(int viewIndex,
                             int sharedTracks,
                             int geomInliers,
                             float angleDeg,
                             float coverage = 0.0f,
                             float baseline = 0.0f,
                             bool knownOverlap = false)
{
    MvsSourceCandidate c;
    c.viewIndex = viewIndex;
    c.sharedTracks = sharedTracks;
    c.geometricInliers = geomInliers;
    c.medianTriangulationAngleDeg = angleDeg;
    c.coverageScore = coverage;
    c.baselineScore = baseline;
    c.knownOverlap = knownOverlap;
    c.sequenceDistance = std::abs(viewIndex - 4);
    return c;
}

} // namespace

TEST(MvsSourcePlanner, VerifiedPairsStayFirstAndQualifiedGeometryBackfillsShortfall)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 6;
    options.rejectAngleOutliers = true;
    options.requireVerifiedPairGeometry = true;
    options.minGeometricInliers = 20;
    options.allowSequenceFallback = false;

    MvsSourceCandidate verified_a = candidate(3, 90, 80, 8.0f, 0.8f, 0.4f, true);
    verified_a.verifiedPairGeometry = true;
    MvsSourceCandidate verified_b = candidate(5, 85, 75, 9.0f, 0.75f, 0.45f, true);
    verified_b.verifiedPairGeometry = true;

    const auto plan = planMvsSourceViewsVerifiedFirst({
        verified_a,
        verified_b,
        candidate(0, 80, 8, 7.0f, 0.70f, 0.35f),
        candidate(1, 95, 12, 10.0f, 0.75f, 0.50f),
        candidate(2, 75, 6, 12.0f, 0.65f, 0.60f),
        candidate(6, 110, 14, 15.0f, 0.80f, 0.70f),
    }, options);

    ASSERT_EQ(plan.selected.size(), 6u);
    EXPECT_EQ(plan.selected[0].tier, MvsSourceTier::VerifiedPair);
    EXPECT_EQ(plan.selected[1].tier, MvsSourceTier::VerifiedPair);
    EXPECT_TRUE(plan.selected[0].verifiedPairGeometry);
    EXPECT_TRUE(plan.selected[1].verifiedPairGeometry);
    for (std::size_t index = 2; index < plan.selected.size(); ++index)
    {
        EXPECT_EQ(plan.selected[index].tier, MvsSourceTier::TrackGeometryBackfill);
        EXPECT_FALSE(plan.selected[index].sequenceFallback);
    }
    EXPECT_FALSE(plan.usedSequenceFallback);
    EXPECT_EQ(plan.requestedSourceCount, 6);
    EXPECT_EQ(plan.sourceViewShortfall, 0);
    EXPECT_EQ(mvsSourcePlanEntryToJson(plan.selected[2])
                  .value(QStringLiteral("source_tier")).toString(),
              QStringLiteral("track_geometry_backfill"));
}

TEST(MvsSourcePlanner, WeakBackfillCandidatesRemainRejectedAndShortfallIsReported)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 6;
    options.rejectAngleOutliers = true;
    options.requireVerifiedPairGeometry = true;
    options.minGeometricInliers = 20;
    options.allowSequenceFallback = false;

    MvsSourceCandidate verified_a = candidate(3, 90, 80, 8.0f, 0.8f, 0.4f, true);
    verified_a.verifiedPairGeometry = true;
    MvsSourceCandidate verified_b = candidate(5, 85, 75, 9.0f, 0.75f, 0.45f, true);
    verified_b.verifiedPairGeometry = true;

    const auto plan = planMvsSourceViewsVerifiedFirst({
        verified_a,
        verified_b,
        candidate(0, 19, 0, 8.0f, 0.8f, 0.4f),
        candidate(1, 80, 0, 0.1f, 0.8f, 0.4f),
        candidate(2, 80, 0, 40.0f, 0.8f, 0.4f),
        candidate(6, 10, 0, 12.0f, 0.8f, 0.4f),
    }, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].tier, MvsSourceTier::VerifiedPair);
    EXPECT_EQ(plan.selected[1].tier, MvsSourceTier::VerifiedPair);
    EXPECT_FALSE(plan.usedSequenceFallback);
    EXPECT_EQ(plan.requestedSourceCount, 6);
    EXPECT_EQ(plan.sourceViewShortfall, 4);
}

TEST(MvsSourcePlanner, VerifiedFirstNeverBackfillsZeroInlierViews)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 5;
    options.viewCount = 16;
    options.maxSources = 4;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 47.0f;
    options.minGeometricInliers = 20;
    options.allowSequenceFallback = false;

    MvsSourceCandidate verified_a = candidate(14, 900, 115, 19.6f, 0.8f, 0.5f, true);
    verified_a.verifiedPairGeometry = true;
    MvsSourceCandidate verified_b = candidate(6, 850, 846, 23.1f, 0.8f, 0.6f, true);
    verified_b.verifiedPairGeometry = true;
    MvsSourceCandidate zero_inlier = candidate(15, 800, 0, 42.9f, 0.8f, 1.0f, true);

    const auto plan = planMvsSourceViewsVerifiedFirst(
        {verified_a, verified_b, zero_inlier}, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].viewIndex, 6);
    EXPECT_EQ(plan.selected[1].viewIndex, 14);
    EXPECT_EQ(plan.sourceViewShortfall, 2);
    EXPECT_TRUE(std::none_of(plan.selected.begin(), plan.selected.end(), [](const auto &entry)
    {
        return entry.geometricInliers <= 0;
    }));
}

TEST(MvsSourcePlanner, VerifiedFirstBackfillRespectsSceneSpecificMaximumAngle)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 2;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 50.0f;
    options.allowSequenceFallback = false;

    MvsSourceCandidate verified = candidate(3, 120, 95, 12.0f, 0.8f, 0.5f, true);
    verified.verifiedPairGeometry = true;
    const auto plan = planMvsSourceViewsVerifiedFirst({
        verified,
        candidate(2, 100, 8, 45.0f, 0.75f, 1.0f, true),
    }, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[1].tier, MvsSourceTier::TrackGeometryBackfill);
    EXPECT_EQ(plan.selected[1].viewIndex, 2);
    EXPECT_EQ(plan.sourceViewShortfall, 0);
}

TEST(MvsSourcePlanner, RanksSharedTracksGeometryAndCoverageBeforeSequence)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 3;
    options.rejectAngleOutliers = true;

    const auto plan = planMvsSourceViews({
        candidate(3, 80, 60, 8.0f, 0.55f, 0.3f),
        candidate(5, 120, 115, 7.5f, 0.70f, 0.4f),
        candidate(1, 20, 15, 9.0f, 0.20f, 0.7f),
        candidate(7, 95, 40, 10.0f, 0.45f, 0.5f),
    }, options);

    ASSERT_EQ(plan.selected.size(), 3u);
    EXPECT_EQ(plan.selected[0].viewIndex, 5);
    EXPECT_EQ(plan.selected[1].viewIndex, 7);
    EXPECT_EQ(plan.selected[2].viewIndex, 3);
    EXPECT_GT(plan.selected[0].score, plan.selected[1].score);
}

TEST(MvsSourcePlanner, RejectsTriangulationAngleOutliersWhenRequested)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 4;
    options.rejectAngleOutliers = true;
    options.minTriangulationAngleDeg = 0.5f;
    options.maxTriangulationAngleDeg = 35.0f;

    const auto plan = planMvsSourceViews({
        candidate(2, 200, 160, 0.1f),
        candidate(3, 160, 150, 12.0f),
        candidate(5, 180, 170, 48.0f),
        candidate(6, 80, 70, 18.0f),
    }, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3);
    EXPECT_EQ(plan.selected[1].viewIndex, 6);
    ASSERT_EQ(plan.rejected.size(), 2u);
    EXPECT_EQ(plan.rejected[0].reason, MvsSourceRejectReason::TriangulationAngle);
    EXPECT_EQ(plan.rejected[1].reason, MvsSourceRejectReason::TriangulationAngle);
}

TEST(MvsSourcePlanner, FallsBackToNearestSequenceWhenNoGeometryExists)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 4;

    const auto plan = planMvsSourceViews({}, options);

    ASSERT_EQ(plan.selected.size(), 4u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3);
    EXPECT_EQ(plan.selected[1].viewIndex, 5);
    EXPECT_EQ(plan.selected[2].viewIndex, 2);
    EXPECT_EQ(plan.selected[3].viewIndex, 6);
    EXPECT_TRUE(plan.usedSequenceFallback);
}

TEST(MvsSourcePlanner, DeduplicatesCandidatesAndKeepsBestScore)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 2;

    const auto plan = planMvsSourceViews({
        candidate(3, 20, 18, 8.0f),
        candidate(3, 120, 100, 8.0f),
        candidate(5, 80, 70, 9.0f),
    }, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3);
    EXPECT_EQ(plan.selected[0].sharedTracks, 120);
    EXPECT_EQ(plan.selected[1].viewIndex, 5);
}

TEST(MvsSourcePlanner, AllowsKnownOverlapPairsWithWeakGeometryAfterStrongPairs)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 3;

    const auto plan = planMvsSourceViews({
        candidate(2, 0, 0, 0.0f, 0.0f, 0.0f, true),
        candidate(3, 60, 50, 8.0f),
        candidate(5, 0, 0, 0.0f),
    }, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3);
    EXPECT_EQ(plan.selected[1].viewIndex, 2);
    EXPECT_FALSE(plan.usedSequenceFallback);
}

TEST(MvsSourcePlanner, ProductionDefaultsRejectWideBaselineSources)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 4;
    options.rejectAngleOutliers = true;

    const auto plan = planMvsSourceViews({
        candidate(3, 180, 170, 7.0f, 0.80f, 0.35f, true),
        candidate(5, 175, 165, 22.0f, 0.75f, 0.70f, true),
        candidate(6, 210, 195, 44.0f, 0.90f, 1.00f, true),
        candidate(7, 220, 205, 53.0f, 0.92f, 1.00f, true),
    }, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3);
    EXPECT_EQ(plan.selected[1].viewIndex, 5);
    EXPECT_TRUE(std::all_of(plan.selected.begin(), plan.selected.end(), [](const auto &entry)
    {
        return entry.medianTriangulationAngleDeg <= 35.0f;
    }));
}

TEST(MvsSourcePlanner, ProductionGateRejectsKnownOverlapWithoutGeometryEvidence)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 3;
    options.minSharedTracks = 20;
    options.minGeometricInliers = 20;
    options.minSourceQualityScore = 0.35f;
    options.allowWeakKnownOverlap = false;

    const auto plan = planMvsSourceViews({
        candidate(2, 0, 0, 0.0f, 0.0f, 0.0f, true),
        candidate(3, 60, 50, 8.0f, 0.75f, 0.40f, true),
        candidate(5, 12, 12, 8.0f, 0.20f, 0.25f, true),
        candidate(6, 45, 42, 9.0f, 0.65f, 0.35f, true),
    }, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3);
    EXPECT_EQ(plan.selected[1].viewIndex, 6);

    const auto rejectedWeakOverlap = std::find_if(
        plan.rejected.begin(),
        plan.rejected.end(),
        [](const auto &rejected)
        {
            return rejected.candidate.viewIndex == 2
                && rejected.reason == MvsSourceRejectReason::LowQuality;
        });
    EXPECT_NE(rejectedWeakOverlap, plan.rejected.end());
}

TEST(MvsSourcePlanner, RejectsProjectedOverlapWithoutVerifiedPairGeometryWhenRequired)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 2;
    options.minSharedTracks = 20;
    options.minGeometricInliers = 20;
    options.allowWeakKnownOverlap = false;
    options.allowSequenceFallback = false;
    options.requireVerifiedPairGeometry = true;

    MvsSourceCandidate projectedOnly = candidate(2, 1500, 0, 9.0f, 0.90f, 0.45f, true);
    projectedOnly.verifiedPairGeometry = false;

    MvsSourceCandidate verified = candidate(3, 120, 95, 8.0f, 0.70f, 0.40f, true);
    verified.verifiedPairGeometry = true;

    const auto plan = planMvsSourceViews({projectedOnly, verified}, options);

    ASSERT_EQ(plan.selected.size(), 1u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3);
    EXPECT_TRUE(plan.selected[0].verifiedPairGeometry);

    const auto rejectedProjected = std::find_if(
        plan.rejected.begin(),
        plan.rejected.end(),
        [](const auto &rejected)
        {
            return rejected.candidate.viewIndex == 2
                && rejected.reason == MvsSourceRejectReason::LowQuality;
        });
    ASSERT_NE(rejectedProjected, plan.rejected.end());
    EXPECT_FALSE(rejectedProjected->candidate.verifiedPairGeometry);
}

TEST(MvsSourcePlanner, PublishesNormalizedSourceQualityScore)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 2;

    const auto plan = planMvsSourceViews({
        candidate(3, 120, 110, 8.0f, 0.8f, 0.7f, true),
        candidate(5, 20, 12, 7.0f, 0.2f, 0.3f, false),
    }, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_GT(plan.selected[0].sourceQualityScore, plan.selected[1].sourceQualityScore);
    EXPECT_GT(plan.selected[0].sourceQualityScore, 0.0f);
    EXPECT_LE(plan.selected[0].sourceQualityScore, 1.0f);

    const QJsonObject json = mvsSourcePlanEntryToJson(plan.selected[0]);
    EXPECT_TRUE(json.contains(QStringLiteral("source_quality_score")));
    EXPECT_GT(json.value(QStringLiteral("source_quality_score")).toDouble(), 0.0);
}

TEST(MvsSourcePlanner, PrefersProductionAerialAngleOverSlightlyMoreDistantWideBaseline)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 2;
    options.rejectAngleOutliers = true;
    options.minTriangulationAngleDeg = 1.0f;
    options.maxTriangulationAngleDeg = 45.0f;
    options.preferredTriangulationAngleDeg = 10.0f;
    options.softMaxTriangulationAngleDeg = 25.0f;

    const auto plan = planMvsSourceViews({
        candidate(2, 150, 145, 38.0f, 0.92f, 1.0f, true),
        candidate(3, 142, 136, 9.0f, 0.88f, 0.45f, true),
        candidate(5, 70, 65, 7.0f, 0.50f, 0.35f, true),
    }, options);

    ASSERT_GE(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3)
        << "Aerial MVS should prefer a well-supported 5-15 degree baseline over a slightly stronger wide baseline.";
    EXPECT_EQ(plan.selected[1].viewIndex, 2);
    EXPECT_GT(plan.selected[0].sourceQualityScore, plan.selected[1].sourceQualityScore);
}
