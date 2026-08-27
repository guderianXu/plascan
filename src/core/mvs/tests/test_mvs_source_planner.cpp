#include "MvsSourcePlanner.h"

#include <QJsonArray>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

using xjw::mvs::mvsSourcePlanEntryToJson;
using xjw::mvs::MvsSourceCandidate;
using xjw::mvs::MvsSourcePairQuality;
using xjw::mvs::MvsSourcePlanEntry;
using xjw::mvs::MvsSourcePlannerOptions;
using xjw::mvs::MvsSourceRejectReason;
using xjw::mvs::MvsSourceTier;
using xjw::mvs::MvsSourceVerificationStatus;
using xjw::mvs::planMvsSourceViews;
using xjw::mvs::planMvsSourceViewsVerifiedFirst;
using xjw::mvs::planMvsRepairSourceViews;
using xjw::mvs::recommendedMvsCrossViewSourceCount;
using xjw::mvs::filterMvsSourcePairQualitiesForImages;

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

MvsSourceCandidate verifiedCandidate(int view_index,
                                     int shared_tracks,
                                     int geometric_inliers,
                                     float angle_degrees)
{
    MvsSourceCandidate result = candidate(
        view_index,
        shared_tracks,
        geometric_inliers,
        angle_degrees,
        0.7f,
        0.5f,
        true);
    result.verifiedPairGeometry = true;
    result.verificationStatus = MvsSourceVerificationStatus::Verified;
    result.pairTotalMatches = std::max(geometric_inliers, 100);
    return result;
}

const xjw::mvs::MvsSourceRankingAuditEntry *rankingAuditForView(
    const xjw::mvs::MvsSourcePlan &plan,
    int view_index)
{
    const auto found = std::find_if(
        plan.rankingCandidates.cbegin(),
        plan.rankingCandidates.cend(),
        [view_index](const auto &audit)
        {
            return audit.candidate.viewIndex == view_index;
        });
    return found == plan.rankingCandidates.cend() ? nullptr : &*found;
}

} // namespace

TEST(MvsSourcePlanner, RepairSourcesRespectRequestedCountForPreferredSources)
{
    std::vector<int> preferredSources;
    for (int source_index = 0; source_index < 12; ++source_index)
    {
        preferredSources.push_back(source_index);
    }
    const std::vector<bool> sourceEligibility(13, true);

    const auto sources = planMvsRepairSourceViews(
        preferredSources, sourceEligibility, 12, 4);

    EXPECT_EQ(sources, (std::vector<int>{0, 1, 2, 3}));
}

TEST(MvsSourcePlanner, RepairSourcesFilterInvalidDuplicateAndIneligiblePreferredSources)
{
    std::vector<bool> sourceEligibility(8, true);
    sourceEligibility[2] = false;
    sourceEligibility[6] = false;

    const auto sources = planMvsRepairSourceViews(
        {-1, 8, 4, 2, 1, 1, 6, 7}, sourceEligibility, 4, 2);

    EXPECT_EQ(sources, (std::vector<int>{1, 7}));
}

TEST(MvsSourcePlanner, RepairSourcesBackfillShortfallFromAlternatingSequenceNeighbors)
{
    std::vector<bool> sourceEligibility(8, true);
    sourceEligibility[3] = false;
    sourceEligibility[6] = false;

    const auto sources = planMvsRepairSourceViews(
        {2, 2, 3}, sourceEligibility, 4, 4);

    EXPECT_EQ(sources, (std::vector<int>{2, 5, 1, 7}));
}

TEST(MvsSourcePlanner, OrbitalCrossViewConsensusUsesFullSixteenSourceBudget)
{
    EXPECT_EQ(recommendedMvsCrossViewSourceCount(
                  xjw::mvs::MvsSceneProfile::OrbitalObject, 8, 24),
              16);
    EXPECT_EQ(recommendedMvsCrossViewSourceCount(
                  xjw::mvs::MvsSceneProfile::OrbitalObject, 8, 10),
              9);
    EXPECT_EQ(recommendedMvsCrossViewSourceCount(
                  xjw::mvs::MvsSceneProfile::Custom, 8, 24),
              8);
}

TEST(MvsSourcePlanner, FiltersRemovedImageReferencesBeforeEnablingVerifiedPairGate)
{
    const std::vector<std::string> currentImages = {
        "E:/data/current/image_01.tif",
        "E:/data/current/image_02.tif",
        "E:/data/current/image_03.tif",
    };
    const std::vector<MvsSourcePairQuality> qualities = {
        {"E:/data/removed/image_01.tif", "E:/data/removed/image_02.tif", 150, 120, true},
        {"E:\\data\\current\\image_01.tif", "e:/DATA/current/image_02.tif", 90, 70, true},
        {"E:/data/current/image_02.tif", "E:/data/current/image_01.tif", 110, 85, true},
        {"E:/data/current/image_02.tif", "E:/data/removed/image_03.tif", 100, 80, true},
    };

    const auto filtered = filterMvsSourcePairQualitiesForImages(qualities, currentImages);

    ASSERT_EQ(filtered.size(), 1u);
    EXPECT_EQ(filtered[0].geometricInliers, 85);
    EXPECT_EQ(filtered[0].imageA, "E:/data/current/image_02.tif");
    EXPECT_EQ(filtered[0].imageB, "E:/data/current/image_01.tif");
}

TEST(MvsSourcePlanner, EntirelyStalePairCatalogProducesNoActiveQuality)
{
    const std::vector<std::string> currentImages = {
        "E:/data/current/image_01.tif",
        "E:/data/current/image_02.tif",
    };
    const std::vector<MvsSourcePairQuality> staleQualities = {
        {"E:/data/removed/old_image_01.tif", "E:/data/removed/old_image_02.tif", 150, 120, true},
    };

    EXPECT_TRUE(filterMvsSourcePairQualitiesForImages(staleQualities, currentImages).empty());
}

TEST(MvsSourcePlanner, RelocatedPairCatalogRebindsUniqueFileNames)
{
    const std::vector<std::string> currentImages = {
        "E:/workspace/images/image_01.tif",
        "E:/workspace/images/image_02.tif",
    };
    const std::vector<MvsSourcePairQuality> relocatedQualities = {
        {"G:/capture/archive/image_01.tif", "G:/capture/archive/image_02.tif", 150, 120, true},
    };

    const auto filtered = filterMvsSourcePairQualitiesForImages(
        relocatedQualities, currentImages);

    ASSERT_EQ(filtered.size(), 1u);
    EXPECT_EQ(filtered.front().imageA, currentImages[0]);
    EXPECT_EQ(filtered.front().imageB, currentImages[1]);
    EXPECT_EQ(filtered.front().geometricInliers, 120);
}

TEST(MvsSourcePlanner, RelocatedPairCatalogRejectsAmbiguousFileNames)
{
    const std::vector<std::string> currentImages = {
        "E:/workspace/a/image_01.tif",
        "E:/workspace/b/image_01.tif",
        "E:/workspace/images/image_02.tif",
    };
    const std::vector<MvsSourcePairQuality> relocatedQualities = {
        {"G:/capture/image_01.tif", "G:/capture/image_02.tif", 150, 120, true},
    };

    EXPECT_TRUE(filterMvsSourcePairQualitiesForImages(
        relocatedQualities, currentImages).empty());
}

TEST(MvsSourcePlanner, AuditedFailureReplacesDuplicateWithMissingStatistics)
{
    const std::vector<std::string> currentImages = {
        "E:/data/current/image_01.tif",
        "E:/data/current/image_02.tif",
    };
    MvsSourcePairQuality missing;
    missing.imageA = currentImages[0];
    missing.imageB = currentImages[1];
    missing.totalMatches = 120;
    missing.verificationReason = "missing_geometric_inlier_statistics";

    MvsSourcePairQuality failed = missing;
    failed.hasVerificationStatistics = true;
    failed.verificationReason = "stored_match_geometry_gate_failed";

    const auto filtered = filterMvsSourcePairQualitiesForImages(
        {missing, failed}, currentImages);

    ASSERT_EQ(filtered.size(), 1u);
    EXPECT_TRUE(filtered.front().hasVerificationStatistics);
    EXPECT_EQ(filtered.front().verificationReason,
              "stored_match_geometry_gate_failed");
}

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
    zero_inlier.verificationStatus = MvsSourceVerificationStatus::Failed;
    zero_inlier.pairTotalMatches = 320;
    zero_inlier.verificationReason = "zero_geometric_inliers";

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

TEST(MvsSourcePlanner, FailedPairWithDirectGeometryCanBackfillHyb2OrbitalShortfall)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 4;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 62.0f;
    options.minGeometricInliers = 20;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;

    MvsSourceCandidate verified =
        candidate(3, 900, 860, 29.8f, 0.9f, 1.0f, true);
    verified.verifiedPairGeometry = true;
    verified.verificationStatus = MvsSourceVerificationStatus::Verified;
    verified.pairTotalMatches = 900;

    MvsSourceCandidate failed_a =
        candidate(2, 480, 34, 59.7f, 0.8f, 1.0f);
    failed_a.verificationStatus = MvsSourceVerificationStatus::Failed;
    failed_a.pairTotalMatches = 49;
    failed_a.pairCoverageScore = 0.25f;
    failed_a.verificationReason = "stored_match_geometry_gate_failed";

    MvsSourceCandidate failed_b =
        candidate(6, 420, 24, 59.5f, 0.75f, 1.0f);
    failed_b.verificationStatus = MvsSourceVerificationStatus::Failed;
    failed_b.pairTotalMatches = 37;
    failed_b.pairCoverageScore = 0.1875f;
    failed_b.verificationReason = "stored_match_geometry_gate_failed";

    MvsSourceCandidate weak_ratio =
        candidate(7, 900, 50, 30.0f, 0.9f, 1.0f);
    weak_ratio.verificationStatus = MvsSourceVerificationStatus::Failed;
    weak_ratio.pairTotalMatches = 200;
    weak_ratio.pairCoverageScore = 0.50f;
    weak_ratio.verificationReason = "stored_match_geometry_gate_failed";

    const auto plan = planMvsSourceViewsVerifiedFirst(
        {weak_ratio, failed_b, verified, failed_a}, options);

    ASSERT_EQ(plan.selected.size(), 3u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3);
    EXPECT_EQ(plan.selected[0].tier, MvsSourceTier::VerifiedPair);
    for (std::size_t index = 1; index < plan.selected.size(); ++index)
    {
        EXPECT_EQ(plan.selected[index].tier, MvsSourceTier::TrackGeometryBackfill);
        EXPECT_EQ(plan.selected[index].verificationStatus,
                  MvsSourceVerificationStatus::Failed);
        EXPECT_FALSE(plan.selected[index].verifiedPairGeometry);
        EXPECT_FALSE(plan.selected[index].sequenceFallback);
        EXPECT_GE(plan.selected[index].geometricInliers, 20);
        EXPECT_GE(plan.selected[index].pairInlierRatio, 0.60f);
    }
    EXPECT_EQ(plan.sourceViewShortfall, 1);
    EXPECT_TRUE(std::none_of(
        plan.selected.cbegin(),
        plan.selected.cend(),
        [](const auto &entry)
        {
            return entry.viewIndex == 7;
        }));

    const auto defaultPlan = planMvsSourceViews({failed_a}, options);
    EXPECT_TRUE(defaultPlan.selected.empty());
}

TEST(MvsSourcePlanner, VerifiedPriorityAndBackfillOrderRemainDeterministic)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 3;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 62.0f;
    options.minGeometricInliers = 20;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;

    MvsSourceCandidate verified_a =
        candidate(3, 120, 90, 15.0f, 0.6f, 0.6f, true);
    verified_a.verifiedPairGeometry = true;
    MvsSourceCandidate verified_b =
        candidate(5, 110, 85, 16.0f, 0.6f, 0.6f, true);
    verified_b.verifiedPairGeometry = true;

    MvsSourceCandidate failed_a =
        candidate(2, 700, 34, 59.7f, 0.9f, 1.0f);
    failed_a.verificationStatus = MvsSourceVerificationStatus::Failed;
    failed_a.pairTotalMatches = 49;
    failed_a.pairCoverageScore = 0.25f;
    MvsSourceCandidate failed_b =
        candidate(6, 600, 24, 59.5f, 0.9f, 1.0f);
    failed_b.verificationStatus = MvsSourceVerificationStatus::Failed;
    failed_b.pairTotalMatches = 37;
    failed_b.pairCoverageScore = 0.1875f;

    const auto first = planMvsSourceViewsVerifiedFirst(
        {failed_b, verified_b, failed_a, verified_a}, options);
    const auto second = planMvsSourceViewsVerifiedFirst(
        {verified_a, failed_a, verified_b, failed_b}, options);

    ASSERT_EQ(first.selected.size(), 3u);
    ASSERT_EQ(second.selected.size(), first.selected.size());
    EXPECT_EQ(first.selected[0].tier, MvsSourceTier::VerifiedPair);
    EXPECT_EQ(first.selected[1].tier, MvsSourceTier::VerifiedPair);
    EXPECT_EQ(first.selected[2].tier, MvsSourceTier::TrackGeometryBackfill);
    for (std::size_t index = 0; index < first.selected.size(); ++index)
    {
        EXPECT_EQ(first.selected[index].viewIndex,
                  second.selected[index].viewIndex);
        EXPECT_EQ(first.selected[index].tier,
                  second.selected[index].tier);
    }
}

TEST(MvsSourcePlanner, FailedBackfillRequiresSampleSizeCoverageAndWilsonConfidence)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 3;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 62.0f;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;

    MvsSourceCandidate verified =
        candidate(3, 120, 95, 30.0f, 0.8f, 0.5f, true);
    verified.verifiedPairGeometry = true;

    MvsSourceCandidate accepted =
        candidate(2, 80, 13, 60.0f, 0.7f, 0.8f);
    accepted.verificationStatus = MvsSourceVerificationStatus::Failed;
    accepted.pairTotalMatches = 17;
    accepted.pairCoverageScore = 0.1875f;

    MvsSourceCandidate too_small =
        candidate(5, 80, 9, 60.0f, 0.7f, 0.8f);
    too_small.verificationStatus = MvsSourceVerificationStatus::Failed;
    too_small.pairTotalMatches = 9;
    too_small.pairCoverageScore = 0.25f;

    MvsSourceCandidate low_coverage =
        candidate(6, 80, 17, 60.0f, 0.7f, 0.8f);
    low_coverage.verificationStatus = MvsSourceVerificationStatus::Failed;
    low_coverage.pairTotalMatches = 19;
    low_coverage.pairCoverageScore = 0.125f;

    MvsSourceCandidate weak_confidence =
        candidate(7, 80, 50, 60.0f, 0.7f, 0.8f);
    weak_confidence.verificationStatus = MvsSourceVerificationStatus::Failed;
    weak_confidence.pairTotalMatches = 200;
    weak_confidence.pairCoverageScore = 0.50f;

    MvsSourceCandidate zero_inlier =
        candidate(8, 80, 0, 60.0f, 0.7f, 0.8f);
    zero_inlier.verificationStatus = MvsSourceVerificationStatus::Failed;
    zero_inlier.pairTotalMatches = 20;
    zero_inlier.pairCoverageScore = 0.50f;

    const auto plan = planMvsSourceViewsVerifiedFirst(
        {weak_confidence, low_coverage, accepted, zero_inlier,
         too_small, verified},
        options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].viewIndex, 3);
    EXPECT_EQ(plan.selected[1].viewIndex, 2);
    EXPECT_EQ(plan.sourceViewShortfall, 1);
}

TEST(MvsSourcePlanner, FailedPairNeverFillsRequestedFourthSource)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 4;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 62.0f;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;

    MvsSourceCandidate verified_a =
        candidate(3, 120, 95, 15.0f, 0.8f, 0.5f, true);
    verified_a.verifiedPairGeometry = true;
    MvsSourceCandidate verified_b =
        candidate(5, 115, 90, 16.0f, 0.8f, 0.5f, true);
    verified_b.verifiedPairGeometry = true;
    MvsSourceCandidate verified_c =
        candidate(2, 110, 85, 30.0f, 0.8f, 0.5f, true);
    verified_c.verifiedPairGeometry = true;

    MvsSourceCandidate failed =
        candidate(6, 80, 24, 60.0f, 0.7f, 0.8f);
    failed.verificationStatus = MvsSourceVerificationStatus::Failed;
    failed.pairTotalMatches = 37;
    failed.pairCoverageScore = 0.1875f;

    const auto plan = planMvsSourceViewsVerifiedFirst(
        {failed, verified_c, verified_b, verified_a}, options);

    ASSERT_EQ(plan.selected.size(), 3u);
    EXPECT_TRUE(std::none_of(
        plan.selected.cbegin(), plan.selected.cend(),
        [](const auto &entry)
        {
            return entry.verificationStatus ==
                MvsSourceVerificationStatus::Failed;
        }));
    EXPECT_EQ(plan.sourceViewShortfall, 1);
}

TEST(MvsSourcePlanner, ExplicitFourSourceCapAllowsQualifiedFailedPairForMajorityNcc)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 4;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 62.0f;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;
    options.failedPairBackfillMaximumTotalSources = 4;

    std::vector<MvsSourceCandidate> candidates;
    for (int view_index : {2, 3, 5})
    {
        MvsSourceCandidate verified =
            candidate(view_index, 120, 95, 20.0f, 0.8f, 0.5f, true);
        verified.verifiedPairGeometry = true;
        candidates.push_back(verified);
    }
    MvsSourceCandidate failed =
        candidate(6, 80, 24, 60.0f, 0.7f, 0.8f);
    failed.verificationStatus = MvsSourceVerificationStatus::Failed;
    failed.pairTotalMatches = 37;
    failed.pairCoverageScore = 0.1875f;
    candidates.push_back(failed);

    const auto plan = planMvsSourceViewsVerifiedFirst(candidates, options);

    ASSERT_EQ(plan.selected.size(), 4u);
    EXPECT_EQ(plan.selected.back().viewIndex, 6);
    EXPECT_EQ(plan.selected.back().verificationStatus,
              MvsSourceVerificationStatus::Failed);
    EXPECT_EQ(plan.sourceViewShortfall, 0);
}

TEST(MvsSourcePlanner, StrictPairAuditMayFillFifthAndSixthSources)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 14;
    options.maxSources = 6;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 65.0f;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;
    options.failedPairBackfillMaximumTotalSources = 4;
    options.allowStrictFailedPairBackfill = true;

    std::vector<MvsSourceCandidate> candidates;
    for (int view_index : {2, 3, 5, 6})
    {
        MvsSourceCandidate verified =
            candidate(view_index, 120, 95, 20.0f, 0.8f, 0.5f, true);
        verified.verifiedPairGeometry = true;
        candidates.push_back(verified);
    }
    for (int view_index : {1, 7})
    {
        MvsSourceCandidate audited =
            candidate(view_index, 90, 58, 35.0f, 0.7f, 0.7f);
        audited.verificationStatus = MvsSourceVerificationStatus::Failed;
        audited.pairTotalMatches = 72;
        audited.pairCoverageScore = 0.42f;
        audited.verificationReason = "production_threshold_not_met";
        candidates.push_back(audited);
    }

    const auto plan = planMvsSourceViewsVerifiedFirst(candidates, options);

    ASSERT_EQ(plan.selected.size(), 6u);
    EXPECT_EQ(plan.selected[4].tier, MvsSourceTier::StrictPairAuditBackfill);
    EXPECT_EQ(plan.selected[5].tier, MvsSourceTier::StrictPairAuditBackfill);
    EXPECT_EQ(plan.sourceViewShortfall, 0);
}

TEST(MvsSourcePlanner, StrictPairAuditStillRejectsWeakFailedPair)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 5;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;
    options.failedPairBackfillMaximumTotalSources = 4;
    options.allowStrictFailedPairBackfill = true;

    std::vector<MvsSourceCandidate> candidates;
    for (int view_index : {2, 3, 5, 6})
    {
        MvsSourceCandidate verified =
            candidate(view_index, 120, 95, 20.0f, 0.8f, 0.5f, true);
        verified.verifiedPairGeometry = true;
        candidates.push_back(verified);
    }
    MvsSourceCandidate weak = candidate(7, 50, 22, 30.0f, 0.7f, 0.7f);
    weak.verificationStatus = MvsSourceVerificationStatus::Failed;
    weak.pairTotalMatches = 50;
    weak.pairCoverageScore = 0.25f;
    candidates.push_back(weak);

    const auto plan = planMvsSourceViewsVerifiedFirst(candidates, options);

    EXPECT_EQ(plan.selected.size(), 4u);
    EXPECT_EQ(plan.sourceViewShortfall, 1);
}

TEST(MvsSourcePlanner, MissingStatisticsMayStillFillRequestedFourthSource)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 4;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 62.0f;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;

    std::vector<MvsSourceCandidate> candidates;
    for (int view_index : {2, 3, 5})
    {
        MvsSourceCandidate verified =
            candidate(view_index, 120, 95, 20.0f, 0.8f, 0.5f, true);
        verified.verifiedPairGeometry = true;
        candidates.push_back(verified);
    }
    MvsSourceCandidate missing =
        candidate(6, 100, 100, 30.0f, 0.7f, 0.5f, true);
    missing.verificationStatus =
        MvsSourceVerificationStatus::MissingStatistics;
    missing.pairTotalMatches = 100;
    candidates.push_back(missing);

    const auto plan = planMvsSourceViewsVerifiedFirst(candidates, options);

    ASSERT_EQ(plan.selected.size(), 4u);
    EXPECT_EQ(plan.selected.back().verificationStatus,
              MvsSourceVerificationStatus::MissingStatistics);
    EXPECT_EQ(plan.sourceViewShortfall, 0);
}

TEST(MvsSourcePlanner, MissingVerificationStatisticsCanBackfillButFailedPairCannot)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 3;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 50.0f;
    options.allowSequenceFallback = false;

    MvsSourceCandidate verified =
        candidate(3, 120, 95, 12.0f, 0.8f, 0.5f, true);
    verified.verifiedPairGeometry = true;
    verified.verificationStatus = MvsSourceVerificationStatus::Verified;
    verified.pairTotalMatches = 130;

    MvsSourceCandidate missing =
        candidate(5, 100, 100, 14.0f, 0.7f, 0.5f, true);
    missing.verificationStatus =
        MvsSourceVerificationStatus::MissingStatistics;
    missing.pairTotalMatches = 140;
    missing.verificationReason = "missing_geometric_inlier_statistics";

    MvsSourceCandidate failed =
        candidate(6, 300, 0, 10.0f, 0.9f, 0.5f, true);
    failed.verificationStatus = MvsSourceVerificationStatus::Failed;
    failed.pairTotalMatches = 320;
    failed.verificationReason = "zero_geometric_inliers";

    const auto plan =
        planMvsSourceViewsVerifiedFirst({verified, missing, failed}, options);

    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.selected[0].tier, MvsSourceTier::VerifiedPair);
    EXPECT_EQ(plan.selected[1].tier, MvsSourceTier::TrackGeometryBackfill);
    EXPECT_EQ(
        plan.selected[1].verificationStatus,
        MvsSourceVerificationStatus::MissingStatistics);
    EXPECT_TRUE(std::none_of(
        plan.selected.cbegin(),
        plan.selected.cend(),
        [](const auto &entry)
        {
            return entry.viewIndex == 6;
        }));
    EXPECT_TRUE(std::any_of(
        plan.rejected.cbegin(),
        plan.rejected.cend(),
        [](const auto &entry)
        {
            return entry.candidate.viewIndex == 6
                && entry.candidate.verificationStatus ==
                    MvsSourceVerificationStatus::Failed;
        }));

    const QJsonObject json = mvsSourcePlanEntryToJson(plan.selected[1]);
    EXPECT_EQ(
        json.value(QStringLiteral("verification_status")).toString(),
        QStringLiteral("missing_statistics"));
    EXPECT_EQ(json.value(QStringLiteral("pair_total_matches")).toInt(), 140);
    EXPECT_EQ(
        json.value(QStringLiteral("verification_reason")).toString(),
        QStringLiteral("missing_geometric_inlier_statistics"));
}

TEST(MvsSourcePlanner,
     MissingStatisticsWithNegligibleDirectMatchesCannotBackfill)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 3;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 60.0f;
    options.allowSequenceFallback = false;
    options.minMissingStatisticsPairMatches = 16;

    MvsSourceCandidate verified =
        candidate(3, 120, 95, 12.0f, 0.8f, 0.5f, true);
    verified.verifiedPairGeometry = true;
    verified.verificationStatus = MvsSourceVerificationStatus::Verified;
    verified.pairTotalMatches = 130;

    MvsSourceCandidate missing =
        candidate(5, 1600, 1600, 56.0f, 0.9f, 1.0f, true);
    missing.verificationStatus =
        MvsSourceVerificationStatus::MissingStatistics;
    missing.pairTotalMatches = 2;
    missing.verificationReason = "stored_match_evidence_insufficient";

    const auto plan =
        planMvsSourceViewsVerifiedFirst({verified, missing}, options);

    ASSERT_EQ(plan.selected.size(), 1u);
    EXPECT_EQ(plan.selected.front().viewIndex, 3);
    EXPECT_EQ(plan.sourceViewShortfall, 2);
    EXPECT_TRUE(std::any_of(
        plan.rejected.cbegin(),
        plan.rejected.cend(),
        [](const auto &entry)
        {
            return entry.candidate.viewIndex == 5 &&
                entry.reason == MvsSourceRejectReason::LowQuality;
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

TEST(MvsSourcePlanner, DefaultRankingKeepsLegacyPlanAndJsonExact)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 10;
    options.maxSources = 2;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 35.0f;

    const std::vector<MvsSourceCandidate> candidates{
        candidate(1, 300, 250, 10.0f),
        candidate(3, 250, 150, 30.0f),
        candidate(2, 160, 150, 20.0f)};
    const auto legacy = planMvsSourceViews(candidates, options);

    ASSERT_EQ(legacy.selected.size(), 2u);
    EXPECT_FALSE(legacy.sourceRankingAudited);
    EXPECT_TRUE(legacy.controlSelected.empty());
    EXPECT_TRUE(legacy.controlRankingCandidates.empty());
    EXPECT_TRUE(legacy.rankingCandidates.empty());
    const QJsonObject legacy_json =
        mvsSourcePlanEntryToJson(legacy.selected.front());
    EXPECT_FALSE(legacy_json.contains(QStringLiteral("legacy_score")));
    EXPECT_FALSE(legacy_json.contains(QStringLiteral("adjusted_score")));
    EXPECT_FALSE(legacy_json.contains(
        QStringLiteral("legacy_rank_within_tier")));
    EXPECT_FALSE(legacy_json.contains(
        QStringLiteral("ranking_soft_maximum_degrees")));
    EXPECT_FALSE(legacy_json.contains(
        QStringLiteral("ranking_effective_maximum_degrees")));

    options.auditSourceRanking = true;
    const auto audited_control = planMvsSourceViews(candidates, options);
    ASSERT_EQ(audited_control.selected.size(), legacy.selected.size());
    for (std::size_t index = 0; index < legacy.selected.size(); ++index)
    {
        EXPECT_EQ(audited_control.selected[index].viewIndex,
                  legacy.selected[index].viewIndex);
        EXPECT_FLOAT_EQ(audited_control.selected[index].score,
                        legacy.selected[index].score);
        EXPECT_FLOAT_EQ(audited_control.selected[index].sourceQualityScore,
                        legacy.selected[index].sourceQualityScore);
    }
}

TEST(MvsSourcePlanner, CompletePoolLetsLegacyScoreSeeEarlyStopCounterexample)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 10;
    options.maxSources = 1;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 35.0f;

    const MvsSourceCandidate early_wide =
        candidate(3, 200, 150, 30.0f);
    const MvsSourceCandidate later_better =
        candidate(2, 160, 150, 10.0f);
    const auto legacy_pool = planMvsSourceViews({early_wide}, options);
    options.auditSourceRanking = true;
    const auto complete_pool = planMvsSourceViews(
        {early_wide, later_better}, options);

    ASSERT_EQ(legacy_pool.selected.size(), 1u);
    ASSERT_EQ(complete_pool.selected.size(), 1u);
    EXPECT_EQ(legacy_pool.selected.front().viewIndex, 3);
    EXPECT_EQ(complete_pool.selected.front().viewIndex, 2);
    EXPECT_GT(complete_pool.selected.front().score,
              legacy_pool.selected.front().score);
    EXPECT_FALSE(complete_pool.sourceRankingApplied)
        << "B expands the pool but must still use the unmodified legacy score.";
}

TEST(MvsSourcePlanner, SoftAngleRankingUsesRegisteredFormulaAndPreservesCount)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 10;
    options.maxSources = 2;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 35.0f;
    options.softMaxTriangulationAngleDeg = 25.0f;
    options.sourceAngleSoftRankingStrength = 1.0f;
    options.auditSourceRanking = true;

    const auto plan = planMvsSourceViews({
        candidate(1, 300, 250, 10.0f),
        candidate(3, 250, 150, 30.0f),
        candidate(2, 160, 150, 20.0f),
    }, options);

    ASSERT_EQ(plan.controlSelected.size(), 2u);
    ASSERT_EQ(plan.treatmentSelected.size(), 2u);
    ASSERT_EQ(plan.controlRankingCandidates.size(), 3u);
    ASSERT_EQ(plan.rankingCandidates.size(), 3u);
    ASSERT_EQ(plan.selected.size(), 2u);
    EXPECT_EQ(plan.controlSelected[1].viewIndex, 3);
    EXPECT_EQ(plan.selected[1].viewIndex, 2);
    EXPECT_TRUE(plan.selectedCountInvariant);
    EXPECT_TRUE(plan.sourceRankingApplied);
    EXPECT_EQ(plan.sourceViewShortfall, 0);

    const auto *wide = rankingAuditForView(plan, 3);
    const auto *narrow = rankingAuditForView(plan, 2);
    const auto control_wide = std::find_if(
        plan.controlRankingCandidates.cbegin(),
        plan.controlRankingCandidates.cend(),
        [](const auto &audit)
        {
            return audit.candidate.viewIndex == 3;
        });
    const auto control_narrow = std::find_if(
        plan.controlRankingCandidates.cbegin(),
        plan.controlRankingCandidates.cend(),
        [](const auto &audit)
        {
            return audit.candidate.viewIndex == 2;
        });
    ASSERT_NE(wide, nullptr);
    ASSERT_NE(narrow, nullptr);
    ASSERT_NE(control_wide, plan.controlRankingCandidates.cend());
    ASSERT_NE(control_narrow, plan.controlRankingCandidates.cend());
    EXPECT_TRUE(control_wide->selectedByPlan);
    EXPECT_FALSE(control_narrow->selectedByPlan);
    EXPECT_FALSE(wide->selectedByPlan);
    EXPECT_TRUE(narrow->selectedByPlan);
    EXPECT_NEAR(wide->candidate.normalizedSoftAnglePenalty, 0.5f, 1.0e-6f);
    EXPECT_NEAR(
        wide->candidate.adjustedScore,
        wide->candidate.score * std::exp(-0.5f),
        1.0e-4f);
    EXPECT_FLOAT_EQ(wide->candidate.rankingSoftMaximumDegrees, 25.0f);
    EXPECT_FLOAT_EQ(wide->candidate.rankingEffectiveMaximumDegrees, 35.0f);
    EXPECT_FLOAT_EQ(narrow->candidate.normalizedSoftAnglePenalty, 0.0f);
    EXPECT_FLOAT_EQ(narrow->candidate.adjustedScore, narrow->candidate.score);
}

TEST(MvsSourcePlanner, SoftRankingStaysInsideVerifiedAndOrdinaryTiers)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 16;
    options.maxSources = 3;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 35.0f;
    options.allowSequenceFallback = false;
    options.sourceAngleSoftRankingStrength = 1.0f;
    options.auditSourceRanking = true;

    const auto verified_plan = planMvsSourceViewsVerifiedFirst({
        verifiedCandidate(1, 300, 250, 10.0f),
        verifiedCandidate(3, 250, 150, 30.0f),
        verifiedCandidate(2, 160, 150, 20.0f),
        verifiedCandidate(5, 120, 100, 15.0f),
    }, options);
    ASSERT_EQ(verified_plan.selected.size(), 3u);
    EXPECT_TRUE(std::all_of(
        verified_plan.selected.cbegin(),
        verified_plan.selected.cend(),
        [](const auto &entry)
        {
            return entry.tier == MvsSourceTier::VerifiedPair;
        }));
    EXPECT_TRUE(verified_plan.sourceRankingApplied);

    const auto ordinary_plan = planMvsSourceViewsVerifiedFirst({
        verifiedCandidate(1, 500, 450, 10.0f),
        candidate(0, 300, 250, 10.0f, 0.7f, 0.5f, true),
        candidate(3, 250, 150, 30.0f, 0.7f, 0.5f, true),
        candidate(2, 160, 150, 20.0f, 0.7f, 0.5f, true),
    }, options);
    ASSERT_EQ(ordinary_plan.selected.size(), 3u);
    EXPECT_EQ(ordinary_plan.selected.front().viewIndex, 1);
    EXPECT_EQ(ordinary_plan.selected.front().tier,
              MvsSourceTier::VerifiedPair);
    EXPECT_EQ(ordinary_plan.selected[1].tier,
              MvsSourceTier::TrackGeometryBackfill);
    EXPECT_EQ(ordinary_plan.selected[2].tier,
              MvsSourceTier::TrackGeometryBackfill);
    EXPECT_TRUE(ordinary_plan.sourceRankingApplied);
}

TEST(MvsSourcePlanner, SoftRankingCoversBoundedAndStrictFailedTiers)
{
    const auto failed_candidate = [](int view_index,
                                     int shared_tracks,
                                     float angle_degrees)
    {
        MvsSourceCandidate result = candidate(
            view_index,
            shared_tracks,
            90,
            angle_degrees,
            0.7f,
            0.5f);
        result.verificationStatus = MvsSourceVerificationStatus::Failed;
        result.pairTotalMatches = 100;
        result.pairCoverageScore = 0.5f;
        return result;
    };

    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 16;
    options.maxSources = 4;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 35.0f;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;
    options.failedPairBackfillMaximumTotalSources = 4;
    options.sourceAngleSoftRankingStrength = 1.0f;
    options.auditSourceRanking = true;
    const std::vector<MvsSourceCandidate> candidates{
        verifiedCandidate(1, 500, 450, 10.0f),
        verifiedCandidate(5, 450, 400, 12.0f),
        failed_candidate(0, 300, 10.0f),
        failed_candidate(3, 250, 30.0f),
        failed_candidate(2, 160, 20.0f)};

    const auto bounded = planMvsSourceViewsVerifiedFirst(candidates, options);
    ASSERT_EQ(bounded.selected.size(), 4u);
    EXPECT_TRUE(bounded.sourceRankingApplied);
    EXPECT_EQ(bounded.selected[2].tier,
              MvsSourceTier::TrackGeometryBackfill);
    EXPECT_EQ(bounded.selected[3].tier,
              MvsSourceTier::TrackGeometryBackfill);

    options.failedPairBackfillMaximumTotalSources = 2;
    options.allowStrictFailedPairBackfill = true;
    const auto strict = planMvsSourceViewsVerifiedFirst(candidates, options);
    ASSERT_EQ(strict.selected.size(), 4u);
    EXPECT_TRUE(strict.sourceRankingApplied);
    EXPECT_EQ(strict.selected[2].tier,
              MvsSourceTier::StrictPairAuditBackfill);
    EXPECT_EQ(strict.selected[3].tier,
              MvsSourceTier::StrictPairAuditBackfill);
}

TEST(MvsSourcePlanner,
     VerifiedFirstBuildsIndependentUniqueControlAndTreatmentPlans)
{
    const auto failed_candidate = [](int view_index,
                                     int shared_tracks,
                                     float angle_degrees)
    {
        MvsSourceCandidate result = candidate(
            view_index,
            shared_tracks,
            90,
            angle_degrees,
            0.7f,
            0.5f);
        result.verificationStatus = MvsSourceVerificationStatus::Failed;
        result.pairTotalMatches = 100;
        result.pairCoverageScore = 0.5f;
        return result;
    };

    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 16;
    options.maxSources = 6;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 65.0f;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;
    options.failedPairBackfillMaximumTotalSources = 4;
    options.allowStrictFailedPairBackfill = true;
    options.sourceAngleSoftRankingStrength = 1.0f;
    options.auditSourceRanking = true;
    const std::vector<MvsSourceCandidate> candidates{
        verifiedCandidate(1, 500, 450, 10.0f),
        verifiedCandidate(5, 450, 400, 12.0f),
        failed_candidate(0, 300, 10.0f),
        failed_candidate(3, 250, 55.0f),
        failed_candidate(2, 160, 20.0f),
        failed_candidate(6, 120, 15.0f)};

    const auto plan = planMvsSourceViewsVerifiedFirst(candidates, options);

    ASSERT_EQ(plan.controlSelected.size(), 6u);
    ASSERT_EQ(plan.treatmentSelected.size(), 6u);
    ASSERT_EQ(plan.selected.size(), 6u);
    EXPECT_TRUE(plan.selectedCountInvariant);
    EXPECT_EQ(plan.sourceViewShortfall, 0);
    EXPECT_TRUE(plan.sourceRankingApplied);

    const auto assert_unique_tiers = [](const auto &selected)
    {
        std::set<int> views;
        for (std::size_t index = 0; index < selected.size(); ++index)
        {
            EXPECT_TRUE(views.insert(selected[index].viewIndex).second);
            if (index < 2)
            {
                EXPECT_EQ(selected[index].tier,
                          MvsSourceTier::VerifiedPair);
            }
            else if (index < 4)
            {
                EXPECT_EQ(selected[index].tier,
                          MvsSourceTier::TrackGeometryBackfill);
            }
            else
            {
                EXPECT_EQ(selected[index].tier,
                          MvsSourceTier::StrictPairAuditBackfill);
            }
        }
        EXPECT_EQ(views.size(), selected.size());
    };
    assert_unique_tiers(plan.controlSelected);
    assert_unique_tiers(plan.treatmentSelected);
    assert_unique_tiers(plan.selected);

    const auto control_wide = std::find_if(
        plan.controlSelected.cbegin(),
        plan.controlSelected.cend(),
        [](const MvsSourcePlanEntry &entry)
        {
            return entry.viewIndex == 3;
        });
    const auto treatment_wide = std::find_if(
        plan.treatmentSelected.cbegin(),
        plan.treatmentSelected.cend(),
        [](const MvsSourcePlanEntry &entry)
        {
            return entry.viewIndex == 3;
        });
    ASSERT_NE(control_wide, plan.controlSelected.cend());
    ASSERT_NE(treatment_wide, plan.treatmentSelected.cend());
    EXPECT_EQ(control_wide->tier, MvsSourceTier::TrackGeometryBackfill);
    EXPECT_EQ(treatment_wide->tier,
              MvsSourceTier::StrictPairAuditBackfill);
    EXPECT_FLOAT_EQ(control_wide->adjustedScore, control_wide->score);
    EXPECT_LT(treatment_wide->adjustedScore, treatment_wide->score);
    EXPECT_FLOAT_EQ(control_wide->rankingEffectiveMaximumDegrees, 65.0f);
    EXPECT_FLOAT_EQ(treatment_wide->rankingEffectiveMaximumDegrees, 55.0f);

    const QJsonObject diagnostics =
        xjw::mvs::mvsSourceRankingDiagnosticsToJson(
            xjw::mvs::MvsSourceRankingPolicy{
                true, true, 16, 6, 6, 6, 1.0f, 25.0f, 65.0f},
            plan);
    EXPECT_TRUE(diagnostics.value(
                    QStringLiteral("count_invariant")).toBool());
    EXPECT_FALSE(diagnostics.value(
                     QStringLiteral("selected_view_set_changed")).toBool());
    EXPECT_TRUE(diagnostics.value(
                    QStringLiteral("selection_changed")).toBool());
    EXPECT_DOUBLE_EQ(diagnostics.value(
                         QStringLiteral("soft_maximum_degrees")).toDouble(),
                     25.0);
    EXPECT_DOUBLE_EQ(diagnostics.value(
                         QStringLiteral("effective_maximum_degrees")).toDouble(),
                     65.0);
    EXPECT_EQ(diagnostics.value(
                  QStringLiteral("control_qualified_candidate_count")).toInt(),
              6);
    EXPECT_GT(diagnostics.value(
                  QStringLiteral("control_qualified_tier_entry_count")).toInt(),
              6);
    EXPECT_EQ(diagnostics.value(
                  QStringLiteral("treatment_qualified_candidate_count")).toInt(),
              6);
    EXPECT_GT(diagnostics.value(
                  QStringLiteral("treatment_qualified_tier_entry_count")).toInt(),
              6);

    const QJsonArray control_candidates = diagnostics.value(
        QStringLiteral("control_candidate_ranking")).toArray();
    int control_bounded_entry_count = 0;
    for (const QJsonValue &value : control_candidates)
    {
        const QJsonObject candidate_json = value.toObject();
        if (candidate_json.value(QStringLiteral("view_index")).toInt() != 3)
        {
            continue;
        }
        ++control_bounded_entry_count;
        EXPECT_EQ(candidate_json.value(
                      QStringLiteral("source_tier")).toString(),
                  QStringLiteral("track_geometry_backfill"));
        EXPECT_TRUE(candidate_json.value(
                        QStringLiteral("selected_by_plan")).toBool());
        EXPECT_DOUBLE_EQ(
            candidate_json.value(QStringLiteral(
                                     "ranking_effective_maximum_degrees"))
                .toDouble(),
            65.0);
    }
    EXPECT_EQ(control_bounded_entry_count, 1);

    const QJsonArray treatment_candidates = diagnostics.value(
        QStringLiteral("treatment_candidate_ranking")).toArray();
    int treatment_bounded_entry_count = 0;
    int treatment_strict_entry_count = 0;
    for (const QJsonValue &value : treatment_candidates)
    {
        const QJsonObject candidate_json = value.toObject();
        if (candidate_json.value(QStringLiteral("view_index")).toInt() != 3)
        {
            continue;
        }
        const QString tier = candidate_json.value(
            QStringLiteral("source_tier")).toString();
        if (tier == QStringLiteral("track_geometry_backfill"))
        {
            ++treatment_bounded_entry_count;
            EXPECT_FALSE(candidate_json.value(
                             QStringLiteral("selected_by_plan")).toBool());
            EXPECT_DOUBLE_EQ(
                candidate_json.value(QStringLiteral(
                                         "ranking_effective_maximum_degrees"))
                    .toDouble(),
                65.0);
        }
        else if (tier == QStringLiteral("strict_pair_audit_backfill"))
        {
            ++treatment_strict_entry_count;
            EXPECT_TRUE(candidate_json.value(
                            QStringLiteral("selected_by_plan")).toBool());
            EXPECT_DOUBLE_EQ(
                candidate_json.value(QStringLiteral(
                                         "ranking_effective_maximum_degrees"))
                    .toDouble(),
                55.0);
        }
    }
    EXPECT_EQ(treatment_bounded_entry_count, 1);
    EXPECT_EQ(treatment_strict_entry_count, 1);
}

TEST(MvsSourcePlanner,
     VerifiedFirstSoftRankingFallsBackToIncompleteControlPlan)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 3;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 35.0f;
    options.allowSequenceFallback = false;
    options.sourceAngleSoftRankingStrength = 1.0f;
    options.auditSourceRanking = true;

    const auto plan = planMvsSourceViewsVerifiedFirst({
        verifiedCandidate(3, 250, 150, 30.0f),
    }, options);

    ASSERT_EQ(plan.controlSelected.size(), 1u);
    ASSERT_EQ(plan.treatmentSelected.size(), 1u);
    ASSERT_EQ(plan.selected.size(), 1u);
    EXPECT_TRUE(plan.selectedCountInvariant);
    EXPECT_FALSE(plan.sourceRankingApplied);
    EXPECT_EQ(plan.sourceRankingDecisionReason,
              "control_source_shortfall");
    EXPECT_EQ(plan.sourceViewShortfall, 2);
    EXPECT_EQ(plan.selected.front().viewIndex,
              plan.controlSelected.front().viewIndex);
    EXPECT_EQ(plan.selected.front().tier,
              plan.controlSelected.front().tier);
    EXPECT_FLOAT_EQ(plan.selected.front().adjustedScore,
                    plan.controlSelected.front().adjustedScore);
    EXPECT_EQ(mvsSourcePlanEntryToJson(plan.selected.front()),
              mvsSourcePlanEntryToJson(plan.controlSelected.front()));
    EXPECT_FLOAT_EQ(plan.controlSelected.front().adjustedScore,
                    plan.controlSelected.front().score);
    EXPECT_FLOAT_EQ(plan.treatmentSelected.front().adjustedScore,
                    plan.treatmentSelected.front().score);
    const auto *treatment_candidate = rankingAuditForView(plan, 3);
    ASSERT_NE(treatment_candidate, nullptr);
    EXPECT_LT(treatment_candidate->candidate.adjustedScore,
              treatment_candidate->candidate.score);

    const std::set<int> control_views{
        plan.controlSelected.front().viewIndex};
    const std::set<int> treatment_views{
        plan.treatmentSelected.front().viewIndex};
    const std::set<int> selected_views{plan.selected.front().viewIndex};
    EXPECT_EQ(control_views.size(), plan.controlSelected.size());
    EXPECT_EQ(treatment_views.size(), plan.treatmentSelected.size());
    EXPECT_EQ(selected_views.size(), plan.selected.size());
}

TEST(MvsSourcePlanner,
     VerifiedFirstCountMismatchFallsBackToCompleteUniqueControlPlan)
{
    const auto failed_candidate = [](int view_index,
                                     int shared_tracks,
                                     int geometric_inliers,
                                     float angle_degrees)
    {
        MvsSourceCandidate result = candidate(
            view_index,
            shared_tracks,
            geometric_inliers,
            angle_degrees,
            0.7f,
            0.5f);
        result.verificationStatus = MvsSourceVerificationStatus::Failed;
        result.pairTotalMatches = 100;
        result.pairCoverageScore = 0.5f;
        return result;
    };

    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 16;
    options.maxSources = 6;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 35.0f;
    options.allowSequenceFallback = false;
    options.allowFailedPairBackfill = true;
    options.failedPairBackfillMaximumTotalSources = 4;
    options.allowStrictFailedPairBackfill = true;
    options.sourceAngleSoftRankingStrength = 1.0f;
    options.auditSourceRanking = true;
    const std::vector<MvsSourceCandidate> candidates{
        verifiedCandidate(1, 500, 450, 10.0f),
        verifiedCandidate(5, 450, 400, 12.0f),
        failed_candidate(0, 300, 90, 10.0f),
        // This pair passes bounded backfill (Wilson > 0.50) but fails
        // the strict tier (Wilson < 0.65).
        failed_candidate(3, 250, 70, 30.0f),
        failed_candidate(2, 160, 90, 20.0f),
        failed_candidate(6, 120, 90, 15.0f)};

    const auto plan = planMvsSourceViewsVerifiedFirst(candidates, options);

    ASSERT_EQ(plan.controlSelected.size(), 6u);
    ASSERT_EQ(plan.treatmentSelected.size(), 5u);
    ASSERT_EQ(plan.selected.size(), 6u);
    EXPECT_FALSE(plan.selectedCountInvariant);
    EXPECT_FALSE(plan.sourceRankingApplied);
    EXPECT_EQ(plan.sourceRankingDecisionReason,
              "selected_count_mismatch_fallback");
    EXPECT_EQ(plan.sourceViewShortfall, 0);

    std::set<int> control_views;
    std::set<int> treatment_views;
    std::set<int> selected_views;
    for (std::size_t index = 0; index < plan.controlSelected.size(); ++index)
    {
        control_views.insert(plan.controlSelected[index].viewIndex);
        selected_views.insert(plan.selected[index].viewIndex);
        EXPECT_EQ(plan.selected[index].viewIndex,
                  plan.controlSelected[index].viewIndex);
        EXPECT_EQ(plan.selected[index].tier,
                  plan.controlSelected[index].tier);
        EXPECT_EQ(mvsSourcePlanEntryToJson(plan.selected[index]),
                  mvsSourcePlanEntryToJson(
                      plan.controlSelected[index]));
    }
    for (const MvsSourcePlanEntry &entry : plan.treatmentSelected)
    {
        treatment_views.insert(entry.viewIndex);
    }
    EXPECT_EQ(control_views.size(), plan.controlSelected.size());
    EXPECT_EQ(treatment_views.size(), plan.treatmentSelected.size());
    EXPECT_EQ(selected_views.size(), plan.selected.size());
}

TEST(MvsSourcePlanner, DuplicateIdentityAndUnknownAngleStayLegacy)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 10;
    options.maxSources = 1;
    options.maxTriangulationAngleDeg = 35.0f;
    options.sourceAngleSoftRankingStrength = 1.0f;
    options.auditSourceRanking = true;

    const auto duplicate_plan = planMvsSourceViews({
        candidate(3, 250, 150, 30.0f),
        candidate(3, 100, 90, 10.0f),
        candidate(2, 160, 150, 20.0f),
    }, options);
    const auto *duplicate = rankingAuditForView(duplicate_plan, 3);
    ASSERT_NE(duplicate, nullptr);
    EXPECT_FLOAT_EQ(duplicate->candidate.medianTriangulationAngleDeg, 30.0f)
        << "Same-view deduplication must remain a legacy evidence decision.";

    const auto unknown_plan = planMvsSourceViews({
        candidate(3, 300, 200, std::numeric_limits<float>::quiet_NaN()),
        candidate(2, 160, 150, 20.0f),
    }, options);
    const auto *unknown = rankingAuditForView(unknown_plan, 3);
    ASSERT_NE(unknown, nullptr);
    EXPECT_FLOAT_EQ(unknown->candidate.normalizedSoftAnglePenalty, 0.0f);
    EXPECT_FLOAT_EQ(unknown->candidate.adjustedScore,
                    unknown->candidate.score);
}

TEST(MvsSourcePlanner, AdjustedRankingIsDeterministicAcrossInputOrder)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 12;
    options.maxSources = 2;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 35.0f;
    options.sourceAngleSoftRankingStrength = 1.0f;
    options.auditSourceRanking = true;
    std::vector<MvsSourceCandidate> candidates{
        candidate(1, 300, 250, 10.0f),
        candidate(3, 250, 150, 30.0f),
        candidate(2, 160, 150, 20.0f),
        candidate(6, 120, 100, 15.0f)};

    const auto first = planMvsSourceViews(candidates, options);
    std::reverse(candidates.begin(), candidates.end());
    const auto second = planMvsSourceViews(candidates, options);

    ASSERT_EQ(first.selected.size(), second.selected.size());
    for (std::size_t index = 0; index < first.selected.size(); ++index)
    {
        EXPECT_EQ(first.selected[index].viewIndex,
                  second.selected[index].viewIndex);
    }
    const xjw::mvs::MvsSourceRankingPolicy policy{
        true, true, 12, 4, 2, 4, 1.0f, 25.0f, 35.0f};
    EXPECT_EQ(
        xjw::mvs::mvsSourceRankingDiagnosticsToJson(policy, first),
        xjw::mvs::mvsSourceRankingDiagnosticsToJson(policy, second));
}

TEST(MvsSourcePlanner, SourceRankingConfigurationRejectsUnsafeCombinations)
{
    using xjw::mvs::validateMvsSourceRankingConfiguration;
    std::string error;

    EXPECT_TRUE(validateMvsSourceRankingConfiguration(
        false, 0.0f, 0.0f, &error));
    EXPECT_TRUE(validateMvsSourceRankingConfiguration(
        true, 0.0f, 25.0f, &error));
    EXPECT_TRUE(validateMvsSourceRankingConfiguration(
        true, 1.0f, 0.0f, &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        false, 1.0f, 0.0f, &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        true, 1.0f, 25.0f, &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        true, -0.1f, 0.0f, &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        true, 4.1f, 0.0f, &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        true, 0.0f, -0.1f, &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        true, 0.0f, 90.1f, &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        true,
        0.0f,
        std::numeric_limits<float>::quiet_NaN(),
        &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        true, 1.0f, -0.1f, &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        true,
        1.0f,
        std::numeric_limits<float>::quiet_NaN(),
        &error));
    EXPECT_FALSE(validateMvsSourceRankingConfiguration(
        true,
        std::numeric_limits<float>::quiet_NaN(),
        0.0f,
        &error));
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

TEST(MvsSourcePlanner, AuditedSequenceFallbackKeepsLegacyRankingFields)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 4;
    options.sourceAngleSoftRankingStrength = 1.0f;
    options.auditSourceRanking = true;

    const auto plan = planMvsSourceViews({}, options);

    ASSERT_EQ(plan.selected.size(), 4u);
    ASSERT_EQ(plan.controlSelected.size(), plan.selected.size());
    ASSERT_EQ(plan.treatmentSelected.size(), plan.selected.size());
    EXPECT_FALSE(plan.sourceRankingApplied);
    EXPECT_TRUE(plan.selectedCountInvariant);
    EXPECT_EQ(plan.sourceRankingDecisionReason,
              "sequence_fallback_unchanged");
    for (std::size_t index = 0; index < plan.selected.size(); ++index)
    {
        const MvsSourcePlanEntry &entry = plan.selected[index];
        EXPECT_TRUE(entry.sourceRankingAudited);
        EXPECT_FLOAT_EQ(entry.adjustedScore, entry.score);
        EXPECT_FLOAT_EQ(entry.normalizedSoftAnglePenalty, 0.0f);
        EXPECT_FLOAT_EQ(entry.rankingSoftMaximumDegrees, 25.0f);
        EXPECT_FLOAT_EQ(entry.rankingEffectiveMaximumDegrees, 35.0f);
        EXPECT_EQ(entry.legacyRankWithinTier, static_cast<int>(index));
        EXPECT_EQ(entry.adjustedRankWithinTier, static_cast<int>(index));

        const QJsonObject json = mvsSourcePlanEntryToJson(entry);
        EXPECT_TRUE(json.contains(QStringLiteral("legacy_score")));
        EXPECT_TRUE(json.contains(QStringLiteral("adjusted_score")));
        EXPECT_TRUE(json.contains(
            QStringLiteral("legacy_rank_within_tier")));
        EXPECT_TRUE(json.contains(
            QStringLiteral("adjusted_rank_within_tier")));
        EXPECT_TRUE(json.contains(
            QStringLiteral("ranking_soft_maximum_degrees")));
        EXPECT_TRUE(json.contains(
            QStringLiteral("ranking_effective_maximum_degrees")));
    }
}

TEST(MvsSourcePlanner,
     ExplicitAngleCapDoesNotBackfillRejectedCandidatesFromSequence)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 2;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 20.0f;
    // DepthMapGenerator disables an unmeasured sequence fallback whenever an
    // explicit source-angle cap is enabled.
    options.allowSequenceFallback = false;

    const auto plan = planMvsSourceViews({
        candidate(3, 160, 150, 24.0f),
        candidate(5, 180, 170, 31.0f),
    }, options);

    EXPECT_TRUE(plan.selected.empty());
    EXPECT_FALSE(plan.usedSequenceFallback);
    EXPECT_EQ(plan.sourceViewShortfall, 2);
    ASSERT_EQ(plan.rejected.size(), 2u);
    EXPECT_TRUE(std::all_of(
        plan.rejected.cbegin(),
        plan.rejected.cend(),
        [](const auto &rejected)
        {
            return rejected.reason ==
                MvsSourceRejectReason::TriangulationAngle;
        }));

    const QJsonObject diagnostics =
        xjw::mvs::mvsSourceAngleDiagnosticsToJson(
            xjw::mvs::MvsSourceAnglePolicy{
                20.0f, 35.0f, 20.0f, true, true, false},
            plan);
    EXPECT_EQ(diagnostics.value(QStringLiteral("scope")).toString(),
              QStringLiteral("patchmatch_source_plan"));
    EXPECT_EQ(diagnostics.value(
                  QStringLiteral("selected_source_count")).toInt(),
              0);
    EXPECT_DOUBLE_EQ(diagnostics.value(
                         QStringLiteral("selected_maximum_degrees"))
                         .toDouble(),
                     0.0);
    EXPECT_EQ(diagnostics.value(
                  QStringLiteral("angle_rejected_candidate_count")).toInt(),
              2);
    EXPECT_FALSE(diagnostics.value(
                     QStringLiteral("sequence_fallback_allowed")).toBool());
}

TEST(MvsSourcePlanner,
     SourceAngleDiagnosticsReportSelectedMaximumAndRejectedCount)
{
    MvsSourcePlannerOptions options;
    options.refIndex = 4;
    options.viewCount = 9;
    options.maxSources = 2;
    options.rejectAngleOutliers = true;
    options.maxTriangulationAngleDeg = 20.0f;
    options.allowSequenceFallback = false;

    const auto plan = planMvsSourceViews({
        candidate(3, 160, 150, 18.0f),
        candidate(5, 180, 170, 24.0f),
    }, options);
    const QJsonObject diagnostics =
        xjw::mvs::mvsSourceAngleDiagnosticsToJson(
            xjw::mvs::MvsSourceAnglePolicy{
                20.0f, 35.0f, 20.0f, true, true, false},
            plan);

    EXPECT_EQ(diagnostics.value(
                  QStringLiteral("selected_source_count")).toInt(),
              1);
    EXPECT_DOUBLE_EQ(diagnostics.value(
                         QStringLiteral("selected_maximum_degrees"))
                         .toDouble(),
                     18.0);
    EXPECT_EQ(diagnostics.value(
                  QStringLiteral("angle_rejected_candidate_count")).toInt(),
              1);
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
