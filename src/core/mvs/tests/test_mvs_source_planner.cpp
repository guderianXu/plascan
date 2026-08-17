#include "MvsSourcePlanner.h"

#include <gtest/gtest.h>

using xjw::mvs::mvsSourcePlanEntryToJson;
using xjw::mvs::MvsSourceCandidate;
using xjw::mvs::MvsSourcePairQuality;
using xjw::mvs::MvsSourcePlannerOptions;
using xjw::mvs::MvsSourceRejectReason;
using xjw::mvs::MvsSourceTier;
using xjw::mvs::MvsSourceVerificationStatus;
using xjw::mvs::planMvsSourceViews;
using xjw::mvs::planMvsSourceViewsVerifiedFirst;
using xjw::mvs::planMvsRepairSourceViews;
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
