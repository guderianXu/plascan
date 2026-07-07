#include <gtest/gtest.h>

#include "ReconstructionPrerequisiteReport.h"

#include <QJsonObject>

using xjw::gui::ReconstructionPrerequisiteReport;
using xjw::gui::ReconstructionPrerequisiteRecommendedAction;
using xjw::gui::reconstructionPrerequisiteActionToString;

TEST(ReconstructionPrerequisiteReportTest, CompleteMatchesAreReadyForTriangulation)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 984;
    report.settledNoMatchPairCount = 584;
    report.missingFeaturePairCount = 0;
    report.missingMatchPairCount = 0;
    report.failedGeometryPairCount = 584;

    EXPECT_TRUE(report.hasEnoughUpstreamData());
    EXPECT_FALSE(report.shouldRunFullRematch());
    EXPECT_FALSE(report.shouldOfferGapFill());
    EXPECT_EQ(report.recommendedAction(), ReconstructionPrerequisiteRecommendedAction::RunSfmWithExistingMatches);
}

TEST(ReconstructionPrerequisiteReportTest, MissingMatchesRequireOnlyGapFill)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 900;
    report.settledNoMatchPairCount = 584;
    report.missingFeaturePairCount = 0;
    report.missingMatchPairCount = 84;
    report.failedGeometryPairCount = 584;

    EXPECT_TRUE(report.hasEnoughUpstreamData());
    EXPECT_TRUE(report.shouldOfferGapFill());
    EXPECT_FALSE(report.shouldRunFullRematch());
    EXPECT_EQ(report.recommendedAction(), ReconstructionPrerequisiteRecommendedAction::FillMissingMatchesOnly);
}

TEST(ReconstructionPrerequisiteReportTest, CompletedMatchingWithNoUsableEdgesRequiresQualityInspection)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 0;
    report.settledNoMatchPairCount = 1568;
    report.missingFeaturePairCount = 0;
    report.missingMatchPairCount = 0;
    report.failedGeometryPairCount = 1568;

    EXPECT_FALSE(report.hasEnoughUpstreamData());
    EXPECT_FALSE(report.shouldRunFullRematch())
        << "A completed matching pass with settled failures is a quality problem, not missing upstream data.";
    EXPECT_FALSE(report.shouldOfferGapFill());
    EXPECT_EQ(reconstructionPrerequisiteActionToString(report.recommendedAction()),
              QStringLiteral("inspect_match_quality"));
}

TEST(ReconstructionPrerequisiteReportTest, MissingFeaturesRequireTiePointPreparation)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 900;
    report.settledNoMatchPairCount = 0;
    report.missingFeaturePairCount = 25;
    report.missingMatchPairCount = 84;

    EXPECT_FALSE(report.hasEnoughUpstreamData());
    EXPECT_TRUE(report.shouldRunFullRematch());
    EXPECT_FALSE(report.shouldOfferGapFill());
    EXPECT_EQ(report.recommendedAction(), ReconstructionPrerequisiteRecommendedAction::PrepareFeaturesAndMatches);
}

TEST(ReconstructionPrerequisiteReportTest, SerializesStableJsonForGuiAndReports)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 984;
    report.settledNoMatchPairCount = 584;
    report.missingFeaturePairCount = 0;
    report.missingMatchPairCount = 0;
    report.failedGeometryPairCount = 584;

    const QJsonObject json = report.toJson();
    EXPECT_EQ(json.value(QStringLiteral("image_count")).toInt(), 444);
    EXPECT_EQ(json.value(QStringLiteral("planned_pair_count")).toInt(), 1568);
    EXPECT_EQ(json.value(QStringLiteral("valid_match_pair_count")).toInt(), 984);
    EXPECT_EQ(json.value(QStringLiteral("settled_no_match_pair_count")).toInt(), 584);
    EXPECT_EQ(json.value(QStringLiteral("missing_feature_pair_count")).toInt(), 0);
    EXPECT_EQ(json.value(QStringLiteral("missing_match_pair_count")).toInt(), 0);
    EXPECT_EQ(json.value(QStringLiteral("failed_geometry_pair_count")).toInt(), 584);
    EXPECT_EQ(json.value(QStringLiteral("recommended_action")).toString(),
              QStringLiteral("run_sfm_with_existing_matches"));
}
