#include <gtest/gtest.h>

#include "preparation/ReconstructionPrerequisiteReport.h"

#include <QJsonObject>

using xjw::aerial_triangulation::ReconstructionPrerequisiteReport;
using xjw::aerial_triangulation::ReconstructionPrerequisiteRecommendedAction;
using xjw::aerial_triangulation::reconstructionPrerequisiteActionToString;

TEST(ReconstructionPrerequisiteReportTest, CompleteMatchesAreReadyForTriangulation)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 984;
    report.settledNoMatchPairCount = 584;
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
    report.missingMatchPairCount = 0;
    report.failedGeometryPairCount = 1568;

    EXPECT_FALSE(report.hasEnoughUpstreamData());
    EXPECT_FALSE(report.shouldRunFullRematch())
        << "A completed matching pass with settled failures is a quality problem, not missing upstream data.";
    EXPECT_FALSE(report.shouldOfferGapFill());
    EXPECT_EQ(reconstructionPrerequisiteActionToString(report.recommendedAction()),
              QStringLiteral("inspect_match_quality"));
}

TEST(ReconstructionPrerequisiteReportTest, MissingImageMatchesRequireFullMatchPreparation)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 0;
    report.settledNoMatchPairCount = 0;
    report.missingMatchPairCount = 1568;

    EXPECT_FALSE(report.hasEnoughUpstreamData());
    EXPECT_TRUE(report.shouldRunFullRematch());
    EXPECT_FALSE(report.shouldOfferGapFill());
    EXPECT_EQ(report.recommendedAction(), ReconstructionPrerequisiteRecommendedAction::PrepareImageMatches);
}

TEST(ReconstructionPrerequisiteReportTest, SerializesStableJsonForGuiAndReports)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 984;
    report.settledNoMatchPairCount = 584;
    report.missingMatchPairCount = 0;
    report.failedGeometryPairCount = 584;

    const QJsonObject json = report.toJson();
    EXPECT_EQ(json.value(QStringLiteral("image_count")).toInt(), 444);
    EXPECT_EQ(json.value(QStringLiteral("planned_pair_count")).toInt(), 1568);
    EXPECT_EQ(json.value(QStringLiteral("valid_match_pair_count")).toInt(), 984);
    EXPECT_EQ(json.value(QStringLiteral("settled_no_match_pair_count")).toInt(), 584);
    EXPECT_FALSE(json.contains(QStringLiteral("missing_feature_pair_count")));
    EXPECT_EQ(json.value(QStringLiteral("missing_match_pair_count")).toInt(), 0);
    EXPECT_EQ(json.value(QStringLiteral("failed_geometry_pair_count")).toInt(), 584);
    EXPECT_EQ(json.value(QStringLiteral("recommended_action")).toString(),
              QStringLiteral("run_sfm_with_existing_matches"));
}
