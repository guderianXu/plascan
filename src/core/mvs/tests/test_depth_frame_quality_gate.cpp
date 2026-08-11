#include "DepthFrameQualityGate.h"
#include "MvsWorkspaceManifest.h"

#include <QJsonArray>
#include <QJsonObject>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace
{

xjw::mvs::DepthFrameQualityInput reliableOrbitalInput()
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::OrbitalObject;
    input.sourceViewCount = 6;
    input.validCoverage = 0.389f;
    input.largestComponentRatio = 0.73f;
    input.meanConfidence = 0.86f;
    input.multiViewConsistency = 0.95f;
    input.depthAtSearchBoundaryRatio = 0.07f;
    input.outputFilterRetentionRatio = 0.999f;
    input.consistencyRetentionRatio = 0.95f;
    input.fusionPostprocessRetentionRatio = 0.48f;
    return input;
}

QJsonObject storedOrbitalArtifact()
{
    return QJsonObject{
        {QStringLiteral("status"), QStringLiteral("completed")},
        {QStringLiteral("scene_profile"), QStringLiteral("orbital_object")},
        {QStringLiteral("acceptance"), QStringLiteral("rejected")},
        {QStringLiteral("fusion_eligible"), false},
        {QStringLiteral("valid_coverage"), 0.389},
        {QStringLiteral("quality_decision"), QJsonObject{
            {QStringLiteral("acceptance"), QStringLiteral("rejected")},
            {QStringLiteral("reasons"), QJsonArray{
                QStringLiteral("destructive_fusion_postprocess_collapse")}}
        }},
        {QStringLiteral("depth_quality"), QJsonObject{
            {QStringLiteral("largest_component_ratio"), 0.73},
            {QStringLiteral("mean_confidence"), 0.86},
            {QStringLiteral("depth_at_search_boundary_ratio"), 0.07}
        }},
        {QStringLiteral("depth_completeness"), QJsonObject{
            {QStringLiteral("consistency_retention_ratio"), 0.95},
            {QStringLiteral("fusion_postprocess_retention_ratio"), 0.48}
        }}
    };
}

} // namespace

TEST(DepthFrameQualityGateTest,
     AcceptsReliableOrbitalCoreAfterLargeHypothesisReduction)
{
    const auto decision = xjw::mvs::evaluateDepthFrame(
        reliableOrbitalInput());

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Accepted);
    EXPECT_EQ(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("destructive_fusion_postprocess_collapse")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     StillRejectsOrbitalFrameWhenRetainedCoreActuallyCollapses)
{
    auto input = reliableOrbitalInput();
    input.fusionPostprocessRetentionRatio = 0.23f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Rejected);
}

TEST(DepthFrameQualityGateTest,
     DoesNotRelaxFusionPostprocessGateForAerialTerrain)
{
    auto input = reliableOrbitalInput();
    input.sceneProfile = xjw::mvs::MvsSceneProfile::AerialTerrain;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Rejected);
}

TEST(DepthFrameQualityMigrationTest,
     ReclassifiesStoredOrbitalFrameRejectedOnlyByLegacyRetentionGate)
{
    const auto qualification = xjw::mvs::qualifyMvsDepthFrameArtifact(
        storedOrbitalArtifact());

    EXPECT_TRUE(qualification.reclassified);
    EXPECT_TRUE(qualification.fusionEligible);
    EXPECT_EQ(qualification.acceptance, QStringLiteral("accepted"));
}

TEST(DepthFrameQualityMigrationTest,
     PreservesStoredRejectionWhenAnotherQualityFailureIsPresent)
{
    QJsonObject artifact = storedOrbitalArtifact();
    QJsonObject decision = artifact.value(
        QStringLiteral("quality_decision")).toObject();
    decision[QStringLiteral("reasons")] = QJsonArray{
        QStringLiteral("destructive_fusion_postprocess_collapse"),
        QStringLiteral("fragmented_depth_support")};
    artifact[QStringLiteral("quality_decision")] = decision;

    const auto qualification = xjw::mvs::qualifyMvsDepthFrameArtifact(
        artifact);

    EXPECT_FALSE(qualification.reclassified);
    EXPECT_FALSE(qualification.fusionEligible);
    EXPECT_EQ(qualification.acceptance, QStringLiteral("rejected"));
}
