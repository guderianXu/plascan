#include "DepthFrameQualityGate.h"
#include "MvsQualityReport.h"
#include "MvsWorkspaceManifest.h"

#include <QJsonArray>
#include <QJsonObject>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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

xjw::mvs::DepthFrameQualityInput reliableCustomInput()
{
    xjw::mvs::DepthFrameQualityInput input;
    input.sceneProfile = xjw::mvs::MvsSceneProfile::Custom;
    input.sourceViewCount = 4;
    input.validCoverage = 0.21f;
    input.largestComponentRatio = 0.38f;
    input.meanConfidence = 0.65f;
    input.multiViewConsistency = 0.65f;
    input.depthAtSearchBoundaryRatio = 0.03f;
    input.outputFilterRetentionRatio = 1.0f;
    input.consistencyRetentionRatio = 0.65f;
    input.fusionPostprocessRetentionRatio = 0.62f;
    input.sparseDepthResidual.available = true;
    input.sparseDepthResidual.projectedSampleCount = 2700;
    input.sparseDepthResidual.validSampleCount = 1900;
    input.sparseDepthResidual.medianAbsoluteLogError = 0.0029f;
    input.discreteGeometryCoreAvailable = true;
    input.discreteGeometryCoreRatio = 0.94f;
    return input;
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

TEST(DepthFrameQualityGateTest,
     KeepsVerifiedCustomCoreForValidationAfterFusionReduction)
{
    const auto input = reliableCustomInput();

    EXPECT_TRUE(xjw::mvs::hasReliableCustomFusionCore(input));
    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_EQ(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("destructive_fusion_postprocess_collapse")),
              decision.reasons.end());
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("custom_fusion_fallback_to_verified_core")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     RejectsCustomFrameWhenVerifiedCoreRetentionActuallyCollapses)
{
    auto input = reliableCustomInput();
    input.fusionPostprocessRetentionRatio = 0.39f;
    input.sparseDepthResidual = {};

    EXPECT_FALSE(xjw::mvs::hasReliableCustomFusionCore(input));
    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("destructive_fusion_postprocess_collapse")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     CompletePoolCannotIndirectlyPromoteAnUnchangedProvisionalPlan)
{
    auto input = reliableCustomInput();
    input.multiViewConsistency = 0.95f;
    input.consistencyRetentionRatio = 0.95f;
    input.fusionPostprocessRetentionRatio = 1.0f;
    input.completeVisibilityCandidatePoolEnabled = true;
    input.completePoolChangedLegacyPlan = false;
    input.initialAcceptanceAvailable = true;
    input.initialAcceptance =
        xjw::mvs::DepthFrameAcceptance::ValidationOnly;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string(
                      "complete_pool_indirect_promotion_not_admitted")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     ChangedCompletePoolPlanUsesJointVerifiedCoreInsteadOfRawRetentionAlone)
{
    auto input = reliableCustomInput();
    input.meanConfidence = 0.78f;
    input.multiViewConsistency = 0.894f;
    input.consistencyRetentionRatio = 0.894f;
    input.fusionPostprocessRetentionRatio = 0.944f;
    input.completeVisibilityCandidatePoolEnabled = true;
    input.completePoolChangedLegacyPlan = true;
    input.initialAcceptanceAvailable = true;
    input.initialAcceptance = xjw::mvs::DepthFrameAcceptance::Accepted;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Accepted);
    EXPECT_EQ(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("depth_consistency_coverage_loss")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     SparseAnchoredCustomSurfaceWithoutDiscreteCoreRemainsAuxiliary)
{
    auto input = reliableCustomInput();
    input.validCoverage = 0.074f;
    input.largestComponentRatio = 0.649f;
    input.meanConfidence = 0.741f;
    input.multiViewConsistency = 0.992f;
    input.consistencyRetentionRatio = 0.992f;
    input.fusionPostprocessRetentionRatio = 0.288f;
    input.discreteGeometryCoreAvailable = true;
    input.discreteGeometryCoreRatio = 0.0f;

    EXPECT_TRUE(
        xjw::mvs::hasReliableCustomSparseAnchoredSurface(input));
    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("custom_sparse_anchored_surface_only")),
              decision.reasons.end());
    EXPECT_EQ(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("destructive_fusion_postprocess_collapse")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     RejectsLowConfidenceCustomFrameWithoutObservableGeometryCore)
{
    auto input = reliableCustomInput();
    input.validCoverage = 0.173f;
    input.largestComponentRatio = 0.957f;
    input.meanConfidence = 0.346f;
    input.multiViewConsistency = 0.943f;
    input.depthAtSearchBoundaryRatio = 0.058f;
    input.consistencyRetentionRatio = 0.943f;
    input.fusionPostprocessRetentionRatio = 1.0f;
    input.sparseDepthResidual.available = false;
    input.sparseDepthResidual.projectedSampleCount = 729;
    input.sparseDepthResidual.validSampleCount = 19;
    input.sparseDepthResidual.medianAbsoluteLogError = 0.00546f;
    input.discreteGeometryCoreRatio = 0.00092f;

    EXPECT_FALSE(xjw::mvs::hasReliableCustomFusionCore(input));
    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("insufficient_custom_geometry_evidence")),
              decision.reasons.end());
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("low_confidence_unverified_custom_depth")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     KeepsUnverifiedButOtherwiseStableCustomFrameValidationOnly)
{
    auto input = reliableCustomInput();
    input.meanConfidence = 0.70f;
    input.multiViewConsistency = 0.95f;
    input.consistencyRetentionRatio = 0.95f;
    input.fusionPostprocessRetentionRatio = 1.0f;
    input.sparseDepthResidual = {};
    input.discreteGeometryCoreAvailable = true;
    input.discreteGeometryCoreRatio = 0.20f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("insufficient_custom_geometry_evidence")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     MissingSparseEvidenceKeepsStrongCustomCoreValidationOnly)
{
    auto input = reliableCustomInput();
    input.multiViewConsistency = 0.95f;
    input.consistencyRetentionRatio = 0.95f;
    input.fusionPostprocessRetentionRatio = 1.0f;
    input.sparseDepthResidual = {};

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("insufficient_custom_geometry_evidence")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     MissingDiscreteCoreKeepsAccurateSparseCustomFrameValidationOnly)
{
    auto input = reliableCustomInput();
    input.multiViewConsistency = 0.95f;
    input.consistencyRetentionRatio = 0.95f;
    input.fusionPostprocessRetentionRatio = 1.0f;
    input.discreteGeometryCoreAvailable = true;
    input.discreteGeometryCoreRatio = 0.20f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("insufficient_custom_geometry_evidence")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     KeepsPreGeometryCustomFrameProvisionalWithoutFailingOpen)
{
    auto input = reliableCustomInput();
    input.meanConfidence = 0.34f;
    input.multiViewConsistencyAvailable = false;
    input.multiViewConsistency = 0.0f;
    input.consistencyRetentionRatio = -1.0f;
    input.fusionPostprocessRetentionRatio = -1.0f;
    input.sparseDepthResidual = {};
    input.discreteGeometryCoreAvailable = false;
    input.discreteGeometryCoreRatio = -1.0f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("insufficient_custom_geometry_evidence")),
              decision.reasons.end());
    EXPECT_EQ(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("low_confidence_unverified_custom_depth")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     SparseAbsoluteDepthResidualUsesValidThreeByThreeMedian)
{
    cv::Mat depth(30, 30, CV_32FC1, cv::Scalar(10.0f));
    depth.at<float>(5, 5) = 100.0f;
    std::vector<xjw::mvs::ProjectedSparseDepthSample> samples;
    for (int row = 0; row < 5; ++row)
    {
        for (int column = 0; column < 5; ++column)
        {
            samples.push_back({
                static_cast<float>(5 + column * 4) /
                    static_cast<float>(depth.cols),
                static_cast<float>(5 + row * 4) /
                    static_cast<float>(depth.rows),
                10.0f});
        }
    }

    const auto summary = xjw::mvs::summarizeSparseDepthResidual(
        depth, samples);

    EXPECT_TRUE(summary.available);
    EXPECT_EQ(summary.projectedSampleCount, 25);
    EXPECT_EQ(summary.validSampleCount, 25);
    EXPECT_NEAR(summary.medianAbsoluteLogError, 0.0f, 1e-7f);
}

TEST(DepthFrameQualityGateTest,
     SparseAbsoluteDepthResidualCanUseScaledZeroRadius)
{
    cv::Mat depth(10, 10, CV_32FC1, cv::Scalar(0.0f));
    depth.at<float>(5, 6) = 10.0f;
    const std::vector<xjw::mvs::ProjectedSparseDepthSample> samples = {
        {0.5f, 0.5f, 10.0f}};

    const auto full_radius = xjw::mvs::summarizeSparseDepthResidual(
        depth, samples, 1);
    const auto scaled_radius = xjw::mvs::summarizeSparseDepthResidual(
        depth, samples, 0);

    EXPECT_EQ(full_radius.neighborhoodRadiusPixels, 1);
    EXPECT_EQ(full_radius.validSampleCount, 1);
    EXPECT_EQ(scaled_radius.neighborhoodRadiusPixels, 0);
    EXPECT_EQ(scaled_radius.projectedSampleCount, 1);
    EXPECT_EQ(scaled_radius.validSampleCount, 0);
}

TEST(DepthFrameQualityGateTest,
     RejectsSufficientSparseAbsoluteDepthResidualAboveTwoPercent)
{
    auto input = reliableOrbitalInput();
    input.sparseDepthResidual.available = true;
    input.sparseDepthResidual.projectedSampleCount = 30;
    input.sparseDepthResidual.validSampleCount = 25;
    input.sparseDepthResidual.medianAbsoluteLogError = 0.021f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("sparse_absolute_depth_residual_rejected")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     AbsoluteGeometryConfidenceIsMonotonicWithMeasuredError)
{
    xjw::mvs::SparseDepthResidualSummary accurate;
    accurate.available = true;
    accurate.projectedSampleCount = 80;
    accurate.validSampleCount = 64;
    accurate.medianAbsoluteLogError = 0.003f;
    auto inaccurate = accurate;
    inaccurate.medianAbsoluteLogError = 0.020f;

    EXPECT_GT(xjw::mvs::geometryErrorConfidence(accurate),
              xjw::mvs::geometryErrorConfidence(inaccurate));
    auto accurate_input = reliableOrbitalInput();
    accurate_input.sparseDepthResidual = accurate;
    auto inaccurate_input = reliableOrbitalInput();
    inaccurate_input.sparseDepthResidual = inaccurate;
    EXPECT_GT(xjw::mvs::evaluateDepthFrame(accurate_input).calibratedConfidence,
              xjw::mvs::evaluateDepthFrame(inaccurate_input).calibratedConfidence);
}

TEST(DepthFrameQualityGateTest,
     MakesIntermediateSparseAbsoluteDepthResidualValidationOnly)
{
    auto input = reliableOrbitalInput();
    input.sparseDepthResidual.available = true;
    input.sparseDepthResidual.projectedSampleCount = 30;
    input.sparseDepthResidual.validSampleCount = 25;
    input.sparseDepthResidual.medianAbsoluteLogError = 0.010f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string(
                      "sparse_absolute_depth_residual_validation_only")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     IgnoresSparseAbsoluteDepthResidualWithoutEnoughValidSamples)
{
    auto input = reliableOrbitalInput();
    input.sparseDepthResidual.available = true;
    input.sparseDepthResidual.projectedSampleCount = 30;
    input.sparseDepthResidual.validSampleCount =
        xjw::mvs::kSparseDepthResidualMinimumSampleCount - 1;
    input.sparseDepthResidual.medianAbsoluteLogError = 0.20f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Accepted);
}

TEST(DepthFrameQualityGateTest,
     SerializesSparseAbsoluteDepthResidualAuditFields)
{
    auto input = reliableOrbitalInput();
    input.sparseDepthResidual.available = true;
    input.sparseDepthResidual.projectedSampleCount = 40;
    input.sparseDepthResidual.validSampleCount = 30;
    input.sparseDepthResidual.medianAbsoluteLogError = 0.010f;

    const QJsonObject decision_json =
        xjw::mvs::depthFrameQualityDecisionToJson(
            xjw::mvs::evaluateDepthFrame(input));
    const QJsonObject residual_json = decision_json.value(
        QStringLiteral("sparse_absolute_depth_residual")).toObject();

    EXPECT_TRUE(residual_json.value(QStringLiteral("available")).toBool());
    EXPECT_EQ(residual_json.value(
                  QStringLiteral("projected_sample_count")).toInt(),
              40);
    EXPECT_EQ(residual_json.value(
                  QStringLiteral("valid_sample_count")).toInt(),
              30);
    EXPECT_DOUBLE_EQ(residual_json.value(
                         QStringLiteral("valid_sample_ratio")).toDouble(),
                     0.75);
    EXPECT_NEAR(residual_json.value(
                    QStringLiteral("median_absolute_log_error")).toDouble(),
                0.010,
                1e-7);
    EXPECT_EQ(residual_json.value(
                  QStringLiteral("minimum_sample_count")).toInt(),
              xjw::mvs::kSparseDepthResidualMinimumSampleCount);
}

TEST(DepthFrameQualityGateTest,
     KeepsConflictingOrbitalDepthOutOfPrimaryFusion)
{
    auto input = reliableOrbitalInput();
    input.adaptiveGeometryEvidenceAvailable = true;
    input.adaptiveEffectiveViewCountMean = 1.42f;
    input.adaptiveConflictRatioMean = 0.65f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("insufficient_adaptive_effective_views")),
              decision.reasons.end());
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("excessive_adaptive_geometry_conflict")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     AcceptsReliableAdaptiveOrbitalEvidence)
{
    auto input = reliableOrbitalInput();
    input.adaptiveGeometryEvidenceAvailable = true;
    input.adaptiveEffectiveViewCountMean = 2.45f;
    input.adaptiveConflictRatioMean = 0.42f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Accepted);
}

TEST(DepthFrameQualityGateTest,
     UsesFinalDiscreteCoreWhenAdaptiveResidualIsMiscalibrated)
{
    auto input = reliableOrbitalInput();
    input.adaptiveGeometryEvidenceAvailable = true;
    input.adaptiveEffectiveViewCountMean = 1.20f;
    input.adaptiveConflictRatioMean = 0.91f;
    input.discreteGeometryCoreAvailable = true;
    input.discreteGeometryCoreRatio = 0.78f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Accepted);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("adaptive_geometry_fallback_to_discrete_core")),
              decision.reasons.end());
    EXPECT_EQ(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("excessive_adaptive_geometry_conflict")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     WeakDiscreteCoreCannotBypassAdaptiveGeometryFailure)
{
    auto input = reliableOrbitalInput();
    input.adaptiveGeometryEvidenceAvailable = true;
    input.adaptiveEffectiveViewCountMean = 1.20f;
    input.adaptiveConflictRatioMean = 0.91f;
    input.discreteGeometryCoreAvailable = true;
    input.discreteGeometryCoreRatio = 0.42f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::ValidationOnly);
}

TEST(DepthFrameQualityGateTest,
     DiscreteFallbackDoesNotOverrideFragmentedSurfaceRejection)
{
    auto input = reliableOrbitalInput();
    input.largestComponentRatio = 0.10f;
    input.adaptiveGeometryEvidenceAvailable = true;
    input.adaptiveEffectiveViewCountMean = 1.20f;
    input.adaptiveConflictRatioMean = 0.91f;
    input.discreteGeometryCoreAvailable = true;
    input.discreteGeometryCoreRatio = 0.90f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("fragmented_depth_support")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     DoesNotInferGeometryFromPhotometricConfidenceWhenUnavailable)
{
    auto input = reliableOrbitalInput();
    input.sceneProfile = xjw::mvs::MvsSceneProfile::AerialTerrain;
    input.validCoverage = 0.99f;
    input.largestComponentRatio = 0.99f;
    input.meanConfidence = 0.31f;
    input.multiViewConsistencyAvailable = false;
    input.multiViewConsistency = 0.0f;
    input.consistencyRetentionRatio = -1.0f;
    input.fusionPostprocessRetentionRatio = -1.0f;

    const auto decision = xjw::mvs::evaluateDepthFrame(input);

    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Accepted);
    EXPECT_EQ(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("low_confidence_full_coverage")),
              decision.reasons.end());
    EXPECT_EQ(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("weak_multiview_consistency")),
              decision.reasons.end());
}

TEST(DepthFrameQualityGateTest,
     UnavailableGeometryCannotQualifyOrbitalFusionCore)
{
    auto input = reliableOrbitalInput();
    input.multiViewConsistencyAvailable = false;

    EXPECT_FALSE(xjw::mvs::hasReliableOrbitalFusionCore(input));

    const auto decision = xjw::mvs::evaluateDepthFrame(input);
    EXPECT_EQ(decision.acceptance,
              xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_NE(std::find(
                  decision.reasons.begin(),
                  decision.reasons.end(),
                  std::string("destructive_fusion_postprocess_collapse")),
              decision.reasons.end());
}
