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
