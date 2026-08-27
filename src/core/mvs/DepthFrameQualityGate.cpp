#include "DepthFrameQualityGate.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace xjw
{
namespace mvs
{

namespace
{

constexpr float kMinimumValidCoverage = 0.02f;
constexpr float kMinimumLargestComponentRatio = 0.15f;
constexpr float kMinimumRetainedFusionCoreRatio = 0.40f;
constexpr float kMinimumGeneralMultiViewConsistency = 0.45f;
constexpr float kLowConfidenceMeanThreshold = 0.45f;
constexpr float kMaximumSearchBoundaryRatio = 0.45f;
constexpr float kConsistencyFallbackCollapseRatio = 0.10f;

float unitValue(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

void lowerAcceptance(DepthFrameAcceptance requested, DepthFrameQualityDecision &decision)
{
    if (requested == DepthFrameAcceptance::Rejected ||
        (requested == DepthFrameAcceptance::ValidationOnly &&
         decision.acceptance == DepthFrameAcceptance::Accepted))
    {
        decision.acceptance = requested;
    }
}

float medianOfSortedValues(const std::vector<float> &values)
{
    if (values.empty())
    {
        return -1.0f;
    }
    const std::size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0U)
    {
        return values[middle];
    }
    return 0.5f * (values[middle - 1] + values[middle]);
}

bool hasSufficientSparseAbsoluteDepthEvidence(
    const DepthFrameQualityInput &input)
{
    const SparseDepthResidualSummary &residual = input.sparseDepthResidual;
    return residual.available &&
        residual.validSampleCount >= kSparseDepthResidualMinimumSampleCount &&
        std::isfinite(residual.medianAbsoluteLogError) &&
        residual.medianAbsoluteLogError >= 0.0f;
}

bool hasAccurateSparseAbsoluteDepthEvidence(
    const DepthFrameQualityInput &input)
{
    return hasSufficientSparseAbsoluteDepthEvidence(input) &&
           input.sparseDepthResidual.medianAbsoluteLogError <=
               sparseDepthResidualValidationThreshold(input.sceneProfile, input.filterMode);
}

bool hasStrongDiscreteGeometryCore(const DepthFrameQualityInput& input)
{
    return input.discreteGeometryCoreAvailable &&
        std::isfinite(input.discreteGeometryCoreRatio) &&
        input.discreteGeometryCoreRatio >= kDiscreteGeometryCoreMinimumRatio;
}

} // namespace

float calibrateDepthConfidence(const DepthConfidenceComponents &components)
{
    constexpr float kFloor = 0.01f;
    float weighted_log_sum = 0.0f;
    float weight_sum = 0.0f;
    const auto add_component = [&](float value, float weight)
    {
        weighted_log_sum += weight * std::log(std::max(kFloor, unitValue(value)));
        weight_sum += weight;
    };
    add_component(components.photometric, 0.22f);
    add_component(components.support, 0.13f);
    add_component(components.uniqueness, 0.10f);
    add_component(components.geometry, 0.30f);
    add_component(components.texture, 0.10f);
    if (std::isfinite(components.absoluteGeometry)
        && components.absoluteGeometry >= 0.0f)
    {
        add_component(components.absoluteGeometry, 0.25f);
    }
    return weight_sum > 0.0f
        ? unitValue(std::exp(weighted_log_sum / weight_sum))
        : 0.0f;
}

float geometryErrorConfidence(const SparseDepthResidualSummary &residual)
{
    if (!residual.available || residual.validSampleCount <= 0
        || !std::isfinite(residual.medianAbsoluteLogError)
        || residual.medianAbsoluteLogError < 0.0f)
    {
        return -1.0f;
    }

    // One confidence e-fold corresponds to one percent absolute log-depth
    // error. Sparse coverage controls how strongly this independent geometry
    // observation may move the frame confidence away from neutral evidence.
    const float error_score = std::exp(
        -residual.medianAbsoluteLogError / 0.01f);
    const float sample_strength = std::min(
        1.0f,
        static_cast<float>(residual.validSampleCount) / 64.0f);
    const float valid_ratio = residual.projectedSampleCount > 0
                                  ? std::clamp(static_cast<float>(residual.validSampleCount) /
                                                   static_cast<float>(residual.projectedSampleCount),
                                               0.0f,
                                               1.0f)
                                  : 0.0f;
    const float evidence_strength = sample_strength * valid_ratio;
    return std::pow(std::max(0.01f, error_score), evidence_strength);
}

float sparseDepthResidualValidationThreshold(MvsSceneProfile sceneProfile, DepthFilterMode filterMode)
{
    if (sceneProfile != MvsSceneProfile::OrbitalObject)
    {
        return kSparseDepthResidualValidationThreshold;
    }

    switch (filterMode)
    {
    case DepthFilterMode::Mild:
        return 0.015f;
    case DepthFilterMode::Moderate:
        return 0.008f;
    case DepthFilterMode::Aggressive:
    default:
        return kSparseDepthResidualValidationThreshold;
    }
}

bool permitsTargetedGapRecovery(MvsSceneProfile sceneProfile,
                                DepthFilterMode filterMode,
                                const SparseDepthResidualSummary &residual)
{
    if (sceneProfile != MvsSceneProfile::OrbitalObject)
    {
        return false;
    }
    if (!residual.available ||
        residual.validSampleCount < kSparseDepthResidualMinimumSampleCount ||
        !std::isfinite(residual.medianAbsoluteLogError) ||
        residual.medianAbsoluteLogError < 0.0f)
    {
        return true;
    }
    return residual.medianAbsoluteLogError <=
        sparseDepthResidualValidationThreshold(sceneProfile, filterMode);
}

DepthConsistencyPublicationSummary summarizeDepthConsistencyPublication(int preConsistencyValidCount,
                                                                        int observedPostConsistencyValidCount,
                                                                        int publishedValidCount,
                                                                        bool originalDepthFallbackApplied,
                                                                        bool consistencyStageExpected)
{
    DepthConsistencyPublicationSummary summary;
    summary.originalDepthFallbackApplied = originalDepthFallbackApplied;

    // A consistency stage is optional for a single-view batch and for frames
    // rejected before cross-view evaluation. Preserve that explicit
    // unavailable state; partially populated statistics still fail closed.
    if (!consistencyStageExpected && !originalDepthFallbackApplied && preConsistencyValidCount == -1 &&
        observedPostConsistencyValidCount == -1 && publishedValidCount == -1)
    {
        return summary;
    }

    if (preConsistencyValidCount <= 0)
    {
        summary.acceptanceCeiling = DepthFrameAcceptance::Rejected;
        summary.diagnosticReason = "invalid_consistency_pre_valid_count";
        return summary;
    }
    if (observedPostConsistencyValidCount < 0)
    {
        summary.acceptanceCeiling = DepthFrameAcceptance::Rejected;
        summary.diagnosticReason = "invalid_consistency_observed_valid_count";
        return summary;
    }
    if (publishedValidCount < 0)
    {
        summary.acceptanceCeiling = DepthFrameAcceptance::Rejected;
        summary.diagnosticReason = "invalid_consistency_published_valid_count";
        return summary;
    }

    const float denominator = static_cast<float>(preConsistencyValidCount);
    // Cross-view repair is part of the consistency stage, so the published
    // valid count may legitimately exceed the pre-stage count. Retention is a
    // loss metric; cap repair growth at one instead of treating it as invalid.
    summary.observedRetentionRatio =
        std::clamp(static_cast<float>(observedPostConsistencyValidCount) / denominator, 0.0f, 1.0f);
    summary.publishedRetentionRatio = std::clamp(static_cast<float>(publishedValidCount) / denominator, 0.0f, 1.0f);

    if (originalDepthFallbackApplied)
    {
        if (publishedValidCount != preConsistencyValidCount)
        {
            summary.acceptanceCeiling = DepthFrameAcceptance::Rejected;
            summary.diagnosticReason = "consistency_fallback_did_not_restore_pre_count";
            return summary;
        }
        if (summary.observedRetentionRatio < kConsistencyFallbackCollapseRatio)
        {
            summary.acceptanceCeiling = DepthFrameAcceptance::Rejected;
            summary.diagnosticReason = "original_depth_fallback_after_consistency_collapse";
            return summary;
        }

        summary.acceptanceCeiling = DepthFrameAcceptance::ValidationOnly;
        summary.diagnosticReason = "original_depth_fallback_not_primary";
        return summary;
    }

    if (publishedValidCount != observedPostConsistencyValidCount)
    {
        summary.acceptanceCeiling = DepthFrameAcceptance::ValidationOnly;
        summary.diagnosticReason = "consistency_publication_count_mismatch_without_fallback";
    }
    return summary;
}

void applyDepthConsistencyPublicationSummary(const DepthConsistencyPublicationSummary& summary,
                                             DepthFrameQualityInput* input)
{
    if (!input)
    {
        return;
    }
    const bool consistency_available = summary.observedRetentionRatio >= 0.0f;
    input->multiViewConsistencyAvailable = consistency_available;
    input->multiViewConsistency = consistency_available ? summary.observedRetentionRatio : 0.0f;
    input->consistencyRetentionRatio = summary.observedRetentionRatio;
    input->consistencyPublicationFallbackApplied = summary.originalDepthFallbackApplied;
    input->consistencyAcceptanceCeiling = summary.acceptanceCeiling;
    input->consistencyPublicationDiagnosticReason = summary.diagnosticReason;
}

DepthFilterSettings depthFilterSettings(DepthFilterMode mode, int availableSourceViews)
{
    DepthFilterSettings settings;
    switch (mode)
    {
    case DepthFilterMode::Mild:
        settings.minComponentArea = 8;
        settings.localDepthOutlierRelThreshold = 0.35f;
        settings.minConsistentViews = 2;
        break;
    case DepthFilterMode::Aggressive:
        settings.minComponentArea = 64;
        settings.localDepthOutlierRelThreshold = 0.15f;
        settings.minConsistentViews = 4;
        break;
    case DepthFilterMode::Moderate:
    default:
        settings.minComponentArea = 24;
        settings.localDepthOutlierRelThreshold = 0.25f;
        settings.minConsistentViews = 3;
        break;
    }

    settings.minConsistentViews = std::min(settings.minConsistentViews,
                                           std::max(1, availableSourceViews));
    return settings;
}

DepthConfidenceThresholds depthConfidenceThresholds(
    MvsSceneProfile sceneProfile,
    DepthFilterMode filterMode,
    int availableSourceViews,
    float configuredPatchMatch,
    float configuredFusion)
{
    DepthConfidenceThresholds thresholds;
    thresholds.patchMatch = std::clamp(configuredPatchMatch, 0.0f, 1.0f);
    thresholds.fusion = std::clamp(configuredFusion, 0.0f, 1.0f);
    if (sceneProfile == MvsSceneProfile::OrbitalObject &&
        filterMode == DepthFilterMode::Mild &&
        availableSourceViews > 0)
    {
        // Ring sectors and grazing views remain locally difficult even when a
        // fourth source is available.  Raising the hard validity threshold at
        // exactly four sources deletes coherent, geometrically supported
        // regions and creates view-count-dependent holes. Preserve the weaker
        // estimates for cross-view evidence and TSDF weighting; additional
        // sources improve the robust NCC itself and must not make the binary
        // photometric gate stricter.
        thresholds.patchMatch = std::min(thresholds.patchMatch, 0.50f);
        thresholds.fusion = std::min(thresholds.fusion, 0.60f);
    }
    return thresholds;
}

int minimumDepthConsistencySourceConfirmations(DepthFilterMode mode,
                                               int availableSourceViews)
{
    if (availableSourceViews <= 1)
    {
        return 0;
    }
    const DepthFilterSettings settings =
        depthFilterSettings(mode, availableSourceViews);
    return std::clamp(settings.minConsistentViews - 1,
                      1,
                      availableSourceViews);
}

int minimumDepthConsistencySourceConfirmations(MvsSceneProfile sceneProfile,
                                               DepthFilterMode mode,
                                               int availableSourceViews)
{
    const int baseline =
        minimumDepthConsistencySourceConfirmations(mode, availableSourceViews);
    if (sceneProfile == MvsSceneProfile::OrbitalObject &&
        mode == DepthFilterMode::Mild && availableSourceViews >= 4)
    {
        return std::max(2, baseline);
    }
    return baseline;
}

float depthConsistencyRelativeThreshold(MvsSceneProfile sceneProfile,
                                        int viewCount,
                                        DepthFilterMode filterMode)
{
    if (viewCount <= 1)
    {
        return 0.25f;
    }
    if (viewCount == 2)
    {
        switch (filterMode)
        {
        case DepthFilterMode::Mild:
            return 0.10f;
        case DepthFilterMode::Aggressive:
            return 0.03f;
        case DepthFilterMode::Moderate:
        default:
            return 0.06f;
        }
    }
    if (sceneProfile == MvsSceneProfile::OrbitalObject)
    {
        switch (filterMode)
        {
        case DepthFilterMode::Mild:
            return 0.0125f;
        case DepthFilterMode::Aggressive:
            return 0.005f;
        case DepthFilterMode::Moderate:
        default:
            return 0.008f;
        }
    }
    switch (filterMode)
    {
    case DepthFilterMode::Mild:
        return 0.03f;
    case DepthFilterMode::Aggressive:
        return 0.0075f;
    case DepthFilterMode::Moderate:
    default:
        return 0.015f;
    }
}

bool useContradictionOnlyDepthConsistency(int sourceViewCount)
{
    return sourceViewCount <= 1;
}

DepthConsistencyEvidence classifyDepthConsistencyEvidence(float expectedDepth,
                                                           float measuredDepth,
                                                           float relativeThreshold)
{
    if (!std::isfinite(expectedDepth) || expectedDepth <= 0.0f ||
        !std::isfinite(measuredDepth) || measuredDepth <= 0.0f)
    {
        return DepthConsistencyEvidence::Unverifiable;
    }

    const float threshold = std::max(0.0f, relativeThreshold);
    const float relative_delta = (measuredDepth - expectedDepth) / expectedDepth;
    if (std::fabs(relative_delta) <= threshold)
    {
        return DepthConsistencyEvidence::Consistent;
    }
    if (relative_delta < -threshold)
    {
        return DepthConsistencyEvidence::Occluded;
    }
    return DepthConsistencyEvidence::Contradicted;
}

DepthFrameQualityDecision evaluateDepthFrame(const DepthFrameQualityInput &input)
{
    DepthFrameQualityDecision decision;
    decision.acceptance = DepthFrameAcceptance::Accepted;
    decision.filterSettings = depthFilterSettings(input.filterMode, input.sourceViewCount);
    decision.sparseDepthResidual = input.sparseDepthResidual;
    decision.sparseDepthResidualValidationThreshold =
        sparseDepthResidualValidationThreshold(input.sceneProfile, input.filterMode);

    DepthConfidenceComponents components;
    components.photometric = input.dualChannelConfidenceAvailable && std::isfinite(input.meanPhotometricConfidence) &&
                                     input.meanPhotometricConfidence >= 0.0f
                                 ? input.meanPhotometricConfidence
                                 : input.meanConfidence;
    components.support =
        input.sourceViewCount > 0 ? std::min(1.0f, static_cast<float>(input.sourceViewCount) / 4.0f) : 0.0f;
    components.uniqueness = 1.0f - input.depthAtSearchBoundaryRatio;
    components.geometry = input.dualChannelConfidenceAvailable &&
                                  std::isfinite(input.meanIndependentGeometryConfidence) &&
                                  input.meanIndependentGeometryConfidence >= 0.0f
                              ? input.meanIndependentGeometryConfidence
                          : input.multiViewConsistencyAvailable ? input.multiViewConsistency
                                                                : 1.0f;
    components.texture = input.largestComponentRatio;
    components.absoluteGeometry = geometryErrorConfidence(input.sparseDepthResidual);
    decision.confidenceComponents = components;
    decision.calibratedConfidence = calibrateDepthConfidence(components);

    DepthFrameAcceptance consistency_acceptance_ceiling = input.consistencyAcceptanceCeiling;
    if (input.consistencyPublicationFallbackApplied && consistency_acceptance_ceiling == DepthFrameAcceptance::Accepted)
    {
        // Fail closed even when a caller records the fallback flag but forgets
        // to forward the helper's explicit ceiling.
        consistency_acceptance_ceiling = DepthFrameAcceptance::ValidationOnly;
    }
    if (consistency_acceptance_ceiling != DepthFrameAcceptance::Accepted)
    {
        lowerAcceptance(consistency_acceptance_ceiling, decision);
        if (!input.consistencyPublicationDiagnosticReason.empty())
        {
            decision.reasons.push_back(input.consistencyPublicationDiagnosticReason);
        }
        else if (input.consistencyPublicationFallbackApplied)
        {
            decision.reasons.emplace_back(consistency_acceptance_ceiling == DepthFrameAcceptance::Rejected
                                              ? "original_depth_fallback_after_consistency_collapse"
                                              : "original_depth_fallback_not_primary");
        }
        else
        {
            decision.reasons.emplace_back("invalid_consistency_publication_statistics");
        }
    }

    if (input.depthAtSearchBoundaryRatio > kMaximumSearchBoundaryRatio)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("depth_search_boundary_collapse");
    }

    if (input.validCoverage < kMinimumValidCoverage)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("insufficient_valid_coverage");
    }

    if (input.outputFilterRetentionRatio >= 0.0f
        && input.outputFilterRetentionRatio < 0.75f)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("destructive_output_filter_collapse");
    }
    else if (input.outputFilterRetentionRatio >= 0.0f
             && input.outputFilterRetentionRatio < 0.90f)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("output_filter_coverage_loss");
    }

    const bool reliable_orbital_fusion_core =
        hasReliableOrbitalFusionCore(input);
    const bool reliable_custom_fusion_core =
        hasReliableCustomFusionCore(input);
    const bool reliable_custom_sparse_surface =
        hasReliableCustomSparseAnchoredSurface(input);
    const bool reliable_fusion_core =
        reliable_orbital_fusion_core || reliable_custom_fusion_core;
    const bool reliable_causal_geometry_correction =
        hasReliableCausalGeometryCorrection(input);
    if (input.fusionPostprocessRetentionRatio >= 0.0f
        && input.fusionPostprocessRetentionRatio < 0.75f
        && !reliable_fusion_core
        && !reliable_custom_sparse_surface
        && !reliable_causal_geometry_correction)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("destructive_fusion_postprocess_collapse");
    }
    else if (input.fusionPostprocessRetentionRatio >= 0.0f
             && input.fusionPostprocessRetentionRatio < 0.75f
             && reliable_custom_sparse_surface
             && !reliable_custom_fusion_core)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back(
            "custom_sparse_anchored_surface_only");
    }
    else if (input.fusionPostprocessRetentionRatio >= 0.0f
             && input.fusionPostprocessRetentionRatio < 0.90f
             && !reliable_fusion_core
             && !reliable_causal_geometry_correction)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("fusion_postprocess_coverage_loss");
    }
    else if (input.fusionPostprocessRetentionRatio >= 0.0f
             && input.fusionPostprocessRetentionRatio < 0.90f
             && reliable_custom_fusion_core
             && !reliable_causal_geometry_correction)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back(
            "custom_fusion_fallback_to_verified_core");
    }

    if (input.hasProjectSupportMask
        && input.validWithinMaskRatio >= 0.0f
        && input.validWithinMaskRatio < 0.80f)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("insufficient_mask_normalized_coverage");
    }

    const bool aerial_edge_neighborhood =
        input.sceneProfile == MvsSceneProfile::AerialTerrain
        && input.sourceViewCount <= 5;
    const float consistency_reject_threshold =
        input.sceneProfile == MvsSceneProfile::AerialTerrain
        ? (aerial_edge_neighborhood ? 0.20f : 0.25f)
        // 环拍物体经过跨视过滤后，保留下来的深度仍是可用于表面约束的
        // 几何核心。只有接近一致性安全回退线的帧才应整帧拒绝；
        // 中等保留率帧由 validation_only 低权重参与 TSDF。
        : 0.10f;
    const float consistency_validation_threshold =
        input.sceneProfile == MvsSceneProfile::AerialTerrain
        ? (aerial_edge_neighborhood ? 0.50f : 0.55f)
        : 0.90f;
    const bool changed_complete_pool_with_verified_core =
        input.completeVisibilityCandidatePoolEnabled &&
        input.completePoolChangedLegacyPlan &&
        reliable_custom_fusion_core;
    if (input.consistencyRetentionRatio >= 0.0f
        && input.consistencyRetentionRatio < consistency_reject_threshold)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("depth_consistency_collapse");
    }
    else if (input.consistencyRetentionRatio >= 0.0f
             && input.consistencyRetentionRatio < consistency_validation_threshold
             && !changed_complete_pool_with_verified_core
             && !reliable_causal_geometry_correction)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("depth_consistency_coverage_loss");
    }

    if (input.largestComponentRatio < kMinimumLargestComponentRatio)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("fragmented_depth_support");
    }

    if (input.multiViewConsistencyAvailable &&
        input.validCoverage >= 0.95f &&
        input.meanConfidence < kLowConfidenceMeanThreshold &&
        input.multiViewConsistency < 0.50f)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("low_confidence_full_coverage");
    }

    const float consistency_threshold = input.sceneProfile == MvsSceneProfile::AerialTerrain
        ? (aerial_edge_neighborhood ? 0.50f : 0.55f)
        : kMinimumGeneralMultiViewConsistency;
    if (input.multiViewConsistencyAvailable &&
        input.sourceViewCount >= 2 &&
        input.multiViewConsistency < consistency_threshold)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("weak_multiview_consistency");
    }

    if (input.sceneProfile == MvsSceneProfile::OrbitalObject &&
        input.adaptiveGeometryEvidenceAvailable)
    {
        const bool insufficient_effective_views =
            !std::isfinite(input.adaptiveEffectiveViewCountMean) ||
            input.adaptiveEffectiveViewCountMean < 1.50f;
        const bool excessive_conflict =
            !std::isfinite(input.adaptiveConflictRatioMean) ||
            input.adaptiveConflictRatioMean > 0.60f;
        const bool use_discrete_geometry_fallback =
            (insufficient_effective_views || excessive_conflict) &&
            hasReliableOrbitalDiscreteGeometryCore(input);
        if (insufficient_effective_views && !use_discrete_geometry_fallback)
        {
            lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
            decision.reasons.emplace_back(
                "insufficient_adaptive_effective_views");
        }
        if (excessive_conflict && !use_discrete_geometry_fallback)
        {
            lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
            decision.reasons.emplace_back("excessive_adaptive_geometry_conflict");
        }
        if (use_discrete_geometry_fallback)
        {
            decision.reasons.emplace_back(
                "adaptive_geometry_fallback_to_discrete_core");
        }
    }

    const SparseDepthResidualSummary &sparse_residual =
        input.sparseDepthResidual;
    const bool has_sufficient_sparse_residual =
        hasSufficientSparseAbsoluteDepthEvidence(input);
    if (has_sufficient_sparse_residual &&
        sparse_residual.medianAbsoluteLogError >
            kSparseDepthResidualRejectionThreshold)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("sparse_absolute_depth_residual_rejected");
    }
    else if (has_sufficient_sparse_residual &&
             sparse_residual.medianAbsoluteLogError > decision.sparseDepthResidualValidationThreshold)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("sparse_absolute_depth_residual_validation_only");
    }

    const bool accurate_sparse_absolute_depth =
        hasAccurateSparseAbsoluteDepthEvidence(input);
    const bool strong_discrete_geometry_core =
        hasStrongDiscreteGeometryCore(input);
    if (input.sceneProfile == MvsSceneProfile::Custom &&
        (!accurate_sparse_absolute_depth || !strong_discrete_geometry_core))
    {
        decision.reasons.emplace_back(
            "insufficient_custom_geometry_evidence");
        if (!accurate_sparse_absolute_depth &&
            !strong_discrete_geometry_core &&
            input.multiViewConsistencyAvailable &&
            input.meanConfidence < kLowConfidenceMeanThreshold)
        {
            lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
            decision.reasons.emplace_back(
                "low_confidence_unverified_custom_depth");
        }
        else
        {
            lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        }
    }

    if (input.completeVisibilityCandidatePoolEnabled &&
        input.initialAcceptanceAvailable &&
        input.initialAcceptance != DepthFrameAcceptance::Accepted &&
        !input.completePoolChangedLegacyPlan &&
        decision.acceptance == DepthFrameAcceptance::Accepted)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back(
            "complete_pool_indirect_promotion_not_admitted");
    }

    if (input.geometryEvidenceTreatmentEnabled &&
        input.initialAcceptanceAvailable &&
        input.initialAcceptance != DepthFrameAcceptance::Accepted &&
        input.geometryCorrectedPixelCount <= 0 &&
        decision.acceptance == DepthFrameAcceptance::Accepted)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back(
            "geometry_treatment_indirect_promotion_not_admitted");
    }
    if (reliable_causal_geometry_correction)
    {
        decision.reasons.emplace_back(
            "relative_retention_explained_by_causal_geometry_core");
    }

    if (decision.reasons.empty())
    {
        decision.reasons.emplace_back("quality_gate_passed");
    }
    return decision;
}

SparseDepthResidualSummary summarizeSparseDepthResidual(
    const cv::Mat &depth,
    const std::vector<ProjectedSparseDepthSample> &samples,
    int neighborhoodRadiusPixels)
{
    SparseDepthResidualSummary summary;
    const int radius = std::clamp(neighborhoodRadiusPixels, 0, 8);
    summary.neighborhoodRadiusPixels = radius;
    if (depth.empty() || depth.type() != CV_32FC1 || samples.empty())
    {
        return summary;
    }

    std::vector<float> absolute_log_errors;
    absolute_log_errors.reserve(samples.size());
    for (const ProjectedSparseDepthSample &sample : samples)
    {
        if (!std::isfinite(sample.uNorm) || !std::isfinite(sample.vNorm) ||
            !std::isfinite(sample.depth) || sample.depth <= 0.0f)
        {
            continue;
        }

        const int column = static_cast<int>(std::lround(
            sample.uNorm * static_cast<float>(depth.cols)));
        const int row = static_cast<int>(std::lround(
            sample.vNorm * static_cast<float>(depth.rows)));
        if (column < 0 || column >= depth.cols || row < 0 || row >= depth.rows)
        {
            continue;
        }
        ++summary.projectedSampleCount;

        std::vector<float> neighborhood;
        neighborhood.reserve(static_cast<std::size_t>(
            (radius * 2 + 1) * (radius * 2 + 1)));
        for (int delta_row = -radius; delta_row <= radius; ++delta_row)
        {
            const int sample_row = row + delta_row;
            if (sample_row < 0 || sample_row >= depth.rows)
            {
                continue;
            }
            const float *depth_row = depth.ptr<float>(sample_row);
            for (int delta_column = -radius;
                 delta_column <= radius;
                 ++delta_column)
            {
                const int sample_column = column + delta_column;
                if (sample_column < 0 || sample_column >= depth.cols)
                {
                    continue;
                }
                const float value = depth_row[sample_column];
                if (std::isfinite(value) && value > 0.0f)
                {
                    neighborhood.push_back(value);
                }
            }
        }
        if (neighborhood.empty())
        {
            continue;
        }

        std::sort(neighborhood.begin(), neighborhood.end());
        const int valid_neighborhood_count =
            static_cast<int>(neighborhood.size());
        const int middle = valid_neighborhood_count / 2;
        const float neighborhood_median =
            (valid_neighborhood_count & 1) != 0
            ? neighborhood[static_cast<std::size_t>(middle)]
            : 0.5f * (neighborhood[static_cast<std::size_t>(middle - 1)] +
                      neighborhood[static_cast<std::size_t>(middle)]);
        const float absolute_log_error = std::fabs(std::log(
            neighborhood_median / sample.depth));
        if (std::isfinite(absolute_log_error))
        {
            absolute_log_errors.push_back(absolute_log_error);
        }
    }

    summary.validSampleCount = static_cast<int>(absolute_log_errors.size());
    if (absolute_log_errors.empty())
    {
        return summary;
    }
    std::sort(absolute_log_errors.begin(), absolute_log_errors.end());
    summary.medianAbsoluteLogError = medianOfSortedValues(absolute_log_errors);
    summary.available =
        summary.validSampleCount >= kSparseDepthResidualMinimumSampleCount;
    return summary;
}

bool hasReliableOrbitalFusionCore(const DepthFrameQualityInput &input)
{
    // Large orbital sequences can create a broad pre-filter hypothesis set.
    // Removing weak hypotheses is not a frame collapse when the remaining
    // surface is still broad, confident, multi-view consistent, and coherent.
    return input.sceneProfile == MvsSceneProfile::OrbitalObject
        && input.multiViewConsistencyAvailable
        && input.fusionPostprocessRetentionRatio >=
            kMinimumRetainedFusionCoreRatio
        && input.validCoverage >= 0.30f
        && input.meanConfidence >= 0.70f
        && input.multiViewConsistency >= kMinimumGeneralMultiViewConsistency
        && input.largestComponentRatio >= kMinimumLargestComponentRatio
        && input.depthAtSearchBoundaryRatio <= kMaximumSearchBoundaryRatio;
}

bool hasReliableCustomFusionCore(const DepthFrameQualityInput &input)
{
    // A generic capture has no ring prior that can justify a purely relative
    // fallback. Require independent absolute anchors and a dense discrete
    // multi-view core before treating low postprocess retention as removal of
    // weak hypotheses rather than collapse of the reconstructed surface.
    return input.sceneProfile == MvsSceneProfile::Custom &&
        input.multiViewConsistencyAvailable &&
        input.fusionPostprocessRetentionRatio >=
            kMinimumRetainedFusionCoreRatio &&
        input.validCoverage >= kMinimumValidCoverage &&
        input.meanConfidence >= kLowConfidenceMeanThreshold &&
        input.multiViewConsistency >= kMinimumGeneralMultiViewConsistency &&
        input.largestComponentRatio >= kMinimumLargestComponentRatio &&
        input.depthAtSearchBoundaryRatio <= kMaximumSearchBoundaryRatio &&
        hasAccurateSparseAbsoluteDepthEvidence(input) &&
        hasStrongDiscreteGeometryCore(input);
}

bool hasReliableCustomSparseAnchoredSurface(
    const DepthFrameQualityInput &input)
{
    // Unlike the full Custom fusion-core exception, this path deliberately
    // does not require a discrete three-view core.  It therefore cannot grant
    // Primary; evaluateDepthFrame later retains it as ValidationOnly.  The
    // remaining conditions are all absolute/product-domain safety evidence,
    // not a relaxation of the relative postprocess-retention threshold.
    return input.sceneProfile == MvsSceneProfile::Custom &&
        input.multiViewConsistencyAvailable &&
        input.validCoverage >= kMinimumValidCoverage &&
        input.meanConfidence >= kLowConfidenceMeanThreshold &&
        input.multiViewConsistency >= kMinimumGeneralMultiViewConsistency &&
        input.largestComponentRatio >= kMinimumLargestComponentRatio &&
        input.depthAtSearchBoundaryRatio <= kMaximumSearchBoundaryRatio &&
        hasAccurateSparseAbsoluteDepthEvidence(input);
}

bool hasReliableCausalGeometryCorrection(
    const DepthFrameQualityInput &input)
{
    return input.geometryEvidenceTreatmentEnabled &&
        input.geometryCorrectedPixelCount > 0 &&
        input.dualChannelConfidenceAvailable &&
        std::isfinite(input.correctedMeanIndependentGeometryConfidence) &&
        input.correctedMeanIndependentGeometryConfidence >= 0.55f &&
        std::isfinite(input.strongIndependentGeometryCoverage) &&
        input.strongIndependentGeometryCoverage > 0.0f &&
        input.validCoverage >= kMinimumValidCoverage &&
        input.largestComponentRatio >= kMinimumLargestComponentRatio &&
        input.depthAtSearchBoundaryRatio <= kMaximumSearchBoundaryRatio &&
        hasAccurateSparseAbsoluteDepthEvidence(input) &&
        hasStrongDiscreteGeometryCore(input);
}

bool hasReliableOrbitalDiscreteGeometryCore(
    const DepthFrameQualityInput &input)
{
    return hasReliableOrbitalFusionCore(input)
        && hasStrongDiscreteGeometryCore(input);
}

const char *depthFrameAcceptanceId(DepthFrameAcceptance acceptance)
{
    switch (acceptance)
    {
    case DepthFrameAcceptance::Accepted:
        return "accepted";
    case DepthFrameAcceptance::ValidationOnly:
        return "validation_only";
    case DepthFrameAcceptance::Rejected:
    default:
        return "rejected";
    }
}

} // namespace mvs
} // namespace xjw
