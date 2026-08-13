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

} // namespace

float calibrateDepthConfidence(const DepthConfidenceComponents &components)
{
    return unitValue(components.photometric) *
           unitValue(components.support) *
           unitValue(components.uniqueness) *
           unitValue(components.geometry) *
           unitValue(components.texture);
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

    DepthConfidenceComponents components;
    components.photometric = input.meanConfidence;
    components.support = input.sourceViewCount > 0
        ? std::min(1.0f, static_cast<float>(input.sourceViewCount) / 4.0f)
        : 0.0f;
    components.uniqueness = 1.0f - input.depthAtSearchBoundaryRatio;
    components.geometry = input.multiViewConsistencyAvailable
        ? input.multiViewConsistency
        : 1.0f;
    components.texture = input.largestComponentRatio;
    decision.calibratedConfidence = calibrateDepthConfidence(components);

    if (input.depthAtSearchBoundaryRatio > 0.45f)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("depth_search_boundary_collapse");
    }

    if (input.validCoverage < 0.02f)
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
    if (input.fusionPostprocessRetentionRatio >= 0.0f
        && input.fusionPostprocessRetentionRatio < 0.75f
        && !reliable_orbital_fusion_core)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("destructive_fusion_postprocess_collapse");
    }
    else if (input.fusionPostprocessRetentionRatio >= 0.0f
             && input.fusionPostprocessRetentionRatio < 0.90f
             && !reliable_orbital_fusion_core)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("fusion_postprocess_coverage_loss");
    }

    if (input.hasConstrainedSupportMask
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
    if (input.consistencyRetentionRatio >= 0.0f
        && input.consistencyRetentionRatio < consistency_reject_threshold)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("depth_consistency_collapse");
    }
    else if (input.consistencyRetentionRatio >= 0.0f
             && input.consistencyRetentionRatio < consistency_validation_threshold)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("depth_consistency_coverage_loss");
    }

    if (input.largestComponentRatio < 0.15f)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("fragmented_depth_support");
    }

    if (input.multiViewConsistencyAvailable &&
        input.validCoverage >= 0.95f &&
        input.meanConfidence < 0.45f &&
        input.multiViewConsistency < 0.50f)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("low_confidence_full_coverage");
    }

    const float consistency_threshold = input.sceneProfile == MvsSceneProfile::AerialTerrain
        ? (aerial_edge_neighborhood ? 0.50f : 0.55f)
        : 0.45f;
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
        sparse_residual.available &&
        sparse_residual.validSampleCount >=
            kSparseDepthResidualMinimumSampleCount &&
        std::isfinite(sparse_residual.medianAbsoluteLogError) &&
        sparse_residual.medianAbsoluteLogError >= 0.0f;
    if (has_sufficient_sparse_residual &&
        sparse_residual.medianAbsoluteLogError >
            kSparseDepthResidualRejectionThreshold)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back(
            "sparse_absolute_depth_residual_rejected");
    }
    else if (has_sufficient_sparse_residual &&
             sparse_residual.medianAbsoluteLogError >
                 kSparseDepthResidualValidationThreshold)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back(
            "sparse_absolute_depth_residual_validation_only");
    }

    if (decision.reasons.empty())
    {
        decision.reasons.emplace_back("quality_gate_passed");
    }
    return decision;
}

SparseDepthResidualSummary summarizeSparseDepthResidual(
    const cv::Mat &depth,
    const std::vector<ProjectedSparseDepthSample> &samples)
{
    SparseDepthResidualSummary summary;
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

        std::array<float, 9> neighborhood{};
        int valid_neighborhood_count = 0;
        for (int delta_row = -1; delta_row <= 1; ++delta_row)
        {
            const int sample_row = row + delta_row;
            if (sample_row < 0 || sample_row >= depth.rows)
            {
                continue;
            }
            const float *depth_row = depth.ptr<float>(sample_row);
            for (int delta_column = -1; delta_column <= 1; ++delta_column)
            {
                const int sample_column = column + delta_column;
                if (sample_column < 0 || sample_column >= depth.cols)
                {
                    continue;
                }
                const float value = depth_row[sample_column];
                if (std::isfinite(value) && value > 0.0f)
                {
                    neighborhood[static_cast<std::size_t>(
                        valid_neighborhood_count++)] = value;
                }
            }
        }
        if (valid_neighborhood_count <= 0)
        {
            continue;
        }

        std::sort(neighborhood.begin(),
                  neighborhood.begin() + valid_neighborhood_count);
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
        && input.fusionPostprocessRetentionRatio >= 0.40f
        && input.validCoverage >= 0.30f
        && input.meanConfidence >= 0.70f
        && input.multiViewConsistency >= 0.45f
        && input.largestComponentRatio >= 0.15f
        && input.depthAtSearchBoundaryRatio <= 0.45f;
}

bool hasReliableOrbitalDiscreteGeometryCore(
    const DepthFrameQualityInput &input)
{
    return hasReliableOrbitalFusionCore(input)
        && input.discreteGeometryCoreAvailable
        && std::isfinite(input.discreteGeometryCoreRatio)
        && input.discreteGeometryCoreRatio >=
            kDiscreteGeometryCoreMinimumRatio;
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
