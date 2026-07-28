#include "DepthFrameQualityGate.h"

#include <algorithm>
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

    DepthConfidenceComponents components;
    components.photometric = input.meanConfidence;
    components.support = input.sourceViewCount > 0
        ? std::min(1.0f, static_cast<float>(input.sourceViewCount) / 4.0f)
        : 0.0f;
    components.uniqueness = 1.0f - input.depthAtSearchBoundaryRatio;
    components.geometry = input.multiViewConsistency;
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

    if (input.validCoverage >= 0.95f &&
        input.meanConfidence < 0.45f &&
        input.multiViewConsistency < 0.50f)
    {
        lowerAcceptance(DepthFrameAcceptance::Rejected, decision);
        decision.reasons.emplace_back("low_confidence_full_coverage");
    }

    const float consistency_threshold = input.sceneProfile == MvsSceneProfile::AerialTerrain
        ? (aerial_edge_neighborhood ? 0.50f : 0.55f)
        : 0.45f;
    if (input.sourceViewCount >= 2 && input.multiViewConsistency < consistency_threshold)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("weak_multiview_consistency");
    }

    if (input.sparseDepthMedianRelativeError > 0.12f)
    {
        lowerAcceptance(DepthFrameAcceptance::ValidationOnly, decision);
        decision.reasons.emplace_back("sparse_depth_residual_high");
    }

    if (decision.reasons.empty())
    {
        decision.reasons.emplace_back("quality_gate_passed");
    }
    return decision;
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
