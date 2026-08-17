#pragma once

#include "MvsTypes.h"

#include <string>
#include <vector>

namespace xjw
{
namespace mvs
{

enum class DepthFrameAcceptance
{
    Accepted,
    ValidationOnly,
    Rejected
};

struct DepthConfidenceComponents
{
    float photometric = 0.0f;
    float support = 0.0f;
    float uniqueness = 0.0f;
    float geometry = 0.0f;
    float texture = 0.0f;
    /// Confidence derived from an independent sparse/reference geometry
    /// residual. Negative means that no absolute geometry evidence exists.
    float absoluteGeometry = -1.0f;
};

struct DepthFilterSettings
{
    int minComponentArea = 24;
    float localDepthOutlierRelThreshold = 0.25f;
    int minConsistentViews = 3;
};

struct DepthConfidenceThresholds
{
    float patchMatch = 0.60f;
    float fusion = 0.65f;
};

struct SparseDepthResidualSummary
{
    bool available = false;
    int projectedSampleCount = 0;
    int validSampleCount = 0;
    float medianAbsoluteLogError = -1.0f;
};

struct DepthFrameQualityInput
{
    MvsSceneProfile sceneProfile = MvsSceneProfile::OrbitalObject;
    DepthFilterMode filterMode = DepthFilterMode::Moderate;
    int sourceViewCount = 0;
    float validCoverage = 0.0f;
    float largestComponentRatio = 0.0f;
    float meanConfidence = 0.0f;
    /// True only after cross-view geometry has produced this metric.  A
    /// pre-geometry frame must not synthesize it from photometric confidence.
    bool multiViewConsistencyAvailable = true;
    float multiViewConsistency = 0.0f;
    float depthAtSearchBoundaryRatio = 0.0f;
    /// Legacy relative-error input retained for source compatibility.  New
    /// callers must provide sparseDepthResidual so sample sufficiency is
    /// explicit and auditable.
    float sparseDepthMedianRelativeError = 0.0f;
    SparseDepthResidualSummary sparseDepthResidual;
    bool hasConstrainedSupportMask = false;
    float validWithinMaskRatio = -1.0f;
    float outputFilterRetentionRatio = -1.0f;
    float consistencyRetentionRatio = -1.0f;
    float fusionPostprocessRetentionRatio = -1.0f;
    bool adaptiveGeometryEvidenceAvailable = false;
    float adaptiveEffectiveViewCountMean = -1.0f;
    float adaptiveConflictRatioMean = -1.0f;
    /// Fraction of final valid pixels jointly supported by at least three
    /// views with a tightly clustered inverse-depth estimate.  Unlike the
    /// adaptive score, this is evaluated only after the hard consistency and
    /// fusion postprocess stages have produced the retained surface core.
    bool discreteGeometryCoreAvailable = false;
    float discreteGeometryCoreRatio = -1.0f;
};

struct DepthFrameQualityDecision
{
    DepthFrameAcceptance acceptance = DepthFrameAcceptance::Rejected;
    DepthFilterSettings filterSettings;
    float calibratedConfidence = 0.0f;
    DepthConfidenceComponents confidenceComponents;
    SparseDepthResidualSummary sparseDepthResidual;
    std::vector<std::string> reasons;
};

enum class DepthConsistencyEvidence
{
    Unverifiable,
    Consistent,
    Occluded,
    Contradicted
};

float calibrateDepthConfidence(const DepthConfidenceComponents &components);
float geometryErrorConfidence(const SparseDepthResidualSummary &residual);

DepthFilterSettings depthFilterSettings(DepthFilterMode mode, int availableSourceViews);

DepthConfidenceThresholds depthConfidenceThresholds(
    MvsSceneProfile sceneProfile,
    DepthFilterMode filterMode,
    int availableSourceViews,
    float configuredPatchMatch,
    float configuredFusion);

int minimumDepthConsistencySourceConfirmations(DepthFilterMode mode,
                                               int availableSourceViews);
int minimumDepthConsistencySourceConfirmations(MvsSceneProfile sceneProfile,
                                               DepthFilterMode mode,
                                               int availableSourceViews);

float depthConsistencyRelativeThreshold(
    MvsSceneProfile sceneProfile,
    int viewCount,
    DepthFilterMode filterMode = DepthFilterMode::Moderate);

bool useContradictionOnlyDepthConsistency(int sourceViewCount);

DepthConsistencyEvidence classifyDepthConsistencyEvidence(float expectedDepth,
                                                           float measuredDepth,
                                                           float relativeThreshold);

DepthFrameQualityDecision evaluateDepthFrame(const DepthFrameQualityInput &input);

/// Compare final positive-Z depth against projected sparse absolute-depth
/// anchors.  Each anchor samples the median of its valid 3x3 neighborhood so
/// isolated holes and one-pixel outliers do not dominate the frame gate.
SparseDepthResidualSummary summarizeSparseDepthResidual(
    const cv::Mat &depth,
    const std::vector<ProjectedSparseDepthSample> &samples);

bool hasReliableOrbitalFusionCore(const DepthFrameQualityInput &input);

/// A continuous adaptive residual can be over-sensitive for very narrow-FOV
/// orbital imagery.  It may only fall back to the retained discrete evidence
/// when a broad, confident frame core and a strong joint support/spread core
/// both survive all final filters.
bool hasReliableOrbitalDiscreteGeometryCore(
    const DepthFrameQualityInput &input);

inline constexpr int kDiscreteGeometryCoreMinimumObservationCount = 3;
inline constexpr float kDiscreteGeometryCoreMaximumInverseDepthSpread = 0.0065f;
inline constexpr float kDiscreteGeometryCoreMinimumRatio = 0.65f;

inline constexpr int kSparseDepthResidualMinimumSampleCount = 20;
inline constexpr float kSparseDepthResidualValidationThreshold = 0.005f;
inline constexpr float kSparseDepthResidualRejectionThreshold = 0.020f;

const char *depthFrameAcceptanceId(DepthFrameAcceptance acceptance);

} // namespace mvs
} // namespace xjw
