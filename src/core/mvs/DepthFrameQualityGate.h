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
    int neighborhoodRadiusPixels = 1;
    float medianAbsoluteLogError = -1.0f;
};

/// Audit summary for the boundary between observed cross-view consistency and
/// the depth product that is eventually published. A safety fallback may
/// restore the original depth for diagnostics or later recovery, but that
/// restored coverage is not evidence that cross-view consistency succeeded.
struct DepthConsistencyPublicationSummary
{
    float observedRetentionRatio = -1.0f;
    float publishedRetentionRatio = -1.0f;
    bool originalDepthFallbackApplied = false;
    DepthFrameAcceptance acceptanceCeiling = DepthFrameAcceptance::Accepted;
    std::string diagnosticReason;
};

struct DepthFrameQualityInput
{
    MvsSceneProfile sceneProfile = MvsSceneProfile::OrbitalObject;
    DepthFilterMode filterMode = DepthFilterMode::Moderate;
    int sourceViewCount = 0;
    float validCoverage = 0.0f;
    float largestComponentRatio = 0.0f;
    float meanConfidence = 0.0f;
    bool dualChannelConfidenceAvailable = false;
    float meanPhotometricConfidence = -1.0f;
    float meanIndependentGeometryConfidence = -1.0f;
    float strongIndependentGeometryCoverage = -1.0f;
    bool geometryEvidenceTreatmentEnabled = false;
    int geometryCorrectedPixelCount = 0;
    float correctedMeanIndependentGeometryConfidence = -1.0f;
    /// True only after cross-view geometry has produced this metric.  A
    /// pre-geometry frame must not synthesize it from photometric confidence.
    bool multiViewConsistencyAvailable = true;
    float multiViewConsistency = 0.0f;
    float depthAtSearchBoundaryRatio = 0.0f;
    SparseDepthResidualSummary sparseDepthResidual;
    /// True only for a semantic project support mask. Technical raster-validity
    /// and content masks still define the evaluated domain, but must not invoke
    /// the project-mask completeness gate.
    bool hasProjectSupportMask = false;
    float validWithinMaskRatio = -1.0f;
    float outputFilterRetentionRatio = -1.0f;
    float consistencyRetentionRatio = -1.0f;
    /// True when the published product restored the pre-consistency depth
    /// after the observed consistency result collapsed. Such a product may
    /// never be promoted to Primary solely because its coverage was restored.
    bool consistencyPublicationFallbackApplied = false;
    /// Upper bound supplied by summarizeDepthConsistencyPublication(). The
    /// default preserves existing callers and quality decisions.
    DepthFrameAcceptance consistencyAcceptanceCeiling = DepthFrameAcceptance::Accepted;
    std::string consistencyPublicationDiagnosticReason;
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
    /// The complete visibility pool is an expert-only experiment.  Admission
    /// may use it only when the reference frame's own selected plan actually
    /// changed relative to the legacy early-stop pool; cross-frame side
    /// effects alone must not promote a provisional frame to Primary.
    bool completeVisibilityCandidatePoolEnabled = false;
    bool completePoolChangedLegacyPlan = false;
    bool initialAcceptanceAvailable = false;
    DepthFrameAcceptance initialAcceptance = DepthFrameAcceptance::Rejected;
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

float calibrateDepthConfidence(const DepthConfidenceComponents& components);
float geometryErrorConfidence(const SparseDepthResidualSummary& residual);

/// Preserve the observed consistency retention independently of any restored
/// publication coverage. An all-unavailable (-1) tuple is accepted only when
/// the caller explicitly marks the consistency stage as not applicable;
/// expected, partially populated, or invalid counts fail closed. A fallback
/// after a real (<10%) consistency collapse is rejected, while any other
/// fallback remains validation-only at best.
DepthConsistencyPublicationSummary summarizeDepthConsistencyPublication(int preConsistencyValidCount,
                                                                        int observedPostConsistencyValidCount,
                                                                        int publishedValidCount,
                                                                        bool originalDepthFallbackApplied,
                                                                        bool consistencyStageExpected);

/// Apply a publication summary without converting an unavailable (-1)
/// observation into synthetic zero-valued multi-view evidence.
void applyDepthConsistencyPublicationSummary(const DepthConsistencyPublicationSummary& summary,
                                             DepthFrameQualityInput* input);

DepthFilterSettings depthFilterSettings(DepthFilterMode mode, int availableSourceViews);

DepthConfidenceThresholds depthConfidenceThresholds(MvsSceneProfile sceneProfile,
                                                    DepthFilterMode filterMode,
                                                    int availableSourceViews,
                                                    float configuredPatchMatch,
                                                    float configuredFusion);

int minimumDepthConsistencySourceConfirmations(DepthFilterMode mode, int availableSourceViews);
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
/// anchors. Each anchor samples the median of the requested grid-domain
/// neighborhood. Callers scale the default full-raster radius onto reduced
/// native depth grids so the physical footprint does not silently grow.
SparseDepthResidualSummary summarizeSparseDepthResidual(
    const cv::Mat &depth,
    const std::vector<ProjectedSparseDepthSample> &samples,
    int neighborhoodRadiusPixels = 1);

bool hasReliableOrbitalFusionCore(const DepthFrameQualityInput &input);

/// General captures may survive an aggressive fusion postprocess only when
/// the retained surface is backed jointly by sparse absolute depth and a
/// strong discrete multi-view core. This exception is deliberately narrower
/// than the orbital fallback because Custom has no ring-geometry prior.
bool hasReliableCustomFusionCore(const DepthFrameQualityInput &input);

/// A Custom frame whose confidence postprocess removes a broad weak
/// hypothesis set may still remain a coverage-only surface when the retained
/// product has accurate independent sparse-depth anchors.  This evidence is
/// intentionally insufficient for Primary because no discrete multi-view
/// core survived.
bool hasReliableCustomSparseAnchoredSurface(
    const DepthFrameQualityInput &input);

/// A treatment may bypass relative-retention penalties only when its changed
/// pixels are backed by independent geometry and the frame still has the
/// existing absolute sparse and discrete geometry cores. It never bypasses
/// absolute coverage, topology, search-boundary, or sparse-error rejection.
bool hasReliableCausalGeometryCorrection(
    const DepthFrameQualityInput &input);

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
