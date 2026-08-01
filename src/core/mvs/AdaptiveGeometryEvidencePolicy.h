#pragma once

#include <span>

namespace xjw::mvs
{

/// Describes how a source view relates to a reference depth hypothesis.
enum class AdaptiveGeometryEvidenceClass
{
    /// No valid source sample exists (outside the image/mask, missing depth, or invalid projection).
    /// It contributes neither positive support nor conflict.
    Unobservable,

    /// A valid source surface lies in front of the reference hypothesis. The reference point is
    /// hidden in this view, so the observation contributes neither positive support nor conflict.
    Occluded,

    /// The reference hypothesis should be visible in the source view. Its footprint-normalized
    /// residual continuously divides source reliability between positive support and conflict.
    Comparable,

    /// Upstream geometry establishes a visible free-space/depth-order contradiction. It contributes
    /// observable conflict only and never positive support.
    Contradictory
};

struct AdaptiveGeometryEvidenceObservation
{
    AdaptiveGeometryEvidenceClass evidenceClass =
        AdaptiveGeometryEvidenceClass::Unobservable;

    /// Euclidean reference/source surface separation in world units.
    float worldResidual = 0.0f;

    /// Joint world-space pixel footprint. For a valid baseline this combines the reference
    /// lateral footprint with the target-view epipolar triangulation uncertainty.
    float worldPixelFootprint = 0.0f;

    /// Source-to-reference round-trip reprojection error in pixels.
    float roundTripResidualPixels = 0.0f;

    /// Source confidence multiplied by pair/angle reliability, nominally in [0, 1].
    float reliabilityWeight = 1.0f;
};

struct AdaptiveGeometryEvidenceOptions
{
    float footprintSigma = 2.0f;
    float roundTripSigmaPixels = 1.5f;
    float priorAlpha = 0.25f;
    float priorBeta = 0.25f;
    float conflictPenalty = 2.0f;
    float targetEffectiveViews = 2.0f;
    float singleViewDiversityFloor = 0.35f;
};

/// Sufficient statistics for independent source-view observations. Instances accumulated for
/// separate image tiles or source batches can be combined by adding their four fields.
struct AdaptiveGeometryEvidenceAccumulator
{
    float positiveSupport = 0.0f;
    float squaredPositiveSupport = 0.0f;
    float conflict = 0.0f;
    float observable = 0.0f;

    void add(const AdaptiveGeometryEvidenceObservation &observation,
             const AdaptiveGeometryEvidenceOptions &options = {});
};

struct AdaptiveGeometryEvidenceResult
{
    float supportWeight = 0.0f;
    float effectiveViewCount = 1.0f;
    float conflictWeight = 0.0f;
    float conflictRatio = 0.0f;
    float observableWeight = 0.0f;
    float positiveSupportWeight = 0.0f;
    float agreementProbability = 0.0f;
    float diversityWeight = 0.0f;
};

AdaptiveGeometryEvidenceResult finalizeAdaptiveGeometryEvidence(
    const AdaptiveGeometryEvidenceAccumulator &accumulator,
    const AdaptiveGeometryEvidenceOptions &options = {});

/// Aggregates independent source-view evidence for one valid reference depth hypothesis.
/// The reference observation has fixed unit mass. Invalid comparable inputs are treated as
/// unobservable so missing data cannot turn into negative evidence.
AdaptiveGeometryEvidenceResult aggregateAdaptiveGeometryEvidence(
    std::span<const AdaptiveGeometryEvidenceObservation> observations,
    const AdaptiveGeometryEvidenceOptions &options = {});

} // namespace xjw::mvs
