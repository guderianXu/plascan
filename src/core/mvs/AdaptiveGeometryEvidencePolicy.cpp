#include <algorithm>
#include <cmath>
#include <limits>

#include "AdaptiveGeometryEvidencePolicy.h"

namespace xjw::mvs
{
namespace
{

struct SanitizedOptions
{
    float footprintSigma = 1.5f;
    float roundTripSigmaPixels = 1.5f;
    float priorAlpha = 0.25f;
    float priorBeta = 0.25f;
    float conflictPenalty = 2.0f;
    float targetEffectiveViews = 2.0f;
    float singleViewDiversityFloor = 0.35f;
};

float finiteOr(float value, float fallback)
{
    return std::isfinite(value) ? value : fallback;
}

SanitizedOptions sanitizeOptions(const AdaptiveGeometryEvidenceOptions &options)
{
    SanitizedOptions sanitized;
    sanitized.footprintSigma = std::max(
        std::numeric_limits<float>::epsilon(),
        finiteOr(options.footprintSigma, sanitized.footprintSigma));
    sanitized.roundTripSigmaPixels = std::max(
        std::numeric_limits<float>::epsilon(),
        finiteOr(options.roundTripSigmaPixels, sanitized.roundTripSigmaPixels));
    sanitized.priorAlpha = std::max(0.0f, finiteOr(options.priorAlpha, sanitized.priorAlpha));
    sanitized.priorBeta = std::max(0.0f, finiteOr(options.priorBeta, sanitized.priorBeta));
    sanitized.conflictPenalty = std::max(
        0.0f,
        finiteOr(options.conflictPenalty, sanitized.conflictPenalty));
    sanitized.targetEffectiveViews = std::max(
        1.0f + std::numeric_limits<float>::epsilon(),
        finiteOr(options.targetEffectiveViews, sanitized.targetEffectiveViews));
    sanitized.singleViewDiversityFloor = std::clamp(
        finiteOr(
            options.singleViewDiversityFloor,
            sanitized.singleViewDiversityFloor),
        0.0f,
        1.0f);
    return sanitized;
}

bool isValidComparableObservation(
    const AdaptiveGeometryEvidenceObservation &observation)
{
    return std::isfinite(observation.worldResidual) &&
        observation.worldResidual >= 0.0f &&
        std::isfinite(observation.worldPixelFootprint) &&
        observation.worldPixelFootprint > std::numeric_limits<float>::epsilon() &&
        std::isfinite(observation.roundTripResidualPixels) &&
        observation.roundTripResidualPixels >= 0.0f;
}

float observationReliability(
    const AdaptiveGeometryEvidenceObservation &observation)
{
    if (!std::isfinite(observation.reliabilityWeight))
    {
        return 0.0f;
    }
    return std::clamp(observation.reliabilityWeight, 0.0f, 1.0f);
}

float agreementWeight(const AdaptiveGeometryEvidenceObservation &observation,
                      const SanitizedOptions &options)
{
    const float normalized_world_residual =
        observation.worldResidual /
        (options.footprintSigma * observation.worldPixelFootprint);
    const float normalized_round_trip_residual =
        observation.roundTripResidualPixels / options.roundTripSigmaPixels;
    const float squared_residual =
        normalized_world_residual * normalized_world_residual +
        normalized_round_trip_residual * normalized_round_trip_residual;
    return std::exp(-0.5f * squared_residual);
}

} // namespace

void AdaptiveGeometryEvidenceAccumulator::add(
    const AdaptiveGeometryEvidenceObservation &observation,
    const AdaptiveGeometryEvidenceOptions &options)
{
    const SanitizedOptions sanitized = sanitizeOptions(options);
    const float reliability = observationReliability(observation);
    if (reliability <= 0.0f)
    {
        return;
    }

    switch (observation.evidenceClass)
    {
    case AdaptiveGeometryEvidenceClass::Unobservable:
    case AdaptiveGeometryEvidenceClass::Occluded:
        break;

    case AdaptiveGeometryEvidenceClass::Comparable:
    {
        if (!isValidComparableObservation(observation))
        {
            break;
        }
        const float agreement = agreementWeight(observation, sanitized);
        const float positive_support = reliability * agreement;
        observable += reliability;
        positiveSupport += positive_support;
        conflict += reliability * (1.0f - agreement);
        squaredPositiveSupport += positive_support * positive_support;
        break;
    }

    case AdaptiveGeometryEvidenceClass::Contradictory:
        observable += reliability;
        conflict += reliability;
        break;
    }
}

AdaptiveGeometryEvidenceResult finalizeAdaptiveGeometryEvidence(
    const AdaptiveGeometryEvidenceAccumulator &accumulator,
    const AdaptiveGeometryEvidenceOptions &options)
{
    const SanitizedOptions sanitized = sanitizeOptions(options);
    AdaptiveGeometryEvidenceResult result;
    result.positiveSupportWeight = std::max(
        0.0f,
        finiteOr(accumulator.positiveSupport, 0.0f));
    result.conflictWeight = std::max(
        0.0f,
        finiteOr(accumulator.conflict, 0.0f));
    result.observableWeight = std::max(
        0.0f,
        finiteOr(accumulator.observable, 0.0f));
    const float squared_positive_support = std::max(
        0.0f,
        finiteOr(accumulator.squaredPositiveSupport, 0.0f));

    const float reference_mass = 1.0f;
    const float positive_mass = reference_mass + result.positiveSupportWeight;
    const float effective_view_denominator =
        reference_mass + squared_positive_support;
    result.effectiveViewCount =
        positive_mass * positive_mass / effective_view_denominator;

    const float agreement_numerator =
        sanitized.priorAlpha + positive_mass;
    const float agreement_denominator =
        sanitized.priorAlpha + sanitized.priorBeta + positive_mass +
        sanitized.conflictPenalty * result.conflictWeight;
    result.agreementProbability = agreement_denominator > 0.0f
        ? std::clamp(agreement_numerator / agreement_denominator, 0.0f, 1.0f)
        : 1.0f;

    const float diversity_progress = std::clamp(
        (result.effectiveViewCount - 1.0f) /
            (sanitized.targetEffectiveViews - 1.0f),
        0.0f,
        1.0f);
    result.diversityWeight = sanitized.singleViewDiversityFloor +
        (1.0f - sanitized.singleViewDiversityFloor) * diversity_progress;
    result.supportWeight = std::clamp(
        result.agreementProbability * result.diversityWeight,
        0.0f,
        1.0f);
    return result;
}

AdaptiveGeometryEvidenceResult aggregateAdaptiveGeometryEvidence(
    std::span<const AdaptiveGeometryEvidenceObservation> observations,
    const AdaptiveGeometryEvidenceOptions &options)
{
    AdaptiveGeometryEvidenceAccumulator accumulator;
    for (const AdaptiveGeometryEvidenceObservation &observation : observations)
    {
        accumulator.add(observation, options);
    }
    return finalizeAdaptiveGeometryEvidence(accumulator, options);
}

} // namespace xjw::mvs
