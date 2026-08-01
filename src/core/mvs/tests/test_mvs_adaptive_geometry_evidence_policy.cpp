#include <array>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "AdaptiveGeometryEvidencePolicy.h"

namespace
{

using xjw::mvs::AdaptiveGeometryEvidenceClass;
using xjw::mvs::AdaptiveGeometryEvidenceAccumulator;
using xjw::mvs::AdaptiveGeometryEvidenceObservation;
using xjw::mvs::AdaptiveGeometryEvidenceResult;

AdaptiveGeometryEvidenceObservation comparableObservation(
    float world_residual,
    float world_pixel_footprint,
    float round_trip_residual_pixels = 0.0f,
    float reliability = 1.0f)
{
    AdaptiveGeometryEvidenceObservation observation;
    observation.evidenceClass = AdaptiveGeometryEvidenceClass::Comparable;
    observation.worldResidual = world_residual;
    observation.worldPixelFootprint = world_pixel_footprint;
    observation.roundTripResidualPixels = round_trip_residual_pixels;
    observation.reliabilityWeight = reliability;
    return observation;
}

TEST(AdaptiveGeometryEvidencePolicyTest,
     SingleReferenceObservationKeepsConfiguredDiversityFloor)
{
    const AdaptiveGeometryEvidenceResult result =
        xjw::mvs::aggregateAdaptiveGeometryEvidence({});

    EXPECT_FLOAT_EQ(result.effectiveViewCount, 1.0f);
    EXPECT_FLOAT_EQ(result.diversityWeight, 0.35f);
    EXPECT_FLOAT_EQ(result.observableWeight, 0.0f);
    EXPECT_FLOAT_EQ(result.conflictWeight, 0.0f);
    EXPECT_NEAR(result.agreementProbability, 5.0f / 6.0f, 1.0e-6f);
    EXPECT_NEAR(result.supportWeight, 7.0f / 24.0f, 1.0e-6f);
}

TEST(AdaptiveGeometryEvidencePolicyTest,
     AgreeingSourcesMonotonicallyIncreaseSupportAndEffectiveViews)
{
    const AdaptiveGeometryEvidenceObservation agreeing =
        comparableObservation(0.0f, 1.0f);
    const std::array one_source{agreeing};
    const std::array two_sources{agreeing, agreeing};

    const AdaptiveGeometryEvidenceResult no_source =
        xjw::mvs::aggregateAdaptiveGeometryEvidence({});
    const AdaptiveGeometryEvidenceResult one =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(one_source);
    const AdaptiveGeometryEvidenceResult two =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(two_sources);

    EXPECT_LT(no_source.supportWeight, one.supportWeight);
    EXPECT_LE(one.supportWeight, two.supportWeight);
    EXPECT_LT(no_source.effectiveViewCount, one.effectiveViewCount);
    EXPECT_LT(one.effectiveViewCount, two.effectiveViewCount);
    EXPECT_FLOAT_EQ(one.effectiveViewCount, 2.0f);
    EXPECT_FLOAT_EQ(two.effectiveViewCount, 3.0f);
}

TEST(AdaptiveGeometryEvidencePolicyTest,
     MissingAndOccludedSourcesDoNotPenalizeReferenceObservation)
{
    AdaptiveGeometryEvidenceObservation missing;
    missing.evidenceClass = AdaptiveGeometryEvidenceClass::Unobservable;
    missing.reliabilityWeight = 1.0f;
    AdaptiveGeometryEvidenceObservation occluded;
    occluded.evidenceClass = AdaptiveGeometryEvidenceClass::Occluded;
    occluded.reliabilityWeight = 1.0f;
    const std::array ignored_sources{missing, occluded};

    const AdaptiveGeometryEvidenceResult baseline =
        xjw::mvs::aggregateAdaptiveGeometryEvidence({});
    const AdaptiveGeometryEvidenceResult with_ignored_sources =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(ignored_sources);

    EXPECT_FLOAT_EQ(with_ignored_sources.supportWeight, baseline.supportWeight);
    EXPECT_FLOAT_EQ(with_ignored_sources.effectiveViewCount, baseline.effectiveViewCount);
    EXPECT_FLOAT_EQ(with_ignored_sources.observableWeight, 0.0f);
    EXPECT_FLOAT_EQ(with_ignored_sources.conflictWeight, 0.0f);
}

TEST(AdaptiveGeometryEvidencePolicyTest,
     ExplicitConflictLowersSupportAndAccumulatesOnlyNegativeEvidence)
{
    AdaptiveGeometryEvidenceObservation contradiction;
    contradiction.evidenceClass = AdaptiveGeometryEvidenceClass::Contradictory;
    contradiction.reliabilityWeight = 0.75f;
    const std::array observations{contradiction};

    const AdaptiveGeometryEvidenceResult baseline =
        xjw::mvs::aggregateAdaptiveGeometryEvidence({});
    const AdaptiveGeometryEvidenceResult conflicted =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(observations);

    EXPECT_LT(conflicted.supportWeight, baseline.supportWeight);
    EXPECT_FLOAT_EQ(conflicted.observableWeight, 0.75f);
    EXPECT_FLOAT_EQ(conflicted.conflictWeight, 0.75f);
    EXPECT_FLOAT_EQ(conflicted.positiveSupportWeight, 0.0f);
    EXPECT_FLOAT_EQ(conflicted.effectiveViewCount, 1.0f);
}

TEST(AdaptiveGeometryEvidencePolicyTest,
     ComparableResidualContinuouslyTransfersObservableMassToConflict)
{
    const std::array close_observation{
        comparableObservation(0.1f, 1.0f, 0.1f)};
    const std::array distant_observation{
        comparableObservation(3.0f, 1.0f, 3.0f)};

    const AdaptiveGeometryEvidenceResult close =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(close_observation);
    const AdaptiveGeometryEvidenceResult distant =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(distant_observation);

    EXPECT_FLOAT_EQ(close.observableWeight, distant.observableWeight);
    EXPECT_GT(close.positiveSupportWeight, distant.positiveSupportWeight);
    EXPECT_LT(close.conflictWeight, distant.conflictWeight);
    EXPECT_GT(close.supportWeight, distant.supportWeight);
}

TEST(AdaptiveGeometryEvidencePolicyTest,
     WorldResidualIsInvariantWhenNormalizedByPixelFootprint)
{
    const std::array near_scale{
        comparableObservation(0.5f, 0.25f, 0.4f, 0.8f)};
    const std::array far_scale{
        comparableObservation(2.0f, 1.0f, 0.4f, 0.8f)};

    const AdaptiveGeometryEvidenceResult near_result =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(near_scale);
    const AdaptiveGeometryEvidenceResult far_result =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(far_scale);

    EXPECT_NEAR(near_result.positiveSupportWeight,
                far_result.positiveSupportWeight,
                1.0e-6f);
    EXPECT_NEAR(near_result.conflictWeight, far_result.conflictWeight, 1.0e-6f);
    EXPECT_NEAR(near_result.effectiveViewCount, far_result.effectiveViewCount, 1.0e-6f);
    EXPECT_NEAR(near_result.supportWeight, far_result.supportWeight, 1.0e-6f);
}

TEST(AdaptiveGeometryEvidencePolicyTest,
     InvalidComparableInputBehavesAsUnobservableInsteadOfConflict)
{
    const std::array invalid_observations{
        comparableObservation(1.0f, 0.0f),
        comparableObservation(-1.0f, 1.0f),
        comparableObservation(
            std::numeric_limits<float>::quiet_NaN(),
            1.0f)};

    const AdaptiveGeometryEvidenceResult baseline =
        xjw::mvs::aggregateAdaptiveGeometryEvidence({});
    const AdaptiveGeometryEvidenceResult invalid =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(invalid_observations);

    EXPECT_FLOAT_EQ(invalid.supportWeight, baseline.supportWeight);
    EXPECT_FLOAT_EQ(invalid.observableWeight, 0.0f);
    EXPECT_FLOAT_EQ(invalid.conflictWeight, 0.0f);
}

TEST(AdaptiveGeometryEvidencePolicyTest,
     SequentialAndBatchAccumulatorsMatchSpanAggregation)
{
    AdaptiveGeometryEvidenceObservation contradiction;
    contradiction.evidenceClass = AdaptiveGeometryEvidenceClass::Contradictory;
    contradiction.reliabilityWeight = 0.35f;
    AdaptiveGeometryEvidenceObservation occluded;
    occluded.evidenceClass = AdaptiveGeometryEvidenceClass::Occluded;
    occluded.reliabilityWeight = 0.9f;
    AdaptiveGeometryEvidenceObservation unobservable;
    unobservable.evidenceClass = AdaptiveGeometryEvidenceClass::Unobservable;
    const std::array observations{
        comparableObservation(0.1f, 0.3f, 0.2f, 0.9f),
        comparableObservation(0.5f, 0.4f, 0.8f, 0.7f),
        contradiction,
        occluded,
        unobservable,
        comparableObservation(0.2f, 0.5f, 0.1f, 0.6f)};

    const AdaptiveGeometryEvidenceResult span_result =
        xjw::mvs::aggregateAdaptiveGeometryEvidence(observations);

    AdaptiveGeometryEvidenceAccumulator sequential;
    for (const AdaptiveGeometryEvidenceObservation &observation : observations)
    {
        sequential.add(observation);
    }
    const AdaptiveGeometryEvidenceResult sequential_result =
        xjw::mvs::finalizeAdaptiveGeometryEvidence(sequential);

    AdaptiveGeometryEvidenceAccumulator first_batch;
    AdaptiveGeometryEvidenceAccumulator second_batch;
    for (std::size_t index = 0; index < observations.size(); ++index)
    {
        AdaptiveGeometryEvidenceAccumulator &batch =
            index < observations.size() / 2 ? first_batch : second_batch;
        batch.add(observations[index]);
    }
    AdaptiveGeometryEvidenceAccumulator combined;
    combined.positiveSupport =
        first_batch.positiveSupport + second_batch.positiveSupport;
    combined.squaredPositiveSupport =
        first_batch.squaredPositiveSupport + second_batch.squaredPositiveSupport;
    combined.conflict = first_batch.conflict + second_batch.conflict;
    combined.observable = first_batch.observable + second_batch.observable;
    const AdaptiveGeometryEvidenceResult batch_result =
        xjw::mvs::finalizeAdaptiveGeometryEvidence(combined);

    const auto expect_equivalent = [&span_result](
                                       const AdaptiveGeometryEvidenceResult &result)
    {
        EXPECT_NEAR(result.supportWeight, span_result.supportWeight, 1.0e-6f);
        EXPECT_NEAR(result.effectiveViewCount,
                    span_result.effectiveViewCount,
                    1.0e-6f);
        EXPECT_NEAR(result.conflictWeight, span_result.conflictWeight, 1.0e-6f);
        EXPECT_NEAR(result.observableWeight, span_result.observableWeight, 1.0e-6f);
        EXPECT_NEAR(result.positiveSupportWeight,
                    span_result.positiveSupportWeight,
                    1.0e-6f);
        EXPECT_NEAR(result.agreementProbability,
                    span_result.agreementProbability,
                    1.0e-6f);
        EXPECT_NEAR(result.diversityWeight, span_result.diversityWeight, 1.0e-6f);
    };
    expect_equivalent(sequential_result);
    expect_equivalent(batch_result);
}

} // namespace
