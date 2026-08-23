#include "PatchMatchPhotometricCost.h"

#include <gtest/gtest.h>

#include <array>

namespace
{

TEST(PatchMatchPhotometricCostTest, RejectsSingleLuckySourceInTwoViewSupport)
{
    const float scores[] = {0.95f, 0.0f};
    EXPECT_FLOAT_EQ(xjw::mvs::robustMultiSourceNcc(scores, 2), 0.0f);
}

TEST(PatchMatchPhotometricCostTest, RequiresBothSourcesWhenOnlyTwoAreAvailable)
{
    const float scores[] = {0.9f, 0.8f};
    EXPECT_NEAR(xjw::mvs::robustMultiSourceNcc(scores, 2), 0.85f, 1e-6f);
}

TEST(PatchMatchPhotometricCostTest, UsesStrongestMajorityForThreeSources)
{
    const float scores[] = {0.9f, 0.8f, 0.0f};
    EXPECT_NEAR(xjw::mvs::robustMultiSourceNcc(scores, 3), 0.85f, 1e-6f);
}

TEST(PatchMatchPhotometricCostTest, RejectsInsufficientMultiViewSupport)
{
    const float scores[] = {0.9f, 0.04f, 0.0f, 0.03f};
    EXPECT_FLOAT_EQ(xjw::mvs::robustMultiSourceNcc(scores, 4), 0.0f);
}

TEST(PatchMatchPhotometricCostTest, KeepsTopRequiredScoresOnly)
{
    const float scores[] = {0.95f, 0.8f, 0.7f, 0.1f, 0.06f};
    EXPECT_NEAR(xjw::mvs::robustMultiSourceNcc(scores, 5),
                (0.95f + 0.8f + 0.7f) / 3.0f,
                1e-6f);
}

TEST(PatchMatchPhotometricCostTest, JointSelectionReturnsCompactDeterministicMask)
{
    const float scores[] = {0.80f, 0.80f, 0.70f, 0.10f};
    const xjw::mvs::JointViewSelection selection =
        xjw::mvs::selectJointSourceViews(scores, 4);

    EXPECT_EQ(selection.sourceMask, 0b0111u);
    EXPECT_EQ(selection.sourceCount, 3);
    EXPECT_EQ(xjw::mvs::selectedSourceCount(selection.sourceMask), 3);
    EXPECT_NEAR(selection.photometricScore, (0.80f + 0.80f + 0.70f) / 3.0f, 1e-6f);
}

TEST(PatchMatchPhotometricCostTest, NeighborPriorChangesSelectionWithoutInflatingScore)
{
    const float scores[] = {0.90f, 0.80f, 0.79f, 0.78f, 0.10f};
    const xjw::mvs::JointViewSelection baseline =
        xjw::mvs::selectJointSourceViews(scores, 5);
    const xjw::mvs::JointViewSelection guided =
        xjw::mvs::selectJointSourceViews(scores, 5, 0b1000u, 0.04f);

    EXPECT_EQ(baseline.sourceMask, 0b00111u);
    EXPECT_EQ(guided.sourceMask, 0b01011u);
    EXPECT_NEAR(guided.photometricScore, (0.90f + 0.80f + 0.78f) / 3.0f, 1e-6f);
}

TEST(PatchMatchPhotometricCostTest, NeighborPriorCannotRescueUnsupportedSource)
{
    const float scores[] = {0.90f, 0.04f, 0.0f, 0.03f};
    const xjw::mvs::JointViewSelection selection =
        xjw::mvs::selectJointSourceViews(scores, 4, 0b1111u, 1.0f);

    EXPECT_FLOAT_EQ(selection.photometricScore, 0.0f);
    EXPECT_EQ(selection.sourceMask, 0u);
}

TEST(PatchMatchPhotometricCostTest, RejectsCorrelationCreatedOnlyBySharedBlackBackground)
{
    const std::array<float, 9> reference{
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.4f, 0.6f, 0.8f};
    const std::array<float, 9> wrong_source{
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.8f, 0.6f, 0.4f, 0.2f};

    xjw::mvs::PatchNccAccumulator unmasked;
    xjw::mvs::PatchNccAccumulator masked;
    for (std::size_t index = 0; index < reference.size(); ++index)
    {
        unmasked.addCandidate(true, reference[index], wrong_source[index]);
        const bool trusted_foreground = index >= 5;
        masked.addCandidate(
            trusted_foreground, reference[index], wrong_source[index]);
    }

    EXPECT_GT(unmasked.score(false), 0.5f);
    EXPECT_NEAR(masked.score(true), 0.0f, 1e-6f);
}

TEST(PatchMatchPhotometricCostTest, KeepsPureForegroundNccUnchanged)
{
    const std::array<float, 8> reference{0.1f, 0.4f, 0.2f, 0.8f, 0.5f, 0.3f, 0.9f, 0.6f};
    const std::array<float, 8> source{0.2f, 0.5f, 0.3f, 0.9f, 0.6f, 0.4f, 1.0f, 0.7f};

    xjw::mvs::PatchNccAccumulator unmasked;
    xjw::mvs::PatchNccAccumulator masked;
    for (std::size_t index = 0; index < reference.size(); ++index)
    {
        unmasked.addCandidate(true, reference[index], source[index]);
        masked.addCandidate(true, reference[index], source[index]);
    }

    EXPECT_NEAR(masked.score(true), unmasked.score(false), 1e-6f);
}

TEST(PatchMatchPhotometricCostTest, RejectsPatchBelowConfiguredValidSampleRatio)
{
    xjw::mvs::PatchNccAccumulator accumulator;
    for (int index = 0; index < 25; ++index)
    {
        const bool valid = index < 8;
        accumulator.addCandidate(valid,
                                 static_cast<float>(index),
                                 static_cast<float>(index) + 1.0f);
    }

    EXPECT_GT(accumulator.score(false), 0.99f);
    EXPECT_FLOAT_EQ(accumulator.score(true, 0.35f), 0.0f);
}

TEST(PatchMatchPhotometricCostTest, CompositeCostSurvivesAffineExposureChange)
{
    xjw::mvs::PatchRobustPhotometricAccumulator accumulator;
    for (int index = 0; index < 9; ++index)
    {
        const float reference = 0.05f * static_cast<float>(index);
        const float source = 0.20f + 1.6f * reference;
        accumulator.addIntensityCandidate(true, reference, source);
        accumulator.addGradientCandidate(true, 0.05f, 0.02f, 0.08f, 0.032f);
        accumulator.addCensusCandidate(
            true, reference - 0.20f, source - 0.52f);
    }

    EXPECT_GT(accumulator.score(false), 0.98f);
}

TEST(PatchMatchPhotometricCostTest, CensusRejectsReversedLocalOrdering)
{
    xjw::mvs::PatchRobustPhotometricAccumulator matching;
    xjw::mvs::PatchRobustPhotometricAccumulator reversed;
    for (int index = 0; index < 9; ++index)
    {
        const float reference = 0.1f * static_cast<float>(index);
        const float source = reference;
        matching.addIntensityCandidate(true, reference, source);
        reversed.addIntensityCandidate(true, reference, source);
        matching.addCensusCandidate(true, reference - 0.4f, source - 0.4f);
        reversed.addCensusCandidate(true, reference - 0.4f, 0.4f - source);
    }

    EXPECT_GT(matching.score(false), reversed.score(false));
}

TEST(PatchMatchPhotometricCostTest, CompositeConfidenceCalibrationIsMonotonic)
{
    EXPECT_FLOAT_EQ(
        xjw::mvs::calibrateRobustPhotometricConfidence(0.0f), 0.0f);
    EXPECT_NEAR(
        xjw::mvs::calibrateRobustPhotometricConfidence(0.5f),
        0.8705506f,
        1.0e-6f);
    EXPECT_LT(
        xjw::mvs::calibrateRobustPhotometricConfidence(0.5f),
        xjw::mvs::calibrateRobustPhotometricConfidence(0.8f));
    EXPECT_FLOAT_EQ(
        xjw::mvs::calibrateRobustPhotometricConfidence(2.0f), 1.0f);
}

} // namespace
