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

} // namespace
