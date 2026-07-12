#include "PatchMatchPhotometricCost.h"

#include <gtest/gtest.h>

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

} // namespace
