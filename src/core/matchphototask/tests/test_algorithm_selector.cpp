#include "MatchPhotosAlgorithmSelector.h"

#include <gtest/gtest.h>

TEST(MatchPhotosAlgorithmSelectorTest, AutoUsesSiftLightGlueAndPrefersCuda)
{
    xjw::matchphotos::MatchPhotosOptions options;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_EQ(plan.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(plan.featureSuffix, QStringLiteral(".sift"));
    EXPECT_EQ(plan.matcherAlgorithm, QStringLiteral("lightglue"));
    EXPECT_EQ(plan.fallbackMatcherAlgorithm, QStringLiteral("sift_bf_l2"));
    EXPECT_TRUE(plan.needsFeatureStage);
    EXPECT_FALSE(plan.endToEndMatcher);
    EXPECT_TRUE(plan.rotationRobust);
    EXPECT_TRUE(plan.preferCuda);
    EXPECT_TRUE(plan.reason.contains(QStringLiteral("旋转鲁棒性")));
}

TEST(MatchPhotosAlgorithmSelectorTest, CudaProfileKeepsSiftLightGlueAndPrefersCuda)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.profile = xjw::matchphotos::MatchPhotosProfile::CudaAccelerated;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_EQ(plan.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(plan.matcherAlgorithm, QStringLiteral("lightglue"));
    EXPECT_TRUE(plan.preferCuda);
}

TEST(MatchPhotosAlgorithmSelectorTest, CpuCompatibleProfileKeepsAutoOnCpu)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.profile = xjw::matchphotos::MatchPhotosProfile::CpuCompatible;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_FALSE(plan.preferCuda);
}

TEST(MatchPhotosAlgorithmSelectorTest, ProfileDoesNotOverrideDisabledGuidedMatching)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.profile = xjw::matchphotos::MatchPhotosProfile::DifficultTexture;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_EQ(plan.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(plan.matcherAlgorithm, QStringLiteral("lightglue"));
    EXPECT_FALSE(plan.enableGuidedMatching);
    EXPECT_GE(plan.maxKeypoints, 12000);
}

TEST(MatchPhotosAlgorithmSelectorTest, ExplicitGuidedMatchingRemainsEnabled)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.profile = xjw::matchphotos::MatchPhotosProfile::DifficultTexture;
    options.enableGuidedMatching = true;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.enableGuidedMatching);
}

TEST(MatchPhotosAlgorithmSelectorTest, ExplicitZeroKeypointLimitMeansUnlimited)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.useExplicitKeypointLimit = true;
    options.maxKeypoints = 0;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_EQ(plan.maxKeypoints, 0);
}
