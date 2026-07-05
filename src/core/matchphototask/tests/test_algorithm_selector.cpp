#include "MatchPhotosAlgorithmSelector.h"

#include <gtest/gtest.h>

TEST(MatchPhotosAlgorithmSelectorTest, AutoUsesSiftLightGlueAsMetashapeLikeDefault)
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
    EXPECT_FALSE(plan.preferCuda);
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

TEST(MatchPhotosAlgorithmSelectorTest, DifficultTextureEnablesGuidedMatching)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.profile = xjw::matchphotos::MatchPhotosProfile::DifficultTexture;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_EQ(plan.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(plan.matcherAlgorithm, QStringLiteral("lightglue"));
    EXPECT_TRUE(plan.enableGuidedMatching);
    EXPECT_GE(plan.maxKeypoints, 12000);
}
