#include "MatchPhotosAlgorithmSelector.h"

#include <gtest/gtest.h>

TEST(MatchPhotosAlgorithmSelectorTest, DefaultUsesRegisteredSiftLightGlue)
{
    xjw::matchphotos::MatchPhotosOptions options;

    const xjw::matchphotos::MatchPhotosAlgorithmPlan plan =
        xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.valid) << qPrintable(plan.validationError);
    EXPECT_EQ(plan.algorithmId, QStringLiteral("sift_lightglue"));
    EXPECT_GT(plan.algorithmVersion, 0u);
    EXPECT_EQ(plan.displayName, QStringLiteral("CUDA SIFT + TensorRT LightGlue"));
    EXPECT_TRUE(plan.extractsFeaturesInMemory);
    EXPECT_TRUE(plan.requiresCuda);
    EXPECT_TRUE(plan.rotationRobust);
    EXPECT_TRUE(plan.preferCuda);
    EXPECT_TRUE(plan.reason.contains(QStringLiteral(".pimatch")));
}

TEST(MatchPhotosAlgorithmSelectorTest, RejectsUnknownAlgorithm)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.algorithmId = QStringLiteral("removed_matcher");

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_FALSE(plan.valid);
    EXPECT_TRUE(plan.validationError.contains(QStringLiteral("未注册")));
}

TEST(MatchPhotosAlgorithmSelectorTest, RejectsCpuForCudaOnlyAlgorithm)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.device = xjw::matchphotos::ComputeDevice::Cpu;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_FALSE(plan.valid);
    EXPECT_TRUE(plan.validationError.contains(QStringLiteral("CUDA")));
}

TEST(MatchPhotosAlgorithmSelectorTest, GuidedMatchingRemainsExplicit)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.profile = xjw::matchphotos::MatchPhotosProfile::DifficultTexture;
    options.enableGuidedMatching = true;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.valid);
    EXPECT_TRUE(plan.enableGuidedMatching);
    EXPECT_GE(plan.maxKeypoints, 12000);
}

TEST(MatchPhotosAlgorithmSelectorTest, ExplicitZeroKeypointLimitMeansUnlimited)
{
    xjw::matchphotos::MatchPhotosOptions options;
    options.useExplicitKeypointLimit = true;
    options.maxKeypoints = 0;

    const auto plan = xjw::matchphotos::MatchPhotosAlgorithmSelector::select(options);

    EXPECT_TRUE(plan.valid);
    EXPECT_EQ(plan.maxKeypoints, 0);
}
