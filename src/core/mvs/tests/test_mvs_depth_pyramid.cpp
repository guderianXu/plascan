#include "DepthPyramidPolicy.h"

#include <gtest/gtest.h>

namespace
{

TEST(MvsDepthPyramidPolicyTest, SmallImagesKeepAFullResolutionFinalLevel)
{
    xjw::mvs::PatchMatchConfig base_config;
    base_config.downsampleFactor = 4;

    const xjw::mvs::DepthPyramidConfig config =
        xjw::mvs::makeDepthPyramidConfig(base_config, 640, 480);

    ASSERT_EQ(config.activeLevelCount, 2);
    EXPECT_EQ(config.levels[0].level, 2);
    EXPECT_EQ(config.levels[0].patchMatch.downsampleFactor, 2);
    EXPECT_EQ(config.levels[1].level, 1);
    EXPECT_EQ(config.levels[1].patchMatch.downsampleFactor, 1);

    const cv::Size final_size = xjw::mvs::depthPyramidWorkingSize(
        640,
        480,
        config.levels[config.activeLevelCount - 1].patchMatch.downsampleFactor);
    EXPECT_GE(std::min(final_size.width, final_size.height), 320);
}

TEST(MvsDepthPyramidPolicyTest, LargeAerialImagesKeepRequestedFinalDownsample)
{
    xjw::mvs::PatchMatchConfig base_config;
    base_config.downsampleFactor = 4;

    const xjw::mvs::DepthPyramidConfig config =
        xjw::mvs::makeDepthPyramidConfig(base_config, 6000, 4000);

    ASSERT_EQ(config.activeLevelCount, 3);
    EXPECT_EQ(config.levels[0].patchMatch.downsampleFactor, 16);
    EXPECT_EQ(config.levels[1].patchMatch.downsampleFactor, 8);
    EXPECT_EQ(config.levels[2].patchMatch.downsampleFactor, 4);
}

} // namespace
