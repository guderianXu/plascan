#include "StreamingDepthFusionService.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>

namespace
{

TEST(StreamingDepthFusionServiceTest, BuildsBalancedReferenceWindows)
{
    EXPECT_EQ(xjw::mvs::streamingFusionWindowIndices(0, 5, 3),
              (std::vector<int>{0, 1, 2, 3}));
    EXPECT_EQ(xjw::mvs::streamingFusionWindowIndices(2, 5, 3),
              (std::vector<int>{2, 1, 3, 0}));
    EXPECT_EQ(xjw::mvs::streamingFusionWindowIndices(4, 5, 3),
              (std::vector<int>{4, 3, 2, 1}));
}

TEST(StreamingDepthFusionServiceTest, RejectsInvalidInputsBeforeLoadingFrames)
{
    xjw::mvs::StreamingDepthFusionResult result;
    std::string error;
    int loadCount = 0;
    const xjw::mvs::FusionFrameLoader loader =
        [&loadCount](int, xjw::mvs::FusionFrameInput *, std::string *) {
            ++loadCount;
            return true;
        };

    EXPECT_FALSE(xjw::mvs::fuseDepthMapsStreaming(
        1, {}, loader, &result, &error));
    EXPECT_EQ(loadCount, 0);
    EXPECT_FALSE(error.empty());
}

TEST(StreamingDepthFusionServiceTest, StopsBeforeLoadingWhenAlreadyCancelled)
{
    xjw::mvs::StreamingDepthFusionConfig config;
    config.cancelFlag = std::make_shared<std::atomic_bool>(true);

    xjw::mvs::StreamingDepthFusionResult result;
    std::string error;
    int loadCount = 0;
    const xjw::mvs::FusionFrameLoader loader =
        [&loadCount](int, xjw::mvs::FusionFrameInput *, std::string *) {
            ++loadCount;
            return true;
        };

    EXPECT_FALSE(xjw::mvs::fuseDepthMapsStreaming(
        2, config, loader, &result, &error));
    EXPECT_EQ(loadCount, 0);
    EXPECT_NE(error.find("cancelled"), std::string::npos);
}

} // namespace
