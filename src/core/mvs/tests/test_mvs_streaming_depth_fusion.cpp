#include "StreamingDepthFusionService.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <memory>
#include <vector>

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

TEST(StreamingDepthFusionServiceTest, RequiresGeometrySupportWhenThresholdIsEnabled)
{
    xjw::mvs::StreamingDepthFusionConfig config;
    config.minConsistentViews = 2;
    config.neighborCount = 1;
    config.useColor = false;

    const xjw::mvs::FusionFrameLoader loader =
        [](int frameIndex, xjw::mvs::FusionFrameInput *frame, std::string *)
        {
            frame->depthMap = cv::Mat(8, 8, CV_32FC1, cv::Scalar(5.0f));
            frame->cameraModel.setIntrinsics(20.0, 20.0, 4.0, 4.0);
            frame->cameraModel.setPose(
                {1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0},
                {static_cast<double>(frameIndex), 0.0, 0.0});
            frame->imgW = frame->depthMap.cols;
            frame->imgH = frame->depthMap.rows;
            return true;
        };

    xjw::mvs::StreamingDepthFusionResult result;
    std::string error;
    EXPECT_FALSE(xjw::mvs::fuseDepthMapsStreaming(
        2, config, loader, &result, &error));
    EXPECT_NE(error.find("几何支持计数"), std::string::npos);
}

TEST(StreamingDepthFusionServiceTest,
     ScalesReprojectionGateForEachNativeTargetFrame)
{
    constexpr int kGridSize = 9;
    constexpr int kFrameCount = 3;
    const std::array<double, kFrameCount> principal_offsets{
        0.0, 0.34, 0.68};

    const auto make_frames = [&](int raster_scale)
    {
        std::vector<xjw::mvs::FusionFrameInput> frames(kFrameCount);
        for (int index = 0; index < kFrameCount; ++index)
        {
            auto &frame = frames[static_cast<std::size_t>(index)];
            frame.depthMap = cv::Mat::zeros(
                kGridSize, kGridSize, CV_32FC1);
            frame.depthMap.at<float>(4, 4) = 8.0f;
            frame.geometrySupportCount = cv::Mat(
                kGridSize, kGridSize, CV_16UC1, cv::Scalar(2));
            frame.cameraModel.setIntrinsics(
                20.0,
                20.0,
                4.0 + principal_offsets[static_cast<std::size_t>(index)],
                4.0);
            frame.cameraModel.setPose(
                {1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0},
                {0.0, 0.0, 0.0});
            frame.cameraModel.setImageSize(
                xjw::CameraImageSize{kGridSize, kGridSize});
            frame.sourceCamera = frame.cameraModel.scaledIntrinsics(
                static_cast<double>(raster_scale),
                static_cast<double>(raster_scale));
            frame.sourceCamera.setImageSize(xjw::CameraImageSize{
                kGridSize * raster_scale,
                kGridSize * raster_scale});
            frame.imgW = kGridSize;
            frame.imgH = kGridSize;
            frame.viewIndex = index;
        }
        return frames;
    };

    xjw::mvs::StreamingDepthFusionConfig config;
    config.minConsistentViews = 2;
    config.depthConsistency = 1.0f;
    config.neighborCount = 2;
    config.workerCount = 1;
    config.useColor = false;

    std::vector<xjw::mvs::FusionFrameInput> native_frames = make_frames(4);
    const xjw::mvs::FusionFrameLoader native_loader =
        [&native_frames](int index,
                         xjw::mvs::FusionFrameInput *frame,
                         std::string *)
        {
            *frame = native_frames[static_cast<std::size_t>(index)];
            return true;
        };
    xjw::mvs::StreamingDepthFusionResult native_result;
    std::string error;
    EXPECT_FALSE(xjw::mvs::fuseDepthMapsStreaming(
        kFrameCount,
        config,
        native_loader,
        &native_result,
        &error));
    EXPECT_TRUE(native_result.points.empty());

    std::vector<xjw::mvs::FusionFrameInput> full_grid_frames = make_frames(1);
    const xjw::mvs::FusionFrameLoader full_grid_loader =
        [&full_grid_frames](int index,
                            xjw::mvs::FusionFrameInput *frame,
                            std::string *)
        {
            *frame = full_grid_frames[static_cast<std::size_t>(index)];
            return true;
        };
    xjw::mvs::StreamingDepthFusionResult full_grid_result;
    error.clear();
    ASSERT_TRUE(xjw::mvs::fuseDepthMapsStreaming(
        kFrameCount,
        config,
        full_grid_loader,
        &full_grid_result,
        &error)) << error;
    EXPECT_EQ(full_grid_result.points.size(),
              static_cast<std::size_t>(kFrameCount));
}

} // namespace
