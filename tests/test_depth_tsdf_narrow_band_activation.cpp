#include "FramePinholeCamera.h"
#include "DepthTsdfNarrowBandActivation.h"
#include "DepthTsdfSurfaceBuilder.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{

xjw::mesh::DepthTsdfLayout makeLayout()
{
    xjw::mesh::DepthTsdfLayout layout;
    layout.ok = true;
    layout.boundsMin = {-1.0f, -1.0f, 0.0f};
    layout.boundsMax = {1.0f, 1.0f, 4.0f};
    layout.cells = {8, 8, 16};
    layout.voxelSize = {0.25f, 0.25f, 0.25f};
    layout.sampleCount = 9U * 9U * 17U;
    return layout;
}

xjw::FramePinholeCamera makeCamera()
{
    xjw::FramePinholeCamera camera;
    camera.setIntrinsics(4.0, 4.0, 2.0, 2.0);
    camera.setPose(
        {1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0},
        {0.0, 0.0, 0.0});
    return camera;
}

xjw::mesh::DepthTsdfNarrowBandActivationOptions makeOptions()
{
    xjw::mesh::DepthTsdfNarrowBandActivationOptions options;
    options.blockSizeSamples = 2;
    options.depthStride = 1;
    options.truncationDistance = 0.30f;
    options.rayStepVoxels = 1.0f;
    return options;
}

xjw::mesh::DepthTsdfNarrowBandFrameView makeView(
    const xjw::FramePinholeCamera &camera,
    const cv::Mat &depth,
    const cv::Mat *depth_valid,
    const cv::Mat *support)
{
    xjw::mesh::DepthTsdfNarrowBandFrameView view;
    view.camera = &camera;
    view.depth = &depth;
    view.depthValidMask = depth_valid;
    view.supportMask = support;
    return view;
}

} // namespace

TEST(DepthTsdfNarrowBandActivationTest, ActivatesOnlyBlocksNearSurface)
{
    const xjw::FramePinholeCamera camera = makeCamera();
    cv::Mat depth = cv::Mat::zeros(5, 5, CV_32FC1);
    cv::Mat valid = cv::Mat::zeros(5, 5, CV_8UC1);
    cv::Mat support(5, 5, CV_8UC1, cv::Scalar(255));
    depth.at<float>(2, 2) = 2.0f;
    valid.at<std::uint8_t>(2, 2) = 255;

    xjw::mesh::DepthTsdfNarrowBandActivation activation;
    ASSERT_TRUE(activation.build(
        makeLayout(),
        {makeView(camera, depth, &valid, &support)},
        makeOptions()));

    EXPECT_TRUE(activation.isSampleActive(4, 4, 8));
    EXPECT_FALSE(activation.isSampleActive(4, 4, 0));
    EXPECT_FALSE(activation.isSampleActive(0, 0, 8));
    EXPECT_EQ(activation.statistics().validSourceSamples, 1U);
    EXPECT_GT(activation.statistics().markedRaySamples, 0U);
    EXPECT_GT(activation.statistics().activeBlocks, 0U);
    EXPECT_LT(
        activation.statistics().activeBlocks,
        activation.statistics().totalBlocks);
}

TEST(DepthTsdfNarrowBandActivationTest, InvalidAndUnsupportedPixelsStayUnknown)
{
    const xjw::FramePinholeCamera camera = makeCamera();
    cv::Mat depth = cv::Mat::zeros(5, 5, CV_32FC1);
    cv::Mat valid = cv::Mat::zeros(5, 5, CV_8UC1);
    cv::Mat support(5, 5, CV_8UC1, cv::Scalar(255));
    depth.at<float>(1, 1) = 2.0f;
    depth.at<float>(2, 2) = 2.0f;
    valid.at<std::uint8_t>(2, 2) = 255;
    support.at<std::uint8_t>(2, 2) = 0;
    depth.at<float>(3, 3) = std::numeric_limits<float>::quiet_NaN();
    valid.at<std::uint8_t>(3, 3) = 255;

    xjw::mesh::DepthTsdfNarrowBandActivation activation;
    ASSERT_TRUE(activation.build(
        makeLayout(),
        {makeView(camera, depth, &valid, &support)},
        makeOptions()));

    EXPECT_EQ(activation.statistics().validSourceSamples, 0U);
    EXPECT_EQ(activation.statistics().markedRaySamples, 0U);
    EXPECT_EQ(activation.statistics().activeBlocks, 0U);
    EXPECT_FALSE(activation.isSampleActive(4, 4, 8));
}

TEST(DepthTsdfNarrowBandActivationTest, HaloExpandsFromCoreBlocksOnce)
{
    const xjw::FramePinholeCamera camera = makeCamera();
    cv::Mat depth = cv::Mat::zeros(5, 5, CV_32FC1);
    cv::Mat valid = cv::Mat::zeros(5, 5, CV_8UC1);
    depth.at<float>(2, 2) = 2.0f;
    valid.at<std::uint8_t>(2, 2) = 255;
    const auto view = makeView(camera, depth, &valid, nullptr);

    xjw::mesh::DepthTsdfNarrowBandActivation without_halo;
    ASSERT_TRUE(without_halo.build(
        makeLayout(), {view}, makeOptions()));
    EXPECT_FALSE(without_halo.isSampleActive(2, 4, 8));

    auto halo_options = makeOptions();
    halo_options.haloBlocks = 1;
    xjw::mesh::DepthTsdfNarrowBandActivation with_halo;
    ASSERT_TRUE(with_halo.build(
        makeLayout(), {view}, halo_options));

    EXPECT_TRUE(with_halo.isSampleActive(2, 4, 8));
    EXPECT_GT(
        with_halo.statistics().activeBlocks,
        without_halo.statistics().activeBlocks);
}

TEST(DepthTsdfNarrowBandActivationTest, CancellationLeavesNoPartialMask)
{
    const xjw::FramePinholeCamera camera = makeCamera();
    cv::Mat depth(5, 5, CV_32FC1, cv::Scalar(2.0f));
    auto options = makeOptions();
    options.isCancelled = []()
    {
        return true;
    };

    xjw::mesh::DepthTsdfNarrowBandActivation activation;
    EXPECT_FALSE(activation.build(
        makeLayout(),
        {makeView(camera, depth, nullptr, nullptr)},
        options));

    EXPECT_TRUE(activation.wasCancelled());
    EXPECT_FALSE(activation.isValid());
    EXPECT_EQ(activation.statistics().activeBlocks, 0U);
    EXPECT_FALSE(activation.isSampleActive(4, 4, 8));
}
