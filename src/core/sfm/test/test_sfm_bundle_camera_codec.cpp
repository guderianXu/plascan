#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <plabundle/camera.h>

#include "pipeline/SfmBundleCameraCodec.h"

namespace
{

    std::array<double, 9> yRotation(double angle)
    {
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        return {cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine};
    }

    placamera::FramePinholeNumericState
    makeState(double angle, bool depth_flipped, int u_axis_sign, int v_axis_sign, int index = 0)
    {
        const placamera::FrameId frame("solver-world");
        const auto suffix = std::to_string(index);
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("ba-definition-" + suffix),
                                                      {1350.0, 1325.0, 620.5, 470.25, 0.0042, u_axis_sign, v_axis_sign},
                                                      {0.012, -0.003, 0.0004, 0.0005, -0.0007},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame,
                                                      depth_flipped);
        return placamera::FramePinholeNumericState::fromModel(
            placamera::FramePinholeModel::create(placamera::CameraInstanceId("ba-instance-" + suffix),
                                                 placamera::ImageId("ba-image-" + suffix),
                                                 definition,
                                                 {1280, 960},
                                                 placamera::Pose::create(frame, {2.0, -3.0, 5.0}, yRotation(angle))));
    }

    std::array<double, 3> cameraToWorld(const placamera::Pose& pose, const std::array<double, 3>& camera_point)
    {
        const auto& rotation = pose.cameraToWorldRotation;
        const auto& center = pose.center;
        return {
            rotation[0] * camera_point[0] + rotation[1] * camera_point[1] + rotation[2] * camera_point[2] + center[0],
            rotation[3] * camera_point[0] + rotation[4] * camera_point[1] + rotation[5] * camera_point[2] + center[1],
            rotation[6] * camera_point[0] + rotation[7] * camera_point[1] + rotation[8] * camera_point[2] + center[2]};
    }

} // namespace

TEST(SfmBundleCameraCodecTest, RandomizedProjectionMatchesPlaCamera)
{
    std::mt19937 generator(20260920U);
    std::uniform_real_distribution<double> angle_distribution(-0.8, 0.8);
    std::uniform_real_distribution<double> xy_distribution(-1.5, 1.5);
    std::uniform_real_distribution<double> depth_distribution(2.0, 20.0);

    for (int camera_index = 0; camera_index < 16; ++camera_index)
    {
        const bool depth_flipped = camera_index % 2 != 0;
        const int u_sign = camera_index % 3 == 0 ? -1 : 1;
        const int v_sign = camera_index % 5 == 0 ? -1 : 1;
        const auto state = makeState(angle_distribution(generator), depth_flipped, u_sign, v_sign, camera_index);
        plabundle::FrameCamera camera;
        std::string error;
        ASSERT_TRUE(xjw::sfm_bundle_camera::encode(state, &camera, &error)) << error;

        for (int point_index = 0; point_index < 64; ++point_index)
        {
            const double positive_depth = depth_distribution(generator);
            const std::array<double, 3> camera_point{xy_distribution(generator),
                                                     xy_distribution(generator),
                                                     depth_flipped ? -positive_depth : positive_depth};
            const auto world = cameraToWorld(state.pose(), camera_point);
            const auto native = state.groundToImage({state.groundFrame(), world});
            ASSERT_TRUE(native) << native.message();
            ASSERT_TRUE(native.value().positiveDepth.has_value());
            plabundle::Projection projection;
            ASSERT_TRUE(plabundle::projectWorldPoint(camera, world, &projection));
            EXPECT_NEAR(projection.pixel[0], native.value().image.sample, 1.0e-11);
            EXPECT_NEAR(projection.pixel[1], native.value().image.line, 1.0e-11);
            EXPECT_NEAR(projection.positiveDepth, *native.value().positiveDepth, 1.0e-12);
        }
    }
}

TEST(SfmBundleCameraCodecTest, RoundTripPreservesIdentityAndNumericState)
{
    auto state = makeState(0.2, false, -1, 1);
    plabundle::FrameCamera camera;
    std::string error;
    ASSERT_TRUE(xjw::sfm_bundle_camera::encode(state, &camera, &error)) << error;
    camera.cameraCenter = {7.0, 8.0, 9.0};
    camera.focalXPixels = 1400.0;
    camera.distortion.k1 = -0.02;
    ASSERT_TRUE(xjw::sfm_bundle_camera::decode(camera, &state, &error)) << error;

    EXPECT_EQ(state.instanceId(), placamera::CameraInstanceId("ba-instance-0"));
    EXPECT_EQ(state.imageId(), placamera::ImageId("ba-image-0"));
    EXPECT_EQ(state.groundFrame(), placamera::FrameId("solver-world"));
    EXPECT_EQ(state.pose().center, camera.cameraCenter);
    EXPECT_DOUBLE_EQ(state.intrinsics().focalX, 1400.0);
    EXPECT_DOUBLE_EQ(state.distortion().radialK1, -0.02);
    EXPECT_EQ(state.imageSize().samples, 1280);
    EXPECT_EQ(state.imageSize().lines, 960);
}

TEST(SfmBundleCameraCodecTest, PoseDeltaMatchesPlaCameraNumericState)
{
    const auto state = makeState(0.2, false, -1, 1);
    plabundle::FrameCamera camera;
    ASSERT_TRUE(xjw::sfm_bundle_camera::encode(state, &camera));
    const std::array<double, 6> delta{0.03, -0.02, 0.01, 1.5, -2.0, 0.75};

    auto native_state = state;
    native_state.applyPoseDelta(delta);
    ASSERT_TRUE(plabundle::applyPoseDelta(&camera, delta));
    for (std::size_t index = 0; index < 9; ++index)
    {
        EXPECT_NEAR(camera.cameraToWorldRotation[index], native_state.pose().cameraToWorldRotation[index], 1.0e-14);
    }
    for (std::size_t index = 0; index < 3; ++index)
    {
        EXPECT_NEAR(camera.cameraCenter[index], native_state.pose().center[index], 1.0e-14);
    }
}

TEST(SfmBundleCameraCodecTest, InvalidInputDoesNotPartiallyMutateTarget)
{
    auto state = makeState(0.0, false, 1, 1);
    const auto original_center = state.pose().center;
    const double original_focal = state.intrinsics().focalX;
    plabundle::FrameCamera camera;
    ASSERT_TRUE(xjw::sfm_bundle_camera::encode(state, &camera));
    camera.cameraCenter = {9.0, 9.0, 9.0};
    camera.pixelPitchMillimeters = 0.0;

    EXPECT_FALSE(xjw::sfm_bundle_camera::decode(camera, &state));
    EXPECT_EQ(state.pose().center, original_center);
    EXPECT_DOUBLE_EQ(state.intrinsics().focalX, original_focal);
}

TEST(SfmBundleCameraCodecTest, BatchConversionPreservesOrderAndCount)
{
    auto first = makeState(0.1, false, 1, 1, 1);
    auto second = makeState(-0.3, true, -1, 1, 2);
    second.setPose(
        placamera::Pose::create(second.groundFrame(), {-4.0, 6.0, 8.0}, second.pose().cameraToWorldRotation));

    std::vector<plabundle::FrameCamera> cameras;
    std::string error;
    ASSERT_TRUE(xjw::sfm_bundle_camera::encodeAll({first, second}, &cameras, &error)) << error;

    ASSERT_EQ(cameras.size(), 2U);
    EXPECT_EQ(cameras[0].cameraCenter, first.pose().center);
    EXPECT_EQ(cameras[1].cameraCenter, second.pose().center);
    EXPECT_FALSE(cameras[0].depthAxisFlipped);
    EXPECT_TRUE(cameras[1].depthAxisFlipped);
}

TEST(SfmBundleCameraCodecTest, InvalidBatchInputDoesNotPartiallyMutateTargets)
{
    std::vector<placamera::FramePinholeNumericState> targets{makeState(0.0, false, 1, 1, 1),
                                                             makeState(0.2, false, 1, 1, 2)};
    const auto original_first_center = targets[0].pose().center;
    const auto original_second_center = targets[1].pose().center;
    const double original_first_focal = targets[0].intrinsics().focalX;

    std::vector<plabundle::FrameCamera> cameras;
    ASSERT_TRUE(xjw::sfm_bundle_camera::encodeAll(targets, &cameras));
    cameras[0].cameraCenter = {100.0, 200.0, 300.0};
    cameras[0].focalXPixels = 1700.0;
    cameras[1].pixelPitchMillimeters = 0.0;

    EXPECT_FALSE(xjw::sfm_bundle_camera::decodeAll(cameras, &targets));
    EXPECT_EQ(targets[0].pose().center, original_first_center);
    EXPECT_EQ(targets[1].pose().center, original_second_center);
    EXPECT_DOUBLE_EQ(targets[0].intrinsics().focalX, original_first_focal);
}
