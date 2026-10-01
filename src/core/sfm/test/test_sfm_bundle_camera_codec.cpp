#include <gtest/gtest.h>

#include <placamera/frame_camera.h>

#include "pipeline/SfmBundleCameraCodec.h"

#include <array>
#include <string>
#include <vector>

namespace
{
    placamera::FramePinholeNumericState makeState(int index, int samples = 1280)
    {
        const placamera::FrameId world("solver-world");
        const auto suffix = std::to_string(index);
        placamera::FrameIntrinsics intrinsics;
        intrinsics.focalX = 1350.0;
        intrinsics.focalY = 1325.0;
        intrinsics.principalX = 620.5;
        intrinsics.principalY = 470.25;
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("ba-definition-" + suffix),
                                                      intrinsics,
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      world);
        const auto model = placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("ba-instance-" + suffix),
            placamera::ImageId("ba-image-" + suffix),
            definition,
            {samples, 960},
            placamera::Pose::create(world, {2.0, -3.0, 5.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
        return placamera::FramePinholeNumericState::fromModel(model);
    }
} // namespace

TEST(SfmBundleCameraCodecTest, BatchBoundaryPreservesCameraIdentityAndOrder)
{
    const std::vector<placamera::FramePinholeNumericState> sources{makeState(1), makeState(2)};
    std::vector<placamera::FramePinholeNumericState> encoded;
    std::string error;
    ASSERT_TRUE(xjw::sfm_bundle_camera::encodeAll(sources, &encoded, &error)) << error;
    ASSERT_EQ(encoded.size(), sources.size());
    EXPECT_EQ(encoded[0].instanceId(), sources[0].instanceId());
    EXPECT_EQ(encoded[1].imageId(), sources[1].imageId());

    encoded[0].applyPoseDelta({0.0, 0.0, 0.0, 0.1, -0.2, 0.3});
    auto decoded = sources;
    ASSERT_TRUE(xjw::sfm_bundle_camera::decodeAll(encoded, &decoded, &error)) << error;
    EXPECT_EQ(decoded[0].pose().center, encoded[0].pose().center);
    EXPECT_EQ(decoded[1].instanceId(), sources[1].instanceId());
}

TEST(SfmBundleCameraCodecTest, RejectedBatchDoesNotPartiallyApplyResults)
{
    std::vector<placamera::FramePinholeNumericState> targets{makeState(1), makeState(2)};
    auto results = targets;
    results[0].applyPoseDelta({0.0, 0.0, 0.0, 0.1, -0.2, 0.3});
    results[1] = makeState(3);

    std::string error;
    EXPECT_FALSE(xjw::sfm_bundle_camera::decodeAll(results, &targets, &error));
    EXPECT_EQ(targets[0].pose().center, makeState(1).pose().center);
    EXPECT_EQ(targets[1].instanceId(), makeState(2).instanceId());
    EXPECT_NE(error.find("identity"), std::string::npos);
}

TEST(SfmBundleCameraCodecTest, RejectsImageGridChange)
{
    auto target = makeState(1);
    std::string error;
    EXPECT_FALSE(xjw::sfm_bundle_camera::decode(makeState(1, 640), &target, &error));
    EXPECT_EQ(target.imageSize().samples, 1280);
}
