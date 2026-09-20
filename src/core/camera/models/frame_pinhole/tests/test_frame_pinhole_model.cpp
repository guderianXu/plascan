#include "camera/models/frame_pinhole/FramePinholeDefinition.h"
#include "camera/models/frame_pinhole/FramePinholeInstance.h"
#include "camera/models/frame_pinhole/FramePinholeOptimization.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/models/frame_pinhole/FramePinholeProjection.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace
{

    using namespace xjw::camera_core;
    using namespace xjw::coordinate_system;
    using namespace xjw::camera_models::frame_pinhole;

    std::shared_ptr<const FramePinholeDefinition>
    makeDefinition(bool depthAxisFlipped = false, int uAxisSign = 1, int vAxisSign = 1, Distortion distortion = {})
    {
        Intrinsics intrinsics;
        intrinsics.focalX = 100.0;
        intrinsics.focalY = 110.0;
        intrinsics.principalX = 50.0;
        intrinsics.principalY = 60.0;
        intrinsics.pixelPitch = 0.01;
        intrinsics.uAxisSign = uAxisSign;
        intrinsics.vAxisSign = vAxisSign;
        return FramePinholeDefinition::create(CameraDefinitionId("definition-1"),
                                              intrinsics,
                                              distortion,
                                              PixelConvention::PixelCenter,
                                              CoordinateFrameId("world"),
                                              depthAxisFlipped);
    }

    FramePinholeInstance makeInstance(std::shared_ptr<const FramePinholeDefinition> definition)
    {
        return FramePinholeInstance::create(CameraInstanceId("instance-1"),
                                            ImageId("image-1"),
                                            std::move(definition),
                                            ImageSize{1000, 800},
                                            Pose::create(CoordinateFrameId("world"),
                                                         {0.0, 0.0, 0.0},
                                                         Rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}));
    }

    TEST(FramePinholeModelTest, ProjectsAndUnprojectsWithSharedPose)
    {
        const auto instance = makeInstance(makeDefinition());

        ProjectionResult projection;
        ASSERT_TRUE(FramePinholeProjection::project(instance, {1.0, 2.0, 10.0}, &projection));
        EXPECT_NEAR(projection.pixel[0], 60.0, 1.0e-12);
        EXPECT_NEAR(projection.pixel[1], 82.0, 1.0e-12);
        EXPECT_DOUBLE_EQ(projection.positiveDepth, 10.0);

        std::array<double, 3> world{};
        ASSERT_TRUE(FramePinholeProjection::unproject(instance, projection.pixel, projection.positiveDepth, &world));
        EXPECT_NEAR(world[0], 1.0, 1.0e-9);
        EXPECT_NEAR(world[1], 2.0, 1.0e-9);
        EXPECT_NEAR(world[2], 10.0, 1.0e-9);
    }

    TEST(FramePinholeModelTest, PreservesDistortedProjectionAfterPositiveDepthNormalization)
    {
        Distortion distortion;
        distortion.radialK1 = 0.03;
        distortion.radialK2 = -0.002;
        distortion.tangentialP1 = 0.0015;
        distortion.tangentialP2 = -0.0007;
        const auto definition = makeDefinition(true, -1, -1, distortion);
        const auto instance = makeInstance(definition);

        ProjectionResult original;
        ASSERT_TRUE(FramePinholeProjection::project(instance, {0.4, -0.3, -5.0}, &original));
        const auto normalized = instance.normalizedForPositiveDepth(CameraDefinitionId("definition-normalized"),
                                                                    CameraInstanceId("instance-normalized"));
        ProjectionResult normalizedProjection;
        ASSERT_TRUE(FramePinholeProjection::project(normalized, {0.4, -0.3, -5.0}, &normalizedProjection));
        EXPECT_NEAR(normalizedProjection.pixel[0], original.pixel[0], 1.0e-9);
        EXPECT_NEAR(normalizedProjection.pixel[1], original.pixel[1], 1.0e-9);
        EXPECT_FALSE(normalized.pinholeDefinition().depthAxisFlipped());
        EXPECT_EQ(normalized.pinholeDefinition().intrinsics().uAxisSign, -1);
        EXPECT_EQ(normalized.pinholeDefinition().intrinsics().vAxisSign, 1);
    }

    TEST(FramePinholeNumericStateTest, PreservesDistortedProjectionAfterPositiveDepthNormalization)
    {
        Distortion distortion;
        distortion.radialK1 = 0.03;
        distortion.radialK2 = -0.002;
        distortion.tangentialP1 = 0.0015;
        distortion.tangentialP2 = -0.0007;
        const auto instance = makeInstance(makeDefinition(true, -1, -1, distortion));

        FramePinholeNumericState state;
        std::string error;
        ASSERT_TRUE(FramePinholeNumericState::fromInstance(instance, &state, &error)) << error;

        const double world[3] = {0.4, -0.3, -5.0};
        double originalPixel[2] = {};
        ASSERT_TRUE(state.projectWorldPoint(world, originalPixel));

        const FramePinholeNumericState normalized = state.normalizedForPositiveDepth();
        double normalizedPixel[2] = {};
        ASSERT_TRUE(normalized.projectWorldPoint(world, normalizedPixel));
        EXPECT_NEAR(normalizedPixel[0], originalPixel[0], 1.0e-9);
        EXPECT_NEAR(normalizedPixel[1], originalPixel[1], 1.0e-9);
        EXPECT_FALSE(normalized.depthAxisFlipped());
        EXPECT_EQ(normalized.uAxisSign(), -1);
        EXPECT_EQ(normalized.vAxisSign(), 1);
    }

    TEST(FramePinholeNumericStateTest, DoesNotPromoteUnboundStateToTypedInstance)
    {
        FramePinholeNumericState state;
        state.setIntrinsics(100.0, 100.0, 50.0, 50.0);
        state.setImageSize(ImageSize{100, 100});
        state.setPose({{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}, {{0.0, 0.0, 0.0}});

        EXPECT_TRUE(state.isValid());
        EXPECT_FALSE(state.hasBoundIdentity());
        EXPECT_EQ(state.toInstance(), nullptr);

        std::string error;
        ASSERT_TRUE(
            state.bindIdentity(CameraInstanceId("instance-1"), ImageId("image-1"), CoordinateFrameId("world"), &error))
            << error;
        EXPECT_TRUE(state.hasBoundIdentity());
        EXPECT_NE(state.toInstance(), nullptr);
    }

    TEST(FramePinholeModelTest, PoseAndCalibrationUpdatesCreateNewValues)
    {
        const auto definition = makeDefinition();
        const auto instance = makeInstance(definition);
        const auto updatedInstance = FramePinholeOptimization::applyPoseUpdate(
            instance, CameraInstanceId("instance-updated"), {0.0, 0.0, 0.0, 1.0, 2.0, 3.0});
        EXPECT_EQ(instance.pose().center, (std::array<double, 3>{0.0, 0.0, 0.0}));
        EXPECT_EQ(updatedInstance.pose().center, (std::array<double, 3>{1.0, 2.0, 3.0}));
        EXPECT_EQ(updatedInstance.instanceId().value(), "instance-updated");

        const auto updatedDefinition = FramePinholeOptimization::applyCalibrationUpdate(
            *definition, CameraDefinitionId("definition-updated"), {10.0, -5.0, 1.0, -2.0});
        EXPECT_DOUBLE_EQ(definition->intrinsics().focalX, 100.0);
        EXPECT_DOUBLE_EQ(updatedDefinition->intrinsics().focalX, 110.0);
        EXPECT_DOUBLE_EQ(updatedDefinition->intrinsics().principalY, 58.0);
    }

    TEST(FramePinholeModelTest, RejectsInvalidIntrinsics)
    {
        Intrinsics intrinsics;
        intrinsics.focalX = -1.0;
        intrinsics.focalY = 100.0;
        intrinsics.principalX = 0.0;
        intrinsics.principalY = 0.0;
        EXPECT_THROW(FramePinholeDefinition::create(CameraDefinitionId("definition-invalid"),
                                                    intrinsics,
                                                    Distortion{},
                                                    PixelConvention::PixelCenter,
                                                    CoordinateFrameId("world")),
                     CameraValidationError);
    }

} // namespace
