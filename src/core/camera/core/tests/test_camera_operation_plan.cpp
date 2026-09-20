#include "camera/core/capabilities/CameraOperationPlan.h"

#include <gtest/gtest.h>

#include <memory>

namespace
{

    std::shared_ptr<const xjw::camera_core::CameraDefinition>
    makeDefinition(const std::string& model, const std::string& frame, xjw::camera_core::CapabilitySet capabilities)
    {
        return std::make_shared<const xjw::camera_core::CameraDefinition>(
            xjw::camera_core::CameraDefinitionId(model + "-definition"),
            model,
            xjw::coordinate_system::CoordinateFrameId(frame),
            1,
            std::move(capabilities));
    }

    std::shared_ptr<const xjw::camera_core::CameraInstance>
    makeInstance(const std::string& instanceId,
                 const std::string& imageId,
                 const std::shared_ptr<const xjw::camera_core::CameraDefinition>& definition)
    {
        return std::make_shared<const xjw::camera_core::CameraInstance>(xjw::camera_core::CameraInstanceId(instanceId),
                                                                        xjw::camera_core::ImageId(imageId),
                                                                        definition,
                                                                        xjw::camera_core::ImageSize{640, 480},
                                                                        std::nullopt,
                                                                        xjw::camera_core::CapabilitySet{});
    }

} // namespace

TEST(CameraOperationPlanTest, StaticOperationsShareOneCapabilityContract)
{
    const auto definition =
        makeDefinition("frame_pinhole",
                       "world",
                       xjw::camera_core::CapabilitySet{xjw::camera_core::CapabilityKind::Projection,
                                                       xjw::camera_core::CapabilityKind::Ray,
                                                       xjw::camera_core::CapabilityKind::StaticPose,
                                                       xjw::camera_core::CapabilityKind::Optimization});
    xjw::camera_core::CameraInstanceSet instances;
    ASSERT_TRUE(instances.add(makeInstance("instance", "image", definition)));

    const auto sfm = xjw::camera_core::planCameraOperation(instances, xjw::camera_core::CameraOperation::StaticSfM);
    const auto ba =
        xjw::camera_core::planCameraOperation(instances, xjw::camera_core::CameraOperation::BundleAdjustment);
    const auto mvs = xjw::camera_core::planCameraOperation(instances, xjw::camera_core::CameraOperation::DenseMvs);

    EXPECT_TRUE(sfm.ok());
    EXPECT_TRUE(ba.ok());
    EXPECT_TRUE(mvs.ok());
    ASSERT_TRUE(sfm.commonWorldFrame.has_value());
    EXPECT_EQ(sfm.commonWorldFrame->value(), "world");
}

TEST(CameraOperationPlanTest, OrthoProjectionRequiresProjectionAndStaticPoseWithoutOptimization)
{
    const auto definition =
        makeDefinition("frame_pinhole",
                       "world",
                       xjw::camera_core::CapabilitySet{xjw::camera_core::CapabilityKind::Projection,
                                                       xjw::camera_core::CapabilityKind::StaticPose});
    xjw::camera_core::CameraInstanceSet instances;
    ASSERT_TRUE(instances.add(makeInstance("instance", "image", definition)));

    const auto ortho =
        xjw::camera_core::planCameraOperation(instances, xjw::camera_core::CameraOperation::OrthoProjection);

    EXPECT_TRUE(ortho.ok()) << ortho.failureMessage();
    EXPECT_STREQ(xjw::camera_core::cameraOperationName(xjw::camera_core::CameraOperation::OrthoProjection),
                 "ortho_projection");
}

TEST(CameraOperationPlanTest, RpcAndPushbroomUseDedicatedContracts)
{
    const auto rpcDefinition =
        makeDefinition("rpc00b",
                       "EPSG:4978",
                       xjw::camera_core::CapabilitySet{xjw::camera_core::CapabilityKind::Projection,
                                                       xjw::camera_core::CapabilityKind::InverseProjection,
                                                       xjw::camera_core::CapabilityKind::Ray});
    xjw::camera_core::CameraInstanceSet rpcInstances;
    ASSERT_TRUE(rpcInstances.add(makeInstance("rpc-instance", "rpc-image", rpcDefinition)));

    const auto rpc =
        xjw::camera_core::planCameraOperation(rpcInstances, xjw::camera_core::CameraOperation::RpcAerialTriangulation);
    const auto staticPlan =
        xjw::camera_core::planCameraOperation(rpcInstances, xjw::camera_core::CameraOperation::StaticSfM);
    const auto orthoPlan =
        xjw::camera_core::planCameraOperation(rpcInstances, xjw::camera_core::CameraOperation::OrthoProjection);
    EXPECT_TRUE(rpc.ok());
    EXPECT_FALSE(staticPlan.ok());
    EXPECT_FALSE(orthoPlan.ok());
    EXPECT_NE(staticPlan.failureMessage().find("static_sfm"), std::string::npos);
    EXPECT_NE(staticPlan.failureMessage().find("static_pose"), std::string::npos);
    EXPECT_NE(orthoPlan.failureMessage().find("ortho_projection"), std::string::npos);
    EXPECT_NE(orthoPlan.failureMessage().find("static_pose"), std::string::npos);

    const auto pushbroomDefinition =
        makeDefinition("planetary_linescan",
                       "orbit",
                       xjw::camera_core::CapabilitySet{xjw::camera_core::CapabilityKind::Projection,
                                                       xjw::camera_core::CapabilityKind::Ray,
                                                       xjw::camera_core::CapabilityKind::Trajectory,
                                                       xjw::camera_core::CapabilityKind::Optimization});
    xjw::camera_core::CameraInstanceSet pushbroomInstances;
    ASSERT_TRUE(pushbroomInstances.add(makeInstance("line-instance", "line-image", pushbroomDefinition)));
    const auto pushbroom = xjw::camera_core::planCameraOperation(
        pushbroomInstances, xjw::camera_core::CameraOperation::PushbroomAerialTriangulation);
    EXPECT_TRUE(pushbroom.ok());
}

TEST(CameraOperationPlanTest, ReportsMixedWorldFramesAtTheWorkflowBoundary)
{
    const auto firstDefinition =
        makeDefinition("frame_pinhole",
                       "world-a",
                       xjw::camera_core::CapabilitySet{xjw::camera_core::CapabilityKind::Projection,
                                                       xjw::camera_core::CapabilityKind::Ray,
                                                       xjw::camera_core::CapabilityKind::StaticPose,
                                                       xjw::camera_core::CapabilityKind::Optimization});
    const auto secondDefinition =
        makeDefinition("frame_pinhole",
                       "world-b",
                       xjw::camera_core::CapabilitySet{xjw::camera_core::CapabilityKind::Projection,
                                                       xjw::camera_core::CapabilityKind::Ray,
                                                       xjw::camera_core::CapabilityKind::StaticPose,
                                                       xjw::camera_core::CapabilityKind::Optimization});
    xjw::camera_core::CameraInstanceSet instances;
    ASSERT_TRUE(instances.add(makeInstance("first", "image-a", firstDefinition)));
    ASSERT_TRUE(instances.add(makeInstance("second", "image-b", secondDefinition)));

    const auto plan =
        xjw::camera_core::planCameraOperation(instances, xjw::camera_core::CameraOperation::BundleAdjustment);
    EXPECT_FALSE(plan.ok());
    EXPECT_EQ(plan.failures.size(), 1U);
    EXPECT_NE(plan.failureMessage().find("image-b"), std::string::npos);
    EXPECT_NE(plan.failureMessage().find("world-b"), std::string::npos);
}

TEST(CameraOperationPlanTest, RejectsEmptyCameraSelection)
{
    const xjw::camera_core::CameraInstanceSet instances;
    const auto plan = xjw::camera_core::planCameraOperation(instances, xjw::camera_core::CameraOperation::DenseMvs);

    EXPECT_FALSE(plan.ok());
    EXPECT_FALSE(plan.inputError.empty());
    EXPECT_NE(plan.failureMessage().find("at least one"), std::string::npos);
}

TEST(CameraOperationPlanTest, SelectsInstancesByCanonicalImageId)
{
    const auto definition =
        makeDefinition("frame_pinhole",
                       "world",
                       xjw::camera_core::CapabilitySet{xjw::camera_core::CapabilityKind::Projection,
                                                       xjw::camera_core::CapabilityKind::Ray,
                                                       xjw::camera_core::CapabilityKind::StaticPose,
                                                       xjw::camera_core::CapabilityKind::Optimization});
    xjw::camera_core::CameraInstanceSet instances;
    ASSERT_TRUE(instances.add(makeInstance("first", "image-a", definition)));
    ASSERT_TRUE(instances.add(makeInstance("second", "image-b", definition)));

    std::string error;
    const auto selected =
        instances.select({xjw::camera_core::ImageId("image-b"), xjw::camera_core::ImageId("image-a")}, &error);
    EXPECT_TRUE(error.empty()) << error;
    ASSERT_EQ(selected.size(), 2U);
    EXPECT_EQ(selected.values().at(0)->imageId().value(), "image-b");
    EXPECT_EQ(selected.values().at(1)->imageId().value(), "image-a");

    const auto duplicate =
        instances.select({xjw::camera_core::ImageId("image-a"), xjw::camera_core::ImageId("image-a")}, &error);
    EXPECT_TRUE(duplicate.empty());
    EXPECT_NE(error.find("duplicate"), std::string::npos);
}
