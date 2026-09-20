#include "camera/reference/compare/CameraReferenceComparator.h"

#include "camera/models/frame_pinhole/FramePinholeDefinition.h"
#include "camera/models/frame_pinhole/FramePinholeInstance.h"
#include "camera/models/rpc/RpcDefinition.h"
#include "camera/models/rpc/RpcInstance.h"

#include <gtest/gtest.h>

#include <memory>

namespace
{

    using namespace xjw::camera_core;
    using namespace xjw::coordinate_system;
    using namespace xjw::camera_models;
    using namespace xjw::camera_reference;

    std::shared_ptr<const frame_pinhole::FramePinholeInstance> pinholeInstance()
    {
        const auto definition = frame_pinhole::FramePinholeDefinition::create(
            CameraDefinitionId("definition"),
            frame_pinhole::Intrinsics{100.0, 100.0, 50.0, 50.0, 1.0, 1, 1},
            frame_pinhole::Distortion{},
            frame_pinhole::PixelConvention::PixelCenter,
            CoordinateFrameId("world"));
        return std::make_shared<const frame_pinhole::FramePinholeInstance>(frame_pinhole::FramePinholeInstance::create(
            CameraInstanceId("instance"),
            ImageId("image"),
            definition,
            ImageSize{100, 100},
            Pose::create(
                CoordinateFrameId("world"), {1.0, 2.0, 3.0}, Rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}})));
    }

    ResolvedCameraReference resolvedReference(const char* frame = "world")
    {
        ResolvedCameraReference reference;
        reference.status = ReferenceResolutionStatus::Resolved;
        reference.targetFrame = CoordinateFrameId(frame);
        reference.pose = Pose::create(
            CoordinateFrameId(frame), {1.0, 2.0, 5.0}, Rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}});
        return reference;
    }

    TEST(CameraReferenceComparatorTest, ComparesStaticPinholePose)
    {
        const CameraReferenceComparison comparison =
            CameraReferenceComparator::compare(*pinholeInstance(), resolvedReference());

        ASSERT_TRUE(comparison.compared());
        ASSERT_TRUE(comparison.positionError.has_value());
        EXPECT_DOUBLE_EQ(*comparison.positionError, 2.0);
        ASSERT_TRUE(comparison.rotationErrorRadians.has_value());
        EXPECT_NEAR(*comparison.rotationErrorRadians, 0.0, 1.0e-12);
    }

    TEST(CameraReferenceComparatorTest, ReportsFrameMismatchBeforeGeometry)
    {
        const CameraReferenceComparison comparison =
            CameraReferenceComparator::compare(*pinholeInstance(), resolvedReference("ecef"));
        EXPECT_EQ(comparison.status, CameraReferenceComparisonStatus::FrameMismatch);
        EXPECT_FALSE(comparison.positionError.has_value());
    }

    TEST(CameraReferenceComparatorTest, ReportsRegionModelWithoutStaticPose)
    {
        rpc::RpcDefinition::Parameters parameters;
        parameters.lineScale = 1.0;
        parameters.sampleScale = 1.0;
        parameters.latitudeScale = 1.0;
        parameters.longitudeScale = 1.0;
        parameters.heightScale = 1.0;
        parameters.lineDenominator[0] = 1.0;
        parameters.sampleDenominator[0] = 1.0;
        const auto instance = rpc::RpcInstance::create(
            CameraInstanceId("rpc-instance"),
            ImageId("rpc-image"),
            rpc::RpcDefinition::create(CameraDefinitionId("rpc-definition"), CoordinateFrameId("world"), parameters),
            ImageSize{100, 100});

        const CameraReferenceComparison comparison = CameraReferenceComparator::compare(instance, resolvedReference());
        EXPECT_EQ(comparison.status, CameraReferenceComparisonStatus::RegionModelWithoutStaticPose);
        EXPECT_NE(comparison.reason.find("static pose"), std::string::npos);
    }

} // namespace
