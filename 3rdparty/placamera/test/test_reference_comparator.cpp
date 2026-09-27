#include "placamera/reference/CameraReferenceComparator.h"

#include <placamera/frame_camera.h>
#include <placamera/rpc_camera.h>

#include <gtest/gtest.h>

namespace
{

    using namespace placamera::reference;

    placamera::FramePinholeModel pinholeInstance()
    {
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("definition"),
                                                      placamera::FrameIntrinsics{100.0, 100.0, 50.0, 50.0, 1.0, 1, 1},
                                                      placamera::BrownConradyDistortion{},
                                                      placamera::PixelConvention::PixelCenter,
                                                      placamera::FrameId("world"));
        return placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("instance"),
            placamera::ImageId("image"),
            definition,
            placamera::ImageSize{100, 100},
            placamera::Pose::create(
                placamera::FrameId("world"), {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    }

    ResolvedCameraReference resolvedReference(const char* frame = "world")
    {
        ResolvedCameraReference reference;
        reference.status = ReferenceResolutionStatus::Resolved;
        reference.targetFrame = placoordinate::CoordinateFrameId(frame);
        reference.pose = placamera::Pose::create(
            placoordinate::CoordinateFrameId(frame), {1.0, 2.0, 5.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        return reference;
    }

    TEST(CameraReferenceComparatorTest, ComparesStaticPinholePose)
    {
        const CameraReferenceComparison comparison =
            CameraReferenceComparator::compare(pinholeInstance(), resolvedReference());

        ASSERT_TRUE(comparison.compared());
        ASSERT_TRUE(comparison.positionError.has_value());
        EXPECT_DOUBLE_EQ(*comparison.positionError, 2.0);
        ASSERT_TRUE(comparison.rotationErrorRadians.has_value());
        EXPECT_NEAR(*comparison.rotationErrorRadians, 0.0, 1.0e-12);
    }

    TEST(CameraReferenceComparatorTest, ReportsFrameMismatchBeforeGeometry)
    {
        const CameraReferenceComparison comparison =
            CameraReferenceComparator::compare(pinholeInstance(), resolvedReference("ecef"));
        EXPECT_EQ(comparison.status, CameraReferenceComparisonStatus::FrameMismatch);
        EXPECT_FALSE(comparison.positionError.has_value());
    }

    TEST(CameraReferenceComparatorTest, ReportsRegionModelWithoutStaticPose)
    {
        placamera::RpcParameters parameters;
        parameters.lineScale = 1.0;
        parameters.sampleScale = 1.0;
        parameters.latitudeScale = 1.0;
        parameters.longitudeScale = 1.0;
        parameters.heightScale = 1.0;
        parameters.lineDenominator[0] = 1.0;
        parameters.sampleDenominator[0] = 1.0;
        const auto instance = placamera::RpcModel::create(
            placamera::CameraInstanceId("rpc-instance"),
            placamera::ImageId("rpc-image"),
            placamera::RpcDefinition::create(
                placamera::CameraDefinitionId("rpc-definition"), placamera::FrameId("world"), parameters),
            placamera::ImageSize{100, 100});

        const CameraReferenceComparison comparison = CameraReferenceComparator::compare(instance, resolvedReference());
        EXPECT_EQ(comparison.status, CameraReferenceComparisonStatus::RegionModelWithoutStaticPose);
        EXPECT_NE(comparison.reason.find("static pose"), std::string::npos);
    }

} // namespace
