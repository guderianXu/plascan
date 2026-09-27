#include <placoordinate/transform/CoordinateTransformService.h>
#include "placamera/reference/CameraReferenceObservation.h"
#include "placamera/reference/CameraReferenceResolver.h"

#include <gtest/gtest.h>

namespace
{

    using namespace placamera::reference;
    using namespace placoordinate;

    CoordinateFrame rootFrame(const char* id)
    {
        return CoordinateFrame::create(CoordinateFrameId(id),
                                       CoordinateFrameKind::Ecef,
                                       LinearUnit::Metre,
                                       AngleUnit::Degree,
                                       std::nullopt,
                                       RigidTransform::identity());
    }

    TEST(CameraReferenceResolverTest, TransformsPointThroughExplicitFrameGraph)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("ecef"));
        transforms.registerFrame(CoordinateFrame::create(
            CoordinateFrameId("local"),
            CoordinateFrameKind::LocalEnu,
            LinearUnit::Metre,
            AngleUnit::Degree,
            CoordinateFrameId("ecef"),
            RigidTransform{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {100.0, 200.0, 300.0}}));

        const auto ecef =
            transforms.transformPoint(CoordinateFrameId("local"), CoordinateFrameId("ecef"), {1.0, 2.0, 3.0});
        EXPECT_EQ(ecef, (std::array<double, 3>{101.0, 202.0, 303.0}));

        const auto local = transforms.transformPoint(CoordinateFrameId("ecef"), CoordinateFrameId("local"), ecef);
        EXPECT_EQ(local, (std::array<double, 3>{1.0, 2.0, 3.0}));
    }

    TEST(CameraReferenceResolverTest, RejectsMissingOrientationConvention)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("local"));

        CameraReferenceObservation observation{
            placamera::ImageId("image-1"), ReferenceSourceId("gnss"), CoordinateFrameId("local")};
        observation.position = std::array<double, 3>{1.0, 2.0, 3.0};
        observation.orientation = placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};

        CameraReferenceResolver resolver(&transforms);
        const ResolvedCameraReference resolved =
            resolver.resolve(observation, CoordinateFrameId("local"), CameraReferenceResolveOptions{});
        EXPECT_EQ(resolved.status, ReferenceResolutionStatus::Unresolved);
        EXPECT_NE(resolved.reason.find("orientation"), std::string::npos);
    }

    TEST(CameraReferenceResolverTest, RejectsUnknownFrame)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("local"));

        CameraReferenceObservation observation{
            placamera::ImageId("image-1"), ReferenceSourceId("gnss"), CoordinateFrameId("missing")};
        observation.position = std::array<double, 3>{1.0, 2.0, 3.0};
        observation.orientation = placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};

        CameraReferenceResolveOptions options;
        options.orientationConvention = "camera_to_world";
        CameraReferenceResolver resolver(&transforms);
        const ResolvedCameraReference resolved = resolver.resolve(observation, CoordinateFrameId("local"), options);
        EXPECT_EQ(resolved.status, ReferenceResolutionStatus::Unresolved);
        EXPECT_NE(resolved.reason.find("unknown coordinate frame"), std::string::npos);
    }

    TEST(CameraReferenceResolverTest, RejectsLeverArmWithoutDirection)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("local"));

        CameraReferenceObservation observation{
            placamera::ImageId("image-1"), ReferenceSourceId("gnss"), CoordinateFrameId("local")};
        observation.position = std::array<double, 3>{1.0, 2.0, 3.0};
        observation.leverArm = LeverArm{{0.1, 0.0, 0.0}, LeverArmDirection::Unknown};

        CameraReferenceResolver resolver(&transforms);
        const ResolvedCameraReference resolved =
            resolver.resolve(observation, CoordinateFrameId("local"), CameraReferenceResolveOptions{});
        EXPECT_EQ(resolved.status, ReferenceResolutionStatus::Unresolved);
        EXPECT_NE(resolved.reason.find("lever arm"), std::string::npos);
    }

    TEST(CameraReferenceResolverTest, RejectsLeverArmWithoutVectorFrame)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("local"));

        CameraReferenceObservation observation{
            placamera::ImageId("image-1"), ReferenceSourceId("gnss"), CoordinateFrameId("local")};
        observation.position = std::array<double, 3>{1.0, 2.0, 3.0};
        observation.orientation = placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        observation.leverArm = LeverArm{{0.1, 0.0, 0.0}, LeverArmDirection::SensorToCamera};

        CameraReferenceResolveOptions options;
        options.orientationConvention = "camera_to_world";
        const ResolvedCameraReference resolved =
            CameraReferenceResolver(&transforms).resolve(observation, CoordinateFrameId("local"), options);
        EXPECT_EQ(resolved.status, ReferenceResolutionStatus::Unresolved);
        EXPECT_NE(resolved.reason.find("vector frame"), std::string::npos);
    }

    TEST(CameraReferenceResolverTest, AppliesLeverArmThroughExplicitVectorFrame)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("world"));
        transforms.registerFrame(
            CoordinateFrame::create(CoordinateFrameId("sensor"),
                                    CoordinateFrameKind::BodyFixed,
                                    LinearUnit::Metre,
                                    AngleUnit::Degree,
                                    CoordinateFrameId("world"),
                                    RigidTransform{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}}));

        CameraReferenceObservation observation{
            placamera::ImageId("image-1"), ReferenceSourceId("gnss"), CoordinateFrameId("world")};
        observation.position = std::array<double, 3>{1.0, 2.0, 3.0};
        observation.orientation = placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        observation.leverArm =
            LeverArm{{0.1, 0.2, 0.3}, LeverArmDirection::SensorToCamera, CoordinateFrameId("sensor")};

        CameraReferenceResolveOptions options;
        options.orientationConvention = "camera_to_world";
        const ResolvedCameraReference resolved =
            CameraReferenceResolver(&transforms).resolve(observation, CoordinateFrameId("world"), options);
        ASSERT_EQ(resolved.status, ReferenceResolutionStatus::Resolved);
        ASSERT_TRUE(resolved.pose.has_value());
        EXPECT_EQ(resolved.pose->center, (std::array<double, 3>{1.1, 2.2, 3.3}));
        EXPECT_TRUE(resolved.leverArmApplied);
    }

    TEST(CameraReferenceResolverTest, PropagatesCovarianceAcrossFrames)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("world"));
        transforms.registerFrame(CoordinateFrame::create(CoordinateFrameId("local"),
                                                         CoordinateFrameKind::LocalEnu,
                                                         LinearUnit::Metre,
                                                         AngleUnit::Degree,
                                                         CoordinateFrameId("world"),
                                                         RigidTransform::identity()));

        CameraReferenceObservation observation{
            placamera::ImageId("image-1"), ReferenceSourceId("gnss"), CoordinateFrameId("local")};
        observation.position = std::array<double, 3>{1.0, 2.0, 3.0};
        observation.orientation = placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        observation.covariance = PoseCovariance::diagonal({1.0, 1.0, 1.0, 0.01, 0.01, 0.01});

        CameraReferenceResolveOptions options;
        options.orientationConvention = "camera_to_world";
        const ResolvedCameraReference resolved =
            CameraReferenceResolver(&transforms).resolve(observation, CoordinateFrameId("world"), options);
        ASSERT_EQ(resolved.status, ReferenceResolutionStatus::Resolved) << resolved.reason;
        ASSERT_TRUE(resolved.covariance);
        EXPECT_DOUBLE_EQ(resolved.covariance->matrixValues()[0], 1.0);
        EXPECT_DOUBLE_EQ(resolved.covariance->matrixValues()[21], 0.01);
    }

    TEST(CameraReferenceResolverTest, RetainsPositionOnlyAndRotationOnlyReferences)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("world"));
        CameraReferenceResolver resolver(&transforms);

        CameraReferenceObservation position_only{
            placamera::ImageId("position"), ReferenceSourceId("gnss"), CoordinateFrameId("world")};
        position_only.position = std::array<double, 3>{1.0, 2.0, 3.0};
        position_only.covariance = PoseCovariance::positionDiagonal({4.0, 9.0, 16.0});
        const ResolvedCameraReference resolved_position =
            resolver.resolve(position_only, CoordinateFrameId("world"), {});
        ASSERT_EQ(resolved_position.status, ReferenceResolutionStatus::Resolved) << resolved_position.reason;
        EXPECT_TRUE(resolved_position.position);
        EXPECT_FALSE(resolved_position.orientation);
        EXPECT_FALSE(resolved_position.pose);
        ASSERT_TRUE(resolved_position.covariance);
        EXPECT_EQ(resolved_position.covariance->components(), PoseCovarianceComponents::PositionOnly);

        CameraReferenceObservation rotation_only{
            placamera::ImageId("rotation"), ReferenceSourceId("imu"), CoordinateFrameId("world")};
        rotation_only.orientation = placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        rotation_only.covariance = PoseCovariance::rotationDiagonal({0.01, 0.04, 0.09});
        CameraReferenceResolveOptions options;
        options.orientationConvention = "camera_to_world";
        const ResolvedCameraReference resolved_rotation =
            resolver.resolve(rotation_only, CoordinateFrameId("world"), options);
        ASSERT_EQ(resolved_rotation.status, ReferenceResolutionStatus::Resolved) << resolved_rotation.reason;
        EXPECT_FALSE(resolved_rotation.position);
        EXPECT_TRUE(resolved_rotation.orientation);
        EXPECT_FALSE(resolved_rotation.pose);
        ASSERT_TRUE(resolved_rotation.covariance);
        EXPECT_EQ(resolved_rotation.covariance->components(), PoseCovarianceComponents::RotationOnly);
    }

    TEST(CameraReferenceResolverTest, ResolvesPoseAndRecordsTransformHash)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("local"));

        CameraReferenceObservation observation{
            placamera::ImageId("image-1"), ReferenceSourceId("gnss"), CoordinateFrameId("local")};
        observation.position = std::array<double, 3>{1.0, 2.0, 3.0};
        observation.orientation = placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};

        CameraReferenceResolveOptions options;
        options.orientationConvention = "camera_to_world";
        CameraReferenceResolver resolver(&transforms);
        const ResolvedCameraReference resolved = resolver.resolve(observation, CoordinateFrameId("local"), options);
        ASSERT_EQ(resolved.status, ReferenceResolutionStatus::Resolved);
        ASSERT_TRUE(resolved.pose.has_value());
        EXPECT_EQ(resolved.pose->center, (std::array<double, 3>{1.0, 2.0, 3.0}));
        EXPECT_FALSE(resolved.transformHash.empty());
        EXPECT_FALSE(resolved.transformProvenanceHash.empty());
        EXPECT_EQ(resolved.pose->frame.value(), "local");
    }

    TEST(CameraReferenceResolverTest, KeepsNormalizationProvenanceStableAcrossObservations)
    {
        CoordinateTransformService transforms;
        transforms.registerFrame(rootFrame("local"));
        CameraReferenceResolveOptions options;
        options.orientationConvention = "camera_to_world";
        CameraReferenceResolver resolver(&transforms);

        CameraReferenceObservation first{
            placamera::ImageId("image-1"), ReferenceSourceId("gnss"), CoordinateFrameId("local")};
        first.position = std::array<double, 3>{1.0, 2.0, 3.0};
        first.orientation = placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        CameraReferenceObservation second = first;
        second.image = placamera::ImageId("image-2");
        second.position = std::array<double, 3>{4.0, 5.0, 6.0};

        const auto firstResolved = resolver.resolve(first, CoordinateFrameId("local"), options);
        const auto secondResolved = resolver.resolve(second, CoordinateFrameId("local"), options);
        ASSERT_EQ(firstResolved.status, ReferenceResolutionStatus::Resolved);
        ASSERT_EQ(secondResolved.status, ReferenceResolutionStatus::Resolved);
        EXPECT_EQ(firstResolved.transformProvenanceHash, secondResolved.transformProvenanceHash);
        EXPECT_NE(firstResolved.transformHash, secondResolved.transformHash);
    }

} // namespace
