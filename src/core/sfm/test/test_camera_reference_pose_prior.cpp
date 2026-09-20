#include "camera/reference/resolve/CameraReferencePosePrior.h"
#include "camera/models/frame_pinhole/FramePinholeDefinition.h"
#include "camera/models/frame_pinhole/FramePinholeInstance.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "pose/CameraReferencePosePriorAdapter.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace
{

    using namespace xjw;
    using namespace xjw::camera_core;
    using namespace xjw::coordinate_system;
    using namespace xjw::camera_models::frame_pinhole;
    using namespace xjw::camera_reference;

    std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> makeCameras(const CoordinateFrameId& frame)
    {
        const auto definition = FramePinholeDefinition::create(CameraDefinitionId("definition"),
                                                               Intrinsics{100.0, 100.0, 50.0, 50.0, 1.0, 1, 1},
                                                               Distortion{},
                                                               PixelConvention::PixelCenter,
                                                               frame);

        std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> cameras;
        for (int index = 0; index < 2; ++index)
        {
            const auto instance =
                FramePinholeInstance::create(CameraInstanceId("instance-" + std::to_string(index)),
                                             ImageId("image-" + std::to_string(index)),
                                             definition,
                                             ImageSize{100, 100},
                                             Pose::create(frame,
                                                          {static_cast<double>(index), 0.0, 0.0},
                                                          Rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}));
            xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
            EXPECT_TRUE(xjw::camera_models::frame_pinhole::FramePinholeNumericState::fromInstance(instance, &state));
            cameras.push_back(std::move(state));
        }
        return cameras;
    }

    ResolvedCameraPosePrior makePrior(const char* image,
                                      const char* frame,
                                      std::optional<PoseCovariance> covariance = std::nullopt,
                                      const char* provenance = "provenance-hash")
    {
        CameraReferenceObservation observation{ImageId(image), ReferenceSourceId("gnss"), CoordinateFrameId(frame)};
        ResolvedCameraReference resolved;
        resolved.status = ReferenceResolutionStatus::Resolved;
        resolved.targetFrame = CoordinateFrameId(frame);
        resolved.pose = Pose::create(
            CoordinateFrameId(frame), {10.0, 20.0, 30.0}, Rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}});
        resolved.covariance = std::move(covariance);
        resolved.transformProvenanceHash = provenance;
        resolved.transformHash = "transform-hash";
        const auto result = makeResolvedCameraPosePrior(observation, resolved);
        EXPECT_TRUE(result.ok()) << result.reason;
        return *result.prior;
    }

    xjw::camera_models::frame_pinhole::FramePinholeNumericState makeUnboundCamera()
    {
        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        camera.setIntrinsics(100.0, 100.0, 50.0, 50.0);
        camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0});
        return camera;
    }

    TEST(CameraReferencePosePriorTest, RejectsUnresolvedReference)
    {
        CameraReferenceObservation observation{
            ImageId("image-0"), ReferenceSourceId("gnss"), CoordinateFrameId("world")};
        ResolvedCameraReference unresolved;
        unresolved.targetFrame = CoordinateFrameId("world");
        unresolved.reason = "orientation is missing";

        const auto result = makeResolvedCameraPosePrior(observation, unresolved);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.reason.find("orientation"), std::string::npos);
    }

    TEST(CameraReferencePosePriorTest, RequiresBothResolutionAndNormalizationProvenance)
    {
        CameraReferenceObservation observation{
            ImageId("image-0"), ReferenceSourceId("gnss"), CoordinateFrameId("world")};
        ResolvedCameraReference resolved;
        resolved.status = ReferenceResolutionStatus::Resolved;
        resolved.targetFrame = CoordinateFrameId("world");
        resolved.pose = Pose::create(
            CoordinateFrameId("world"), {0.0, 0.0, 0.0}, Rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}});
        resolved.transformHash = "resolution-hash";

        const auto result = makeResolvedCameraPosePrior(observation, resolved);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.reason.find("normalization provenance"), std::string::npos);
    }

    TEST(CameraReferencePosePriorAdapterTest, MapsImageKeyedPriorsAndDiagonalCovariance)
    {
        const auto cameras = makeCameras(CoordinateFrameId("world"));
        const auto covariance = PoseCovariance::diagonal({4.0, 9.0, 16.0, 0.01, 0.04, 0.09});
        const std::vector<ResolvedCameraPosePrior> references{
            makePrior("image-1", "world", covariance),
        };

        const auto result = CameraReferencePosePriorAdapter::toBundleAdjustPriors(cameras, references);
        ASSERT_TRUE(result.ok()) << result.error;
        ASSERT_EQ(result.priors.size(), cameras.size());
        EXPECT_FALSE(result.priors[0].enabled);
        ASSERT_TRUE(result.priors[1].enabled);
        EXPECT_EQ(result.matchedReferenceCount, 1u);
        EXPECT_EQ(result.priors[1].cameraCenter, (std::array<double, 3>{10.0, 20.0, 30.0}));
        EXPECT_NEAR(result.priors[1].positionSigmaMeters, 4.0, 1.0e-12);
        EXPECT_NEAR(result.priors[1].rotationSigmaDegrees, std::sqrt(0.09) * 180.0 / 3.14159265358979323846, 1.0e-12);
    }

    TEST(CameraReferencePosePriorAdapterTest, RejectsMatchedReferenceFrameMismatch)
    {
        const auto cameras = makeCameras(CoordinateFrameId("world"));
        const std::vector<ResolvedCameraPosePrior> references{
            makePrior("image-0", "ecef"),
        };

        const auto result = CameraReferencePosePriorAdapter::toBundleAdjustPriors(cameras, references);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error.find("image-0"), std::string::npos);
        EXPECT_NE(result.error.find("world"), std::string::npos);
        EXPECT_NE(result.error.find("ecef"), std::string::npos);
    }

    TEST(CameraReferencePosePriorAdapterTest, RejectsDuplicateMatchedReferences)
    {
        const auto cameras = makeCameras(CoordinateFrameId("world"));
        const std::vector<ResolvedCameraPosePrior> references{
            makePrior("image-0", "world"),
            makePrior("image-0", "world"),
        };

        const auto result = CameraReferencePosePriorAdapter::toBundleAdjustPriors(cameras, references);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error.find("duplicate"), std::string::npos);
    }

    TEST(CameraReferencePosePriorAdapterTest, RejectsMixedFrameNormalizationProvenance)
    {
        const auto cameras = makeCameras(CoordinateFrameId("world"));
        const std::vector<ResolvedCameraPosePrior> references{
            makePrior("image-0", "world"),
            makePrior("image-1", "world", std::nullopt, "different-provenance"),
        };

        const auto result = CameraReferencePosePriorAdapter::toBundleAdjustPriors(cameras, references);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error.find("normalization provenance"), std::string::npos);
    }

    TEST(CameraReferencePosePriorAdapterTest, IgnoresOutOfWindowReferencesWithoutMixingProvenance)
    {
        const auto cameras = makeCameras(CoordinateFrameId("world"));
        const std::vector<ResolvedCameraPosePrior> references{
            makePrior("image-0", "world"),
            makePrior("outside-window", "world", std::nullopt, "different-provenance"),
        };

        const auto result = CameraReferencePosePriorAdapter::toBundleAdjustPriors(cameras, references);
        ASSERT_TRUE(result.ok()) << result.error;
        EXPECT_EQ(result.matchedReferenceCount, 1u);
        EXPECT_EQ(result.ignoredReferenceCount, 1u);
        ASSERT_EQ(result.ignoredReferenceImages.size(), 1u);
        EXPECT_EQ(result.ignoredReferenceImages.front(), ImageId("outside-window"));
    }

    TEST(CameraReferencePosePriorAdapterTest, RejectsUnboundNumericCameraIdentity)
    {
        const std::vector<xjw::camera_models::frame_pinhole::FramePinholeNumericState> cameras{makeUnboundCamera()};
        const std::vector<ResolvedCameraPosePrior> references{makePrior("image-0", "uninitialized-frame")};

        const auto result = CameraReferencePosePriorAdapter::toBundleAdjustPriors(cameras, references);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error.find("explicit image identity"), std::string::npos);
    }

    TEST(FramePinholeNumericStateTest, BindsExplicitIdentityOnlyOnce)
    {
        auto camera = makeUnboundCamera();
        EXPECT_FALSE(camera.hasBoundIdentity());

        std::string error;
        EXPECT_TRUE(
            camera.bindIdentity(CameraInstanceId("instance"), ImageId("image"), CoordinateFrameId("world"), &error))
            << error;
        EXPECT_TRUE(camera.hasBoundIdentity());
        EXPECT_EQ(camera.imageId(), ImageId("image"));
        EXPECT_EQ(camera.worldFrame(), CoordinateFrameId("world"));

        EXPECT_FALSE(camera.bindIdentity(
            CameraInstanceId("other-instance"), ImageId("other-image"), CoordinateFrameId("other-world"), &error));
        EXPECT_NE(error.find("already"), std::string::npos);
    }

} // namespace
