#include "placamera/reference/CameraReferencePosePrior.h"
#include "pose/CameraReferencePosePriorAdapter.h"

#include <gtest/gtest.h>
#include <placamera/frame_camera.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace
{

    using namespace xjw;
    using namespace placoordinate;
    using namespace placamera::reference;

    std::vector<CameraReferenceTarget> makeCameras(const CoordinateFrameId& frame)
    {
        std::vector<CameraReferenceTarget> cameras;
        for (int index = 0; index < 2; ++index)
        {
            cameras.push_back({placamera::ImageId("image-" + std::to_string(index)), frame});
        }
        return cameras;
    }

    ResolvedCameraPosePrior makePrior(const char* image,
                                      const char* frame,
                                      std::optional<PoseCovariance> covariance = std::nullopt,
                                      const char* provenance = "provenance-hash")
    {
        CameraReferenceObservation observation{
            placamera::ImageId(image), ReferenceSourceId("gnss"), CoordinateFrameId(frame)};
        ResolvedCameraReference resolved;
        resolved.status = ReferenceResolutionStatus::Resolved;
        resolved.targetFrame = CoordinateFrameId(frame);
        resolved.pose =
            placamera::Pose::create(CoordinateFrameId(frame),
                                    {10.0, 20.0, 30.0},
                                    placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}});
        resolved.covariance = std::move(covariance);
        resolved.transformProvenanceHash = provenance;
        resolved.transformHash = "transform-hash";
        const auto result = makeResolvedCameraPosePrior(observation, resolved);
        EXPECT_TRUE(result) << result.message();
        return result.value();
    }

    TEST(CameraReferencePosePriorTest, RejectsUnresolvedReference)
    {
        CameraReferenceObservation observation{
            placamera::ImageId("image-0"), ReferenceSourceId("gnss"), CoordinateFrameId("world")};
        ResolvedCameraReference unresolved;
        unresolved.targetFrame = CoordinateFrameId("world");
        unresolved.reason = "orientation is missing";

        const auto result = makeResolvedCameraPosePrior(observation, unresolved);
        EXPECT_FALSE(result);
        EXPECT_NE(result.message().find("orientation"), std::string::npos);
    }

    TEST(CameraReferencePosePriorTest, RequiresBothResolutionAndNormalizationProvenance)
    {
        CameraReferenceObservation observation{
            placamera::ImageId("image-0"), ReferenceSourceId("gnss"), CoordinateFrameId("world")};
        ResolvedCameraReference resolved;
        resolved.status = ReferenceResolutionStatus::Resolved;
        resolved.targetFrame = CoordinateFrameId("world");
        resolved.pose =
            placamera::Pose::create(CoordinateFrameId("world"),
                                    {0.0, 0.0, 0.0},
                                    placamera::RotationMatrix{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}});
        resolved.transformHash = "resolution-hash";

        const auto result = makeResolvedCameraPosePrior(observation, resolved);
        EXPECT_FALSE(result);
        EXPECT_NE(result.message().find("normalization provenance"), std::string::npos);
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
        EXPECT_FALSE(result.priors[0].has_value());
        ASSERT_TRUE(result.priors[1].has_value());
        EXPECT_EQ(result.matchedReferenceCount, 1u);
        EXPECT_EQ(result.priors[1]->cameraCenter, (std::array<double, 3>{10.0, 20.0, 30.0}));
        EXPECT_NEAR(result.priors[1]->positionSigmaMeters, 4.0, 1.0e-12);
        EXPECT_NEAR(result.priors[1]->rotationSigmaDegrees, std::sqrt(0.09) * 180.0 / 3.14159265358979323846, 1.0e-12);
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
        EXPECT_EQ(result.ignoredReferenceImages.front(), placamera::ImageId("outside-window"));
    }

    TEST(CameraReferencePosePriorAdapterTest, RejectsMixedTargetFrames)
    {
        auto cameras = makeCameras(CoordinateFrameId("world"));
        cameras[1].worldFrame = CoordinateFrameId("other-world");
        const std::vector<ResolvedCameraPosePrior> references{makePrior("image-0", "world")};

        const auto result = CameraReferencePosePriorAdapter::toBundleAdjustPriors(cameras, references);
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error.find("image-1"), std::string::npos);
        EXPECT_NE(result.error.find("other-world"), std::string::npos);
    }

    TEST(CameraReferencePosePriorAdapterTest, RejectsDuplicateTargetImages)
    {
        auto targets = makeCameras(CoordinateFrameId("world"));
        targets[1].imageId = targets[0].imageId;

        const auto result = CameraReferencePosePriorAdapter::toBundleAdjustPriors(targets, {});
        EXPECT_FALSE(result.ok());
        EXPECT_NE(result.error.find("duplicate image"), std::string::npos);
    }

} // namespace
