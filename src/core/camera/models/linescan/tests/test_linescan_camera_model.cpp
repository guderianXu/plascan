#include "camera/core/capabilities/CapabilityRequirements.h"
#include "camera/models/linescan/LineScanDefinition.h"
#include "camera/models/linescan/LineScanInstance.h"
#include "camera/models/linescan/LineScanOptics.h"
#include "camera/models/linescan/LineScanOptimization.h"
#include "camera/models/linescan/LineScanProjection.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>

namespace
{

    using namespace xjw::camera_core;
    using namespace xjw::coordinate_system;
    using namespace xjw::camera_models::linescan;

    std::shared_ptr<const LineScanDefinition> makeDefinition()
    {
        return LineScanDefinition::create(CameraDefinitionId("linescan-definition"),
                                          CoordinateFrameId("body-fixed"),
                                          LineScanOptics{10.0, 0.01, 5.0, 0.0});
    }

    LineScanTrajectory makeTrajectory()
    {
        return LineScanTrajectory::create({
            TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.0),
                             {0.0, -1.0, 0.0},
                             Rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}},
            TrajectorySample{TimeReference::create(TimeScale::Tdb, 1.0),
                             {0.0, 1.0, 0.0},
                             Rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}},
        });
    }

    LineScanInstance makeInstance()
    {
        return LineScanInstance::create(CameraInstanceId("linescan-instance"),
                                        ImageId("linescan-image"),
                                        makeDefinition(),
                                        ImageSize{1000, 11},
                                        makeTrajectory(),
                                        LineTiming{0.5, 0.0, 0.1, TimeScale::Tdb});
    }

    TEST(LineScanCameraModelTest, InterpolatesTrajectoryAndMapsLineTime)
    {
        const LineScanInstance instance = makeInstance();
        double time = 0.0;
        ASSERT_TRUE(instance.timeForLine(5.5, &time));
        EXPECT_DOUBLE_EQ(time, 0.5);

        double line = 0.0;
        ASSERT_TRUE(instance.lineForTime(0.5, &line));
        EXPECT_DOUBLE_EQ(line, 5.5);

        const Pose pose = instance.trajectory().poseAt(TimeReference::create(TimeScale::Tdb, time),
                                                       instance.definition().worldFrame());
        EXPECT_NEAR(pose.center[1], 0.0, 1.0e-12);
    }

    TEST(LineScanCameraModelTest, MapsPiecewiseLineRatesWithoutACompatibilityTimingPath)
    {
        LineTiming timing;
        timing.timeScale = TimeScale::Tdb;
        timing.segments = {{0.5, 0.0, 0.1}, {5.5, 0.7, 0.2}};
        const LineScanInstance instance = LineScanInstance::create(CameraInstanceId("segmented-instance"),
                                                                   ImageId("segmented-image"),
                                                                   makeDefinition(),
                                                                   ImageSize{1000, 11},
                                                                   makeTrajectory(),
                                                                   timing);
        double time = 0.0;
        ASSERT_TRUE(instance.timeForLine(4.5, &time));
        EXPECT_NEAR(time, 0.4, 1.0e-12);
        ASSERT_TRUE(instance.timeForLine(5.5, &time));
        EXPECT_NEAR(time, 0.7, 1.0e-12);
        double line = 0.0;
        ASSERT_TRUE(instance.lineForTime(0.9, &line));
        EXPECT_NEAR(line, 6.5, 1.0e-12);
    }

    TEST(LineScanCameraModelTest, ProjectsBySolvingAcquisitionLine)
    {
        const LineScanInstance instance = makeInstance();
        const std::array<double, 3> world{{2.0, 0.0, 10.0}};
        LineScanProjectionResult projection;
        ASSERT_TRUE(LineScanProjection::project(instance, world, &projection));
        EXPECT_NEAR(projection.sample, 205.0, 1.0e-9);
        EXPECT_NEAR(projection.line, 5.5, 1.0e-7);
        EXPECT_NEAR(projection.lineResidualPixels, 0.0, 1.0e-7);
        EXPECT_NEAR(projection.timeSeconds, 0.5, 1.0e-7);

        LineScanRay ray;
        ASSERT_TRUE(LineScanProjection::ray(instance, projection.sample, projection.line, &ray));
        EXPECT_NEAR(ray.origin[1], 0.0, 1.0e-7);
        EXPECT_NEAR(ray.direction[0], 2.0 / std::sqrt(104.0), 1.0e-7);
        EXPECT_NEAR(ray.direction[2], 10.0 / std::sqrt(104.0), 1.0e-7);

        const CapabilityCheckResult staticPoseCheck =
            requireCapabilities(instance, CapabilitySet{CapabilityKind::StaticPose});
        EXPECT_FALSE(staticPoseCheck.ok());
    }

    TEST(LineScanCameraModelTest, AppliesTrajectoryBiasAndRejectsInvalidInput)
    {
        const LineScanTrajectoryBias current{{{1.0, 2.0, 3.0}}, {{0.1, 0.2, 0.3}}, 4.0};
        const LineScanTrajectoryBias updated =
            LineScanOptimization::applyUpdate(current, std::array<double, 7>{{0.5, -1.0, 2.0, 0.01, 0.02, 0.03, -0.5}});
        EXPECT_DOUBLE_EQ(updated.translationMeters[0], 1.5);
        EXPECT_DOUBLE_EQ(updated.translationMeters[1], 1.0);
        EXPECT_DOUBLE_EQ(updated.rotationVectorRadians[2], 0.33);
        EXPECT_DOUBLE_EQ(updated.timeOffsetSeconds, 3.5);

        std::array<double, 7> invalid{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN()}};
        EXPECT_THROW(LineScanOptimization::applyUpdate(current, invalid), CameraValidationError);
    }

    TEST(LineScanCameraModelTest, OpticsTransformSupportsDetectorAndNormalizedRadialModels)
    {
        const auto radialDefinition = LineScanDefinition::create(CameraDefinitionId("radial-definition"),
                                                                 CoordinateFrameId("body-fixed"),
                                                                 LineScanOptics{10.0, 0.01, 5.0, 0.01});
        FocalPlaneCoordinate focal;
        ASSERT_TRUE(LineScanOpticsTransform::pixelToUndistortedFocal(*radialDefinition, 205.0, &focal));
        double sample = 0.0;
        double lineResidual = 0.0;
        ASSERT_TRUE(LineScanOpticsTransform::undistortedFocalToPixel(
            *radialDefinition, focal, &sample, &lineResidual));
        EXPECT_NEAR(sample, 205.0, 1.0e-10);
        EXPECT_NEAR(lineResidual, 0.0, 1.0e-12);

        LineScanOptics detectorOptics;
        detectorOptics.focalLengthMillimeters = 100.0;
        detectorOptics.distortionK1 = 1.0e-5;
        detectorOptics.distortionModel = LineScanDistortionModel::LroNacFocalPlane;
        LineScanDetectorGeometry detector;
        detector.detectorSampleOrigin = 100.0;
        detector.focalToPixelLines = {0.0, 100.0, 0.0};
        detector.focalToPixelSamples = {0.0, 0.0, 100.0};
        detectorOptics.detectorGeometry = detector;
        const auto detectorDefinition = LineScanDefinition::create(CameraDefinitionId("detector-definition"),
                                                                   CoordinateFrameId("body-fixed"),
                                                                   detectorOptics);
        ASSERT_TRUE(LineScanOpticsTransform::pixelToUndistortedFocal(*detectorDefinition, 110.0, &focal));
        ASSERT_TRUE(LineScanOpticsTransform::undistortedFocalToPixel(
            *detectorDefinition, focal, &sample, &lineResidual));
        EXPECT_NEAR(sample, 110.0, 1.0e-10);
        EXPECT_NEAR(lineResidual, 0.0, 1.0e-10);
    }

    TEST(LineScanCameraModelTest, RejectsInvalidTrajectory)
    {
        EXPECT_THROW(LineScanTrajectory::create({
                         TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.0)},
                     }),
                     CameraValidationError);
    }

    TEST(LineScanCameraModelTest, RejectsTimingWithDifferentTimeScale)
    {
        EXPECT_THROW(LineScanInstance::create(CameraInstanceId("mismatched-instance"),
                                              ImageId("mismatched-image"),
                                              makeDefinition(),
                                              ImageSize{1000, 11},
                                              makeTrajectory(),
                                              LineTiming{0.5, 0.0, 0.1, TimeScale::Utc}),
                     CameraValidationError);
    }

} // namespace
