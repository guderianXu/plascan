#include <placamera/linescan_camera.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace
{

    using namespace placamera;

    std::shared_ptr<const LineScanDefinition> makeDefinition()
    {
        LineScanOptics optics;
        optics.focalLengthMillimeters = 10.0;
        optics.samplePitchMillimeters = 0.01;
        optics.principalSample = 5.0;
        return LineScanDefinition::create(CameraDefinitionId("linescan-definition"), FrameId("body-fixed"), optics);
    }

    LineScanTrajectory makeTrajectory()
    {
        const RotationMatrix identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        return LineScanTrajectory::create({
            TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.0), {0.0, -1.0, 0.0}, identity},
            TrajectorySample{TimeReference::create(TimeScale::Tdb, 1.0), {0.0, 1.0, 0.0}, identity},
        });
    }

    LineScanModel makeModel(LineTiming timing = {0.5, 0.0, 0.1, TimeScale::Tdb, {}})
    {
        return LineScanModel::create(CameraInstanceId("linescan-instance"),
                                     ImageId("linescan-image"),
                                     makeDefinition(),
                                     ImageSize{1000, 11},
                                     makeTrajectory(),
                                     std::move(timing));
    }

    MetashapeCalibration makeCompleteCalibration()
    {
        MetashapeCalibration calibration;
        calibration.f = 980.0;
        calibration.cx = 512.25;
        calibration.cy = 384.75;
        calibration.b1 = 3.0;
        calibration.b2 = -0.4;
        calibration.k1 = 0.01;
        calibration.k2 = -0.001;
        calibration.k3 = 0.0002;
        calibration.k4 = -0.00003;
        calibration.p1 = 0.0005;
        calibration.p2 = -0.0004;
        calibration.p3 = 0.00007;
        calibration.p4 = -0.000005;
        calibration.principalPointDecomposition = PrincipalPointDecomposition{400.0, 300.0, 112.25, 84.75};
        return calibration;
    }

    LineScanOptics makeCompleteOptics()
    {
        LineScanOptics optics;
        optics.focalLengthMillimeters = 35.0;
        optics.samplePitchMillimeters = 0.01;
        optics.completeCalibration = makeCompleteCalibration();
        return optics;
    }

    TEST(LineScanModelTest, InterpolatesTrajectoryAndMapsLineTime)
    {
        const LineScanModel model = makeModel();
        const auto time = model.timeForLine(5.5);
        ASSERT_TRUE(time) << time.message();
        EXPECT_DOUBLE_EQ(time.value().seconds, 0.5);

        const auto line = model.lineForTime(TimeReference::create(TimeScale::Tdb, 0.5));
        ASSERT_TRUE(line) << line.message();
        EXPECT_DOUBLE_EQ(line.value(), 5.5);

        const auto pose = model.trajectory().poseAt(time.value(), model.groundFrame());
        ASSERT_TRUE(pose) << pose.message();
        EXPECT_NEAR(pose.value().center[1], 0.0, 1.0e-12);
    }

    TEST(LineScanModelTest, MapsPiecewiseLineRates)
    {
        LineTiming timing;
        timing.timeScale = TimeScale::Tdb;
        timing.segments = {{0.5, 0.0, 0.1}, {5.5, 0.7, 0.2}};
        const LineScanModel model = makeModel(timing);

        const auto first = model.timeForLine(4.5);
        const auto second = model.timeForLine(5.5);
        const auto inverse = model.lineForTime(TimeReference::create(TimeScale::Tdb, 0.9));
        ASSERT_TRUE(first);
        ASSERT_TRUE(second);
        ASSERT_TRUE(inverse);
        EXPECT_NEAR(first.value().seconds, 0.4, 1.0e-12);
        EXPECT_NEAR(second.value().seconds, 0.7, 1.0e-12);
        EXPECT_NEAR(inverse.value(), 6.5, 1.0e-12);
    }

    TEST(LineScanModelTest, ProjectsBySolvingAcquisitionLineAndBuildsLocus)
    {
        const LineScanModel line_scan = makeModel();
        const RasterModel& model = line_scan;
        const GroundCoordinate ground{FrameId("body-fixed"), {2.0, 0.0, 10.0}};
        const auto projection = model.groundToImage(ground);
        ASSERT_TRUE(projection) << projection.message();
        EXPECT_NEAR(projection.value().image.sample, 205.0, 1.0e-9);
        EXPECT_NEAR(projection.value().image.line, 5.5, 1.0e-7);
        ASSERT_TRUE(projection.value().positiveDepth.has_value());
        EXPECT_NEAR(*projection.value().positiveDepth, 10.0, 1.0e-9);
        ASSERT_TRUE(projection.value().acquisitionTime.has_value());
        EXPECT_NEAR(projection.value().acquisitionTime->seconds, 0.5, 1.0e-7);
        EXPECT_LT(projection.achievedPrecisionPixels(), 1.0e-6);

        const auto locus = model.imageToImagingLocus(projection.value().image);
        ASSERT_TRUE(locus) << locus.message();
        EXPECT_NEAR(locus.value().origin.position[1], 0.0, 1.0e-7);
        EXPECT_NEAR(locus.value().direction[0], 2.0 / std::sqrt(104.0), 1.0e-7);
        EXPECT_NEAR(locus.value().direction[2], 10.0 / std::sqrt(104.0), 1.0e-7);
        EXPECT_TRUE(model.capabilities().contains(CapabilityKind::Trajectory));
        EXPECT_FALSE(model.capabilities().contains(CapabilityKind::StaticPose));
    }

    TEST(LineScanModelTest, AppliesTrajectoryBiasUpdates)
    {
        const LineScanTrajectoryBias current{{{1.0, 2.0, 3.0}}, {{0.1, 0.2, 0.3}}, 4.0};
        const LineScanTrajectoryBias updated =
            applyLineScanTrajectoryBiasUpdate(current, std::array<double, 7>{{0.5, -1.0, 2.0, 0.01, 0.02, 0.03, -0.5}});
        EXPECT_DOUBLE_EQ(updated.translationMeters[0], 1.5);
        EXPECT_DOUBLE_EQ(updated.translationMeters[1], 1.0);
        EXPECT_DOUBLE_EQ(updated.rotationVectorRadians[2], 0.33);
        EXPECT_DOUBLE_EQ(updated.timeOffsetSeconds, 3.5);

        std::array<double, 7> invalid{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN()}};
        EXPECT_THROW(applyLineScanTrajectoryBiasUpdate(current, invalid), CameraValidationError);
    }

    TEST(LineScanModelTest, ProjectsAtLineWithExplicitBiasWithoutChangingModel)
    {
        const LineScanModel model = makeModel();
        const GroundCoordinate ground{FrameId("body-fixed"), {2.0, 0.0, 10.0}};
        LineScanTrajectoryBias bias;
        bias.translationMeters = {1.0, 0.0, 0.0};
        const auto explicit_projection = model.projectAtLine(ground, 5.5, bias);
        const auto biased_model = model.withTrajectoryBias(model.instanceId(), bias);
        const auto stored_projection = biased_model.projectAtLine(ground, 5.5);
        ASSERT_TRUE(explicit_projection) << explicit_projection.message();
        ASSERT_TRUE(stored_projection) << stored_projection.message();
        EXPECT_NEAR(explicit_projection.value().projection.image.sample,
                    stored_projection.value().projection.image.sample,
                    1.0e-12);
        EXPECT_NEAR(
            explicit_projection.value().lineResidualPixels, stored_projection.value().lineResidualPixels, 1.0e-12);
        EXPECT_DOUBLE_EQ(model.trajectoryBias().translationMeters[0], 0.0);

        bias.timeOffsetSeconds = std::numeric_limits<double>::quiet_NaN();
        EXPECT_FALSE(model.projectAtLine(ground, 5.5, bias));
    }

    TEST(LineScanModelTest, SupportsNormalizedAndDetectorFocalMappings)
    {
        LineScanOptics radial_optics;
        radial_optics.focalLengthMillimeters = 10.0;
        radial_optics.samplePitchMillimeters = 0.01;
        radial_optics.principalSample = 5.0;
        radial_optics.distortionK1 = 0.01;
        const auto radial =
            LineScanDefinition::create(CameraDefinitionId("radial"), FrameId("body-fixed"), radial_optics);
        const auto focal = lineScanPixelToUndistortedFocal(*radial, 205.0);
        ASSERT_TRUE(focal) << focal.message();
        const auto restored = lineScanUndistortedFocalToPixel(*radial, focal.value());
        ASSERT_TRUE(restored) << restored.message();
        EXPECT_NEAR(restored.value().sample, 205.0, 1.0e-10);
        EXPECT_NEAR(restored.value().lineResidualPixels, 0.0, 1.0e-12);

        LineScanOptics detector_optics;
        detector_optics.focalLengthMillimeters = 100.0;
        detector_optics.distortionK1 = 1.0e-5;
        detector_optics.distortionModel = LineScanDistortionModel::LroNacFocalPlane;
        LineScanDetectorGeometry detector;
        detector.detectorSampleOrigin = 100.0;
        detector.focalToPixelLines = {0.0, 100.0, 0.0};
        detector.focalToPixelSamples = {0.0, 0.0, 100.0};
        detector_optics.detectorGeometry = detector;
        const auto detector_definition =
            LineScanDefinition::create(CameraDefinitionId("detector"), FrameId("body-fixed"), detector_optics);
        const auto detector_focal = lineScanPixelToUndistortedFocal(*detector_definition, 110.0);
        ASSERT_TRUE(detector_focal) << detector_focal.message();
        const auto detector_restored = lineScanUndistortedFocalToPixel(*detector_definition, detector_focal.value());
        ASSERT_TRUE(detector_restored) << detector_restored.message();
        EXPECT_NEAR(detector_restored.value().sample, 110.0, 1.0e-10);
        EXPECT_NEAR(detector_restored.value().lineResidualPixels, 0.0, 1.0e-10);
    }

    TEST(LineScanDefinitionTest, AppliesCompleteMetashapeCalibrationAndExplicitPixelConvention)
    {
        const LineScanOptics optics = makeCompleteOptics();
        const auto zero_based = LineScanDefinition::create(
            CameraDefinitionId("complete-zero"), FrameId("body-fixed"), optics, LineScanPixelConvention::ZeroBased);
        const auto pixel_center = LineScanDefinition::create(
            CameraDefinitionId("complete-center"), FrameId("body-fixed"), optics, LineScanPixelConvention::PixelCenter);
        const FocalPlaneCoordinate focal{3.5, -1.75};

        const auto zero_projection = lineScanUndistortedFocalToPixel(*zero_based, focal);
        const auto center_projection = lineScanUndistortedFocalToPixel(*pixel_center, focal);
        ASSERT_TRUE(zero_projection) << zero_projection.message();
        ASSERT_TRUE(center_projection) << center_projection.message();
        EXPECT_NEAR(zero_projection.value().sample, 610.60218524324046, 1.0e-12);
        EXPECT_NEAR(zero_projection.value().lineResidualPixels, -49.01787737313555, 1.0e-12);
        EXPECT_NEAR(center_projection.value().sample, zero_projection.value().sample + 0.5, 1.0e-12);
        EXPECT_DOUBLE_EQ(center_projection.value().lineResidualPixels, zero_projection.value().lineResidualPixels);

        const EvaluationOptions precise{1.0e-12, 50, false};
        const auto zero_focal = lineScanPixelToUndistortedFocal(*zero_based, zero_projection.value().sample, precise);
        const auto center_focal =
            lineScanPixelToUndistortedFocal(*pixel_center, center_projection.value().sample, precise);
        ASSERT_TRUE(zero_focal) << zero_focal.message();
        ASSERT_TRUE(center_focal) << center_focal.message();
        EXPECT_NEAR(zero_focal.value().xMillimeters, center_focal.value().xMillimeters, 1.0e-12);
        EXPECT_NEAR(zero_focal.value().yMillimeters, center_focal.value().yMillimeters, 1.0e-12);

        const auto restored = lineScanUndistortedFocalToPixel(*zero_based, zero_focal.value());
        ASSERT_TRUE(restored) << restored.message();
        EXPECT_NEAR(restored.value().sample, zero_projection.value().sample, 1.0e-9);
        EXPECT_NEAR(restored.value().lineResidualPixels, 0.0, 1.0e-9);
    }

    TEST(LineScanDefinitionTest, ComposesCompleteCalibrationWithDetectorGeometry)
    {
        LineScanOptics optics = makeCompleteOptics();
        LineScanDetectorGeometry detector;
        detector.detectorSampleSumming = 2.0;
        detector.detectorLineSumming = 1.5;
        detector.detectorSampleOrigin = 10.0;
        detector.detectorLineOrigin = -4.0;
        detector.startingDetectorSample = 6.0;
        detector.startingDetectorLine = 3.0;
        detector.focalToPixelSamples = {100.0, 2.0, 40.0};
        detector.focalToPixelLines = {-20.0, 30.0, -5.0};
        optics.detectorGeometry = detector;

        const auto pixel_center =
            LineScanDefinition::create(CameraDefinitionId("complete-detector"), FrameId("body-fixed"), optics);
        const auto zero_based = LineScanDefinition::create(CameraDefinitionId("complete-detector-zero"),
                                                           FrameId("body-fixed"),
                                                           optics,
                                                           LineScanPixelConvention::ZeroBased);
        const FocalPlaneCoordinate focal{3.5, -1.75};

        const auto center_projection = lineScanUndistortedFocalToPixel(*pixel_center, focal);
        const auto zero_projection = lineScanUndistortedFocalToPixel(*zero_based, focal);
        ASSERT_TRUE(center_projection) << center_projection.message();
        ASSERT_TRUE(zero_projection) << zero_projection.message();
        EXPECT_NEAR(center_projection.value().sample, 43.179946377805294, 1.0e-12);
        EXPECT_NEAR(center_projection.value().lineResidualPixels, 3.3043662944192747, 1.0e-12);
        EXPECT_NEAR(zero_projection.value().sample, 42.679946377805294, 1.0e-12);
        EXPECT_DOUBLE_EQ(zero_projection.value().lineResidualPixels, center_projection.value().lineResidualPixels);

        const auto inverse = lineScanPixelToUndistortedFocal(*pixel_center, center_projection.value().sample);
        ASSERT_TRUE(inverse) << inverse.message();
        const auto restored = lineScanUndistortedFocalToPixel(*pixel_center, inverse.value());
        ASSERT_TRUE(restored) << restored.message();
        EXPECT_NEAR(restored.value().sample, center_projection.value().sample, 1.0e-9);
        EXPECT_NEAR(restored.value().lineResidualPixels, 0.0, 1.0e-9);
    }

    TEST(LineScanTrajectoryTest, InterpolatesFrameComposedStates)
    {
        FrameComposedTrajectory composed;
        composed.inertialStates = {
            {TimeReference::create(TimeScale::Tdb, 0.0), {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}},
            {TimeReference::create(TimeScale::Tdb, 1.0), {2.0, 0.0, 0.0}, {2.0, 0.0, 0.0}},
        };
        composed.inertialToWorld.samples = {
            {TimeReference::create(TimeScale::Tdb, 0.0), {1.0, 0.0, 0.0, 0.0}},
            {TimeReference::create(TimeScale::Tdb, 1.0), {1.0, 0.0, 0.0, 0.0}},
        };
        composed.inertialToSensor.samples = composed.inertialToWorld.samples;
        const LineScanTrajectory trajectory = LineScanTrajectory::createFrameComposed(std::move(composed));
        const auto pose = trajectory.poseAt(TimeReference::create(TimeScale::Tdb, 0.5), FrameId("body-fixed"));
        ASSERT_TRUE(pose) << pose.message();
        EXPECT_NEAR(pose.value().center[0], 1.0, 1.0e-12);
        EXPECT_EQ(pose.value().cameraToWorldRotation, (RotationMatrix{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    }

    TEST(LineScanModelTest, RejectsInvalidTrajectoryTimingAndFrames)
    {
        EXPECT_THROW(LineScanTrajectory::create({TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.0)}}),
                     CameraValidationError);

        EXPECT_THROW(LineScanModel::create(CameraInstanceId("mismatched"),
                                           ImageId("image"),
                                           makeDefinition(),
                                           ImageSize{1000, 11},
                                           makeTrajectory(),
                                           LineTiming{0.5, 0.0, 0.1, TimeScale::Utc, {}}),
                     CameraValidationError);

        const LineScanModel model = makeModel();
        const auto mismatch = model.groundToImage(GroundCoordinate{FrameId("other"), {2.0, 0.0, 10.0}});
        EXPECT_FALSE(mismatch);
        EXPECT_EQ(mismatch.errorCode(), CameraErrorCode::FrameMismatch);
    }

} // namespace
