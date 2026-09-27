#include <placamera/frame_camera.h>
#include <placamera/linescan_camera.h>
#include <placamera/reference/MetashapeCameraReferenceAdapter.h>
#include <placamera/rpc_camera.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <fstream>
#include <string>

namespace
{
    using Json = nlohmann::json;
    using namespace placamera;
    using namespace placamera::reference;

    const Json& fixture()
    {
        static const Json value = []
        {
            std::ifstream stream(PLACAMERA_METALIGN_CAMERA_FIXTURE);
            if (!stream)
            {
                throw std::runtime_error("cannot open metalign camera differential fixture");
            }
            return Json::parse(stream);
        }();
        return value;
    }

    RpcParameters rpcParameters(const Json& source)
    {
        const Json& normalization = source.at("normalization");
        RpcParameters parameters;
        parameters.lineOffset = normalization.at("line_offset");
        parameters.sampleOffset = normalization.at("sample_offset");
        parameters.latitudeOffset = normalization.at("latitude_offset");
        parameters.longitudeOffset = normalization.at("longitude_offset");
        parameters.heightOffset = normalization.at("height_offset");
        parameters.lineScale = normalization.at("line_scale");
        parameters.sampleScale = normalization.at("sample_scale");
        parameters.latitudeScale = normalization.at("latitude_scale");
        parameters.longitudeScale = normalization.at("longitude_scale");
        parameters.heightScale = normalization.at("height_scale");
        parameters.lineNumerator[2] = 1.0;
        parameters.lineDenominator[0] = 1.0;
        parameters.sampleNumerator[1] = 1.0;
        parameters.sampleDenominator[0] = 1.0;
        return parameters;
    }

    MetashapeCalibration completeCalibration(const Json& source)
    {
        const Json& parameters = source.at("parameters");
        MetashapeCalibration calibration;
        calibration.f = parameters.at("f");
        calibration.cx = parameters.at("cx");
        calibration.cy = parameters.at("cy");
        calibration.b1 = parameters.at("b1");
        calibration.b2 = parameters.at("b2");
        calibration.k1 = parameters.at("k1");
        calibration.k2 = parameters.at("k2");
        calibration.k3 = parameters.at("k3");
        calibration.k4 = parameters.at("k4");
        calibration.p1 = parameters.at("p1");
        calibration.p2 = parameters.at("p2");
        calibration.p3 = parameters.at("p3");
        calibration.p4 = parameters.at("p4");
        const auto center = source.at("image_center").get<std::array<double, 2>>();
        const auto offset = source.at("principal_offset").get<std::array<double, 2>>();
        calibration.principalPointDecomposition =
            PrincipalPointDecomposition{center[0], center[1], offset[0], offset[1]};
        return calibration;
    }

    TEST(MetalignDifferentialFixtureTest, MatchesLineScanProjectionAndSceneTimeOffset)
    {
        const Json& source = fixture().at("line_scan");
        LineScanOptics optics;
        optics.focalLengthMillimeters = source.at("focal_pixels");
        optics.samplePitchMillimeters = 1.0;
        // metalign stores a zero-based principal sample; PlaCamera's optical
        // principal is expressed in its CSM pixel-centre coordinate.
        optics.principalSample = source.at("principal_sample").get<double>() + 0.5;
        const auto definition = LineScanDefinition::create(
            CameraDefinitionId("metalign-line"), FrameId("fixture-world"), optics, LineScanPixelConvention::ZeroBased);
        const RotationMatrix identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        std::vector<TrajectorySample> samples;
        for (const Json& encoded : source.at("samples"))
        {
            samples.emplace_back(TimeReference::create(TimeScale::Relative, encoded.at("time_seconds")),
                                 encoded.at("center").get<Vector3>(),
                                 identity);
        }
        const auto trajectory = LineScanTrajectory::create(std::move(samples));
        const auto point = source.at("point").get<Vector3>();
        const GroundCoordinate ground{FrameId("fixture-world"), point};
        const LineTiming timing{0.0, 0.0, 0.01, TimeScale::Relative, {}};

        for (const char* case_name : {"zero_offset", "positive_offset"})
        {
            LineScanTrajectoryBias bias;
            bias.timeOffsetSeconds = source.at(case_name).at("time_offset_seconds");
            const auto model = LineScanModel::create(CameraInstanceId(std::string("line-") + case_name),
                                                     ImageId(std::string("image-") + case_name),
                                                     definition,
                                                     ImageSize{101, 101},
                                                     trajectory,
                                                     timing,
                                                     bias);
            const auto projected = model.groundToImage(ground);
            ASSERT_TRUE(projected) << projected.message();
            const auto expected = source.at(case_name).at("expected_pixel").get<std::array<double, 2>>();
            EXPECT_NEAR(projected.value().image.sample, expected[0], 1.0e-7);
            EXPECT_NEAR(projected.value().image.line, expected[1], 1.0e-7);
            ASSERT_TRUE(projected.value().acquisitionTime);
            EXPECT_NEAR(projected.value().acquisitionTime->seconds,
                        source.at(case_name).at("expected_time").get<double>(),
                        1.0e-9);
        }
    }

    TEST(MetalignDifferentialFixtureTest, MatchesCompleteLineScanCalibrationAndDetectorComposition)
    {
        const Json& source = fixture().at("line_scan").at("complete_calibration");
        LineScanOptics optics;
        optics.focalLengthMillimeters = source.at("physical_focal_mm");
        optics.samplePitchMillimeters = source.at("sample_pitch_mm");
        optics.completeCalibration = completeCalibration(source);
        const auto focal_values = source.at("focal_coordinate_mm").get<std::array<double, 2>>();
        const FocalPlaneCoordinate focal{focal_values[0], focal_values[1]};

        const auto zero_definition = LineScanDefinition::create(CameraDefinitionId("metalign-complete-line"),
                                                                FrameId("fixture-world"),
                                                                optics,
                                                                LineScanPixelConvention::ZeroBased);
        const auto projected = lineScanUndistortedFocalToPixel(*zero_definition, focal);
        ASSERT_TRUE(projected) << projected.message();
        const auto expected = source.at("expected_zero_based_projection").get<std::array<double, 2>>();
        EXPECT_NEAR(projected.value().sample, expected[0], 1.0e-12);
        EXPECT_NEAR(projected.value().lineResidualPixels, expected[1], 1.0e-12);

        const Json& detector_source = source.at("detector_geometry");
        LineScanDetectorGeometry detector;
        detector.detectorSampleSumming = detector_source.at("sample_summing");
        detector.detectorLineSumming = detector_source.at("line_summing");
        detector.detectorSampleOrigin = detector_source.at("sample_origin");
        detector.detectorLineOrigin = detector_source.at("line_origin");
        detector.startingDetectorSample = detector_source.at("starting_sample");
        detector.startingDetectorLine = detector_source.at("starting_line");
        detector.focalToPixelSamples = detector_source.at("focal_to_pixel_samples").get<std::array<double, 3>>();
        detector.focalToPixelLines = detector_source.at("focal_to_pixel_lines").get<std::array<double, 3>>();
        optics.detectorGeometry = detector;
        const auto detector_definition = LineScanDefinition::create(
            CameraDefinitionId("metalign-complete-detector"), FrameId("fixture-world"), optics);
        const auto detector_projected = lineScanUndistortedFocalToPixel(*detector_definition, focal);
        ASSERT_TRUE(detector_projected) << detector_projected.message();
        const auto detector_expected =
            source.at("expected_pixel_center_detector_projection").get<std::array<double, 2>>();
        EXPECT_NEAR(detector_projected.value().sample, detector_expected[0], 1.0e-12);
        EXPECT_NEAR(detector_projected.value().lineResidualPixels, detector_expected[1], 1.0e-12);
    }

    TEST(MetalignDifferentialFixtureTest, PreservesExactPrincipalPointDecomposition)
    {
        const Json& source = fixture().at("exact_principal_point");
        const auto center = source.at("image_center").get<std::array<double, 2>>();
        const auto offset = source.at("principal_offset").get<std::array<double, 2>>();
        const auto absolute = source.at("expected_absolute_principal").get<std::array<double, 2>>();
        FrameCalibration calibration;
        calibration.f = source.at("f");
        calibration.cx = absolute[0];
        calibration.cy = absolute[1];
        calibration.principalPointDecomposition =
            PrincipalPointDecomposition{center[0], center[1], offset[0], offset[1]};
        const auto definition = FramePinholeDefinition::create(CameraDefinitionId("metalign-exact-principal"),
                                                               calibration,
                                                               PixelConvention::PixelCenter,
                                                               FrameId("fixture-world"));
        const auto pixel = source.at("pixel").get<std::array<double, 2>>();
        const auto normalized = definition->undistortPixel({pixel[0], pixel[1]});
        ASSERT_TRUE(normalized) << normalized.message();
        const auto expected = source.at("expected_normalized").get<std::array<double, 2>>();
        EXPECT_DOUBLE_EQ(normalized.value()[0], expected[0]);
        EXPECT_DOUBLE_EQ(normalized.value()[1], expected[1]);
        EXPECT_DOUBLE_EQ(definition->intrinsics().principalX, absolute[0]);
        EXPECT_DOUBLE_EQ(definition->intrinsics().principalY, absolute[1]);
    }

    TEST(MetalignDifferentialFixtureTest, MatchesGroundRpcAndKeepsNormalizedImageRpcExplicit)
    {
        const Json& source = fixture().at("rpc");
        const auto definition =
            RpcDefinition::create(CameraDefinitionId("metalign-rpc"), FrameId("fixture-ecef"), rpcParameters(source));
        const auto ground_values = source.at("ground").get<std::array<double, 3>>();
        const GeodeticCoordinate ground{ground_values[0], ground_values[1], ground_values[2]};

        const auto identity = RpcModel::create(
            CameraInstanceId("rpc-identity"), ImageId("rpc-image-identity"), definition, ImageSize{1200, 1000});
        const auto uncorrected = identity.groundToImageGeodetic(ground);
        ASSERT_TRUE(uncorrected) << uncorrected.message();
        const auto expected_uncorrected = source.at("expected_uncorrected").get<std::array<double, 2>>();
        EXPECT_NEAR(uncorrected.value().image.sample, expected_uncorrected[0], 1.0e-10);
        EXPECT_NEAR(uncorrected.value().image.line, expected_uncorrected[1], 1.0e-10);

        const Json& ground_source = source.at("ground_correction");
        RpcGroundCorrection ground_correction;
        ground_correction.sampleOffsetPixels = ground_source.at("sample_offset");
        ground_correction.lineOffsetPixels = ground_source.at("line_offset");
        ground_correction.sampleLongitudePixelsPerDegree = ground_source.at("sample_longitude");
        ground_correction.sampleLatitudePixelsPerDegree = ground_source.at("sample_latitude");
        ground_correction.sampleHeightPixelsPerMeter = ground_source.at("sample_height");
        ground_correction.lineLongitudePixelsPerDegree = ground_source.at("line_longitude");
        ground_correction.lineLatitudePixelsPerDegree = ground_source.at("line_latitude");
        ground_correction.lineHeightPixelsPerMeter = ground_source.at("line_height");
        const auto ground_model = RpcModel::createWithCorrection(CameraInstanceId("rpc-ground"),
                                                                 ImageId("rpc-image-ground"),
                                                                 definition,
                                                                 ImageSize{1200, 1000},
                                                                 RpcCorrection::groundCoordinates(ground_correction));
        const auto ground_projected = ground_model.groundToImageGeodetic(ground);
        ASSERT_TRUE(ground_projected) << ground_projected.message();
        const auto expected_ground = ground_source.at("expected_pixel").get<std::array<double, 2>>();
        EXPECT_NEAR(ground_projected.value().image.sample, expected_ground[0], 1.0e-10);
        EXPECT_NEAR(ground_projected.value().image.line, expected_ground[1], 1.0e-10);
        EXPECT_EQ(ground_model.correctionDomain(), RpcCorrectionDomain::GroundCoordinates);

        const Json& image_source = source.at("normalized_image_correction");
        RpcImageCorrection image_correction;
        image_correction.sampleOffsetPixels = image_source.at("sample_offset");
        image_correction.sampleSamplePixels = image_source.at("sample_sample");
        image_correction.sampleLinePixels = image_source.at("sample_line");
        image_correction.lineOffsetPixels = image_source.at("line_offset");
        image_correction.lineSamplePixels = image_source.at("line_sample");
        image_correction.lineLinePixels = image_source.at("line_line");
        const auto image_model = RpcModel::create(CameraInstanceId("rpc-image-domain"),
                                                  ImageId("rpc-image-normalized"),
                                                  definition,
                                                  ImageSize{1200, 1000},
                                                  image_correction);
        const auto image_projected = image_model.groundToImageGeodetic(ground);
        ASSERT_TRUE(image_projected) << image_projected.message();
        const auto expected_image = image_source.at("expected_pixel").get<std::array<double, 2>>();
        EXPECT_NEAR(image_projected.value().image.sample, expected_image[0], 1.0e-10);
        EXPECT_NEAR(image_projected.value().image.line, expected_image[1], 1.0e-10);
        EXPECT_EQ(image_model.correctionDomain(), RpcCorrectionDomain::NormalizedImage);
        EXPECT_NE(image_projected.value().image.line, ground_projected.value().image.line);
    }

    TEST(MetalignDifferentialFixtureTest, ConvertsAnisotropicYprCovarianceToCanonicalTangent)
    {
        const Json& source = fixture().at("reference");
        MetashapeCameraReference reference;
        reference.yawPitchRollDegrees = source.at("yaw_pitch_roll_degrees").get<Vector3>();
        reference.yawPitchRollCovarianceDegreesSquared =
            source.at("yaw_pitch_roll_covariance_degrees_squared").get<std::array<double, 9>>();
        const auto adapted = adaptMetashapeCameraReference(
            ImageId("reference-image"), ReferenceSourceId("metalign"), FrameId("fixture-world"), reference);
        ASSERT_TRUE(adapted) << adapted.message();
        ASSERT_TRUE(adapted.value().orientation);
        ASSERT_TRUE(adapted.value().covariance);

        const auto expected_rotation = source.at("expected_rotation").get<RotationMatrix>();
        for (std::size_t index = 0; index < expected_rotation.size(); ++index)
        {
            EXPECT_NEAR((*adapted.value().orientation)[index], expected_rotation[index], 1.0e-14);
        }
        const auto expected_covariance =
            source.at("expected_left_tangent_covariance_radians_squared").get<std::array<double, 9>>();
        const auto& covariance = adapted.value().covariance->matrixValues();
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                EXPECT_NEAR(covariance[(row + 3) * 6 + column + 3], expected_covariance[row * 3 + column], 1.0e-15);
            }
        }
        EXPECT_EQ(adapted.value().covariance->components(), PoseCovarianceComponents::RotationOnly);
    }

    TEST(MetalignDifferentialFixtureTest, RecordsFrozenReferenceRevision)
    {
        EXPECT_EQ(fixture().at("format"), "placamera-metalignment-camera-contract-v1");
        EXPECT_EQ(fixture().at("provenance").at("commit"), "e5c7b819f49114df3b93010cb17998593a6b53aa");
    }

} // namespace
