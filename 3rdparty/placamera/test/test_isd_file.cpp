#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <placamera/isd.h>

namespace
{

    using namespace placamera;

    class TemporaryIsdFile
    {
    public:
        TemporaryIsdFile(const std::string& name, std::string_view contents)
        {
            std::error_code error;
            const std::filesystem::path directory = std::filesystem::path(PLACAMERA_TEST_TMP_ROOT) / "isd-files";
            std::filesystem::create_directories(directory, error);
            if (error)
            {
                return;
            }
            _path = directory / name;
            std::ofstream output(_path, std::ios::binary | std::ios::trunc);
            output << contents;
            _valid = output.good();
        }

        ~TemporaryIsdFile()
        {
            std::error_code error;
            std::filesystem::remove(_path, error);
            std::filesystem::remove(_path.parent_path(), error);
        }

        const std::filesystem::path& path() const
        {
            return _path;
        }

        bool valid() const
        {
            return _valid;
        }

    private:
        std::filesystem::path _path;
        bool _valid = false;
    };

    const char* kSyntheticIsd = R"JSON({
  "image_lines": 100,
  "image_samples": 200,
  "name_platform": "SYNTHETIC ORBITER",
  "name_sensor": "SYNTHETIC LINE SCANNER",
  "name_model": "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL",
  "interpolation_method": "lagrange",
  "naif_keywords": {"BODY_FRAME_CODE": 31001},
  "line_scan_rate": [[0.5, -0.5, 0.01]],
  "starting_ephemeris_time": 999.5,
  "center_ephemeris_time": 1000.0,
  "body_rotation": {
    "ephemeris_times": [998.0, 1002.0],
    "quaternions": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    "constant_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  },
  "instrument_pointing": {
    "ephemeris_times": [998.0, 1002.0],
    "quaternions": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    "constant_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  },
  "detector_sample_summing": 1.0,
  "detector_line_summing": 1.0,
  "focal_length_model": {"focal_length": 100.0},
  "detector_center": {"line": 0.0, "sample": 100.0},
  "starting_detector_line": 0.0,
  "starting_detector_sample": 0.0,
  "focal2pixel_lines": [0.0, 100.0, 0.0],
  "focal2pixel_samples": [0.0, 0.0, 100.0],
  "optical_distortion": {"lrolrocnac": {"coefficients": [0.00001]}},
  "instrument_position": {
    "ephemeris_times": [998.0, 1002.0],
    "positions": [[-2.005, 0.0, 0.0], [1.995, 0.0, 0.0]],
    "velocities": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
  }
})JSON";

    bool importSynthetic(PlanetaryLineScanIsdImport* imported, std::string* error)
    {
        const TemporaryIsdFile file("synthetic.isd", kSyntheticIsd);
        if (!file.valid())
        {
            return false;
        }
        auto result = importPlanetaryLineScanIsd(file.path(),
                                                 CameraDefinitionId("synthetic-definition"),
                                                 CameraInstanceId("synthetic-instance"),
                                                 ImageId("synthetic-image"));
        if (!result)
        {
            *error = result.message();
            return false;
        }
        *imported = result.takeValue();
        return true;
    }

} // namespace

TEST(PlanetaryLineScanIsdIO, ImportsRequiredUsgsCsmFields)
{
    PlanetaryLineScanIsdImport imported;
    std::string error;
    ASSERT_TRUE(importSynthetic(&imported, &error)) << error;
    ASSERT_NE(imported.instance, nullptr);
    EXPECT_EQ(imported.instance->imageSize().lines, 100);
    EXPECT_EQ(imported.instance->imageSize().samples, 200);
    EXPECT_EQ(imported.metadata.modelName, "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL");
    EXPECT_EQ(imported.metadata.platformName, "SYNTHETIC ORBITER");
    EXPECT_EQ(imported.metadata.interpolationMethod, "lagrange");
    EXPECT_EQ(imported.metadata.targetName, "MOON");
    EXPECT_EQ(imported.instance->groundFrame().value(), "MOON_ME");
    EXPECT_EQ(imported.metadata.bodyFixedFrameCode, 31001);
    EXPECT_DOUBLE_EQ(imported.instance->lineScanDefinition().optics().focalLengthMillimeters, 100.0);
}

TEST(PlanetaryLineScanIsdIO, RejectsUnacknowledgedInterpolationModel)
{
    std::string isd(kSyntheticIsd);
    const std::size_t interpolation = isd.find("\"lagrange\"");
    ASSERT_NE(interpolation, std::string::npos);
    isd.replace(interpolation, std::string("\"lagrange\"").size(), "\"spline\"");
    const TemporaryIsdFile file("unsupported_interpolation.isd", isd);
    ASSERT_TRUE(file.valid());

    const auto imported = importPlanetaryLineScanIsd(
        file.path(), CameraDefinitionId("definition"), CameraInstanceId("instance"), ImageId("image"));
    EXPECT_FALSE(imported);
    EXPECT_NE(imported.message().find("interpolation_method='lagrange'"), std::string::npos) << imported.message();
}

TEST(PlanetaryLineScanIsdIO, PreservesPixelCenterTimingAndDetectorGeometry)
{
    PlanetaryLineScanIsdImport imported;
    std::string error;
    ASSERT_TRUE(importSynthetic(&imported, &error)) << error;

    const auto first_time = imported.instance->timeForLine(0.5);
    const auto middle_time = imported.instance->timeForLine(50.5);
    ASSERT_TRUE(first_time);
    ASSERT_TRUE(middle_time);
    EXPECT_DOUBLE_EQ(first_time.value().seconds, 999.505);
    EXPECT_DOUBLE_EQ(middle_time.value().seconds, 1000.005);
    const auto line = imported.instance->lineForTime(first_time.value());
    ASSERT_TRUE(line);
    EXPECT_NEAR(line.value(), 0.5, 1.0e-12);

    const auto ray = imported.instance->imageToImagingLocus({110.0, 50.5});
    ASSERT_TRUE(ray) << ray.message();
    EXPECT_NEAR(ray.value().origin.position[0], 0.0, 1.0e-9);
    EXPECT_GT(ray.value().direction[1], 0.0);
    EXPECT_GT(ray.value().direction[2], 0.999);
}

TEST(PlanetaryLineScanIsdIO, ImportsCompleteCalibrationWithoutCollapsingPrincipalPoint)
{
    std::string isd(kSyntheticIsd);
    const std::size_t closing_brace = isd.rfind('}');
    ASSERT_NE(closing_brace, std::string::npos);
    isd.insert(closing_brace,
               R"JSON(,
  "complete_calibration": {
    "f": 980.0,
    "cx": 112.25,
    "cy": 44.5,
    "b1": 3.0,
    "b2": -0.4,
    "k1": 0.01,
    "k2": -0.001,
    "k3": 0.0002,
    "k4": -0.00003,
    "p1": 0.0005,
    "p2": -0.0004,
    "p3": 0.00007,
    "p4": -0.000005,
    "sample_pitch_mm": 0.01,
    "image_center": [100.0, 50.0],
    "cx_offset": 12.25,
    "cy_offset": -5.5
  }
)JSON");
    const auto imported = parsePlanetaryLineScanIsd(isd,
                                                    CameraDefinitionId("complete-definition"),
                                                    CameraInstanceId("complete-instance"),
                                                    ImageId("complete-image"));
    ASSERT_TRUE(imported) << imported.message();
    const LineScanOptics& optics = imported.value().instance->lineScanDefinition().optics();
    ASSERT_TRUE(optics.completeCalibration);
    const MetashapeCalibration& calibration = *optics.completeCalibration;
    EXPECT_DOUBLE_EQ(calibration.f, 980.0);
    EXPECT_DOUBLE_EQ(calibration.cx, 112.25);
    EXPECT_DOUBLE_EQ(calibration.cy, 44.5);
    EXPECT_DOUBLE_EQ(calibration.b1, 3.0);
    EXPECT_DOUBLE_EQ(calibration.b2, -0.4);
    EXPECT_DOUBLE_EQ(calibration.k4, -0.00003);
    EXPECT_DOUBLE_EQ(calibration.p4, -0.000005);
    ASSERT_TRUE(calibration.principalPointDecomposition);
    EXPECT_EQ(*calibration.principalPointDecomposition, (PrincipalPointDecomposition{100.0, 50.0, 12.25, -5.5}));
    EXPECT_DOUBLE_EQ(optics.samplePitchMillimeters, 0.01);
}

TEST(PlanetaryLineScanIsdIO, ProjectsAtLineAndAppliesTypedTrajectoryBias)
{
    PlanetaryLineScanIsdImport imported;
    std::string error;
    ASSERT_TRUE(importSynthetic(&imported, &error)) << error;
    const GroundCoordinate ground{imported.instance->groundFrame(), {0.0, 0.0, 1000.0}};

    const auto nominal = imported.instance->projectAtLine(ground, 50.5);
    ASSERT_TRUE(nominal) << nominal.message();
    EXPECT_NEAR(nominal.value().projection.image.sample, 100.0, 1.0e-10);
    EXPECT_NEAR(nominal.value().lineResidualPixels, 0.0, 1.0e-9);
    ASSERT_TRUE(nominal.value().projection.positiveDepth.has_value());
    EXPECT_NEAR(*nominal.value().projection.positiveDepth, 1000.0, 1.0e-9);

    LineScanTrajectoryBias bias;
    bias.translationMeters = {10.0, 0.0, 0.0};
    const auto biased = imported.instance->projectAtLine(ground, 50.5, bias);
    ASSERT_TRUE(biased) << biased.message();
    EXPECT_NEAR(biased.value().lineResidualPixels, -100.0, 1.0e-8);
}

TEST(PlanetaryLineScanIsdIO, IterativeProjectionFindsAcquisitionLine)
{
    PlanetaryLineScanIsdImport imported;
    std::string error;
    ASSERT_TRUE(importSynthetic(&imported, &error)) << error;
    const auto projection =
        imported.instance->groundToImage(GroundCoordinate{imported.instance->groundFrame(), {0.0, 0.0, 1000.0}});
    ASSERT_TRUE(projection) << projection.message();
    EXPECT_NEAR(projection.value().image.line, 50.5, 1.0e-8);
    EXPECT_NEAR(projection.value().image.sample, 100.0, 1.0e-8);
    ASSERT_TRUE(projection.value().positiveDepth.has_value());
    EXPECT_NEAR(*projection.value().positiveDepth, 1000.0, 1.0e-9);
    ASSERT_TRUE(projection.value().acquisitionTime.has_value());
    EXPECT_NEAR(projection.value().acquisitionTime->seconds, 1000.005, 1.0e-12);
}

TEST(PlanetaryLineScanIsdIO, OptionalLroFixtureUsesMoonMeTrajectory)
{
#ifndef PLANETARY_TEST_DATA_DIR
    GTEST_SKIP() << "PlaScan optional planetary fixture root is not configured";
#else
    const std::filesystem::path fixture = std::filesystem::path(PLANETARY_TEST_DATA_DIR) /
                                          "photogrammetry_benchmarks/isis_lro_lola_lidar/extracted/ISIS3/isis/"
                                          "tests/data/lidarObservationPair/lidarObservationImage1.isd";
    if (!std::filesystem::exists(fixture))
    {
        GTEST_SKIP() << "Optional downloaded ISIS LRO fixture is not present";
    }

    const auto imported_result = importPlanetaryLineScanIsd(
        fixture.string(), CameraDefinitionId("lro-definition"), CameraInstanceId("lro-instance"), ImageId("lro-image"));
    ASSERT_TRUE(imported_result) << imported_result.message();
    const auto& imported = imported_result.value();
    EXPECT_EQ(imported.instance->imageSize().lines, 52224);
    EXPECT_EQ(imported.instance->imageSize().samples, 5064);
    EXPECT_NEAR(imported.metadata.focalLengthMillimeters, 699.62, 1.0e-12);

    constexpr double shotTime = 317845268.6627772;
    constexpr double latitudeDegrees = 60.8347278;
    constexpr double longitudeDegrees = 99.0976484;
    constexpr double radiusMeters = 1735261.573;
    constexpr double pi = 3.14159265358979323846;
    const double latitude = latitudeDegrees * pi / 180.0;
    const double longitude = longitudeDegrees * pi / 180.0;
    const std::array<double, 3> ground{{radiusMeters * std::cos(latitude) * std::cos(longitude),
                                        radiusMeters * std::cos(latitude) * std::sin(longitude),
                                        radiusMeters * std::sin(latitude)}};
    const auto pose = imported.instance->trajectory().poseAt(TimeReference::create(TimeScale::Tdb, shotTime),
                                                             imported.instance->groundFrame());
    ASSERT_TRUE(pose) << pose.message();
    const double range = std::hypot(std::hypot(ground[0] - pose.value().center[0], ground[1] - pose.value().center[1]),
                                    ground[2] - pose.value().center[2]);
    EXPECT_NEAR(range, 56552.95, 0.15);

    const GroundCoordinate target{imported.instance->groundFrame(), ground};
    const auto fixed_line = imported.instance->projectAtLine(target, 18995.176438712828 - 0.5);
    ASSERT_TRUE(fixed_line) << fixed_line.message();
    EXPECT_NEAR(fixed_line.value().lineResidualPixels, 0.0, 20.0);
    EXPECT_NEAR(fixed_line.value().projection.image.sample, 4979.609104416888 - 0.5, 5.0);

    const auto projected = imported.instance->groundToImage(target);
    ASSERT_TRUE(projected) << projected.message();
    EXPECT_NEAR(projected.value().image.line, 18995.176438712828 - 0.5, 20.0);
    EXPECT_NEAR(projected.value().image.sample, 4979.609104416888 - 0.5, 5.0);
#endif
}
