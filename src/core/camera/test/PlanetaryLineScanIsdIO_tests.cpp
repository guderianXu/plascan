#include "PlanetaryLineScanIsdIO.h"

#include "camera/models/linescan/LineScanProjection.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QTemporaryDir>

#include <cmath>
#include <filesystem>
#include <string>

namespace
{

    using namespace xjw::camera_core;
    using namespace xjw::coordinate_system;
    using namespace xjw::camera_models::linescan;

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

    bool importSynthetic(QTemporaryDir* directory,
                         PlanetaryLineScanIsdImport* imported,
                         std::string* error)
    {
        if (!directory->isValid())
        {
            return false;
        }
        const QString path = directory->filePath(QStringLiteral("synthetic.isd"));
        QFile file(path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate) || file.write(kSyntheticIsd) < 0)
        {
            return false;
        }
        file.close();
        return importPlanetaryLineScanIsd(path.toStdString(),
                                          CameraDefinitionId("synthetic-definition"),
                                          CameraInstanceId("synthetic-instance"),
                                          ImageId("synthetic-image"),
                                          imported,
                                          error);
    }

} // namespace

TEST(PlanetaryLineScanIsdIO, ImportsRequiredUsgsCsmFields)
{
    QTemporaryDir directory;
    PlanetaryLineScanIsdImport imported;
    std::string error;
    ASSERT_TRUE(importSynthetic(&directory, &imported, &error)) << error;
    ASSERT_NE(imported.instance, nullptr);
    EXPECT_EQ(imported.instance->imageSize().lines, 100);
    EXPECT_EQ(imported.instance->imageSize().samples, 200);
    EXPECT_EQ(imported.metadata.modelName, "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL");
    EXPECT_EQ(imported.metadata.platformName, "SYNTHETIC ORBITER");
    EXPECT_EQ(imported.metadata.interpolationMethod, "lagrange");
    EXPECT_EQ(imported.metadata.targetName, "MOON");
    EXPECT_EQ(imported.instance->definition().worldFrame().value(), "MOON_ME");
    EXPECT_EQ(imported.metadata.bodyFixedFrameCode, 31001);
    EXPECT_DOUBLE_EQ(imported.instance->lineScanDefinition().optics().focalLengthMillimeters, 100.0);
}

TEST(PlanetaryLineScanIsdIO, RejectsUnacknowledgedInterpolationModel)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    QByteArray isd(kSyntheticIsd);
    isd.replace("\"lagrange\"", "\"spline\"");
    const QString path = directory.filePath(QStringLiteral("unsupported_interpolation.isd"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    ASSERT_EQ(file.write(isd), isd.size());
    file.close();

    PlanetaryLineScanIsdImport imported;
    std::string error;
    EXPECT_FALSE(importPlanetaryLineScanIsd(path.toStdString(),
                                            CameraDefinitionId("definition"),
                                            CameraInstanceId("instance"),
                                            ImageId("image"),
                                            &imported,
                                            &error));
    EXPECT_NE(error.find("interpolation_method='lagrange'"), std::string::npos) << error;
}

TEST(PlanetaryLineScanIsdIO, PreservesPixelCenterTimingAndDetectorGeometry)
{
    QTemporaryDir directory;
    PlanetaryLineScanIsdImport imported;
    std::string error;
    ASSERT_TRUE(importSynthetic(&directory, &imported, &error)) << error;

    double firstTime = 0.0;
    double middleTime = 0.0;
    ASSERT_TRUE(imported.instance->timeForLine(0.5, &firstTime));
    ASSERT_TRUE(imported.instance->timeForLine(50.5, &middleTime));
    EXPECT_DOUBLE_EQ(firstTime, 999.505);
    EXPECT_DOUBLE_EQ(middleTime, 1000.005);
    double line = 0.0;
    ASSERT_TRUE(imported.instance->lineForTime(firstTime, &line));
    EXPECT_NEAR(line, 0.5, 1.0e-12);

    LineScanRay ray;
    ASSERT_TRUE(LineScanProjection::ray(*imported.instance, 110.0, 50.5, &ray));
    EXPECT_NEAR(ray.origin[0], 0.0, 1.0e-9);
    EXPECT_GT(ray.direction[1], 0.0);
    EXPECT_GT(ray.direction[2], 0.999);
}

TEST(PlanetaryLineScanIsdIO, ProjectsAtLineAndAppliesTypedTrajectoryBias)
{
    QTemporaryDir directory;
    PlanetaryLineScanIsdImport imported;
    std::string error;
    ASSERT_TRUE(importSynthetic(&directory, &imported, &error)) << error;
    const std::array<double, 3> ground{{0.0, 0.0, 1000.0}};

    LineScanProjectionResult nominal;
    ASSERT_TRUE(LineScanProjection::projectAtLine(*imported.instance, ground, 50.5, &nominal));
    EXPECT_NEAR(nominal.sample, 100.0, 1.0e-10);
    EXPECT_NEAR(nominal.lineResidualPixels, 0.0, 1.0e-9);
    EXPECT_NEAR(nominal.positiveDepth, 1000.0, 1.0e-9);

    LineScanTrajectoryBias bias;
    bias.translationMeters = {10.0, 0.0, 0.0};
    LineScanProjectionResult biased;
    ASSERT_TRUE(LineScanProjection::projectAtLine(*imported.instance, ground, 50.5, bias, &biased));
    EXPECT_NEAR(biased.lineResidualPixels, -100.0, 1.0e-8);
}

TEST(PlanetaryLineScanIsdIO, IterativeProjectionFindsAcquisitionLine)
{
    QTemporaryDir directory;
    PlanetaryLineScanIsdImport imported;
    std::string error;
    ASSERT_TRUE(importSynthetic(&directory, &imported, &error)) << error;
    LineScanProjectionResult projection;
    ASSERT_TRUE(LineScanProjection::project(*imported.instance, {0.0, 0.0, 1000.0}, &projection));
    EXPECT_NEAR(projection.line, 50.5, 1.0e-8);
    EXPECT_NEAR(projection.sample, 100.0, 1.0e-8);
    EXPECT_NEAR(projection.positiveDepth, 1000.0, 1.0e-9);
    EXPECT_NEAR(projection.timeSeconds, 1000.005, 1.0e-12);
}

TEST(PlanetaryLineScanIsdIO, OptionalLroFixtureUsesMoonMeTrajectory)
{
    const std::filesystem::path fixture = std::filesystem::path(PLANETARY_TEST_DATA_DIR) /
                                          "photogrammetry_benchmarks/isis_lro_lola_lidar/extracted/ISIS3/isis/"
                                          "tests/data/lidarObservationPair/lidarObservationImage1.isd";
    if (!std::filesystem::exists(fixture))
    {
        GTEST_SKIP() << "Optional downloaded ISIS LRO fixture is not present";
    }

    PlanetaryLineScanIsdImport imported;
    std::string error;
    ASSERT_TRUE(importPlanetaryLineScanIsd(fixture.string(),
                                           CameraDefinitionId("lro-definition"),
                                           CameraInstanceId("lro-instance"),
                                           ImageId("lro-image"),
                                           &imported,
                                           &error))
        << error;
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
    const Pose pose = imported.instance->trajectory().poseAt(TimeReference::create(TimeScale::Tdb, shotTime),
                                                             imported.instance->definition().worldFrame());
    const double range = std::hypot(std::hypot(ground[0] - pose.center[0], ground[1] - pose.center[1]),
                                    ground[2] - pose.center[2]);
    EXPECT_NEAR(range, 56552.95, 0.15);

    LineScanProjectionResult fixedLine;
    ASSERT_TRUE(LineScanProjection::projectAtLine(
        *imported.instance, ground, 18995.176438712828 - 0.5, &fixedLine));
    EXPECT_NEAR(fixedLine.lineResidualPixels, 0.0, 20.0);
    EXPECT_NEAR(fixedLine.sample, 4979.609104416888 - 0.5, 5.0);

    LineScanProjectionResult projected;
    ASSERT_TRUE(LineScanProjection::project(*imported.instance, ground, &projected));
    EXPECT_NEAR(projected.line, 18995.176438712828 - 0.5, 20.0);
    EXPECT_NEAR(projected.sample, 4979.609104416888 - 0.5, 5.0);
}
