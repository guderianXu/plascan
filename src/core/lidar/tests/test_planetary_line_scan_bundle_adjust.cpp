#include "PlanetaryLineScanBundleAdjust.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace
{

struct ScopedTestDirectory
{
    std::filesystem::path path;

    ~ScopedTestDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::string syntheticLineScanIsd(double cameraY)
{
    std::ostringstream json;
    json << R"JSON({
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
  "optical_distortion": {"lrolrocnac": {"coefficients": [0.0]}},
  "instrument_position": {
    "ephemeris_times": [998.0, 1002.0],
    "positions": [[-2.005, )JSON" << cameraY << R"JSON(, 0.0], [1.995, )JSON"
         << cameraY << R"JSON(, 0.0]],
    "velocities": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
  }
})JSON";
    return json.str();
}

bool writeTextFile(const std::filesystem::path &path, const std::string &contents)
{
    std::ofstream stream(path, std::ios::binary);
    stream << contents;
    return stream.good();
}

double distance(const std::array<double, 3> &left, const std::array<double, 3> &right)
{
    return std::hypot(std::hypot(left[0] - right[0], left[1] - right[1]),
                      left[2] - right[2]);
}

} // namespace

TEST(PlanetaryLineScanBundleAdjustTest, TriangulatesCrossingForwardRays)
{
    xjw::PlanetaryLineScanCamera::ImagingRay first;
    first.centerBodyFixedMeters = {{-1.0, 0.0, 0.0}};
    first.directionBodyFixed = {{1.0, 0.0, 1.0}};
    xjw::PlanetaryLineScanCamera::ImagingRay second;
    second.centerBodyFixedMeters = {{1.0, 0.0, 0.0}};
    second.directionBodyFixed = {{-1.0, 0.0, 1.0}};

    std::array<double, 3> point{};
    double separation = -1.0;
    ASSERT_TRUE(xjw::lidar::triangulatePlanetaryLineScanRays(
        first, second, &point, &separation));
    EXPECT_NEAR(point[0], 0.0, 1.0e-12);
    EXPECT_NEAR(point[1], 0.0, 1.0e-12);
    EXPECT_NEAR(point[2], 1.0, 1.0e-12);
    EXPECT_NEAR(separation, 0.0, 1.0e-12);
}

TEST(PlanetaryLineScanBundleAdjustTest, RejectsParallelRays)
{
    xjw::PlanetaryLineScanCamera::ImagingRay first;
    first.centerBodyFixedMeters = {{0.0, 0.0, 0.0}};
    first.directionBodyFixed = {{0.0, 0.0, 1.0}};
    xjw::PlanetaryLineScanCamera::ImagingRay second;
    second.centerBodyFixedMeters = {{1.0, 0.0, 0.0}};
    second.directionBodyFixed = {{0.0, 0.0, 1.0}};

    std::array<double, 3> point{};
    EXPECT_FALSE(xjw::lidar::triangulatePlanetaryLineScanRays(
        first, second, &point));
}

TEST(PlanetaryLineScanBundleAdjustTest, NamesScientificAndIsisCompatibleTimeModes)
{
    EXPECT_STREQ(xjw::lidar::planetaryLaserLineScanTimeModeName(
                     xjw::lidar::PlanetaryLaserLineScanTimeMode::ShotEphemerisTime),
                 "shot_et");
    EXPECT_STREQ(xjw::lidar::planetaryLaserLineScanTimeModeName(
                     xjw::lidar::PlanetaryLaserLineScanTimeMode::IsisSimultaneousMeasureLine),
                 "isis_simultaneous_measure_line");
}

TEST(PlanetaryLineScanBundleAdjustTest, ConvertsIsisPixelsAndLaserRangeAnchorsFreeNetwork)
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    ScopedTestDirectory directory{
        std::filesystem::temp_directory_path() /
        ("plascan_linescan_ba_" + std::to_string(nonce))};
    ASSERT_TRUE(std::filesystem::create_directories(directory.path));

    std::vector<xjw::lidar::PlanetaryLineScanBaCamera> cameras(2);
    cameras[0].serialNumber = "CAMERA_A";
    cameras[1].serialNumber = "CAMERA_B";
    for (int index = 0; index < 2; ++index)
    {
        const auto isdPath = directory.path / ("camera_" + std::to_string(index) + ".isd");
        ASSERT_TRUE(writeTextFile(isdPath, syntheticLineScanIsd(index == 0 ? -0.001 : 0.001)));
        std::string error;
        ASSERT_TRUE(cameras[index].model.loadFromIsd(isdPath.string(), &error)) << error;
    }

    xjw::lidar::IsisControlNetwork network;
    network.networkId = "synthetic_linescan";
    network.targetName = "MOON";
    for (int index = 0; index < 8; ++index)
    {
        const std::array<double, 3> ground{{
            -0.3 + 0.08 * index,
            -0.0005 + 0.00014 * index,
            1000.0 + 0.5 * (index % 3)}};
        xjw::lidar::IsisControlPoint point;
        point.id = "POINT_" + std::to_string(index);
        point.type = xjw::lidar::IsisControlPointType::Free;
        for (const auto &camera : cameras)
        {
            xjw::PlanetaryLineScanCamera::ImageCoordinate csm;
            ASSERT_TRUE(camera.model.groundToImage(
                ground,
                xjw::PlanetaryLineScanCamera::PixelConvention::CsmPixelCenter,
                &csm));
            // Serialize as ISIS PVL coordinates: the first pixel centre is (1, 1).
            point.measures.push_back({camera.serialNumber,
                                      csm.sample + 0.5,
                                      csm.line + 0.5,
                                      false});
        }
        network.points.push_back(std::move(point));
    }

    double shotEt = 0.0;
    ASSERT_TRUE(cameras[0].model.absoluteEtForLine(
        50.5,
        xjw::PlanetaryLineScanCamera::PixelConvention::CsmPixelCenter,
        &shotEt));
    std::array<double, 3> nominalCenter{};
    ASSERT_TRUE(cameras[0].model.sensorCenterBodyFixedAtEt(shotEt, &nominalCenter));
    const std::array<double, 3> fixedLaserPoint{{0.0, 0.0, 1000.0}};
    std::array<double, 3> displacedCenter = nominalCenter;
    displacedCenter[2] += 20.0;

    xjw::lidar::PlanetaryLaserDataset laser;
    laser.sourceFormat = xjw::lidar::PlanetaryLaserSourceFormat::PlaScanSiJsonV1;
    laser.sensorModel = xjw::lidar::PlanetaryLaserSensorModel::LineScan;
    laser.rangeType = xjw::lidar::PlanetaryLaserRangeType::OneWay;
    laser.reference.targetName = "MOON";
    laser.reference.bodyFixedFrame = "MOON_ME";
    laser.reference.laserFrame = "SYNTHETIC_LASER";
    laser.reference.timeSystem = xjw::lidar::PlanetaryLaserTimeSystem::TdbEtSeconds;
    laser.reference.latitudeType = "planetocentric";
    laser.reference.longitudeDirection = "positive_east";
    xjw::lidar::PlanetaryLaserShot shot;
    shot.id = "LASER_0";
    shot.pointMode = xjw::lidar::PlanetaryLaserPointMode::Fixed;
    shot.ephemerisTimeSeconds = shotEt;
    shot.observedRangeMeters = distance(fixedLaserPoint, displacedCenter);
    shot.rangeSigmaMeters = 0.1;
    shot.pointBodyFixedMeters = fixedLaserPoint;
    shot.simultaneousImageIds = {"CAMERA_A"};
    shot.leverArmSpecified = true;
    laser.shots.push_back(shot);

    xjw::lidar::PlanetaryLineScanBaOptions options;
    options.backend = xjw::BABackend::PlaMatrixCpu;
    options.maximumIterations = 30;
    options.imageHuberDeltaPixels = 0.0;
    options.laserRangeWeight = 2.5;
    options.laserRangeHuberDeltaSigma = 3.0;
    options.cameraAngleSigmaDegrees = 0.05;

    xjw::lidar::PlanetaryLineScanBaResult noLaser;
    std::string error;
    ASSERT_TRUE(xjw::lidar::runPlanetaryLineScanBundleAdjust(
        cameras, network, &laser, options, &noLaser, &error)) << error;
    ASSERT_TRUE(noLaser.success) << error;
    EXPECT_EQ(noLaser.usedBackend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_LT(noLaser.initialImageRmsPixels, 1.0e-6);
    EXPECT_GT(noLaser.refinedLaserRangeRmsMeters, 10.0);

    options.enableLaserRangeConstraints = true;
    xjw::lidar::PlanetaryLineScanBaResult withLaser;
    ASSERT_TRUE(xjw::lidar::runPlanetaryLineScanBundleAdjust(
        cameras, network, &laser, options, &withLaser, &error)) << error;
    ASSERT_TRUE(withLaser.success) << error;
    EXPECT_EQ(withLaser.usedBackend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_EQ(withLaser.controlPointCount, 8);
    EXPECT_EQ(withLaser.imageObservationCount, 16);
    EXPECT_EQ(withLaser.activeLaserRangeCount, 1);
    EXPECT_LT(withLaser.refinedLaserRangeRmsMeters,
              noLaser.refinedLaserRangeRmsMeters * 0.1);

    for (const xjw::BABackend backend :
         {xjw::BABackend::PlaMatrixCuda, xjw::BABackend::PlaMatrixOpenCl})
    {
        if (!xjw::BundleAdjust::isBackendAvailable(backend))
        {
            continue;
        }
        options.backend = backend;
        options.allowBackendFallback = false;
        xjw::lidar::PlanetaryLineScanBaResult accelerated;
        ASSERT_TRUE(xjw::lidar::runPlanetaryLineScanBundleAdjust(
            cameras, network, &laser, options, &accelerated, &error)) << error;
        EXPECT_EQ(accelerated.usedBackend, backend);
        EXPECT_TRUE(accelerated.usedGpu);
        EXPECT_FALSE(accelerated.deviceName.empty());
        EXPECT_NEAR(accelerated.refinedImageRmsPixels, withLaser.refinedImageRmsPixels, 1.0e-6);
        EXPECT_NEAR(accelerated.refinedLaserRangeRmsMeters, withLaser.refinedLaserRangeRmsMeters, 1.0e-6);
    }

    options.backend = xjw::BABackend::Auto;
    options.minPlaMatrixCudaCameras = 1000;
    options.minPlaMatrixCudaObservations = 1000000;
    options.minPlaMatrixOpenClCameras = 1000;
    options.minPlaMatrixOpenClObservations = 1000000;
    options.minPlaMatrixDenseCameras = 1000;
    xjw::lidar::PlanetaryLineScanBaResult automatic;
    ASSERT_TRUE(xjw::lidar::runPlanetaryLineScanBundleAdjust(cameras, network, &laser, options, &automatic, &error))
        << error;
    EXPECT_EQ(automatic.requestedBackend, xjw::BABackend::Auto);
    EXPECT_EQ(automatic.usedBackend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_FALSE(automatic.backendFallback);

    laser.shots.front().pointMode = xjw::lidar::PlanetaryLaserPointMode::Constrained;
    laser.shots.front().pointCovarianceBodyFixedMetersSquared = {{
        0.01, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.01}};
    options.backend = xjw::BABackend::PlaMatrixCpu;
    xjw::lidar::PlanetaryLineScanBaResult constrained;
    ASSERT_TRUE(xjw::lidar::runPlanetaryLineScanBundleAdjust(
        cameras, network, &laser, options, &constrained, &error)) << error;
    ASSERT_TRUE(constrained.success);
    ASSERT_EQ(constrained.laserShots.size(), 1u);
    EXPECT_LT(constrained.refinedLaserRangeRmsMeters,
              noLaser.refinedLaserRangeRmsMeters * 0.1);

    options.backend = xjw::BABackend::PlaMatrixCpu;
    options.allowBackendFallback = true;
    options.cancelFlag = std::make_shared<std::atomic<bool>>(true);
    xjw::lidar::PlanetaryLineScanBaResult cancelled;
    EXPECT_FALSE(xjw::lidar::runPlanetaryLineScanBundleAdjust(
        cameras, network, &laser, options, &cancelled, &error));
    EXPECT_FALSE(cancelled.success);
    EXPECT_FALSE(cancelled.solutionUsable);
    EXPECT_FALSE(cancelled.backendFallback);
    EXPECT_EQ(cancelled.usedBackend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_EQ(cancelled.terminationType, "CANCELLED");
    for (const auto &camera : cancelled.cameras)
    {
        EXPECT_EQ(camera.translationBodyFixedMeters,
                  (std::array<double, 3>{{0.0, 0.0, 0.0}}));
        EXPECT_EQ(camera.angleAxisBodyFixedRadians,
                  (std::array<double, 3>{{0.0, 0.0, 0.0}}));
    }
}
