#include "PlanetaryLineScanCamera.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QTemporaryDir>

#include <cmath>
#include <filesystem>
#include <string>

namespace
{

using LineScanCamera = xjw::PlanetaryLineScanCamera;

const char *kSyntheticIsd = R"JSON({
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

bool loadSyntheticCamera(QTemporaryDir *directory, LineScanCamera *camera, std::string *error)
{
    if (!directory->isValid())
    {
        return false;
    }
    const QString path = directory->filePath(QStringLiteral("synthetic.isd"));
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)
        || file.write(kSyntheticIsd) < 0)
    {
        return false;
    }
    file.close();
    return camera->loadFromIsd(path.toUtf8().toStdString(), error);
}

} // namespace

TEST(PlanetaryLineScanCameraIsd, ParsesRequiredUsgsCsmFields)
{
    QTemporaryDir directory;
    LineScanCamera camera;
    std::string error;
    ASSERT_TRUE(loadSyntheticCamera(&directory, &camera, &error)) << error;
    EXPECT_TRUE(camera.isValid());
    EXPECT_EQ(camera.imageLines(), 100);
    EXPECT_EQ(camera.imageSamples(), 200);
    EXPECT_EQ(camera.modelName(), "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL");
    EXPECT_EQ(camera.platformName(), "SYNTHETIC ORBITER");
    EXPECT_EQ(camera.declaredInterpolationMethod(), "lagrange");
    EXPECT_EQ(camera.targetName(), "MOON");
    EXPECT_EQ(camera.bodyFixedFrameName(), "MOON_ME");
    EXPECT_EQ(camera.bodyFixedFrameCode(), 31001);
    EXPECT_DOUBLE_EQ(camera.focalLengthMillimeters(), 100.0);
}

TEST(PlanetaryLineScanCameraIsd, RejectsUnacknowledgedInterpolationModel)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    QByteArray isd(kSyntheticIsd);
    ASSERT_TRUE(isd.contains("\"lagrange\""));
    isd.replace("\"lagrange\"", "\"spline\"");
    const QString path = directory.filePath(QStringLiteral("unsupported_interpolation.isd"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    ASSERT_EQ(file.write(isd), isd.size());
    file.close();

    LineScanCamera camera;
    std::string error;
    EXPECT_FALSE(camera.loadFromIsd(path.toUtf8().toStdString(), &error));
    EXPECT_NE(error.find("interpolation_method='lagrange'"), std::string::npos) << error;
}

TEST(PlanetaryLineScanCameraTime, PreservesCsmAndOpenCvPixelCenterConventions)
{
    QTemporaryDir directory;
    LineScanCamera camera;
    std::string error;
    ASSERT_TRUE(loadSyntheticCamera(&directory, &camera, &error)) << error;

    double csm_et = 0.0;
    double opencv_et = 0.0;
    ASSERT_TRUE(camera.absoluteEtForLine(0.5, LineScanCamera::PixelConvention::CsmPixelCenter, &csm_et));
    ASSERT_TRUE(camera.absoluteEtForLine(0.0, LineScanCamera::PixelConvention::OpenCvZeroBased, &opencv_et));
    EXPECT_DOUBLE_EQ(csm_et, 999.505);
    EXPECT_DOUBLE_EQ(opencv_et, csm_et);

    double csm_line = 0.0;
    double opencv_line = 0.0;
    ASSERT_TRUE(camera.lineForAbsoluteEt(csm_et, LineScanCamera::PixelConvention::CsmPixelCenter, &csm_line));
    ASSERT_TRUE(camera.lineForAbsoluteEt(csm_et, LineScanCamera::PixelConvention::OpenCvZeroBased, &opencv_line));
    EXPECT_NEAR(csm_line, 0.5, 1.0e-12);
    EXPECT_NEAR(opencv_line, 0.0, 1.0e-12);
    EXPECT_FALSE(camera.absoluteEtForLine(-0.01, LineScanCamera::PixelConvention::OpenCvZeroBased, &opencv_et));
}

TEST(PlanetaryLineScanCameraOptics, FocalDistortionAndRayRoundTrip)
{
    QTemporaryDir directory;
    LineScanCamera camera;
    std::string error;
    ASSERT_TRUE(loadSyntheticCamera(&directory, &camera, &error)) << error;

    std::array<double, 2> distorted{};
    std::array<double, 2> undistorted{};
    std::array<double, 2> restored{};
    ASSERT_TRUE(camera.pixelToDistortedFocalPlane(
        110.0, LineScanCamera::PixelConvention::CsmPixelCenter, &distorted));
    EXPECT_NEAR(distorted[0], 0.0, 1.0e-15);
    EXPECT_NEAR(distorted[1], 0.1, 1.0e-15);
    ASSERT_TRUE(camera.removeOpticalDistortion(distorted, &undistorted));
    ASSERT_TRUE(camera.applyOpticalDistortion(undistorted, &restored));
    EXPECT_NEAR(restored[0], distorted[0], 1.0e-12);
    EXPECT_NEAR(restored[1], distorted[1], 1.0e-10);

    LineScanCamera::ImagingRay csm_ray;
    LineScanCamera::ImagingRay opencv_ray;
    ASSERT_TRUE(camera.pixelRayBodyFixed(
        110.0, 50.5, LineScanCamera::PixelConvention::CsmPixelCenter, &csm_ray));
    ASSERT_TRUE(camera.pixelRayBodyFixed(
        109.5, 50.0, LineScanCamera::PixelConvention::OpenCvZeroBased, &opencv_ray));
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_NEAR(csm_ray.centerBodyFixedMeters[axis], opencv_ray.centerBodyFixedMeters[axis], 1.0e-12);
        EXPECT_NEAR(csm_ray.directionBodyFixed[axis], opencv_ray.directionBodyFixed[axis], 1.0e-12);
    }
    EXPECT_GT(csm_ray.directionBodyFixed[1], 0.0);
    EXPECT_GT(csm_ray.directionBodyFixed[2], 0.999);
}

TEST(PlanetaryLineScanCameraProjection, FixedLineResidualSupportsBundleAdjustmentBias)
{
    QTemporaryDir directory;
    LineScanCamera camera;
    std::string error;
    ASSERT_TRUE(loadSyntheticCamera(&directory, &camera, &error)) << error;
    const LineScanCamera::Vector3 ground{{0.0, 0.0, 1000.0}};

    LineScanCamera::FixedLineProjection nominal;
    ASSERT_TRUE(camera.projectAtObservedLine(
        ground, 50.5, LineScanCamera::PixelConvention::CsmPixelCenter, &nominal));
    EXPECT_NEAR(nominal.sample, 100.0, 1.0e-10);
    EXPECT_NEAR(nominal.detectorLineResidualPixels, 0.0, 1.0e-9);
    EXPECT_NEAR(nominal.sensorDepthMeters, 1000.0, 1.0e-9);

    const LineScanCamera::PoseBias translated = LineScanCamera::bodyFixedSmallAngleBias(
        {0.0, 0.0, 0.0}, {10.0, 0.0, 0.0});
    LineScanCamera::FixedLineProjection biased;
    ASSERT_TRUE(camera.projectAtObservedLine(
        ground, 50.5, LineScanCamera::PixelConvention::CsmPixelCenter, &biased, translated));
    EXPECT_NEAR(biased.detectorLineResidualPixels, -100.0, 1.0e-8);
}

TEST(PlanetaryLineScanCameraProjection, IterativeGroundToImageFindsPushbroomLine)
{
    QTemporaryDir directory;
    LineScanCamera camera;
    std::string error;
    ASSERT_TRUE(loadSyntheticCamera(&directory, &camera, &error)) << error;
    const LineScanCamera::Vector3 ground{{0.0, 0.0, 1000.0}};

    LineScanCamera::ImageCoordinate csm_image;
    LineScanCamera::ImageCoordinate opencv_image;
    LineScanCamera::ImageCoordinate configured_image;
    LineScanCamera::GroundToImageOptions options;
    ASSERT_TRUE(camera.groundToImage(
        ground, LineScanCamera::PixelConvention::CsmPixelCenter, &csm_image));
    ASSERT_TRUE(camera.groundToImage(
        ground, LineScanCamera::PixelConvention::OpenCvZeroBased, &opencv_image));
    ASSERT_TRUE(camera.groundToImage(
        ground, LineScanCamera::PixelConvention::CsmPixelCenter, &configured_image, options));
    EXPECT_NEAR(csm_image.line, 50.5, 1.0e-8);
    EXPECT_NEAR(csm_image.sample, 100.0, 1.0e-8);
    EXPECT_NEAR(configured_image.line, csm_image.line, 1.0e-12);
    EXPECT_NEAR(configured_image.sample, csm_image.sample, 1.0e-12);
    EXPECT_NEAR(opencv_image.line, csm_image.line - 0.5, 1.0e-8);
    EXPECT_NEAR(opencv_image.sample, csm_image.sample - 0.5, 1.0e-8);
}

TEST(PlanetaryLineScanCameraModel, ImplementsCommonOpenCvGeometry)
{
    QTemporaryDir directory;
    LineScanCamera camera;
    std::string error;
    ASSERT_TRUE(loadSyntheticCamera(&directory, &camera, &error)) << error;

    const xjw::CameraModel &model = camera;
    EXPECT_EQ(model.modelType(), xjw::CameraModelType::PlanetaryLineScan);
    ASSERT_TRUE(model.imageSize().has_value());
    EXPECT_EQ(model.imageSize()->samples, 200);
    EXPECT_EQ(model.imageSize()->lines, 100);
    EXPECT_EQ(model.worldFrameName(), "MOON_ME");

    const xjw::CameraImageCoordinate pixel{99.5, 50.0};
    xjw::CameraImagingRay common_ray;
    LineScanCamera::ImagingRay direct_ray;
    ASSERT_TRUE(model.rayForPixel(pixel, &common_ray));
    ASSERT_TRUE(camera.pixelRayBodyFixed(
        pixel.sample,
        pixel.line,
        LineScanCamera::PixelConvention::OpenCvZeroBased,
        &direct_ray));
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_NEAR(common_ray.originMeters[axis],
                    direct_ray.centerBodyFixedMeters[axis],
                    1.0e-12);
        EXPECT_NEAR(common_ray.direction[axis],
                    direct_ray.directionBodyFixed[axis],
                    1.0e-12);
    }
    ASSERT_TRUE(common_ray.ephemerisTimeSeconds.has_value());
    EXPECT_DOUBLE_EQ(*common_ray.ephemerisTimeSeconds,
                     direct_ray.ephemerisTimeSeconds);

    xjw::CameraGroundProjection projection;
    ASSERT_TRUE(model.groundToImage({0.0, 0.0, 1000.0}, &projection));
    EXPECT_NEAR(projection.image.sample, 99.5, 1.0e-8);
    EXPECT_NEAR(projection.image.line, 50.0, 1.0e-8);
    EXPECT_NEAR(projection.positiveDepthMeters, 1000.0, 1.0e-9);
    ASSERT_TRUE(projection.ephemerisTimeSeconds.has_value());
    EXPECT_NEAR(*projection.ephemerisTimeSeconds, 1000.005, 1.0e-12);
}

TEST(PlanetaryLineScanCameraFixture, LroIsdUsesMoonMeMetersAndTimeDependentPose)
{
    const std::filesystem::path fixture = std::filesystem::path(PLANETARY_TEST_DATA_DIR)
        / "photogrammetry_benchmarks" / "isis_lro_lola_lidar" / "extracted" / "ISIS3"
        / "isis" / "tests" / "data" / "lidarObservationPair" / "lidarObservationImage1.isd";
    if (!std::filesystem::exists(fixture))
    {
        GTEST_SKIP() << "Optional downloaded ISIS LRO fixture is not present";
    }

    LineScanCamera camera;
    std::string error;
    ASSERT_TRUE(camera.loadFromIsd(fixture.string(), &error)) << error;
    EXPECT_EQ(camera.imageLines(), 52224);
    EXPECT_EQ(camera.imageSamples(), 5064);
    EXPECT_NEAR(camera.focalLengthMillimeters(), 699.62, 1.0e-12);

    constexpr double shot_et = 317845268.6627772;
    constexpr double latitude_degrees = 60.8347278;
    constexpr double longitude_degrees = 99.0976484;
    constexpr double radius_meters = 1735261.573;
    constexpr double pi = 3.14159265358979323846;
    const double latitude = latitude_degrees * pi / 180.0;
    const double longitude = longitude_degrees * pi / 180.0;
    const LineScanCamera::Vector3 ground{{
        radius_meters * std::cos(latitude) * std::cos(longitude),
        radius_meters * std::cos(latitude) * std::sin(longitude),
        radius_meters * std::sin(latitude)
    }};
    LineScanCamera::Vector3 center{};
    ASSERT_TRUE(camera.sensorCenterBodyFixedAtEt(shot_et, &center));
    const double dx = ground[0] - center[0];
    const double dy = ground[1] - center[1];
    const double dz = ground[2] - center[2];
    const double range = std::sqrt(dx * dx + dy * dy + dz * dz);
    EXPECT_NEAR(range, 56552.95, 0.15);

    LineScanCamera::FixedLineProjection fixed_line;
    ASSERT_TRUE(camera.projectAtObservedLine(
        ground, 18995.176438712828 - 0.5,
        LineScanCamera::PixelConvention::CsmPixelCenter, &fixed_line));
    EXPECT_NEAR(fixed_line.detectorLineResidualPixels, 0.0, 20.0);
    EXPECT_NEAR(fixed_line.sample, 4979.609104416888 - 0.5, 5.0);

    LineScanCamera::ImageCoordinate projected;
    ASSERT_TRUE(camera.groundToImage(
        ground, LineScanCamera::PixelConvention::CsmPixelCenter, &projected));
    EXPECT_NEAR(projected.line, 18995.176438712828 - 0.5, 20.0);
    EXPECT_NEAR(projected.sample, 4979.609104416888 - 0.5, 5.0);

    LineScanCamera::Matrix3 first_rotation{};
    LineScanCamera::Matrix3 last_rotation{};
    ASSERT_TRUE(camera.sensorToBodyFixedAtEt(317845261.8, &first_rotation));
    ASSERT_TRUE(camera.sensorToBodyFixedAtEt(317845280.8, &last_rotation));
    EXPECT_GT(std::abs(first_rotation[0] - last_rotation[0]), 1.0e-4);
}
