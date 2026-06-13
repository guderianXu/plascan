#include <gtest/gtest.h>

#include "Camera.h"
#include "CameraFormatConverter.h"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <optional>
#include <sstream>

namespace
{

void writeText(const std::filesystem::path &path, const std::string &text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    ASSERT_TRUE(out.good()) << path;
    out << text;
}

std::string readText(const std::filesystem::path &path)
{
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

} // namespace

TEST(CameraFormatConverterTest, KnownFormatsExposeCliNames)
{
    const auto formats = xjw::camera::supportedFormatNames();

    EXPECT_NE(std::find(formats.begin(), formats.end(), "auto"), formats.end());
    EXPECT_NE(std::find(formats.begin(), formats.end(), "middlebury-par"), formats.end());
    EXPECT_NE(std::find(formats.begin(), formats.end(), "epfl-camera"), formats.end());

    EXPECT_EQ(xjw::camera::parseCameraFormat("middlebury_par"), xjw::camera::CameraFormat::MiddleburyPar);
    EXPECT_EQ(xjw::camera::parseCameraFormat("epfl-camera"), xjw::camera::CameraFormat::EpflCamera);
    EXPECT_EQ(xjw::camera::parseCameraFormat("unknown"), std::nullopt);
}

TEST(CameraFormatConverterTest, MiddleburyParConvertsToTsaiAndImageCameraList)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "plascan_camera_convert_middlebury_test";
    std::filesystem::remove_all(root);

    const std::filesystem::path source = root / "extracted" / "dinoSparseRing";
    writeText(source / "dinoSR0001.png", "fake image");
    writeText(source / "dinoSR0002.png", "fake image");
    writeText(source / "dinoSR_par.txt",
              "2\n"
              "dinoSR0001.png 120 0 40 0 130 50 0 0 1 0 -1 0 1 0 0 0 0 1 2 -3 4\n"
              "dinoSR0002.png 121 0 41 0 131 51 0 0 1 1 0 0 0 1 0 0 0 1 4 5 6\n");

    xjw::camera::CameraConversionOptions options;
    options.format = xjw::camera::CameraFormat::MiddleburyPar;
    options.inputPath = source;
    options.outputDir = root / "out";
    options.datasetId = "dino";
    options.overwrite = true;

    const auto result = xjw::camera::convertCameraDataset(options);
    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(result.cameraCount, 2);
    EXPECT_EQ(result.inputFormat, xjw::camera::CameraFormat::MiddleburyPar);
    EXPECT_TRUE(std::filesystem::exists(result.imageCameraList));
    EXPECT_TRUE(std::filesystem::exists(result.summaryPath));

    const std::string lis = readText(result.imageCameraList);
    EXPECT_NE(lis.find("dinoSR0001.png cameras/dinoSR0001.tsai"), std::string::npos);
    EXPECT_NE(lis.find("dinoSR0002.png cameras/dinoSR0002.tsai"), std::string::npos);

    xjw::Camera camera;
    ASSERT_TRUE(camera.loadFromFile((options.outputDir / "cameras" / "dinoSR0001.tsai").string()));
    EXPECT_DOUBLE_EQ(camera.focalX(), 120.0);
    EXPECT_DOUBLE_EQ(camera.focalY(), 130.0);
    EXPECT_DOUBLE_EQ(camera.principalX(), 40.0);
    EXPECT_DOUBLE_EQ(camera.principalY(), 50.0);

    const auto center = camera.cameraCenter();
    EXPECT_DOUBLE_EQ(center[0], 3.0);
    EXPECT_DOUBLE_EQ(center[1], 2.0);
    EXPECT_DOUBLE_EQ(center[2], -4.0);

    const auto rotation = camera.cameraToWorldRotation();
    EXPECT_DOUBLE_EQ(rotation[0], 0.0);
    EXPECT_DOUBLE_EQ(rotation[1], 1.0);
    EXPECT_DOUBLE_EQ(rotation[2], 0.0);
    EXPECT_DOUBLE_EQ(rotation[3], -1.0);
    EXPECT_DOUBLE_EQ(rotation[4], 0.0);
    EXPECT_DOUBLE_EQ(rotation[5], 0.0);
    EXPECT_DOUBLE_EQ(rotation[6], 0.0);
    EXPECT_DOUBLE_EQ(rotation[7], 0.0);
    EXPECT_DOUBLE_EQ(rotation[8], 1.0);

    const std::string summary = readText(result.summaryPath);
    EXPECT_NE(summary.find("\"input_format\": \"middlebury-par\""), std::string::npos);
    EXPECT_NE(summary.find("\"camera_count\": 2"), std::string::npos);

    std::filesystem::remove_all(root);
}

TEST(CameraFormatConverterTest, EpflCameraConvertsWithSkewWarning)
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "plascan_camera_convert_epfl_test";
    std::filesystem::remove_all(root);

    const std::filesystem::path source = root / "epfl";
    writeText(source / "rdimage.000.ppm", "fake image");
    writeText(source / "rdimage.000.ppm.camera",
              "3954.75 -8.5 1619.9\n"
              "0 3948.0 1151.4\n"
              "0 0 1\n"
              "0 0 0\n"
              "0 -1 0\n"
              "1 0 0\n"
              "0 0 1\n"
              "60 -11 -35\n"
              "3072 2048\n");

    xjw::camera::CameraConversionOptions options;
    options.format = xjw::camera::CameraFormat::EpflCamera;
    options.inputPath = source;
    options.outputDir = root / "out";
    options.overwrite = true;

    const auto result = xjw::camera::convertCameraDataset(options);
    ASSERT_TRUE(result.success) << result.errorMessage;
    ASSERT_EQ(result.cameraCount, 1);
    ASSERT_FALSE(result.warnings.empty());
    EXPECT_NE(result.warnings.front().find("skew"), std::string::npos);

    xjw::Camera camera;
    ASSERT_TRUE(camera.loadFromFile((options.outputDir / "cameras" / "rdimage.000.ppm.tsai").string()));
    EXPECT_DOUBLE_EQ(camera.focalX(), 3954.75);
    EXPECT_DOUBLE_EQ(camera.focalY(), 3948.0);
    EXPECT_DOUBLE_EQ(camera.principalX(), 1619.9);
    EXPECT_DOUBLE_EQ(camera.principalY(), 1151.4);

    const auto center = camera.cameraCenter();
    EXPECT_DOUBLE_EQ(center[0], 60.0);
    EXPECT_DOUBLE_EQ(center[1], -11.0);
    EXPECT_DOUBLE_EQ(center[2], -35.0);

    const std::string summary = readText(result.summaryPath);
    EXPECT_NE(summary.find("\"input_format\": \"epfl-camera\""), std::string::npos);
    EXPECT_NE(summary.find("skew"), std::string::npos);

    std::filesystem::remove_all(root);
}
