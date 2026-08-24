#include "CameraFormatConverter.h"
#include "ColmapImageUndistorter.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>

namespace
{

cv::Mat checkerboard(int width, int height)
{
    cv::Mat image(height, width, CV_8UC3);
    for (int row = 0; row < height; ++row)
    {
        cv::Vec3b *pixels = image.ptr<cv::Vec3b>(row);
        for (int column = 0; column < width; ++column)
        {
            const std::uint8_t value =
                ((row / 4 + column / 4) % 2) == 0 ? 32 : 224;
            pixels[column] = cv::Vec3b(value, value, value);
        }
    }
    return image;
}

TEST(ColmapImageUndistorterTest, PinholeMappingIsPixelExact)
{
    const cv::Mat source = checkerboard(64, 48);
    const xjw::camera::ColmapRasterModel camera{
        "PINHOLE", 64, 48, {60.0, 59.0, 32.5, 24.5}};

    const auto result = xjw::camera::undistortColmapRaster(source, camera);

    EXPECT_DOUBLE_EQ(result.focalScale, 1.0);
    EXPECT_EQ(cv::countNonZero(result.validMask), 64 * 48);
    EXPECT_EQ(cv::countNonZero(result.image.reshape(1) != source.reshape(1)), 0);
    EXPECT_DOUBLE_EQ(result.rasterIndexIntrinsics[2], 32.0);
    EXPECT_DOUBLE_EQ(result.rasterIndexIntrinsics[5], 24.0);
}

TEST(ColmapImageUndistorterTest, ThinPrismFisheyeProducesFullValidPinholeRaster)
{
    const cv::Mat source = checkerboard(96, 72);
    const xjw::camera::ColmapRasterModel camera{
        "THIN_PRISM_FISHEYE",
        96,
        72,
        {52.0, 51.0, 48.5, 36.5,
         0.21, 0.20, 0.0005, -0.0002, -0.16, 0.40, 0.001, -0.001}};

    const auto result = xjw::camera::undistortColmapRaster(source, camera);

    EXPECT_GE(result.focalScale, 1.0);
    EXPECT_EQ(result.image.size(), source.size());
    EXPECT_EQ(result.image.type(), source.type());
    EXPECT_EQ(cv::countNonZero(result.validMask), 96 * 72);
    EXPECT_DOUBLE_EQ(
        result.rasterIndexIntrinsics[0], camera.params[0] * result.focalScale);
    EXPECT_DOUBLE_EQ(
        result.rasterIndexIntrinsics[4], camera.params[1] * result.focalScale);
    EXPECT_DOUBLE_EQ(result.rasterIndexIntrinsics[2], 48.0);
    EXPECT_DOUBLE_EQ(result.rasterIndexIntrinsics[5], 36.0);
}

TEST(ColmapImageUndistorterTest, RejectsMalformedModelBeforeReadingPixels)
{
    const xjw::camera::ColmapRasterModel camera{
        "THIN_PRISM_FISHEYE", 32, 24, {1.0, 2.0}};
    EXPECT_FALSE(xjw::camera::isSupportedColmapPreUndistortModel(camera));
    EXPECT_THROW(
        xjw::camera::undistortColmapRaster(cv::Mat(24, 32, CV_8UC1), camera),
        std::invalid_argument);
}

TEST(ColmapImageUndistorterTest, ConverterPublishesPreparedRasterMaskAndPinholeCameraAtomically)
{
    const std::filesystem::path root = std::filesystem::temp_directory_path() /
        "plascan_colmap_preundistort_test";
    std::filesystem::remove_all(root);
    const std::filesystem::path sparse = root / "dataset" / "sparse";
    const std::filesystem::path images = root / "dataset" / "images";
    std::filesystem::create_directories(sparse);
    std::filesystem::create_directories(images);
    {
        std::ofstream cameras(sparse / "cameras.txt");
        cameras << "1 THIN_PRISM_FISHEYE 96 72 52 51 48.5 36.5 "
                << "0.21 0.20 0.0005 -0.0002 -0.16 0.40 0.001 -0.001\n";
        std::ofstream poses(sparse / "images.txt");
        poses << "1 1 0 0 0 0 0 0 1 frame.jpg\n\n";
        std::ofstream points(sparse / "points3D.txt");
        points << "# unused\n";
    }
    ASSERT_TRUE(cv::imwrite((images / "frame.jpg").string(), checkerboard(96, 72)));

    xjw::camera::CameraConversionOptions options;
    options.format = xjw::camera::CameraFormat::ColmapText;
    options.inputPath = sparse;
    options.outputDir = root / "output";
    options.preUndistortColmapImages = true;
    const auto result = xjw::camera::convertCameraDataset(options);

    ASSERT_TRUE(result.success) << result.errorMessage;
    EXPECT_EQ(result.preUndistortedImageCount, 1);
    EXPECT_TRUE(std::filesystem::is_regular_file(
        options.outputDir / "images" / "frame.png"));
    const cv::Mat valid = cv::imread(
        (options.outputDir / "valid_masks" / "frame_valid.png").string(),
        cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(valid.empty());
    EXPECT_EQ(cv::countNonZero(valid), 96 * 72);
    EXPECT_TRUE(std::filesystem::is_regular_file(result.preUndistortManifestPath));
    {
        std::ifstream list(result.imageCameraList);
        const std::string list_text{
            std::istreambuf_iterator<char>(list), std::istreambuf_iterator<char>()};
        EXPECT_NE(list_text.find("images/frame.png cameras/frame.tsai"),
                  std::string::npos);
    }
    EXPECT_TRUE(result.warnings.empty());
    std::filesystem::remove_all(root);
}

} // namespace
