#include "MvsImagePreprocessor.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace
{

xjw::Camera makeCamera()
{
    xjw::Camera camera;
    camera.setIntrinsics(40.0, 42.0, 32.0, 24.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {0.0, 0.0, 0.0});
    return camera;
}

cv::Mat makeGradientImage()
{
    cv::Mat image(48, 64, CV_8UC1);
    for (int row = 0; row < image.rows; ++row)
    {
        for (int column = 0; column < image.cols; ++column)
        {
            image.at<unsigned char>(row, column) =
                static_cast<unsigned char>((row * 7 + column * 3) % 251);
        }
    }
    return image;
}

} // namespace

TEST(MvsImagePreprocessor, RejectsEmptyImageAndInvalidCamera)
{
    cv::Mat prepared;
    xjw::Camera prepared_camera;
    std::string error;

    EXPECT_FALSE(xjw::mvs::prepareMvsImage(
        cv::Mat(), makeCamera(), &prepared, &prepared_camera, &error));
    EXPECT_FALSE(error.empty());

    error.clear();
    EXPECT_FALSE(xjw::mvs::prepareMvsImage(
        makeGradientImage(), xjw::Camera(), &prepared, &prepared_camera, &error));
    EXPECT_FALSE(error.empty());
}

TEST(MvsImagePreprocessor, ZeroDistortionOnlyNormalizesCameraAxes)
{
    xjw::Camera source_camera = makeCamera();
    source_camera.setAxisDirections(-1, 1);
    source_camera.setDepthAxisFlipped(true);
    const cv::Mat source = makeGradientImage();

    cv::Mat prepared;
    xjw::Camera prepared_camera;
    std::string error;
    ASSERT_TRUE(xjw::mvs::prepareMvsImage(
        source, source_camera, &prepared, &prepared_camera, &error)) << error;

    EXPECT_EQ(cv::norm(source, prepared, cv::NORM_INF), 0.0);
    EXPECT_EQ(prepared_camera.uAxisSign(), 1);
    EXPECT_EQ(prepared_camera.vAxisSign(), 1);
    EXPECT_FALSE(prepared_camera.depthAxisFlipped());
}

TEST(MvsImagePreprocessor, BrownDistortionRemapsPixelsAndClearsOutputDistortion)
{
    xjw::Camera source_camera = makeCamera();
    source_camera.setDistortion(0.35, -0.08, 0.01, 0.006, -0.004);
    const cv::Mat source = makeGradientImage();

    cv::Mat prepared;
    xjw::Camera prepared_camera;
    std::string error;
    ASSERT_TRUE(xjw::mvs::prepareMvsImage(
        source, source_camera, &prepared, &prepared_camera, &error)) << error;

    EXPECT_EQ(prepared.size(), source.size());
    EXPECT_EQ(prepared.type(), source.type());
    EXPECT_GT(cv::norm(source, prepared, cv::NORM_INF), 0.0);
    const xjw::Camera::Distortion distortion = prepared_camera.distortion();
    EXPECT_DOUBLE_EQ(distortion.radialK1, 0.0);
    EXPECT_DOUBLE_EQ(distortion.radialK2, 0.0);
    EXPECT_DOUBLE_EQ(distortion.radialK3, 0.0);
    EXPECT_DOUBLE_EQ(distortion.tangentialP1, 0.0);
    EXPECT_DOUBLE_EQ(distortion.tangentialP2, 0.0);
}
