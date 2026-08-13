#include "MvsImagePreprocessor.h"

#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>

namespace
{

xjw::FramePinholeCamera makeCamera()
{
    xjw::FramePinholeCamera camera;
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
    xjw::FramePinholeCamera prepared_camera;
    std::string error;

    EXPECT_FALSE(xjw::mvs::prepareMvsImage(
        cv::Mat(), makeCamera(), &prepared, &prepared_camera, &error));
    EXPECT_FALSE(error.empty());

    error.clear();
    EXPECT_FALSE(xjw::mvs::prepareMvsImage(
        makeGradientImage(), xjw::FramePinholeCamera(), &prepared, &prepared_camera, &error));
    EXPECT_FALSE(error.empty());
}

TEST(MvsImagePreprocessor, ZeroDistortionOnlyNormalizesCameraAxes)
{
    xjw::FramePinholeCamera source_camera = makeCamera();
    source_camera.setAxisDirections(-1, 1);
    source_camera.setDepthAxisFlipped(true);
    const cv::Mat source = makeGradientImage();

    cv::Mat prepared;
    xjw::FramePinholeCamera prepared_camera;
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
    xjw::FramePinholeCamera source_camera = makeCamera();
    source_camera.setDistortion(0.35, -0.08, 0.01, 0.006, -0.004);
    const cv::Mat source = makeGradientImage();

    cv::Mat prepared;
    xjw::FramePinholeCamera prepared_camera;
    std::string error;
    ASSERT_TRUE(xjw::mvs::prepareMvsImage(
        source, source_camera, &prepared, &prepared_camera, &error)) << error;

    EXPECT_EQ(prepared.size(), source.size());
    EXPECT_EQ(prepared.type(), source.type());
    EXPECT_GT(cv::norm(source, prepared, cv::NORM_INF), 0.0);
    const xjw::FramePinholeCamera::Distortion distortion = prepared_camera.distortion();
    EXPECT_DOUBLE_EQ(distortion.radialK1, 0.0);
    EXPECT_DOUBLE_EQ(distortion.radialK2, 0.0);
    EXPECT_DOUBLE_EQ(distortion.radialK3, 0.0);
    EXPECT_DOUBLE_EQ(distortion.tangentialP1, 0.0);
    EXPECT_DOUBLE_EQ(distortion.tangentialP2, 0.0);
}

TEST(MvsImagePreprocessor, BrownDistortionUsesOneMapForImageAndValidMask)
{
    xjw::FramePinholeCamera source_camera = makeCamera();
    source_camera.setDistortion(0.35, -0.08, 0.01, 0.006, -0.004);
    cv::Mat source_valid_mask(48, 64, CV_8UC1, cv::Scalar(255));
    cv::rectangle(source_valid_mask, cv::Rect(22, 17, 16, 14), cv::Scalar(0), cv::FILLED);
    const cv::Mat source = source_valid_mask.clone();

    cv::Mat prepared;
    cv::Mat prepared_valid_mask;
    xjw::FramePinholeCamera prepared_camera;
    std::string error;
    ASSERT_TRUE(xjw::mvs::prepareMvsImageAndMask(source,
                                                source_valid_mask,
                                                source_camera,
                                                &prepared,
                                                &prepared_valid_mask,
                                                &prepared_camera,
                                                &error)) << error;

    const auto intrinsics = source_camera.intrinsics();
    const auto distortion = source_camera.distortion();
    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3)
        << intrinsics.focalX, 0.0, intrinsics.principalX,
           0.0, intrinsics.focalY, intrinsics.principalY,
           0.0, 0.0, 1.0);
    const cv::Mat distortion_coefficients = (cv::Mat_<double>(1, 5)
        << distortion.radialK1,
           distortion.radialK2,
           distortion.tangentialP1,
           distortion.tangentialP2,
           distortion.radialK3);
    cv::Mat map_x;
    cv::Mat map_y;
    cv::initUndistortRectifyMap(camera_matrix,
                                distortion_coefficients,
                                cv::Mat(),
                                camera_matrix,
                                source.size(),
                                CV_32FC1,
                                map_x,
                                map_y);
    cv::Mat expected_image;
    cv::Mat expected_mask;
    cv::remap(source,
              expected_image,
              map_x,
              map_y,
              cv::INTER_LINEAR,
              cv::BORDER_CONSTANT,
              cv::Scalar(0));
    cv::remap(source_valid_mask,
              expected_mask,
              map_x,
              map_y,
              cv::INTER_NEAREST,
              cv::BORDER_CONSTANT,
              cv::Scalar(0));
    for (int row = 0; row < map_x.rows; ++row)
    {
        const float *map_x_row = map_x.ptr<float>(row);
        const float *map_y_row = map_y.ptr<float>(row);
        std::uint8_t *mask_row = expected_mask.ptr<std::uint8_t>(row);
        for (int column = 0; column < map_x.cols; ++column)
        {
            if (map_x_row[column] < 0.0f ||
                map_x_row[column] > static_cast<float>(source.cols - 1) ||
                map_y_row[column] < 0.0f ||
                map_y_row[column] > static_cast<float>(source.rows - 1))
            {
                mask_row[column] = 0;
            }
        }
    }

    ASSERT_EQ(prepared.size(), source.size());
    ASSERT_EQ(prepared_valid_mask.size(), source_valid_mask.size());
    EXPECT_EQ(cv::norm(prepared, expected_image, cv::NORM_INF), 0.0);
    EXPECT_EQ(cv::norm(prepared_valid_mask, expected_mask, cv::NORM_INF), 0.0);
    EXPECT_GT(cv::norm(prepared_valid_mask, source_valid_mask, cv::NORM_INF), 0.0);
    EXPECT_EQ(prepared_valid_mask.at<std::uint8_t>(0, 0), 0);
    EXPECT_EQ(prepared_valid_mask.at<std::uint8_t>(0, prepared_valid_mask.cols - 1), 0);
    EXPECT_EQ(prepared_valid_mask.at<std::uint8_t>(prepared_valid_mask.rows - 1, 0), 0);
    EXPECT_EQ(prepared_valid_mask.at<std::uint8_t>(
                  prepared_valid_mask.rows - 1, prepared_valid_mask.cols - 1),
              0);
}
