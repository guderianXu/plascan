#include "MvsImagePreprocessor.h"

#include "io/PathIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFileInfo>
#include <QTemporaryDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>

namespace
{

    placamera::BrownConradyDistortion brownDistortion()
    {
        return {0.35, -0.08, 0.01, 0.006, -0.004};
    }

    placamera::FramePinholeModel
    makeCamera(placamera::BrownConradyDistortion distortion = {},
               int u_axis_sign = 1,
               int v_axis_sign = 1,
               bool depth_flipped = false,
               placamera::PixelConvention pixel_convention = placamera::PixelConvention::PixelCenter,
               placamera::ImageSize image_size = {64, 48},
               std::string instance_id = "prepared-instance-7",
               std::string image_id = "prepared-image-7")
    {
        placamera::FrameIntrinsics intrinsics;
        intrinsics.focalX = 40.0;
        intrinsics.focalY = 42.0;
        intrinsics.principalX = 32.0;
        intrinsics.principalY = 24.0;
        intrinsics.uAxisSign = u_axis_sign;
        intrinsics.vAxisSign = v_axis_sign;
        const placamera::FrameId ground_frame("prepared-world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId(instance_id + "-definition"),
                                                      intrinsics,
                                                      distortion,
                                                      pixel_convention,
                                                      ground_frame,
                                                      depth_flipped);
        const auto pose =
            placamera::Pose::create(ground_frame, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        return placamera::FramePinholeModel::create(
            placamera::CameraInstanceId(instance_id), placamera::ImageId(image_id), definition, image_size, pose);
    }

    cv::Mat makeGradientImage()
    {
        cv::Mat image(48, 64, CV_8UC1);
        for (int row = 0; row < image.rows; ++row)
        {
            for (int column = 0; column < image.cols; ++column)
            {
                image.at<unsigned char>(row, column) = static_cast<unsigned char>((row * 7 + column * 3) % 251);
            }
        }
        return image;
    }

} // namespace

TEST(MvsImagePreprocessor, RejectsEmptyImageAndInvalidCamera)
{
    cv::Mat prepared;
    std::shared_ptr<const placamera::FramePinholeModel> prepared_camera;
    std::string error;

    EXPECT_FALSE(xjw::mvs::prepareMvsImage(cv::Mat(), makeCamera(), &prepared, &prepared_camera, &error));
    EXPECT_FALSE(error.empty());

    error.clear();
    EXPECT_FALSE(
        xjw::mvs::prepareMvsImage(makeGradientImage(),
                                  makeCamera({}, 1, 1, false, placamera::PixelConvention::PixelCenter, {32, 24}),
                                  &prepared,
                                  &prepared_camera,
                                  &error));
    EXPECT_FALSE(error.empty());
}

TEST(MvsImagePreprocessor, ZeroDistortionOnlyNormalizesCameraAxes)
{
    const auto source_camera = makeCamera({}, -1, 1, true);
    const cv::Mat source = makeGradientImage();

    cv::Mat prepared;
    std::shared_ptr<const placamera::FramePinholeModel> prepared_camera;
    std::string error;
    ASSERT_TRUE(xjw::mvs::prepareMvsImage(source, source_camera, &prepared, &prepared_camera, &error)) << error;

    EXPECT_EQ(cv::norm(source, prepared, cv::NORM_INF), 0.0);
    ASSERT_TRUE(prepared_camera);
    EXPECT_EQ(prepared_camera->pinholeDefinition().intrinsics().uAxisSign, 1);
    EXPECT_EQ(prepared_camera->pinholeDefinition().intrinsics().vAxisSign, 1);
    EXPECT_FALSE(prepared_camera->pinholeDefinition().depthAxisFlipped());
}

TEST(MvsImagePreprocessor, BrownDistortionRemapsPixelsAndClearsOutputDistortion)
{
    const auto source_camera = makeCamera(brownDistortion());
    const cv::Mat source = makeGradientImage();

    cv::Mat prepared;
    std::shared_ptr<const placamera::FramePinholeModel> prepared_camera;
    std::string error;
    ASSERT_TRUE(xjw::mvs::prepareMvsImage(source, source_camera, &prepared, &prepared_camera, &error)) << error;

    EXPECT_EQ(prepared.size(), source.size());
    EXPECT_EQ(prepared.type(), source.type());
    EXPECT_GT(cv::norm(source, prepared, cv::NORM_INF), 0.0);
    ASSERT_TRUE(prepared_camera);
    const auto distortion = prepared_camera->pinholeDefinition().distortion();
    EXPECT_DOUBLE_EQ(distortion.radialK1, 0.0);
    EXPECT_DOUBLE_EQ(distortion.radialK2, 0.0);
    EXPECT_DOUBLE_EQ(distortion.radialK3, 0.0);
    EXPECT_DOUBLE_EQ(distortion.tangentialP1, 0.0);
    EXPECT_DOUBLE_EQ(distortion.tangentialP2, 0.0);
}

TEST(MvsImagePreprocessor, BrownDistortionUsesOneMapForImageAndValidMask)
{
    const auto source_camera = makeCamera(brownDistortion());
    cv::Mat source_valid_mask(48, 64, CV_8UC1, cv::Scalar(255));
    cv::rectangle(source_valid_mask, cv::Rect(22, 17, 16, 14), cv::Scalar(0), cv::FILLED);
    const cv::Mat source = source_valid_mask.clone();

    cv::Mat prepared;
    cv::Mat prepared_valid_mask;
    std::shared_ptr<const placamera::FramePinholeModel> prepared_camera;
    std::string error;
    ASSERT_TRUE(xjw::mvs::prepareMvsImageAndMask(
        source, source_valid_mask, source_camera, &prepared, &prepared_valid_mask, &prepared_camera, &error))
        << error;

    const auto intrinsics = source_camera.pinholeDefinition().intrinsics();
    const auto distortion = source_camera.pinholeDefinition().distortion();
    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << intrinsics.focalX,
                                   0.0,
                                   intrinsics.principalX,
                                   0.0,
                                   intrinsics.focalY,
                                   intrinsics.principalY,
                                   0.0,
                                   0.0,
                                   1.0);
    const cv::Mat distortion_coefficients = (cv::Mat_<double>(1, 5) << distortion.radialK1,
                                             distortion.radialK2,
                                             distortion.tangentialP1,
                                             distortion.tangentialP2,
                                             distortion.radialK3);
    cv::Mat map_x;
    cv::Mat map_y;
    cv::initUndistortRectifyMap(
        camera_matrix, distortion_coefficients, cv::Mat(), camera_matrix, source.size(), CV_32FC1, map_x, map_y);
    cv::Mat expected_image;
    cv::Mat expected_mask;
    cv::remap(source, expected_image, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
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

TEST(MvsImagePreprocessor,
     PersistsLosslessColorRasterAndPreparedMaskForBrownCamera)
{
    QTemporaryDir temporary_directory;
    ASSERT_TRUE(temporary_directory.isValid());

    const auto source_camera = makeCamera(brownDistortion());
    cv::Mat source_color;
    cv::cvtColor(makeGradientImage(), source_color, cv::COLOR_GRAY2BGR);
    source_color.at<cv::Vec3b>(18, 27) = cv::Vec3b(17, 103, 241);
    const QString source_path = QDir(temporary_directory.path()).filePath(QStringLiteral("distorted source.png"));
    ASSERT_TRUE(xjw::common::io::writeImage(source_path, source_color));

    xjw::mvs::MvsPreparedRasterArtifact mismatched_artifact;
    std::string mismatched_error;
    EXPECT_FALSE(xjw::mvs::saveMvsPreparedRasterArtifact(
        xjw::common::io::toUtf8Path(source_path),
        makeCamera({}, 1, 1, false, placamera::PixelConvention::PixelCenter, {32, 24}),
        cv::Mat(),
        xjw::common::io::toUtf8Path(temporary_directory.path()),
        6,
        &mismatched_artifact,
        &mismatched_error));
    EXPECT_NE(mismatched_error.find("尺寸"), std::string::npos);
    EXPECT_FALSE(mismatched_artifact.camera);

    const auto corner_definition =
        placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("prepared-corner-definition"),
                                                  source_camera.pinholeDefinition().intrinsics(),
                                                  source_camera.pinholeDefinition().distortion(),
                                                  placamera::PixelConvention::PixelCorner,
                                                  source_camera.groundFrame());
    const auto corner_model = placamera::FramePinholeModel::create(source_camera.instanceId(),
                                                                   source_camera.imageId(),
                                                                   corner_definition,
                                                                   source_camera.imageSize(),
                                                                   source_camera.pose());
    xjw::mvs::MvsPreparedRasterArtifact corner_artifact;
    std::string corner_error;
    EXPECT_FALSE(xjw::mvs::saveMvsPreparedRasterArtifact(xjw::common::io::toUtf8Path(source_path),
                                                         corner_model,
                                                         cv::Mat(),
                                                         xjw::common::io::toUtf8Path(temporary_directory.path()),
                                                         6,
                                                         &corner_artifact,
                                                         &corner_error));
    EXPECT_NE(corner_error.find("像素中心"), std::string::npos);
    EXPECT_FALSE(corner_artifact.camera);

    cv::Mat expected_color;
    std::shared_ptr<const placamera::FramePinholeModel> expected_camera;
    std::string error;
    ASSERT_TRUE(xjw::mvs::prepareMvsImage(source_color, source_camera, &expected_color, &expected_camera, &error))
        << error;
    cv::Mat source_valid_mask(source_color.size(), CV_8UC1, cv::Scalar(255));
    cv::rectangle(source_valid_mask, cv::Rect(20, 14, 18, 16), cv::Scalar(0), cv::FILLED);
    cv::Mat unused_gray;
    cv::Mat prepared_valid_mask;
    ASSERT_TRUE(xjw::mvs::prepareMvsImageAndMask(makeGradientImage(),
                                                 source_valid_mask,
                                                 source_camera,
                                                 &unused_gray,
                                                 &prepared_valid_mask,
                                                 &expected_camera,
                                                 &error))
        << error;

    xjw::mvs::MvsPreparedRasterArtifact artifact;
    ASSERT_TRUE(xjw::mvs::saveMvsPreparedRasterArtifact(xjw::common::io::toUtf8Path(source_path),
                                                        source_camera,
                                                        prepared_valid_mask,
                                                        xjw::common::io::toUtf8Path(temporary_directory.path()),
                                                        7,
                                                        &artifact,
                                                        &error))
        << error;

    EXPECT_TRUE(QFileInfo::exists(xjw::common::io::fromUtf8Path(artifact.imagePath)));
    EXPECT_TRUE(QFileInfo::exists(xjw::common::io::fromUtf8Path(artifact.validMaskPath)));
    const cv::Mat stored_color = xjw::common::io::readImage(artifact.imagePath, cv::IMREAD_COLOR);
    const cv::Mat stored_mask = xjw::common::io::readImage(artifact.validMaskPath, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(stored_color.empty());
    ASSERT_FALSE(stored_mask.empty());
    EXPECT_EQ(cv::norm(stored_color, expected_color, cv::NORM_INF), 0.0);
    EXPECT_EQ(cv::norm(stored_mask, prepared_valid_mask, cv::NORM_INF), 0.0);
    EXPECT_GT(cv::norm(stored_color, source_color, cv::NORM_INF), 0.0);
    ASSERT_TRUE(artifact.camera);
    EXPECT_DOUBLE_EQ(artifact.camera->pinholeDefinition().distortion().radialK1, 0.0);
    EXPECT_DOUBLE_EQ(artifact.camera->pinholeDefinition().distortion().tangentialP2, 0.0);
    EXPECT_EQ(artifact.camera->imageSize().samples, source_color.cols);
    EXPECT_EQ(artifact.camera->imageSize().lines, source_color.rows);
    EXPECT_EQ(artifact.camera->instanceId().value(), "prepared-instance-7");
    EXPECT_EQ(artifact.camera->imageId().value(), "prepared-image-7");
    EXPECT_EQ(artifact.camera->groundFrame().value(), "prepared-world");
}
