#include <array>

#include <gtest/gtest.h>

#include "geometry/OpenCvCameraAdapter.h"

namespace
{
    placamera::FramePinholeModel
    makeCamera(placamera::FrameIntrinsics intrinsics, bool depth_axis_flipped, std::array<double, 3> center)
    {
        const placamera::FrameId frame("opencv-adapter-world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("opencv-adapter-definition"),
                                                      intrinsics,
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame,
                                                      depth_axis_flipped);
        return placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("opencv-adapter-instance"),
            placamera::ImageId("opencv-adapter-image"),
            definition,
            placamera::ImageSize{100, 80},
            placamera::Pose::create(frame, center, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    }
} // namespace

TEST(OpenCvCameraAdapterTest, BuildsSignedAndPositiveDepthIntrinsics)
{
    const auto camera = makeCamera({100.0, 200.0, 10.0, 20.0, 1.0, -1, 1}, true, {0.0, 0.0, 0.0});

    const cv::Mat signedMatrix = xjw::openCvCameraMatrix(camera.pinholeDefinition(), false);
    const cv::Mat positiveDepthMatrix = xjw::openCvCameraMatrix(camera.pinholeDefinition(), true);

    EXPECT_DOUBLE_EQ(signedMatrix.at<double>(0, 0), -100.0);
    EXPECT_DOUBLE_EQ(signedMatrix.at<double>(1, 1), 200.0);
    EXPECT_DOUBLE_EQ(positiveDepthMatrix.at<double>(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(positiveDepthMatrix.at<double>(1, 1), -200.0);
    EXPECT_DOUBLE_EQ(positiveDepthMatrix.at<double>(0, 2), 10.0);
    EXPECT_DOUBLE_EQ(positiveDepthMatrix.at<double>(1, 2), 20.0);
}

TEST(OpenCvCameraAdapterTest, BuildsPhysicalSignedProjectionMatrix)
{
    const auto camera = makeCamera({100.0, 200.0, 10.0, 20.0}, false, {1.0, 2.0, 3.0});

    const cv::Mat projection = xjw::openCvProjectionMatrix(camera);

    ASSERT_EQ(projection.rows, 3);
    ASSERT_EQ(projection.cols, 4);
    EXPECT_DOUBLE_EQ(projection.at<double>(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(projection.at<double>(0, 3), -130.0);
    EXPECT_DOUBLE_EQ(projection.at<double>(1, 3), -460.0);
    EXPECT_DOUBLE_EQ(projection.at<double>(2, 3), -3.0);
}
