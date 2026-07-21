#include "geometry/OpenCvCameraAdapter.h"

#include <gtest/gtest.h>

TEST(OpenCvCameraAdapterTest, BuildsSignedAndPositiveDepthIntrinsics)
{
    xjw::Camera camera;
    camera.setIntrinsics(100.0, 200.0, 10.0, 20.0);
    camera.setAxisDirections(-1, 1);
    camera.setDepthAxisFlipped(true);

    const cv::Mat signedMatrix = xjw::openCvCameraMatrix(camera, false);
    const cv::Mat positiveDepthMatrix = xjw::openCvCameraMatrix(camera, true);

    EXPECT_DOUBLE_EQ(signedMatrix.at<double>(0, 0), -100.0);
    EXPECT_DOUBLE_EQ(signedMatrix.at<double>(1, 1), 200.0);
    EXPECT_DOUBLE_EQ(positiveDepthMatrix.at<double>(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(positiveDepthMatrix.at<double>(1, 1), -200.0);
    EXPECT_DOUBLE_EQ(positiveDepthMatrix.at<double>(0, 2), 10.0);
    EXPECT_DOUBLE_EQ(positiveDepthMatrix.at<double>(1, 2), 20.0);
}

TEST(OpenCvCameraAdapterTest, BuildsPhysicalSignedProjectionMatrix)
{
    xjw::Camera camera;
    camera.setIntrinsics(100.0, 200.0, 10.0, 20.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {1.0, 2.0, 3.0});

    const cv::Mat projection = xjw::openCvProjectionMatrix(camera);

    ASSERT_EQ(projection.rows, 3);
    ASSERT_EQ(projection.cols, 4);
    EXPECT_DOUBLE_EQ(projection.at<double>(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(projection.at<double>(0, 3), -130.0);
    EXPECT_DOUBLE_EQ(projection.at<double>(1, 3), -460.0);
    EXPECT_DOUBLE_EQ(projection.at<double>(2, 3), -3.0);
}
