#include "EpipolarRectifier.h"
#include "DisparityTriangulator.h"
#include "Camera.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace xjw;
using namespace xjw::mvs;

namespace
{

void applyHomography(const cv::Mat &H, double u, double v, double &ox, double &oy)
{
    const double *h = H.ptr<double>(0);
    const double w = h[6] * u + h[7] * v + h[8];
    ox = (h[0] * u + h[1] * v + h[2]) / w;
    oy = (h[3] * u + h[4] * v + h[5]) / w;
}

Camera makeCamera(double cx, double cy, double tx)
{
    Camera cam;
    cam.setIntrinsics(2000.0, 2000.0, cx, cy);
    std::array<double, 9> R = {1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0};
    std::array<double, 3> C = {tx, 0.0, 0.0};
    cam.setPose(R, C);
    return cam;
}

Camera makeYawedCamera(double cx, double cy, double tx, double yawDegrees)
{
    Camera cam;
    cam.setIntrinsics(2000.0, 2000.0, cx, cy);
    const double yaw = yawDegrees * M_PI / 180.0;
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    std::array<double, 9> R = {c, 0.0, s,
                               0.0, 1.0, 0.0,
                               -s, 0.0, c};
    std::array<double, 3> C = {tx, 0.0, 0.0};
    cam.setPose(R, C);
    return cam;
}

Camera makeVerticalBaselineCamera(double cx, double cy, double ty)
{
    Camera cam;
    cam.setIntrinsics(2000.0, 2000.0, cx, cy);
    std::array<double, 9> R = {1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0};
    std::array<double, 3> C = {0.0, ty, 0.0};
    cam.setPose(R, C);
    return cam;
}


} // namespace

TEST(EpipolarRectifier, RectificationMakesEpipolarRowsMatchAcrossDepths)
{
    Camera leftCamera = makeCamera(32.0, 24.0, 0.0);
    Camera rightCamera = makeCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           leftCamera.toPositiveDepthModel(),
                                           rightCamera.toPositiveDepthModel(),
                                           rect,
                                           &error)) << error;

    const double leftWorld[3] = {0.1, 0.0, 5.0};
    const double rightWorld[3] = {0.1, 0.0, 8.0};
    const double *worlds[2] = {leftWorld, rightWorld};

    for (const double *world : worlds)
    {
        double leftPixel[2] = {0.0, 0.0};
        double rightPixel[2] = {0.0, 0.0};
        ASSERT_TRUE(leftCamera.projectWorldPoint(world, leftPixel));
        ASSERT_TRUE(rightCamera.projectWorldPoint(world, rightPixel));

        double leftRectX = 0.0;
        double leftRectY = 0.0;
        double rightRectX = 0.0;
        double rightRectY = 0.0;
        applyHomography(rect.H1, leftPixel[0], leftPixel[1], leftRectX, leftRectY);
        applyHomography(rect.H2, rightPixel[0], rightPixel[1], rightRectX, rightRectY);

        EXPECT_NEAR(leftRectY, rightRectY, 1.0);
    }
}

TEST(EpipolarRectifier, ParallelStereoKeepsPositiveRectifiedDisparity)
{
    Camera leftCamera = makeCamera(32.0, 24.0, 0.0);
    Camera rightCamera = makeCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           leftCamera.toPositiveDepthModel(),
                                           rightCamera.toPositiveDepthModel(),
                                           rect,
                                           &error)) << error;

    const double world[3] = {0.1, 0.0, 5.0};
    double leftPixel[2] = {0.0, 0.0};
    double rightPixel[2] = {0.0, 0.0};
    ASSERT_TRUE(leftCamera.projectWorldPoint(world, leftPixel));
    ASSERT_TRUE(rightCamera.projectWorldPoint(world, rightPixel));

    double leftRectX = 0.0;
    double leftRectY = 0.0;
    double rightRectX = 0.0;
    double rightRectY = 0.0;
    applyHomography(rect.H1, leftPixel[0], leftPixel[1], leftRectX, leftRectY);
    applyHomography(rect.H2, rightPixel[0], rightPixel[1], rightRectX, rightRectY);

    EXPECT_GT(leftRectX - rightRectX, 0.0);
}


TEST(EpipolarRectifier, YawedStereoKeepsEpipolarRowsAlignedAcrossPoints)
{
    Camera leftCamera = makeCamera(512.0, 384.0, 0.0);
    Camera rightCamera = makeYawedCamera(512.0, 384.0, 0.2, 15.0);

    cv::Mat leftImage(768, 1024, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(768, 1024, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           leftCamera.toPositiveDepthModel(),
                                           rightCamera.toPositiveDepthModel(),
                                           rect,
                                           &error)) << error;

    const double xs[] = {-0.4, -0.2, 0.0, 0.2, 0.4};
    const double ys[] = {-0.25, 0.0, 0.25};
    const double zs[] = {4.0, 6.0, 9.0};

    for (double x : xs)
    {
        for (double y : ys)
        {
            for (double z : zs)
            {
                const double world[3] = {x, y, z};
                double leftPixel[2] = {0.0, 0.0};
                double rightPixel[2] = {0.0, 0.0};
                ASSERT_TRUE(leftCamera.projectWorldPoint(world, leftPixel));
                ASSERT_TRUE(rightCamera.projectWorldPoint(world, rightPixel));

                double leftRectX = 0.0;
                double leftRectY = 0.0;
                double rightRectX = 0.0;
                double rightRectY = 0.0;
                applyHomography(rect.H1, leftPixel[0], leftPixel[1], leftRectX, leftRectY);
                applyHomography(rect.H2, rightPixel[0], rightPixel[1], rightRectX, rightRectY);

                EXPECT_NEAR(leftRectY, rightRectY, 0.5) << "world=" << x << "," << y << "," << z;
            }
        }
    }
}


TEST(EpipolarRectifier, VerticalBaselineSetsTransposedForHorizontalDisparity)
{
    Camera leftCamera = makeCamera(32.0, 24.0, 0.0);
    Camera rightCamera = makeVerticalBaselineCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           leftCamera.toPositiveDepthModel(),
                                           rightCamera.toPositiveDepthModel(),
                                           rect,
                                           &error)) << error;

    EXPECT_TRUE(rect.transposed);
}


TEST(EpipolarRectifier, TransposedRectifiedCamerasProjectIntoTransposedPixels)
{
    Camera leftCamera = makeCamera(32.0, 24.0, 0.0);
    Camera rightCamera = makeVerticalBaselineCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           leftCamera.toPositiveDepthModel(),
                                           rightCamera.toPositiveDepthModel(),
                                           rect,
                                           &error)) << error;
    ASSERT_TRUE(rect.transposed);

    const double world[3] = {0.1, 0.05, 5.0};
    double leftPixel[2] = {0.0, 0.0};
    double rightPixel[2] = {0.0, 0.0};
    ASSERT_TRUE(leftCamera.projectWorldPoint(world, leftPixel));
    ASSERT_TRUE(rightCamera.projectWorldPoint(world, rightPixel));

    double leftRectX = 0.0;
    double leftRectY = 0.0;
    double rightRectX = 0.0;
    double rightRectY = 0.0;
    applyHomography(rect.H1, leftPixel[0], leftPixel[1], leftRectX, leftRectY);
    applyHomography(rect.H2, rightPixel[0], rightPixel[1], rightRectX, rightRectY);

    float leftProjX = 0.0f;
    float leftProjY = 0.0f;
    float rightProjX = 0.0f;
    float rightProjY = 0.0f;
    ASSERT_TRUE(rect.rectCamLeft.project(static_cast<float>(world[0]),
                                         static_cast<float>(world[1]),
                                         static_cast<float>(world[2]),
                                         leftProjX,
                                         leftProjY));
    ASSERT_TRUE(rect.rectCamRight.project(static_cast<float>(world[0]),
                                          static_cast<float>(world[1]),
                                          static_cast<float>(world[2]),
                                          rightProjX,
                                          rightProjY));

    EXPECT_NEAR(leftProjX, static_cast<float>(leftRectX), 1.0f);
    EXPECT_NEAR(leftProjY, static_cast<float>(leftRectY), 1.0f);
    EXPECT_NEAR(rightProjX, static_cast<float>(rightRectX), 1.0f);
    EXPECT_NEAR(rightProjY, static_cast<float>(rightRectY), 1.0f);
}


TEST(EpipolarRectifier, TransposedRectifiedCameraRawFieldsMatchProjection)
{
    Camera leftCamera = makeCamera(32.0, 24.0, 0.0);
    Camera rightCamera = makeVerticalBaselineCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           leftCamera.toPositiveDepthModel(),
                                           rightCamera.toPositiveDepthModel(),
                                           rect,
                                           &error)) << error;
    ASSERT_TRUE(rect.transposed);

    const double world[3] = {0.1, 0.05, 5.0};

    float projX = 0.0f;
    float projY = 0.0f;
    ASSERT_TRUE(rect.rectCamLeft.project(static_cast<float>(world[0]),
                                         static_cast<float>(world[1]),
                                         static_cast<float>(world[2]),
                                         projX,
                                         projY));

    const float cameraX = rect.rectCamLeft.R_cw[0] * static_cast<float>(world[0])
                        + rect.rectCamLeft.R_cw[1] * static_cast<float>(world[1])
                        + rect.rectCamLeft.R_cw[2] * static_cast<float>(world[2])
                        + rect.rectCamLeft.T[0];
    const float cameraY = rect.rectCamLeft.R_cw[3] * static_cast<float>(world[0])
                        + rect.rectCamLeft.R_cw[4] * static_cast<float>(world[1])
                        + rect.rectCamLeft.R_cw[5] * static_cast<float>(world[2])
                        + rect.rectCamLeft.T[1];
    const float cameraZ = rect.rectCamLeft.R_cw[6] * static_cast<float>(world[0])
                        + rect.rectCamLeft.R_cw[7] * static_cast<float>(world[1])
                        + rect.rectCamLeft.R_cw[8] * static_cast<float>(world[2])
                        + rect.rectCamLeft.T[2];
    ASSERT_GT(cameraZ, 0.0f);

    const float rawX = rect.rectCamLeft.fx * cameraX / cameraZ + rect.rectCamLeft.cx;
    const float rawY = rect.rectCamLeft.fy * cameraY / cameraZ + rect.rectCamLeft.cy;

    EXPECT_NEAR(rawX, projX, 1.0f);
    EXPECT_NEAR(rawY, projY, 1.0f);
}



TEST(DisparityTriangulator, TransposedRectifiedDepthTriangulationKeepsLowReprojectionError)
{
    Camera leftCamera = makeCamera(32.0, 24.0, 0.0);
    Camera rightCamera = makeVerticalBaselineCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           leftCamera.toPositiveDepthModel(),
                                           rightCamera.toPositiveDepthModel(),
                                           rect,
                                           &error)) << error;
    ASSERT_TRUE(rect.transposed);

    const double candidateWorlds[][3] = {
        {0.0, 0.0, 5.0},
        {0.02, 0.0, 5.0},
        {0.0, 0.02, 5.0},
        {0.02, 0.02, 5.0},
        {-0.02, 0.0, 5.0},
        {0.0, -0.02, 5.0}
    };

    float rectX = 0.0f;
    float rectY = 0.0f;
    float cameraZ = 0.0f;
    bool foundWorld = false;
    for (const auto &world : candidateWorlds)
    {
        if (!rect.rectCamLeft.project(static_cast<float>(world[0]),
                                      static_cast<float>(world[1]),
                                      static_cast<float>(world[2]),
                                      rectX,
                                      rectY))
        {
            continue;
        }

        const int px = static_cast<int>(std::round(rectX));
        const int py = static_cast<int>(std::round(rectY));
        if (px < 0 || px >= rect.rectLeft.cols || py < 0 || py >= rect.rectLeft.rows)
        {
            continue;
        }

        cameraZ = rect.rectCamLeft.R_cw[6] * static_cast<float>(world[0])
                + rect.rectCamLeft.R_cw[7] * static_cast<float>(world[1])
                + rect.rectCamLeft.R_cw[8] * static_cast<float>(world[2])
                + rect.rectCamLeft.T[2];
        ASSERT_GT(cameraZ, 0.0f);
        foundWorld = true;
        break;
    }
    ASSERT_TRUE(foundWorld);

    const int px = static_cast<int>(std::round(rectX));
    const int py = static_cast<int>(std::round(rectY));

    cv::Mat depthMap(rect.rectLeft.rows, rect.rectLeft.cols, CV_32F, cv::Scalar(0.0f));
    cv::Mat validMask(rect.rectLeft.rows, rect.rectLeft.cols, CV_8U, cv::Scalar(0));
    depthMap.at<float>(py, px) = cameraZ;
    validMask.at<uint8_t>(py, px) = 255;

    TriangulationConfig cfg;
    cfg.maxTriangulationError = 1.0f;
    cfg.numThreads = 1;
    cfg.transposed = true;

    TriangulationResult result = DisparityTriangulator::triangulateFromDepth(
        depthMap,
        validMask,
        rect.H1inv,
        leftCamera,
        rightCamera,
        rect.rectCamLeft,
        cfg);

    ASSERT_EQ(result.validPoints, 1);
    EXPECT_LT(result.medianError, 2.0f);
}
