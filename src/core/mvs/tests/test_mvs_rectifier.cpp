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

Camera makeDinoRingCamera(const std::array<double, 9> &rotation_camera_to_world,
                          const std::array<double, 3> &center)
{
    Camera camera;
    camera.setIntrinsics(3310.4, 3325.5, 316.73, 200.55);
    camera.setPose(rotation_camera_to_world, center);
    return camera;
}


} // namespace

TEST(EpipolarRectifier, RejectsImagesThatStillCarryLensDistortion)
{
    Camera left_camera = makeCamera(32.0, 24.0, 0.0);
    Camera right_camera = makeCamera(32.0, 24.0, 0.2);
    left_camera.setDistortion(0.1, 0.0, 0.0, 0.0, 0.0);

    cv::Mat left_image(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat right_image(48, 64, CV_8U, cv::Scalar(96));
    EpipolarRectifier::RectifiedPair rectified;
    std::string error;
    EXPECT_FALSE(EpipolarRectifier::rectify(left_image,
                                            right_image,
                                            left_camera.normalizedForPositiveDepth(),
                                            right_camera.normalizedForPositiveDepth(),
                                            rectified,
                                            &error));
    EXPECT_NE(error.find("去畸变"), std::string::npos);
}

TEST(EpipolarRectifier, RejectsConvergentPairWithoutUsableRectifiedCanvas)
{
    const Camera left_camera = makeDinoRingCamera(
        {-0.143964578361, -0.903665806035, -0.403315364598,
         0.969652632813, -0.0474333525503, -0.239841305752,
         0.197606171538, -0.425604192333, 0.883069362015},
        {0.243378250328, 0.170140221186, -0.604858522898});
    const Camera right_camera = makeDinoRingCamera(
        {-0.231436872629, -0.658608111657, -0.716012081872,
         0.96422332027, -0.0574942602048, -0.258781378564,
         0.129269371656, -0.750286404865, 0.648350496196},
        {0.447988592591, 0.182631346554, -0.449273743884});

    cv::Mat left_image(480, 640, CV_8U, cv::Scalar(64));
    cv::Mat right_image(480, 640, CV_8U, cv::Scalar(96));
    EpipolarRectifier::RectifiedPair rectified;
    std::string error;

    EXPECT_FALSE(EpipolarRectifier::rectify(left_image,
                                            right_image,
                                            left_camera.normalizedForPositiveDepth(),
                                            right_camera.normalizedForPositiveDepth(),
                                            rectified,
                                            &error));
    EXPECT_NE(error.find("有效区域"), std::string::npos);
}

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
                                           leftCamera.normalizedForPositiveDepth(),
                                           rightCamera.normalizedForPositiveDepth(),
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
                                           leftCamera.normalizedForPositiveDepth(),
                                           rightCamera.normalizedForPositiveDepth(),
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
                                           leftCamera.normalizedForPositiveDepth(),
                                           rightCamera.normalizedForPositiveDepth(),
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
                                           leftCamera.normalizedForPositiveDepth(),
                                           rightCamera.normalizedForPositiveDepth(),
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
                                           leftCamera.normalizedForPositiveDepth(),
                                           rightCamera.normalizedForPositiveDepth(),
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

    double leftProjected[2] = {};
    double rightProjected[2] = {};
    ASSERT_TRUE(rect.rectCamLeft.projectWorldPoint(world, leftProjected));
    ASSERT_TRUE(rect.rectCamRight.projectWorldPoint(world, rightProjected));

    EXPECT_NEAR(leftProjected[0], leftRectX, 1.0);
    EXPECT_NEAR(leftProjected[1], leftRectY, 1.0);
    EXPECT_NEAR(rightProjected[0], rightRectX, 1.0);
    EXPECT_NEAR(rightProjected[1], rightRectY, 1.0);
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
                                           leftCamera.normalizedForPositiveDepth(),
                                           rightCamera.normalizedForPositiveDepth(),
                                           rect,
                                           &error)) << error;
    ASSERT_TRUE(rect.transposed);

    const double world[3] = {0.1, 0.05, 5.0};

    double projected[2] = {0.0, 0.0};
    double cameraZ = 0.0;
    ASSERT_TRUE(rect.rectCamLeft.projectWorldPointWithDepth(world, projected, cameraZ));

    double camera_point[3] = {0.0, 0.0, 0.0};
    rect.rectCamLeft.worldToCamera(world, camera_point);
    ASSERT_GT(cameraZ, 0.0);
    const Camera::Intrinsics intrinsics = rect.rectCamLeft.intrinsics();
    const double rawX = intrinsics.focalX * camera_point[0] / cameraZ + intrinsics.principalX;
    const double rawY = intrinsics.focalY * camera_point[1] / cameraZ + intrinsics.principalY;

    EXPECT_NEAR(rawX, projected[0], 1.0);
    EXPECT_NEAR(rawY, projected[1], 1.0);
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
                                           leftCamera.normalizedForPositiveDepth(),
                                           rightCamera.normalizedForPositiveDepth(),
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

    double rectX = 0.0;
    double rectY = 0.0;
    double cameraZ = 0.0;
    bool foundWorld = false;
    for (const auto &world : candidateWorlds)
    {
        double projected[2] = {0.0, 0.0};
        if (!rect.rectCamLeft.projectWorldPointWithDepth(world, projected, cameraZ))
        {
            continue;
        }
        rectX = projected[0];
        rectY = projected[1];

        const int px = static_cast<int>(std::round(rectX));
        const int py = static_cast<int>(std::round(rectY));
        if (px < 0 || px >= rect.rectLeft.cols || py < 0 || py >= rect.rectLeft.rows)
        {
            continue;
        }

        ASSERT_GT(cameraZ, 0.0);
        foundWorld = true;
        break;
    }
    ASSERT_TRUE(foundWorld);

    const int px = static_cast<int>(std::round(rectX));
    const int py = static_cast<int>(std::round(rectY));

    cv::Mat depthMap(rect.rectLeft.rows, rect.rectLeft.cols, CV_32F, cv::Scalar(0.0f));
    cv::Mat validMask(rect.rectLeft.rows, rect.rectLeft.cols, CV_8U, cv::Scalar(0));
    depthMap.at<float>(py, px) = static_cast<float>(cameraZ);
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

TEST(DisparityTriangulator, LeftReferenceDisparityReprojectsToRightAtXMinusD)
{
    constexpr int imageSize = 256;
    constexpr int leftX = 128;
    constexpr int imageY = 128;
    constexpr float disparity = 80.0f;
    Camera leftCamera = makeCamera(128.0, 128.0, 0.0);
    Camera rightCamera = makeCamera(128.0, 128.0, 0.2);
    cv::Mat identityStorage = cv::Mat::zeros(3, 4, CV_64F);
    identityStorage.at<double>(0, 0) = 1.0;
    identityStorage.at<double>(1, 1) = 1.0;
    identityStorage.at<double>(2, 2) = 1.0;
    const cv::Mat identity = identityStorage(cv::Rect(0, 0, 3, 3));
    ASSERT_FALSE(identity.isContinuous());

    cv::Mat disparityMap(imageSize, imageSize, CV_32F, cv::Scalar(0.0f));
    cv::Mat validMask(imageSize, imageSize, CV_8U, cv::Scalar(0));
    disparityMap.at<float>(imageY, leftX) = disparity;
    validMask.at<uint8_t>(imageY, leftX) = 255;

    TriangulationConfig config;
    config.maxTriangulationError = 0.01f;
    config.numThreads = 1;
    const TriangulationResult result = DisparityTriangulator::triangulate(
        disparityMap,
        validMask,
        identity,
        identity,
        leftCamera,
        rightCamera,
        config);

    ASSERT_EQ(result.validPoints, 1);
    const cv::Vec3d localPoint = result.pointCloud.at<cv::Vec3d>(imageY, leftX);
    const double world[3] = {
        localPoint[0] + result.pointOffset[0],
        localPoint[1] + result.pointOffset[1],
        localPoint[2] + result.pointOffset[2]};
    double leftProjection[2] = {0.0, 0.0};
    double rightProjection[2] = {0.0, 0.0};
    ASSERT_TRUE(leftCamera.projectWorldPoint(world, leftProjection));
    ASSERT_TRUE(rightCamera.projectWorldPoint(world, rightProjection));

    EXPECT_NEAR(leftProjection[0], leftX, 0.25);
    EXPECT_NEAR(leftProjection[1], imageY, 0.25);
    EXPECT_NEAR(rightProjection[0], leftX - disparity, 0.25);
    EXPECT_NEAR(rightProjection[1], imageY, 0.25);
}

TEST(DisparityTriangulator, RejectsInvalidDisparityInputContractsBeforeWorkersStart)
{
    const Camera leftCamera = makeCamera(2.0, 2.0, 0.0);
    const Camera rightCamera = makeCamera(2.0, 2.0, 0.2);
    const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat disparity(4, 4, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat validMask(4, 4, CV_8UC1, cv::Scalar(255));

    const TriangulationResult wrongType = DisparityTriangulator::triangulate(
        cv::Mat(4, 4, CV_16UC1, cv::Scalar(1)),
        validMask,
        identity,
        identity,
        leftCamera,
        rightCamera);
    EXPECT_NE(wrongType.errorMessage.find("CV_32FC1"), std::string::npos);
    EXPECT_TRUE(wrongType.pointCloud.empty());

    const TriangulationResult wrongMask = DisparityTriangulator::triangulate(
        disparity,
        cv::Mat(3, 4, CV_8UC1, cv::Scalar(255)),
        identity,
        identity,
        leftCamera,
        rightCamera);
    EXPECT_NE(wrongMask.errorMessage.find("尺寸"), std::string::npos);
    EXPECT_TRUE(wrongMask.pointCloud.empty());

    const TriangulationResult wrongHomography = DisparityTriangulator::triangulate(
        disparity,
        validMask,
        cv::Mat::eye(2, 3, CV_64F),
        identity,
        leftCamera,
        rightCamera);
    EXPECT_NE(wrongHomography.errorMessage.find("3x3"), std::string::npos);
    EXPECT_TRUE(wrongHomography.pointCloud.empty());

    const TriangulationResult invalidCamera = DisparityTriangulator::triangulate(
        disparity,
        validMask,
        identity,
        identity,
        Camera(),
        rightCamera);
    EXPECT_NE(invalidCamera.errorMessage.find("左相机"), std::string::npos);
    EXPECT_TRUE(invalidCamera.pointCloud.empty());
}

TEST(DisparityTriangulator, RejectsInvalidDepthInputContractsBeforeWorkersStart)
{
    const Camera leftCamera = makeCamera(2.0, 2.0, 0.0);
    const Camera rightCamera = makeCamera(2.0, 2.0, 0.2);
    const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat depth(4, 4, CV_32FC1, cv::Scalar(2.0f));
    const cv::Mat validMask(4, 4, CV_8UC1, cv::Scalar(255));

    const TriangulationResult wrongMaskType =
        DisparityTriangulator::triangulateFromDepth(
            depth,
            cv::Mat(4, 4, CV_32FC1, cv::Scalar(1.0f)),
            identity,
            leftCamera,
            rightCamera,
            leftCamera);
    EXPECT_NE(wrongMaskType.errorMessage.find("CV_8UC1"), std::string::npos);
    EXPECT_TRUE(wrongMaskType.pointCloud.empty());

    const TriangulationResult wrongHomographyType =
        DisparityTriangulator::triangulateFromDepth(
            depth,
            validMask,
            cv::Mat::eye(3, 3, CV_32F),
            leftCamera,
            rightCamera,
            leftCamera);
    EXPECT_NE(wrongHomographyType.errorMessage.find("CV_64FC1"), std::string::npos);
    EXPECT_TRUE(wrongHomographyType.pointCloud.empty());
}
