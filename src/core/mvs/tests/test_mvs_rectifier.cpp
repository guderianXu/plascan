#include "EpipolarRectifier.h"
#include "DisparityTriangulator.h"

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

placamera::FramePinholeModel makeCameraWithPose(placamera::FrameIntrinsics intrinsics,
                                                const std::array<double, 9>& rotation,
                                                const std::array<double, 3>& center,
                                                const std::string& id = "fixture",
                                                const std::string& frame_name = "test-world",
                                                placamera::ImageSize size = {1024, 768},
                                                placamera::BrownConradyDistortion distortion = {},
                                                const std::string& image_id = {})
{
    const placamera::FrameId frame(frame_name);
    const auto definition = placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId(id + "-definition"),
                                                                      intrinsics,
                                                                      distortion,
                                                                      placamera::PixelConvention::PixelCenter,
                                                                      frame);
    return placamera::FramePinholeModel::create(placamera::CameraInstanceId(id),
                                                placamera::ImageId(image_id.empty() ? id + "-image" : image_id),
                                                definition,
                                                size,
                                                placamera::Pose::create(frame, center, rotation));
}

placamera::FramePinholeModel makeCamera(double cx, double cy, double tx)
{
    return makeCameraWithPose({2000.0, 2000.0, cx, cy}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {tx, 0.0, 0.0});
}

placamera::FramePinholeModel makeYawedCamera(double cx, double cy, double tx, double yawDegrees)
{
    const double yaw = yawDegrees * M_PI / 180.0;
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    return makeCameraWithPose({2000.0, 2000.0, cx, cy}, {c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c}, {tx, 0.0, 0.0});
}

placamera::FramePinholeModel makeVerticalBaselineCamera(double cx, double cy, double ty)
{
    return makeCameraWithPose({2000.0, 2000.0, cx, cy}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {0.0, ty, 0.0});
}

placamera::FramePinholeModel makeDinoRingCamera(const std::array<double, 9>& rotation_camera_to_world,
                                                const std::array<double, 3>& center)
{
    cv::Mat approximate_rotation(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            approximate_rotation.at<double>(row, column) = rotation_camera_to_world[row * 3 + column];
        }
    }
    const cv::SVD decomposition(approximate_rotation);
    const cv::Mat proper_rotation = decomposition.u * decomposition.vt;
    std::array<double, 9> rotation{};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            rotation[row * 3 + column] = proper_rotation.at<double>(row, column);
        }
    }
    return makeCameraWithPose({3310.4, 3325.5, 316.73, 200.55}, rotation, center);
}

placamera::FramePinholeModel makeNativeCamera(const placamera::FramePinholeModel& camera,
                                              const std::string& id,
                                              int width,
                                              int height,
                                              const std::string& frame = "test-world")
{
    const placamera::FrameId ground_frame(frame);
    const auto definition = placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId(id + "-definition"),
                                                                      camera.pinholeDefinition().intrinsics(),
                                                                      camera.pinholeDefinition().distortion(),
                                                                      placamera::PixelConvention::PixelCenter,
                                                                      ground_frame,
                                                                      camera.pinholeDefinition().depthAxisFlipped());
    const auto pose = placamera::Pose::create(ground_frame, camera.pose().center, camera.pose().cameraToWorldRotation);
    return placamera::FramePinholeModel::create(placamera::CameraInstanceId(id),
                                                placamera::ImageId(id + "-image"),
                                                definition,
                                                placamera::ImageSize{width, height},
                                                pose);
}

bool projectPoint(const placamera::FramePinholeModel& camera,
                  const double world[3],
                  double pixel[2],
                  double* depth = nullptr)
{
    const auto result = camera.groundToImage({camera.groundFrame(), {world[0], world[1], world[2]}});
    if (!result || !result.value().positiveDepth)
    {
        return false;
    }
    pixel[0] = result.value().image.sample;
    pixel[1] = result.value().image.line;
    if (depth != nullptr)
    {
        *depth = *result.value().positiveDepth;
    }
    return true;
}

bool unprojectPoint(const placamera::FramePinholeModel& camera, const double pixel[2], double depth, double world[3])
{
    const auto result = camera.imageToGroundAtDepth({pixel[0], pixel[1]}, depth);
    if (!result)
    {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        world[axis] = result.value().position[axis];
    }
    return true;
}

} // namespace

TEST(EpipolarRectifier, RejectsImagesThatStillCarryLensDistortion)
{
    const auto left_camera = makeCameraWithPose({2000.0, 2000.0, 32.0, 24.0},
                                                {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                                                {0.0, 0.0, 0.0},
                                                "distorted-left",
                                                "test-world",
                                                {64, 48},
                                                {0.1, 0.0, 0.0, 0.0, 0.0});
    const auto right_camera = makeCamera(32.0, 24.0, 0.2);

    cv::Mat left_image(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat right_image(48, 64, CV_8U, cv::Scalar(96));
    EpipolarRectifier::RectifiedPair rectified;
    std::string error;
    EXPECT_FALSE(
        EpipolarRectifier::rectify(left_image,
                                   right_image,
                                   makeNativeCamera(left_camera, "rect-left", left_image.cols, left_image.rows),
                                   makeNativeCamera(right_camera, "rect-right", right_image.cols, right_image.rows),
                                   rectified,
                                   &error));
    EXPECT_NE(error.find("去畸变"), std::string::npos);
}

TEST(EpipolarRectifier, RejectsMixedCameraGroundFrames)
{
    const auto left_camera = makeNativeCamera(makeCamera(32.0, 24.0, 0.0), "mixed-left", 64, 48, "left-world");
    const auto right_camera = makeNativeCamera(makeCamera(32.0, 24.0, 0.2), "mixed-right", 64, 48, "right-world");
    const cv::Mat image(48, 64, CV_8U, cv::Scalar(64));
    EpipolarRectifier::RectifiedPair result;
    std::string error;
    EXPECT_FALSE(EpipolarRectifier::rectify(image, image, left_camera, right_camera, result, &error));
    EXPECT_NE(error.find("坐标系"), std::string::npos);
    EXPECT_FALSE(result.rectCamLeft.has_value());
}

TEST(EpipolarRectifier, RejectsConvergentPairWithoutUsableRectifiedCanvas)
{
    const placamera::FramePinholeModel left_camera =
        makeDinoRingCamera({-0.143964578361,
                            -0.903665806035,
                            -0.403315364598,
                            0.969652632813,
                            -0.0474333525503,
                            -0.239841305752,
                            0.197606171538,
                            -0.425604192333,
                            0.883069362015},
                           {0.243378250328, 0.170140221186, -0.604858522898});
    const placamera::FramePinholeModel right_camera =
        makeDinoRingCamera({-0.231436872629,
                            -0.658608111657,
                            -0.716012081872,
                            0.96422332027,
                            -0.0574942602048,
                            -0.258781378564,
                            0.129269371656,
                            -0.750286404865,
                            0.648350496196},
                           {0.447988592591, 0.182631346554, -0.449273743884});

    cv::Mat left_image(480, 640, CV_8U, cv::Scalar(64));
    cv::Mat right_image(480, 640, CV_8U, cv::Scalar(96));
    EpipolarRectifier::RectifiedPair rectified;
    std::string error;

    EXPECT_FALSE(
        EpipolarRectifier::rectify(left_image,
                                   right_image,
                                   makeNativeCamera(left_camera, "rect-left", left_image.cols, left_image.rows),
                                   makeNativeCamera(right_camera, "rect-right", right_image.cols, right_image.rows),
                                   rectified,
                                   &error));
    EXPECT_NE(error.find("有效区域"), std::string::npos);
}

TEST(EpipolarRectifier, RectificationMakesEpipolarRowsMatchAcrossDepths)
{
    placamera::FramePinholeModel leftCamera = makeCamera(32.0, 24.0, 0.0);
    placamera::FramePinholeModel rightCamera = makeCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(
        EpipolarRectifier::rectify(leftImage,
                                   rightImage,
                                   makeNativeCamera(leftCamera, "rect-left", leftImage.cols, leftImage.rows),
                                   makeNativeCamera(rightCamera, "rect-right", rightImage.cols, rightImage.rows),
                                   rect,
                                   &error))
        << error;

    const double leftWorld[3] = {0.1, 0.0, 5.0};
    const double rightWorld[3] = {0.1, 0.0, 8.0};
    const double* worlds[2] = {leftWorld, rightWorld};

    for (const double* world : worlds)
    {
        double leftPixel[2] = {0.0, 0.0};
        double rightPixel[2] = {0.0, 0.0};
        ASSERT_TRUE(projectPoint(leftCamera, world, leftPixel));
        ASSERT_TRUE(projectPoint(rightCamera, world, rightPixel));

        double leftRectX = 0.0;
        double leftRectY = 0.0;
        double rightRectX = 0.0;
        double rightRectY = 0.0;
        applyHomography(rect.H1, leftPixel[0], leftPixel[1], leftRectX, leftRectY);
        applyHomography(rect.H2, rightPixel[0], rightPixel[1], rightRectX, rightRectY);

        EXPECT_NEAR(leftRectY, rightRectY, 1.0);
    }
}

TEST(EpipolarRectifier, PreservesCameraIdentityOnDerivedRectifiedViews)
{
    const auto leftCamera = makeCameraWithPose({2000.0, 2000.0, 32.0, 24.0},
                                               {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                                               {0.0, 0.0, 0.0},
                                               "rect-instance-left",
                                               "rect-frame",
                                               {64, 48},
                                               {},
                                               "rect-image-left");
    const auto rightCamera = makeCameraWithPose({2000.0, 2000.0, 32.0, 24.0},
                                                {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                                                {0.2, 0.0, 0.0},
                                                "rect-instance-right",
                                                "rect-frame",
                                                {64, 48},
                                                {},
                                                "rect-image-right");

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));
    EpipolarRectifier::RectifiedPair rectified;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage, rightImage, leftCamera, rightCamera, rectified, &error)) << error;

    ASSERT_TRUE(rectified.rectCamLeft.has_value());
    ASSERT_TRUE(rectified.rectCamRight.has_value());
    EXPECT_EQ(rectified.rectCamLeft->imageId().value(), "rect-image-left-rectified");
    EXPECT_EQ(rectified.rectCamRight->imageId().value(), "rect-image-right-rectified");
    EXPECT_EQ(rectified.rectCamLeft->instanceId().value(), "rect-instance-left-rectified");
    EXPECT_EQ(rectified.rectCamRight->instanceId().value(), "rect-instance-right-rectified");
    EXPECT_EQ(rectified.rectCamLeft->groundFrame().value(), "rect-frame");
    EXPECT_EQ(rectified.rectCamRight->groundFrame().value(), "rect-frame");
}

TEST(EpipolarRectifier, ParallelStereoKeepsPositiveRectifiedDisparity)
{
    placamera::FramePinholeModel leftCamera = makeCamera(32.0, 24.0, 0.0);
    placamera::FramePinholeModel rightCamera = makeCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           makeNativeCamera(leftCamera,
                                                            "rect-left", leftImage.cols, leftImage.rows),
                                           makeNativeCamera(rightCamera,
                                                            "rect-right", rightImage.cols, rightImage.rows),
                                           rect,
                                           &error)) << error;

    const double world[3] = {0.1, 0.0, 5.0};
    double leftPixel[2] = {0.0, 0.0};
    double rightPixel[2] = {0.0, 0.0};
    ASSERT_TRUE(projectPoint(leftCamera, world, leftPixel));
    ASSERT_TRUE(projectPoint(rightCamera, world, rightPixel));

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
    placamera::FramePinholeModel leftCamera = makeCamera(512.0, 384.0, 0.0);
    placamera::FramePinholeModel rightCamera = makeYawedCamera(512.0, 384.0, 0.2, 15.0);

    cv::Mat leftImage(768, 1024, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(768, 1024, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           makeNativeCamera(leftCamera,
                                                            "rect-left", leftImage.cols, leftImage.rows),
                                           makeNativeCamera(rightCamera,
                                                            "rect-right", rightImage.cols, rightImage.rows),
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
                ASSERT_TRUE(projectPoint(leftCamera, world, leftPixel));
                ASSERT_TRUE(projectPoint(rightCamera, world, rightPixel));

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

TEST(EpipolarRectifier, RotatedRectifiedDepthRoundTripsToOriginalCameraZ)
{
    constexpr int width = 64;
    constexpr int height = 48;
    constexpr double focal = 100.0;
    constexpr double principal_x = 32.0;
    constexpr double principal_y = 24.0;
    constexpr double angle_degrees = 10.0;
    const double angle = angle_degrees * M_PI / 180.0;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const auto original_camera = makeCameraWithPose({focal, focal, principal_x, principal_y},
                                                    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                                                    {0.0, 0.0, 0.0},
                                                    "original-depth",
                                                    "test-world",
                                                    {width, height});
    // Pose receives camera-to-world. Its transpose is the +10 degree
    // world-to-rectified-camera rotation used by the homography below.
    const auto rectified_camera = makeCameraWithPose({focal, focal, principal_x, principal_y},
                                                     {cosine, 0.0, -sine, 0.0, 1.0, 0.0, sine, 0.0, cosine},
                                                     {0.0, 0.0, 0.0},
                                                     "rectified-depth",
                                                     "test-world",
                                                     {width, height});
    const cv::Mat camera_matrix =
        (cv::Mat_<double>(3, 3) << focal, 0.0, principal_x, 0.0, focal, principal_y, 0.0, 0.0, 1.0);
    const cv::Mat rotation_world_to_rectified =
        (cv::Mat_<double>(3, 3) << cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine);
    EpipolarRectifier::RectifiedPair rectified;
    rectified.H1 = camera_matrix * rotation_world_to_rectified * camera_matrix.inv();
    rectified.H1inv = rectified.H1.inv();
    const auto original_native = makeNativeCamera(original_camera, "original-depth", width, height);
    const auto rectified_native = makeNativeCamera(rectified_camera, "rectified-depth", width, height);
    rectified.rectCamLeft = rectified_native;
    rectified.refIsRight = false;

    constexpr double original_depth = 6.0;
    constexpr int original_column = 32;
    constexpr int original_row = 24;
    const double original_pixel[2] = {original_column, original_row};
    double world[3] = {};
    ASSERT_TRUE(unprojectPoint(original_camera, original_pixel, original_depth, world));
    double rectified_pixel[2] = {};
    double rectified_depth = 0.0;
    ASSERT_TRUE(projectPoint(rectified_camera, world, rectified_pixel, &rectified_depth));
    ASSERT_GT(std::fabs(rectified_depth - original_depth), 0.01);
    const int rectified_column = static_cast<int>(std::lround(rectified_pixel[0]));
    const int rectified_row = static_cast<int>(std::lround(rectified_pixel[1]));
    ASSERT_GE(rectified_column, 0);
    ASSERT_LT(rectified_column, width);
    ASSERT_GE(rectified_row, 0);
    ASSERT_LT(rectified_row, height);

    cv::Mat rectified_depth_map(height, width, CV_32FC1, cv::Scalar(0.0f));
    rectified_depth_map.at<float>(rectified_row, rectified_column) = static_cast<float>(rectified_depth);
    const cv::Mat original_depth_map =
        EpipolarRectifier::unrectifyDepth(rectified_depth_map, rectified, original_native, width, height);

    ASSERT_FALSE(original_depth_map.empty());
    ASSERT_GT(original_depth_map.at<float>(original_row, original_column), 0.0f);
    EXPECT_NEAR(original_depth_map.at<float>(original_row, original_column), original_depth, 1.0e-3);

    float rectified_near = 0.0f;
    float rectified_far = 0.0f;
    ASSERT_TRUE(EpipolarRectifier::rectifiedDepthRange(
        original_native, rectified_native, width, height, 4.0f, 9.0f, rectified_near, rectified_far));
    for (const int row : {0, height - 1})
    {
        for (const int column : {0, width - 1})
        {
            const double pixel[2] = {static_cast<double>(column), static_cast<double>(row)};
            for (const double depth : {4.0, 9.0})
            {
                double range_world[3] = {};
                ASSERT_TRUE(unprojectPoint(original_camera, pixel, depth, range_world));
                const auto projected_depth = rectified_camera.signedDepth(
                    {rectified_camera.groundFrame(), {range_world[0], range_world[1], range_world[2]}});
                ASSERT_TRUE(projected_depth);
                const double depth_in_rectified_camera = projected_depth.value();
                EXPECT_GE(depth_in_rectified_camera, rectified_near);
                EXPECT_LE(depth_in_rectified_camera, rectified_far);
            }
        }
    }
}

TEST(EpipolarRectifier, VerticalBaselineSetsTransposedForHorizontalDisparity)
{
    placamera::FramePinholeModel leftCamera = makeCamera(32.0, 24.0, 0.0);
    placamera::FramePinholeModel rightCamera =
        makeVerticalBaselineCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           makeNativeCamera(leftCamera,
                                                            "rect-left", leftImage.cols, leftImage.rows),
                                           makeNativeCamera(rightCamera,
                                                            "rect-right", rightImage.cols, rightImage.rows),
                                           rect,
                                           &error)) << error;

    EXPECT_TRUE(rect.transposed);
}


TEST(EpipolarRectifier, TransposedRectifiedCamerasProjectIntoTransposedPixels)
{
    placamera::FramePinholeModel leftCamera = makeCamera(32.0, 24.0, 0.0);
    placamera::FramePinholeModel rightCamera =
        makeVerticalBaselineCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           makeNativeCamera(leftCamera,
                                                            "rect-left", leftImage.cols, leftImage.rows),
                                           makeNativeCamera(rightCamera,
                                                            "rect-right", rightImage.cols, rightImage.rows),
                                           rect,
                                           &error)) << error;
    ASSERT_TRUE(rect.transposed);

    const double world[3] = {0.1, 0.05, 5.0};
    double leftPixel[2] = {0.0, 0.0};
    double rightPixel[2] = {0.0, 0.0};
    ASSERT_TRUE(projectPoint(leftCamera, world, leftPixel));
    ASSERT_TRUE(projectPoint(rightCamera, world, rightPixel));

    double leftRectX = 0.0;
    double leftRectY = 0.0;
    double rightRectX = 0.0;
    double rightRectY = 0.0;
    applyHomography(rect.H1, leftPixel[0], leftPixel[1], leftRectX, leftRectY);
    applyHomography(rect.H2, rightPixel[0], rightPixel[1], rightRectX, rightRectY);

    ASSERT_TRUE(rect.rectCamLeft.has_value());
    ASSERT_TRUE(rect.rectCamRight.has_value());
    const placamera::GroundCoordinate ground{rect.rectCamLeft->groundFrame(), {world[0], world[1], world[2]}};
    const auto left_projected = rect.rectCamLeft->groundToImage(ground);
    const auto right_projected = rect.rectCamRight->groundToImage(ground);
    ASSERT_TRUE(left_projected) << left_projected.message();
    ASSERT_TRUE(right_projected) << right_projected.message();

    EXPECT_NEAR(left_projected.value().image.sample, leftRectX, 1.0);
    EXPECT_NEAR(left_projected.value().image.line, leftRectY, 1.0);
    EXPECT_NEAR(right_projected.value().image.sample, rightRectX, 1.0);
    EXPECT_NEAR(right_projected.value().image.line, rightRectY, 1.0);
}


TEST(EpipolarRectifier, TransposedRectifiedCameraRawFieldsMatchProjection)
{
    placamera::FramePinholeModel leftCamera = makeCamera(32.0, 24.0, 0.0);
    placamera::FramePinholeModel rightCamera =
        makeVerticalBaselineCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           makeNativeCamera(leftCamera,
                                                            "rect-left", leftImage.cols, leftImage.rows),
                                           makeNativeCamera(rightCamera,
                                                            "rect-right", rightImage.cols, rightImage.rows),
                                           rect,
                                           &error)) << error;
    ASSERT_TRUE(rect.transposed);

    const double world[3] = {0.1, 0.05, 5.0};

    ASSERT_TRUE(rect.rectCamLeft.has_value());
    const placamera::GroundCoordinate ground{rect.rectCamLeft->groundFrame(), {world[0], world[1], world[2]}};
    const auto projected = rect.rectCamLeft->groundToImage(ground);
    ASSERT_TRUE(projected) << projected.message();
    ASSERT_TRUE(projected.value().positiveDepth.has_value());
    const double cameraZ = *projected.value().positiveDepth;

    const auto& pose = rect.rectCamLeft->pose();
    const auto& rotation = pose.cameraToWorldRotation;
    const std::array<double, 3> offset{world[0] - pose.center[0], world[1] - pose.center[1], world[2] - pose.center[2]};
    const std::array<double, 3> camera_point{
        rotation[0] * offset[0] + rotation[3] * offset[1] + rotation[6] * offset[2],
        rotation[1] * offset[0] + rotation[4] * offset[1] + rotation[7] * offset[2],
        rotation[2] * offset[0] + rotation[5] * offset[1] + rotation[8] * offset[2]};
    ASSERT_GT(cameraZ, 0.0);
    const auto& intrinsics = rect.rectCamLeft->pinholeDefinition().intrinsics();
    const double rawX = intrinsics.uAxisSign * intrinsics.focalX * camera_point[0] / cameraZ + intrinsics.principalX;
    const double rawY = intrinsics.vAxisSign * intrinsics.focalY * camera_point[1] / cameraZ + intrinsics.principalY;

    EXPECT_NEAR(rawX, projected.value().image.sample, 1.0);
    EXPECT_NEAR(rawY, projected.value().image.line, 1.0);
}

TEST(DisparityTriangulator, TransposedRectifiedDepthTriangulationKeepsLowReprojectionError)
{
    placamera::FramePinholeModel leftCamera = makeCamera(32.0, 24.0, 0.0);
    placamera::FramePinholeModel rightCamera =
        makeVerticalBaselineCamera(32.0, 24.0, 0.2);

    cv::Mat leftImage(48, 64, CV_8U, cv::Scalar(64));
    cv::Mat rightImage(48, 64, CV_8U, cv::Scalar(96));

    EpipolarRectifier::RectifiedPair rect;
    std::string error;
    ASSERT_TRUE(EpipolarRectifier::rectify(leftImage,
                                           rightImage,
                                           makeNativeCamera(leftCamera,
                                                            "rect-left", leftImage.cols, leftImage.rows),
                                           makeNativeCamera(rightCamera,
                                                            "rect-right", rightImage.cols, rightImage.rows),
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
        ASSERT_TRUE(rect.rectCamLeft.has_value());
        const placamera::GroundCoordinate ground{rect.rectCamLeft->groundFrame(), {world[0], world[1], world[2]}};
        const auto projected = rect.rectCamLeft->groundToImage(ground);
        if (!projected || !projected.value().positiveDepth)
        {
            continue;
        }
        rectX = projected.value().image.sample;
        rectY = projected.value().image.line;
        cameraZ = *projected.value().positiveDepth;

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
        makeNativeCamera(leftCamera, "depth-left", depthMap.cols, depthMap.rows),
        makeNativeCamera(rightCamera, "depth-right", depthMap.cols, depthMap.rows),
        *rect.rectCamLeft,
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
    placamera::FramePinholeModel leftCamera = makeCamera(128.0, 128.0, 0.0);
    placamera::FramePinholeModel rightCamera = makeCamera(128.0, 128.0, 0.2);
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
    const auto left_native = makeNativeCamera(leftCamera, "disparity-left", imageSize, imageSize);
    const auto right_native = makeNativeCamera(rightCamera, "disparity-right", imageSize, imageSize);
    const TriangulationResult result = DisparityTriangulator::triangulate(
        disparityMap,
        validMask,
        identity,
        identity,
        left_native,
        right_native,
        config);

    ASSERT_EQ(result.validPoints, 1);
    const cv::Vec3d localPoint = result.pointCloud.at<cv::Vec3d>(imageY, leftX);
    const double world[3] = {localPoint[0] + result.pointOffset[0],
                             localPoint[1] + result.pointOffset[1],
                             localPoint[2] + result.pointOffset[2]};
    const placamera::GroundCoordinate ground{left_native.groundFrame(), {world[0], world[1], world[2]}};
    const auto left_projection = left_native.groundToImage(ground);
    const auto right_projection = right_native.groundToImage(ground);
    ASSERT_TRUE(left_projection) << left_projection.message();
    ASSERT_TRUE(right_projection) << right_projection.message();

    EXPECT_NEAR(left_projection.value().image.sample, leftX, 0.25);
    EXPECT_NEAR(left_projection.value().image.line, imageY, 0.25);
    EXPECT_NEAR(right_projection.value().image.sample, leftX - disparity, 0.25);
    EXPECT_NEAR(right_projection.value().image.line, imageY, 0.25);
}

TEST(DisparityTriangulator, RejectsInvalidDisparityInputContractsBeforeWorkersStart)
{
    const placamera::FramePinholeModel leftCamera = makeCamera(2.0, 2.0, 0.0);
    const placamera::FramePinholeModel rightCamera = makeCamera(2.0, 2.0, 0.2);
    const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat disparity(4, 4, CV_32FC1, cv::Scalar(1.0f));
    const cv::Mat validMask(4, 4, CV_8UC1, cv::Scalar(255));
    const auto left_native = makeNativeCamera(leftCamera, "invalid-left", 4, 4);
    const auto right_native = makeNativeCamera(rightCamera, "invalid-right", 4, 4);

    const TriangulationResult wrongType = DisparityTriangulator::triangulate(
        cv::Mat(4, 4, CV_16UC1, cv::Scalar(1)), validMask, identity, identity, left_native, right_native);
    EXPECT_NE(wrongType.errorMessage.find("CV_32FC1"), std::string::npos);
    EXPECT_TRUE(wrongType.pointCloud.empty());

    const TriangulationResult wrongMask = DisparityTriangulator::triangulate(
        disparity, cv::Mat(3, 4, CV_8UC1, cv::Scalar(255)), identity, identity, left_native, right_native);
    EXPECT_NE(wrongMask.errorMessage.find("尺寸"), std::string::npos);
    EXPECT_TRUE(wrongMask.pointCloud.empty());

    const TriangulationResult wrongHomography = DisparityTriangulator::triangulate(
        disparity, validMask, cv::Mat::eye(2, 3, CV_64F), identity, left_native, right_native);
    EXPECT_NE(wrongHomography.errorMessage.find("3x3"), std::string::npos);
    EXPECT_TRUE(wrongHomography.pointCloud.empty());

    const TriangulationResult mismatchedFrame =
        DisparityTriangulator::triangulate(disparity,
                                           validMask,
                                           identity,
                                           identity,
                                           makeNativeCamera(leftCamera, "other-frame", 4, 4, "other-world"),
                                           right_native);
    EXPECT_NE(mismatchedFrame.errorMessage.find("坐标系"), std::string::npos);
    EXPECT_TRUE(mismatchedFrame.pointCloud.empty());
}

TEST(DisparityTriangulator, RejectsInvalidDepthInputContractsBeforeWorkersStart)
{
    const placamera::FramePinholeModel leftCamera = makeCamera(2.0, 2.0, 0.0);
    const placamera::FramePinholeModel rightCamera = makeCamera(2.0, 2.0, 0.2);
    const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat depth(4, 4, CV_32FC1, cv::Scalar(2.0f));
    const cv::Mat validMask(4, 4, CV_8UC1, cv::Scalar(255));
    const auto left_native = makeNativeCamera(leftCamera, "depth-contract-left", 4, 4);
    const auto right_native = makeNativeCamera(rightCamera, "depth-contract-right", 4, 4);

    const TriangulationResult wrongMaskType = DisparityTriangulator::triangulateFromDepth(
        depth, cv::Mat(4, 4, CV_32FC1, cv::Scalar(1.0f)), identity, left_native, right_native, left_native);
    EXPECT_NE(wrongMaskType.errorMessage.find("CV_8UC1"), std::string::npos);
    EXPECT_TRUE(wrongMaskType.pointCloud.empty());

    const TriangulationResult wrongHomographyType =
        DisparityTriangulator::triangulateFromDepth(
            depth,
            validMask,
            cv::Mat::eye(3, 3, CV_32F),
            left_native,
            right_native,
            left_native);
    EXPECT_NE(wrongHomographyType.errorMessage.find("CV_64FC1"), std::string::npos);
    EXPECT_TRUE(wrongHomographyType.pointCloud.empty());

    const TriangulationResult mismatchedRectifiedFrame = DisparityTriangulator::triangulateFromDepth(
        depth,
        validMask,
        identity,
        left_native,
        right_native,
        makeNativeCamera(leftCamera, "depth-other-frame", 4, 4, "other-world"));
    EXPECT_NE(mismatchedRectifiedFrame.errorMessage.find("坐标系"), std::string::npos);
    EXPECT_TRUE(mismatchedRectifiedFrame.pointCloud.empty());
}
