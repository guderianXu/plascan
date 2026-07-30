// ============================================================
// 文件：Camera_tests.cpp
// 功能：使用纯合成参数验证 Camera 的最小状态、单位换算、前后方判定、
//       depth-axis flip，以及 Camera 正深度规范化后的投影一致性。
//
// 这些测试不读取磁盘，单位姿态下可直接把世界 Z 看作相机 Z，便于 review
// 时核对每个符号和主点预期。
// ============================================================

#include "Camera.h"
#include <gtest/gtest.h>

using namespace xjw;

TEST(CameraBasic, DefaultInvalid)
{
    // 默认对象只有安全初值，尚未通过文件或 setter 建立可用投影模型。
    Camera cam;
    EXPECT_FALSE(cam.isValid());
}

TEST(CameraIntrinsics, MillimeterConversion)
{
    // 0.5 mm/pixel 意味着 120 mm 焦距在运行态等于 240 pixel。
    Camera cam;
    cam.setIntrinsicsMillimeters(120.0, 120.0, 10.0, 10.0, 0.5);
    EXPECT_NEAR(cam.focalXMillimeters(), 120.0, 1e-9);
    EXPECT_NEAR(cam.focalX(), 120.0 / 0.5, 1e-9);
}

TEST(CameraStructuredState, AccessorsReturnCompleteSnapshots)
{
    Camera camera;
    camera.setIntrinsics(800.0, 810.0, 320.0, 240.0);
    camera.setPixelPitch(0.004);
    camera.setAxisDirections(-1, 1);

    Camera::Distortion configured_distortion;
    configured_distortion.radialK1 = 0.1;
    configured_distortion.radialK2 = -0.01;
    configured_distortion.radialK3 = 0.001;
    configured_distortion.tangentialP1 = 0.0002;
    configured_distortion.tangentialP2 = -0.0003;
    camera.setDistortion(configured_distortion);

    const std::array<double, 9> rotation{{0.0, -1.0, 0.0,
                                          1.0,  0.0, 0.0,
                                          0.0,  0.0, 1.0}};
    const std::array<double, 3> center{{1.0, 2.0, 3.0}};
    camera.setPose(rotation, center);
    camera.setDepthAxisFlipped(true);

    Camera::Intrinsics intrinsics = camera.intrinsics();
    Camera::Distortion distortion = camera.distortion();
    Camera::Pose pose = camera.pose();
    EXPECT_DOUBLE_EQ(intrinsics.focalX, 800.0);
    EXPECT_DOUBLE_EQ(intrinsics.focalY, 810.0);
    EXPECT_DOUBLE_EQ(intrinsics.principalX, 320.0);
    EXPECT_DOUBLE_EQ(intrinsics.principalY, 240.0);
    EXPECT_DOUBLE_EQ(intrinsics.pixelPitch, 0.004);
    EXPECT_EQ(intrinsics.uAxisSign, -1);
    EXPECT_EQ(intrinsics.vAxisSign, 1);
    EXPECT_DOUBLE_EQ(distortion.radialK1, configured_distortion.radialK1);
    EXPECT_DOUBLE_EQ(distortion.radialK2, configured_distortion.radialK2);
    EXPECT_DOUBLE_EQ(distortion.radialK3, configured_distortion.radialK3);
    EXPECT_DOUBLE_EQ(distortion.tangentialP1, configured_distortion.tangentialP1);
    EXPECT_DOUBLE_EQ(distortion.tangentialP2, configured_distortion.tangentialP2);
    EXPECT_EQ(pose.cameraToWorldRotation, rotation);
    EXPECT_EQ(pose.cameraCenter, center);
    EXPECT_TRUE(pose.depthAxisFlipped);

    // Getter 返回的是值快照；修改快照不能绕过 Camera 的受控更新接口。
    intrinsics.focalX = 1.0;
    distortion.radialK1 = 2.0;
    pose.cameraCenter[0] = 4.0;
    EXPECT_DOUBLE_EQ(camera.intrinsics().focalX, 800.0);
    EXPECT_DOUBLE_EQ(camera.distortion().radialK1, configured_distortion.radialK1);
    EXPECT_DOUBLE_EQ(camera.pose().cameraCenter[0], 1.0);
}

TEST(CameraProjection, FrontAndBack)
{
    // 单位 R_cw、原点 C 和常规 +Z 光轴下，轴线上正 Z 点应落在主点。
    Camera cam;
    cam.setIntrinsics(100.0, 100.0, 50.0, 50.0);
    std::array<double,9> R = {1,0,0,0,1,0,0,0,1};
    std::array<double,3> C = {0,0,0};
    cam.setPose(R, C);

    double worldFront[3] = {0.0, 0.0, 10.0};
    double pixel[2] = {0.0, 0.0};
    EXPECT_TRUE(cam.projectWorldPoint(worldFront, pixel));
    EXPECT_NEAR(pixel[0], 50.0, 1e-9);
    EXPECT_NEAR(pixel[1], 50.0, 1e-9);

    double worldBack[3] = {0.0, 0.0, -10.0};
    EXPECT_FALSE(cam.projectWorldPoint(worldBack, pixel));
    EXPECT_TRUE(cam.projectWorldPointSigned(worldBack, pixel));
}

TEST(CameraProjection, FrontAndBackWithDepthFlip)
{
    // depth flip 只改变“哪一侧是物理前方”；透视除法仍保留 Z_cam 自身符号。
    Camera cam;
    cam.setIntrinsics(100.0, 100.0, 50.0, 50.0);
    std::array<double,9> R = {1,0,0,0,1,0,0,0,1};
    std::array<double,3> C = {0,0,0};
    cam.setPose(R, C);
    cam.setDepthAxisFlipped(true);

    double worldFront[3] = {0.0, 0.0, -10.0};
    double pixel[2] = {0.0, 0.0};
    EXPECT_TRUE(cam.projectWorldPoint(worldFront, pixel));
    EXPECT_NEAR(pixel[0], 50.0, 1e-9);
    EXPECT_NEAR(pixel[1], 50.0, 1e-9);

    double worldBack[3] = {0.0, 0.0, 10.0};
    EXPECT_FALSE(cam.projectWorldPoint(worldBack, pixel));
    EXPECT_TRUE(cam.projectWorldPointSigned(worldFront, pixel));
}

TEST(CameraPositiveDepth, NormalizationPreservesDistortedProjection)
{
    Camera camera;
    camera.setIntrinsics(700.0, 710.0, 320.0, 240.0);
    camera.setAxisDirections(-1, -1);
    camera.setDepthAxisFlipped(true);
    camera.setDistortion(0.03, -0.002, 0.0001, 0.0015, -0.0007);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {1.0, -2.0, 0.5});

    const double world[3] = {1.4, -1.7, -3.5};
    double original_pixel[2] = {0.0, 0.0};
    ASSERT_TRUE(camera.projectWorldPoint(world, original_pixel));

    const Camera normalized = camera.normalizedForPositiveDepth();
    const Camera::Intrinsics normalized_intrinsics = normalized.intrinsics();
    const Camera::Distortion normalized_distortion = normalized.distortion();
    EXPECT_EQ(normalized_intrinsics.uAxisSign, 1);
    EXPECT_EQ(normalized_intrinsics.vAxisSign, 1);
    EXPECT_FALSE(normalized.depthAxisFlipped());
    EXPECT_DOUBLE_EQ(normalized_distortion.radialK1, 0.03);
    EXPECT_DOUBLE_EQ(normalized_distortion.tangentialP1, -0.0015);
    EXPECT_DOUBLE_EQ(normalized_distortion.tangentialP2, 0.0007);

    double normalized_pixel[2] = {0.0, 0.0};
    ASSERT_TRUE(normalized.projectWorldPoint(world, normalized_pixel));
    EXPECT_NEAR(normalized_pixel[0], original_pixel[0], 1e-9);
    EXPECT_NEAR(normalized_pixel[1], original_pixel[1], 1e-9);
}

TEST(CameraPositiveDepth, UsesPhysicalForwardAxis)
{
    Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {0.0, 0.0, 0.0});

    const double positive_z[3] = {0.0, 0.0, 5.0};
    const double negative_z[3] = {0.0, 0.0, -5.0};
    EXPECT_DOUBLE_EQ(camera.positiveDepth(positive_z), 5.0);
    EXPECT_TRUE(camera.isPointInFront(positive_z));
    EXPECT_FALSE(camera.isPointInFront(negative_z));

    camera.setDepthAxisFlipped(true);
    EXPECT_DOUBLE_EQ(camera.positiveDepth(negative_z), 5.0);
    EXPECT_TRUE(camera.isPointInFront(negative_z));
    EXPECT_FALSE(camera.isPointInFront(positive_z));
}

TEST(CameraPositiveDepth, DistortedProjectionUnprojectionRoundTrip)
{
    Camera camera;
    camera.setIntrinsics(920.0, 910.0, 512.0, 384.0);
    camera.setAxisDirections(-1, 1);
    camera.setDepthAxisFlipped(true);
    camera.setDistortion(0.015, -0.001, 0.00005, -0.0004, 0.0008);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {2.0, 3.0, 4.0});

    const double world[3] = {2.8, 2.6, -6.0};
    double pixel[2] = {0.0, 0.0};
    double positive_depth = 0.0;
    ASSERT_TRUE(camera.projectWorldPointWithDepth(world, pixel, positive_depth));
    EXPECT_NEAR(positive_depth, 10.0, 1e-12);

    double restored_world[3] = {0.0, 0.0, 0.0};
    ASSERT_TRUE(camera.unprojectPixel(pixel, positive_depth, restored_world));
    EXPECT_NEAR(restored_world[0], world[0], 1e-8);
    EXPECT_NEAR(restored_world[1], world[1], 1e-8);
    EXPECT_NEAR(restored_world[2], world[2], 1e-8);
}
