#include "Camera.h"
#include "PositiveDepthCameraModel.h"
#include <gtest/gtest.h>

using namespace xjw;

TEST(CameraBasic, DefaultInvalid)
{
    Camera cam;
    EXPECT_FALSE(cam.isValid());
}

TEST(CameraIntrinsics, MillimeterConversion)
{
    Camera cam;
    cam.setIntrinsicsMillimeters(120.0, 120.0, 10.0, 10.0, 0.5);
    EXPECT_NEAR(cam.focalXMillimeters(), 120.0, 1e-9);
    EXPECT_NEAR(cam.focalX(), 120.0 / 0.5, 1e-9);
}

TEST(CameraProjection, FrontAndBack)
{
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

TEST(CameraToPositiveDepthModel, ProjectConsistency)
{
    Camera cam;
    cam.setIntrinsics(100.0, 100.0, 32.0, 24.0);
    std::array<double,9> R = {1,0,0,0,1,0,0,0,1};
    std::array<double,3> C = {0,0,0};
    cam.setPose(R, C);

    PositiveDepthCameraModel pdm(cam);

    float px = 0.0f, py = 0.0f;
    bool ok = pdm.project(0.0f, 0.0f, 10.0f, px, py);
    EXPECT_TRUE(ok);

    double world[3] = {0.0, 0.0, 10.0};
    double pixelD[2] = {0.0, 0.0};
    EXPECT_TRUE(cam.projectWorldPoint(world, pixelD));
    EXPECT_NEAR(px, static_cast<float>(pixelD[0]), 1e-4f);
    EXPECT_NEAR(py, static_cast<float>(pixelD[1]), 1e-4f);
}

TEST(CameraToPositiveDepthModel, ProjectConsistencyWithDepthFlip)
{
    Camera cam;
    cam.setIntrinsics(100.0, 100.0, 32.0, 24.0);
    std::array<double,9> R = {1,0,0,0,1,0,0,0,1};
    std::array<double,3> C = {0,0,0};
    cam.setPose(R, C);
    cam.setDepthAxisFlipped(true);

    PositiveDepthCameraModel pdm(cam);

    float px = 0.0f, py = 0.0f;
    bool ok = pdm.project(0.0f, 0.0f, -10.0f, px, py);
    EXPECT_TRUE(ok);

    double world[3] = {0.0, 0.0, -10.0};
    double pixelD[2] = {0.0, 0.0};
    EXPECT_TRUE(cam.projectWorldPoint(world, pixelD));
    EXPECT_NEAR(px, static_cast<float>(pixelD[0]), 1e-4f);
    EXPECT_NEAR(py, static_cast<float>(pixelD[1]), 1e-4f);
}
