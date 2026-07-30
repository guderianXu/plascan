#include <gtest/gtest.h>

#include "BundleAdjustProjection.h"
#include "Camera.h"

#include <cmath>

TEST(BundleAdjustProjectionTest, MatchesCameraProjectWorldPointForTsaiCamera)
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 980.0, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{-2.0, 1.0, 0.0}});
    camera.setAxisDirections(-1, 1);
    camera.setDistortion(1.0e-4, -2.0e-7, 3.0e-10, 1.0e-5, -2.0e-5);

    const double world[3] = {0.3, -0.2, 40.0};
    double cameraPixel[2] = {0.0, 0.0};
    ASSERT_TRUE(camera.projectWorldPoint(world, cameraPixel));

    const auto projectionCamera = xjw::ba::makeProjectionCamera(camera);
    double projectionPixel[2] = {0.0, 0.0};
    ASSERT_TRUE(xjw::ba::project(projectionCamera, world, projectionPixel));

    EXPECT_NEAR(projectionPixel[0], cameraPixel[0], 1e-9);
    EXPECT_NEAR(projectionPixel[1], cameraPixel[1], 1e-9);
}

TEST(BundleAdjustProjectionTest, PoseDeltaProjectionMatchesCameraUpdate)
{
    xjw::Camera camera;
    camera.setIntrinsics(900.0, 870.0, 320.0, 240.0);
    camera.setPose({{0.995004165278, -0.099833416647, 0.0,
                     0.099833416647, 0.995004165278, 0.0,
                     0.0, 0.0, 1.0}},
                   {{1.0, -2.0, 0.5}});
    camera.setAxisDirections(-1, 1);
    camera.setDistortion(2.0e-4, -3.0e-7, 1.0e-10, 2.0e-5, -1.0e-5);

    const double delta[6] = {0.025, -0.018, 0.011, 0.12, -0.08, 0.04};
    const double world[3] = {2.5, -0.6, 24.0};

    xjw::Camera updated = camera;
    updated.applyDeltaPose(delta);
    double expected[2] = {0.0, 0.0};
    ASSERT_TRUE(updated.projectWorldPoint(world, expected));

    const auto projectionCamera = xjw::ba::makeProjectionCamera(camera);
    double actual[2] = {0.0, 0.0};
    ASSERT_TRUE(xjw::ba::projectWithPoseDelta(
        projectionCamera, delta, world, actual));

    EXPECT_NEAR(actual[0], expected[0], 1e-9);
    EXPECT_NEAR(actual[1], expected[1], 1e-9);
}

TEST(BundleAdjustProjectionTest, SharedFocalProjectionMatchesCameraUpdate)
{
    xjw::Camera camera;
    camera.setIntrinsics(800.0, 760.0, 400.0, 300.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{0.5, -0.25, 0.0}});

    const double delta[6] = {-0.01, 0.02, 0.005, 0.03, 0.02, -0.01};
    const double world[3] = {1.2, 0.4, 18.0};
    const double sharedFocalLog[1] = {std::log(960.0)};

    xjw::Camera updated = camera;
    updated.applyDeltaPose(delta);
    updated.setIntrinsics(960.0, 912.0, 400.0, 300.0);
    double expected[2] = {0.0, 0.0};
    ASSERT_TRUE(updated.projectWorldPoint(world, expected));

    const auto projectionCamera = xjw::ba::makeProjectionCamera(camera);
    double actual[2] = {0.0, 0.0};
    ASSERT_TRUE(xjw::ba::projectWithPoseDeltaAndSharedFocal(
        projectionCamera, delta, world, sharedFocalLog, actual));

    EXPECT_NEAR(actual[0], expected[0], 1e-9);
    EXPECT_NEAR(actual[1], expected[1], 1e-9);
}
