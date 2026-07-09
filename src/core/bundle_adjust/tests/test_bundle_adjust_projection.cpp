#include <gtest/gtest.h>

#include "BundleAdjustProjection.h"
#include "Camera.h"

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

