#include "geometry/ProjectionGeometry.h"

#include <gtest/gtest.h>

namespace
{

xjw::Camera makeCamera()
{
    xjw::Camera camera;
    camera.setIntrinsics(100.0, 200.0, 10.0, 20.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {0.0, 0.0, 0.0});
    camera.setAxisDirections(-1, 1);
    return camera;
}

} // namespace

TEST(ProjectionGeometryTest, ProjectsPhysicalFrontPointWithAxisDirections)
{
    const xjw::Camera camera = makeCamera();
    const xjw::ProjectionResult result =
        xjw::projectForReprojection(camera, {1.0, 2.0, 10.0});

    ASSERT_TRUE(result.success);
    EXPECT_FALSE(result.usedSignedFallback);
    EXPECT_DOUBLE_EQ(result.pixel[0], 0.0);
    EXPECT_DOUBLE_EQ(result.pixel[1], 60.0);
}

TEST(ProjectionGeometryTest, SignedFallbackMatchesCurrentCameraProjection)
{
    const xjw::Camera camera = makeCamera();
    const xjw::ProjectionResult result =
        xjw::projectForReprojection(camera, {1.0, 2.0, -10.0});

    ASSERT_TRUE(result.success);
    EXPECT_TRUE(result.usedSignedFallback);
    EXPECT_DOUBLE_EQ(result.pixel[0], 20.0);
    EXPECT_DOUBLE_EQ(result.pixel[1], -20.0);
    EXPECT_DOUBLE_EQ(
        xjw::reprojectionErrorPx(camera, {1.0, 2.0, -10.0}, {23.0, -16.0}),
        5.0);
}

TEST(ProjectionGeometryTest, FlippedDepthUsesNegativeCameraZAsPhysicalFront)
{
    xjw::Camera camera = makeCamera();
    camera.setDepthAxisFlipped(true);

    const xjw::ProjectionResult result =
        xjw::projectForReprojection(camera, {1.0, 2.0, -10.0});

    ASSERT_TRUE(result.success);
    EXPECT_FALSE(result.usedSignedFallback);
    EXPECT_DOUBLE_EQ(result.positiveDepth, 10.0);
}
