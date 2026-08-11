#include "FramePinholeCamera.h"
#include "DepthRayMetric.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

namespace xjw::mesh
{
namespace
{

FramePinholeCamera makeCamera()
{
    FramePinholeCamera camera;
    camera.setIntrinsics(100.0, 100.0, 0.0, 0.0);
    camera.setPose(
        {1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0},
        {0.0, 0.0, 0.0});
    return camera;
}

TEST(DepthRayMetricTest, CentrePixelUsesCameraZAsRayDistance)
{
    const DepthRayMetricSample sample =
        DepthRayMetric::evaluate(makeCamera(), {0.0, 0.0}, 5.0);

    ASSERT_TRUE(sample.valid);
    EXPECT_DOUBLE_EQ(sample.cameraZDepth, 5.0);
    EXPECT_NEAR(sample.rayDistancePerCameraZ, 1.0, 1.0e-12);
    EXPECT_NEAR(sample.rayDistance, 5.0, 1.0e-12);
    EXPECT_NEAR(sample.worldPoint[0], 0.0, 1.0e-12);
    EXPECT_NEAR(sample.worldPoint[1], 0.0, 1.0e-12);
    EXPECT_NEAR(sample.worldPoint[2], 5.0, 1.0e-12);
    EXPECT_NEAR(sample.unitWorldRay[2], 1.0, 1.0e-12);
    EXPECT_NEAR(sample.worldPixelFootprintX, 0.05, 2.0e-6);
    EXPECT_NEAR(sample.worldPixelFootprintY, 0.05, 2.0e-6);
    EXPECT_NEAR(sample.worldPixelFootprint, 0.05, 2.0e-6);
}

TEST(DepthRayMetricTest, OffAxisPixelSeparatesCameraZAndRayDistance)
{
    const DepthRayMetricSample sample =
        DepthRayMetric::evaluate(makeCamera(), {100.0, 0.0}, 5.0);

    ASSERT_TRUE(sample.valid);
    EXPECT_NEAR(sample.rayDistancePerCameraZ, std::sqrt(2.0), 1.0e-12);
    EXPECT_NEAR(sample.rayDistance, 5.0 * std::sqrt(2.0), 1.0e-12);
    EXPECT_NEAR(sample.unitWorldRay[0], 1.0 / std::sqrt(2.0), 1.0e-12);
    EXPECT_NEAR(sample.unitWorldRay[2], 1.0 / std::sqrt(2.0), 1.0e-12);

    std::array<double, 3> camera_z_offset_point{};
    double offset_ray_distance = 0.0;
    ASSERT_TRUE(DepthRayMetric::pointAtCameraZOffset(
        sample, 1.0, &camera_z_offset_point, &offset_ray_distance));
    EXPECT_NEAR(camera_z_offset_point[0], 6.0, 1.0e-12);
    EXPECT_NEAR(camera_z_offset_point[2], 6.0, 1.0e-12);
    EXPECT_NEAR(offset_ray_distance, 6.0 * std::sqrt(2.0), 1.0e-12);

    std::array<double, 3> ray_offset_point{};
    double offset_camera_z = 0.0;
    ASSERT_TRUE(DepthRayMetric::pointAtRayDistanceOffset(
        sample, std::sqrt(2.0), &ray_offset_point, &offset_camera_z));
    EXPECT_NEAR(offset_camera_z, 6.0, 1.0e-12);
    EXPECT_NEAR(ray_offset_point[0], 6.0, 1.0e-12);
    EXPECT_NEAR(ray_offset_point[2], 6.0, 1.0e-12);
}

TEST(DepthRayMetricTest, FlippedDepthAxisKeepsPositiveDepthConvention)
{
    FramePinholeCamera camera = makeCamera();
    camera.setCameraCenter({1.0, 2.0, 3.0});
    camera.setDepthAxisFlipped(true);

    const DepthRayMetricSample sample =
        DepthRayMetric::evaluate(camera, {0.0, 0.0}, 5.0);
    ASSERT_TRUE(sample.valid);
    EXPECT_NEAR(sample.worldPoint[0], 1.0, 1.0e-12);
    EXPECT_NEAR(sample.worldPoint[1], 2.0, 1.0e-12);
    EXPECT_NEAR(sample.worldPoint[2], -2.0, 1.0e-12);
    EXPECT_NEAR(sample.unitWorldRay[2], -1.0, 1.0e-12);

    std::array<double, 3> point{};
    double camera_z = 0.0;
    ASSERT_TRUE(DepthRayMetric::pointAtRayDistanceOffset(
        sample, 2.0, &point, &camera_z));
    EXPECT_NEAR(camera_z, 7.0, 1.0e-12);
    EXPECT_NEAR(point[2], -4.0, 1.0e-12);

    EXPECT_FALSE(DepthRayMetric::pointAtRayDistanceOffset(
        sample, -5.0, &point, &camera_z));
    EXPECT_FALSE(DepthRayMetric::pointAtCameraZOffset(
        sample, -5.0, &point));
}

TEST(DepthRayMetricTest, RejectsInvalidCameraDepthPixelAndOffset)
{
    const FramePinholeCamera invalid_camera;
    EXPECT_FALSE(DepthRayMetric::evaluate(
        invalid_camera, {0.0, 0.0}, 1.0).valid);

    const FramePinholeCamera camera = makeCamera();
    EXPECT_FALSE(DepthRayMetric::evaluate(camera, {0.0, 0.0}, 0.0).valid);
    EXPECT_FALSE(DepthRayMetric::evaluate(camera, {0.0, 0.0}, -1.0).valid);
    EXPECT_FALSE(DepthRayMetric::evaluate(
        camera,
        {std::numeric_limits<double>::infinity(), 0.0},
        1.0).valid);
    EXPECT_FALSE(DepthRayMetric::evaluate(
        camera,
        {0.0, 0.0},
        std::numeric_limits<double>::quiet_NaN()).valid);

    const DepthRayMetricSample sample =
        DepthRayMetric::evaluate(camera, {0.0, 0.0}, 1.0);
    ASSERT_TRUE(sample.valid);
    std::array<double, 3> point{};
    EXPECT_FALSE(DepthRayMetric::pointAtCameraZOffset(
        sample,
        std::numeric_limits<double>::infinity(),
        &point));
    EXPECT_FALSE(DepthRayMetric::pointAtRayDistanceOffset(
        sample,
        std::numeric_limits<double>::quiet_NaN(),
        &point));
    EXPECT_FALSE(DepthRayMetric::pointAtCameraZOffset(
        sample, 0.1, nullptr));
}

} // namespace
} // namespace xjw::mesh
