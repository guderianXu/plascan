#include "geometry/TriangulationQuality.h"

#include <gtest/gtest.h>

#include <cmath>

TEST(TriangulationQualityTest, ComputesMinimumCameraRayAngle)
{
    xjw::Camera left;
    left.setIntrinsics(100.0, 100.0, 0.0, 0.0);
    left.setPose({1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0},
                 {-1.0, 0.0, 0.0});
    xjw::Camera right = left;
    right.setCameraCenter({1.0, 0.0, 0.0});

    xjw::BATrack track;
    track.observations.push_back({0, 0.0, 0.0});
    track.observations.push_back({1, 0.0, 0.0});

    const double angle = xjw::minimumTriangulationAngleDeg(
        {left, right}, track, {0.0, 0.0, 10.0});

    EXPECT_NEAR(angle, 2.0 * std::atan(0.1) * 180.0 / 3.14159265358979323846, 1e-12);
}

TEST(TriangulationQualityTest, ComputesPairRmsReprojectionError)
{
    xjw::Camera camera;
    camera.setIntrinsics(100.0, 100.0, 0.0, 0.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {0.0, 0.0, 0.0});

    EXPECT_DOUBLE_EQ(
        xjw::pairRmsReprojectionErrorPx(
            camera, {3.0, 4.0}, camera, {0.0, 0.0}, {0.0, 0.0, 10.0}),
        std::sqrt(12.5));
}
