#include "geometry/TriangulationQuality.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

TEST(TriangulationQualityTest, ComputesMinimumCameraRayAngle)
{
    xjw::FramePinholeCamera left;
    left.setIntrinsics(100.0, 100.0, 0.0, 0.0);
    left.setPose({1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0},
                 {-1.0, 0.0, 0.0});
    xjw::FramePinholeCamera right = left;
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
    xjw::FramePinholeCamera camera;
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

TEST(TriangulationQualityTest, ReconstructionUncertaintyRespondsToBaseline)
{
    xjw::FramePinholeCamera left;
    left.setIntrinsics(800.0, 800.0, 0.0, 0.0);
    left.setPose({1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0},
                 {-1.0, 0.0, 0.0});
    xjw::FramePinholeCamera wideRight = left;
    wideRight.setCameraCenter({1.0, 0.0, 0.0});
    xjw::FramePinholeCamera narrowRight = left;
    narrowRight.setCameraCenter({-0.8, 0.0, 0.0});

    const std::array<double, 3> point{0.0, 0.0, 10.0};
    const double wide = xjw::reconstructionUncertainty(
        {{&left, 2.0}, {&wideRight, 2.0}}, point);
    const double narrow = xjw::reconstructionUncertainty(
        {{&left, 2.0}, {&narrowRight, 2.0}}, point);

    EXPECT_TRUE(std::isfinite(wide));
    EXPECT_TRUE(std::isfinite(narrow));
    EXPECT_GE(wide, 1.0);
    EXPECT_GT(narrow, wide);
}

TEST(TriangulationQualityTest, ProjectionAccuracyAveragesEveryObservationScale)
{
    xjw::FramePinholeCamera camera;
    const std::vector<xjw::TiePointQualityObservation> observations{
        {&camera, 1.0}, {&camera, 2.0}, {&camera, 3.0}};
    EXPECT_DOUBLE_EQ(xjw::projectionAccuracy(observations), 2.0);

    std::vector<xjw::TiePointQualityObservation> incomplete = observations;
    incomplete.back().measurementScale = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(xjw::projectionAccuracy(incomplete)));
}

TEST(TriangulationQualityTest, CleanTiePointQualityMatchesReferenceContract)
{
    std::vector<xjw::FramePinholeCamera> cameras(3);
    const std::array<double, 3> camera_x{-1.0, 0.0, 1.0};
    const std::array<double, 3> scales{1.0, 2.0, 3.0};
    const std::array<double, 3> point{0.0, 0.0, 5.0};
    std::vector<xjw::TiePointQualityObservation> observations;
    for (std::size_t index = 0; index < cameras.size(); ++index)
    {
        xjw::FramePinholeCamera& camera = cameras[index];
        camera.setIntrinsics(1000.0, 1000.0, 0.0, 0.0);
        camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, {camera_x[index], 0.0, 0.0});
        double projected[2]{};
        ASSERT_TRUE(camera.projectWorldPoint(point.data(), projected));
        observations.push_back({&camera, scales[index], {projected[0] + (index == 0 ? 2.0 : 0.0), projected[1]}});
    }

    const xjw::CleanTiePointQuality quality = xjw::evaluateCleanTiePointQuality(observations, point);

    EXPECT_DOUBLE_EQ(quality.reprojectionError, 2.0);
    EXPECT_EQ(quality.imageCount, 3U);
    EXPECT_DOUBLE_EQ(quality.projectionAccuracy, 2.0);
    ASSERT_TRUE(quality.hasProjectionGeometry);
    EXPECT_NEAR(quality.reconstructionUncertainty, std::sqrt(37.5), 1.0e-9);

    observations.front().measurementScale = 0.0;
    const xjw::CleanTiePointQuality zeroScale = xjw::evaluateCleanTiePointQuality(observations, point);
    EXPECT_DOUBLE_EQ(zeroScale.reprojectionError, 2.0);
    EXPECT_DOUBLE_EQ(zeroScale.projectionAccuracy, 5.0 / 3.0);
    EXPECT_NEAR(zeroScale.reconstructionUncertainty, quality.reconstructionUncertainty, 1.0e-12);
}
