#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "Camera.h"

#include <array>
#include <cmath>
#include <vector>

namespace
{

xjw::Camera makeCamera()
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{0.0, 0.0, 0.0}});
    return camera;
}

xjw::BATrack makeDepthAmbiguousTrack()
{
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 12.0}};
    track.observations.push_back(xjw::BAObservation{0, 512.0, 384.0});
    track.observations.push_back(xjw::BAObservation{1, 512.0, 384.0});

    xjw::BALaserPlaneConstraint constraint;
    constraint.point = {{0.0, 0.0, 10.0}};
    constraint.normal = {{0.0, 0.0, 1.0}};
    constraint.weight = 5.0;
    constraint.initialSignedDistance = 2.0;
    track.laserPlaneConstraints.push_back(constraint);
    return track;
}

double pointToLaserPlaneDistance(const std::array<double, 3> &point)
{
    return std::abs(point[2] - 10.0);
}

} // namespace

TEST(BundleAdjustLidarConstraintTest, LaserPlaneConstraintReducesPointToPlaneDistance)
{
    const std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    const std::vector<xjw::BATrack> tracks{makeDepthAmbiguousTrack()};

    xjw::BAOptions options;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableLaserPlaneConstraints = true;
    options.laserPlaneWeight = 1.0;
    options.laserHuberDeltaMeters = 10.0;
    options.maxIterations = 4;
    options.maxPointIterations = 12;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);

    ASSERT_EQ(result.points.size(), 1u);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.laserConstraintCount, 1);
    EXPECT_NEAR(result.laserRmsBeforeMeters, 2.0, 1e-9);
    EXPECT_LT(result.laserRmsAfterMeters, 0.05);
    EXPECT_LT(pointToLaserPlaneDistance(result.points.front().point), 0.05);
}

TEST(BundleAdjustLidarConstraintTest, DisabledLaserPlaneConstraintLeavesDepthAmbiguousPointUnchanged)
{
    const std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    const std::vector<xjw::BATrack> tracks{makeDepthAmbiguousTrack()};

    xjw::BAOptions options;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableLaserPlaneConstraints = false;
    options.maxIterations = 2;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);

    ASSERT_EQ(result.points.size(), 1u);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.laserConstraintCount, 0);
    EXPECT_NEAR(result.points.front().point[2], 12.0, 1e-9);
}
