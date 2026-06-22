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

TEST(BundleAdjustControlPointConstraintTest, SoftPointConstraintReducesControlPointDistance)
{
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 12.0}};
    track.observations.push_back(xjw::BAObservation{0, 512.0, 384.0});
    track.observations.push_back(xjw::BAObservation{1, 512.0, 384.0});

    xjw::BAControlPointConstraint constraint;
    constraint.point = {{0.0, 0.0, 10.0}};
    constraint.sigmaMeters = 0.05;
    constraint.weight = 1.0;
    track.controlPointConstraints.push_back(constraint);

    xjw::BAOptions options;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableControlPointConstraints = true;
    options.controlPointHuberDeltaMeters = 10.0;
    options.maxIterations = 4;
    options.maxPointIterations = 12;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints({makeCamera(), makeCamera()}, {track}, options);

    ASSERT_EQ(result.points.size(), 1u);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.controlPointConstraintCount, 1);
    EXPECT_NEAR(result.controlPointRmsBeforeMeters, 2.0, 1e-9);
    EXPECT_LT(result.controlPointRmsAfterMeters, 0.05);
    EXPECT_LT(std::abs(result.points.front().point[2] - 10.0), 0.05);
}

TEST(BundleAdjustControlPointConstraintTest, DisabledPointConstraintLeavesDepthAmbiguousPointUnchanged)
{
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 12.0}};
    track.observations.push_back(xjw::BAObservation{0, 512.0, 384.0});
    track.observations.push_back(xjw::BAObservation{1, 512.0, 384.0});

    xjw::BAControlPointConstraint constraint;
    constraint.point = {{0.0, 0.0, 10.0}};
    constraint.sigmaMeters = 0.05;
    track.controlPointConstraints.push_back(constraint);

    xjw::BAOptions options;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableControlPointConstraints = false;
    options.maxIterations = 2;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints({makeCamera(), makeCamera()}, {track}, options);

    ASSERT_EQ(result.points.size(), 1u);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.controlPointConstraintCount, 0);
    EXPECT_NEAR(result.points.front().point[2], 12.0, 1e-9);
}

TEST(BundleAdjustScaleBarConstraintTest, SoftScaleBarConstraintReducesEndpointDistanceError)
{
    xjw::BATrack left;
    left.initialPoint = {{0.0, 0.0, 10.0}};
    left.observations.push_back(xjw::BAObservation{0, 512.0, 384.0});
    left.observations.push_back(xjw::BAObservation{1, 512.0, 384.0});

    xjw::BATrack right;
    right.initialPoint = {{12.0, 0.0, 10.0}};
    right.observations.push_back(xjw::BAObservation{0, 1712.0, 384.0});
    right.observations.push_back(xjw::BAObservation{1, 1712.0, 384.0});

    xjw::BAScaleBarConstraint scaleBar;
    scaleBar.trackIndexA = 0;
    scaleBar.trackIndexB = 1;
    scaleBar.measuredDistanceMeters = 10.0;
    scaleBar.sigmaMeters = 0.05;
    scaleBar.weight = 1.0;

    xjw::BAOptions options;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableScaleBarConstraints = true;
    options.scaleBarWeight = 1000.0;
    options.scaleBarHuberDeltaMeters = 10.0;
    options.scaleBarConstraints.push_back(scaleBar);
    options.maxIterations = 8;
    options.maxPointIterations = 20;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints({makeCamera(), makeCamera()},
                                                                   {left, right},
                                                                   options);

    ASSERT_EQ(result.points.size(), 2u);
    ASSERT_TRUE(result.points[0].valid);
    ASSERT_TRUE(result.points[1].valid);
    EXPECT_EQ(result.scaleBarConstraintCount, 1);
    EXPECT_NEAR(result.scaleBarRmsBeforeMeters, 2.0, 1e-9);
    EXPECT_LT(result.scaleBarRmsAfterMeters, 0.2);
}
