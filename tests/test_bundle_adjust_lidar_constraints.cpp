#include <gtest/gtest.h>

#include <plabundle/solver.h>

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace
{

    plabundle::FrameCamera makeCamera()
    {
        plabundle::FrameCamera camera;
        camera.cameraToWorldRotation = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        camera.cameraCenter = {0.0, 0.0, 0.0};
        camera.focalXPixels = 1000.0;
        camera.focalYPixels = 1000.0;
        camera.principalXPixel = 512.0;
        camera.principalYPixel = 384.0;
        camera.imageSize = plabundle::ImageSize{1024, 768};
        return camera;
    }

    plabundle::Track makeDepthAmbiguousTrack()
    {
        plabundle::Track track;
        track.initialPoint = {{0.0, 0.0, 12.0}};
        track.observations.push_back({0, 512.0, 384.0});
        track.observations.push_back({1, 512.0, 384.0});

        plabundle::LaserPlaneConstraint constraint;
        constraint.point = {{0.0, 0.0, 10.0}};
        constraint.normal = {{0.0, 0.0, 1.0}};
        constraint.weight = 5.0;
        constraint.initialSignedDistance = 2.0;
        track.laserPlaneConstraints.push_back(constraint);
        return track;
    }

    plabundle::Result solve(const std::vector<plabundle::FrameCamera>& cameras,
                            std::vector<plabundle::Track> tracks,
                            const plabundle::SolveOptions& options,
                            std::vector<plabundle::ScaleBarConstraint> scale_bars = {})
    {
        plabundle::Problem problem;
        problem.cameras = cameras;
        problem.tracks = std::move(tracks);
        problem.scaleBarConstraints = std::move(scale_bars);
        return plabundle::Solver().solve(problem, options);
    }

    double pointToLaserPlaneDistance(const std::array<double, 3>& point)
    {
        return std::abs(point[2] - 10.0);
    }

} // namespace

TEST(BundleAdjustLidarConstraintTest, LaserPlaneConstraintReducesPointToPlaneDistance)
{
    const std::vector<plabundle::FrameCamera> cameras{makeCamera(), makeCamera()};
    const std::vector<plabundle::Track> tracks{makeDepthAmbiguousTrack()};

    plabundle::SolveOptions options;
    options.calibration.refineCameraPose = false;
    options.solver.enablePointFilter = false;
    options.constraints.laserPlaneWeight = 1.0;
    options.constraints.laserHuberDeltaMeters = 10.0;
    options.solver.maxIterations = 4;

    const plabundle::Result result = solve(cameras, tracks, options);

    ASSERT_EQ(result.points.size(), 1U);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.quality.laserConstraintCount, 1);
    EXPECT_NEAR(result.quality.laserRmsBeforeMeters, 2.0, 1e-9);
    EXPECT_LT(result.quality.laserRmsAfterMeters, 0.05);
    EXPECT_LT(pointToLaserPlaneDistance(result.points.front().point), 0.05);
}

TEST(BundleAdjustLidarConstraintTest, OmittedLaserPlaneConstraintLeavesDepthAmbiguousPointUnchanged)
{
    const std::vector<plabundle::FrameCamera> cameras{makeCamera(), makeCamera()};
    std::vector<plabundle::Track> tracks{makeDepthAmbiguousTrack()};
    tracks.front().laserPlaneConstraints.clear();

    plabundle::SolveOptions options;
    options.calibration.refineCameraPose = false;
    options.solver.enablePointFilter = false;
    options.solver.maxIterations = 2;

    const plabundle::Result result = solve(cameras, tracks, options);

    ASSERT_EQ(result.points.size(), 1U);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.quality.laserConstraintCount, 0);
    EXPECT_NEAR(result.points.front().point[2], 12.0, 1e-9);
}

TEST(BundleAdjustControlPointConstraintTest, SoftPointConstraintReducesControlPointDistance)
{
    plabundle::Track track;
    track.initialPoint = {{0.0, 0.0, 12.0}};
    track.observations.push_back({0, 512.0, 384.0});
    track.observations.push_back({1, 512.0, 384.0});

    plabundle::ControlPointConstraint constraint;
    constraint.point = {{0.0, 0.0, 10.0}};
    constraint.sigmaMeters = 0.05;
    constraint.weight = 1.0;
    track.controlPointConstraints.push_back(constraint);

    plabundle::SolveOptions options;
    options.calibration.refineCameraPose = false;
    options.solver.enablePointFilter = false;
    options.constraints.controlPointHuberDeltaMeters = 10.0;
    options.solver.maxIterations = 4;

    const plabundle::Result result = solve({makeCamera(), makeCamera()}, {track}, options);

    ASSERT_EQ(result.points.size(), 1U);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.quality.controlPointConstraintCount, 1);
    EXPECT_NEAR(result.quality.controlPointRmsBeforeMeters, 2.0, 1e-9);
    EXPECT_LT(result.quality.controlPointRmsAfterMeters, 0.05);
    EXPECT_LT(std::abs(result.points.front().point[2] - 10.0), 0.05);
}

TEST(BundleAdjustControlPointConstraintTest, OmittedPointConstraintLeavesDepthAmbiguousPointUnchanged)
{
    plabundle::Track track;
    track.initialPoint = {{0.0, 0.0, 12.0}};
    track.observations.push_back({0, 512.0, 384.0});
    track.observations.push_back({1, 512.0, 384.0});

    plabundle::SolveOptions options;
    options.calibration.refineCameraPose = false;
    options.solver.enablePointFilter = false;
    options.solver.maxIterations = 2;

    const plabundle::Result result = solve({makeCamera(), makeCamera()}, {track}, options);

    ASSERT_EQ(result.points.size(), 1U);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.quality.controlPointConstraintCount, 0);
    EXPECT_NEAR(result.points.front().point[2], 12.0, 1e-9);
}

TEST(BundleAdjustScaleBarConstraintTest, SoftScaleBarConstraintReducesEndpointDistanceError)
{
    plabundle::Track left;
    left.initialPoint = {{0.0, 0.0, 10.0}};
    left.observations.push_back({0, 512.0, 384.0});
    left.observations.push_back({1, 512.0, 384.0});

    plabundle::Track right;
    right.initialPoint = {{12.0, 0.0, 10.0}};
    right.observations.push_back({0, 1712.0, 384.0});
    right.observations.push_back({1, 1712.0, 384.0});

    plabundle::ScaleBarConstraint scale_bar;
    scale_bar.trackIndexA = 0;
    scale_bar.trackIndexB = 1;
    scale_bar.measuredDistanceMeters = 10.0;
    scale_bar.sigmaMeters = 0.05;
    scale_bar.weight = 1.0;

    plabundle::SolveOptions options;
    options.calibration.refineCameraPose = false;
    options.solver.enablePointFilter = false;
    options.constraints.scaleBarWeight = 1000.0;
    options.constraints.scaleBarHuberDeltaMeters = 10.0;
    options.solver.maxIterations = 8;

    const plabundle::Result result = solve({makeCamera(), makeCamera()}, {left, right}, options, {scale_bar});

    ASSERT_EQ(result.points.size(), 2U);
    ASSERT_TRUE(result.points[0].valid);
    ASSERT_TRUE(result.points[1].valid);
    EXPECT_EQ(result.quality.scaleBarConstraintCount, 1);
    EXPECT_NEAR(result.quality.scaleBarRmsBeforeMeters, 2.0, 1e-9);
    EXPECT_LT(result.quality.scaleBarRmsAfterMeters, 0.2);
}
