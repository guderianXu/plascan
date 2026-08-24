#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "BundleAdjustSolver.h"
#include "BundleAdjustPlaMatrixConstraints.h"
#include "FramePinholeCamera.h"

namespace
{

xjw::FramePinholeCamera makeConstraintCamera(double center_x, double center_y)
{
    xjw::FramePinholeCamera camera;
    camera.setIntrinsics(960.0, 950.0, 512.0, 384.0);
    camera.setDistortion(-0.012, 0.0008, -0.00005, 0.0002, -0.00015);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{center_x, center_y, 0.0}});
    return camera;
}

std::array<double, 2> projectPoint(const xjw::FramePinholeCamera& camera,
                                   const std::array<double, 3>& point)
{
    const double world[3] = {point[0], point[1], point[2]};
    double pixel[2] = {0.0, 0.0};
    EXPECT_TRUE(camera.projectWorldPoint(world, pixel));
    return {{pixel[0], pixel[1]}};
}

double distance(const std::array<double, 3>& left,
                const std::array<double, 3>& right)
{
    double squared = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double delta = left[static_cast<std::size_t>(axis)] -
                             right[static_cast<std::size_t>(axis)];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

std::vector<xjw::BATrack> makeConstraintTracks(
    const std::vector<xjw::FramePinholeCamera>& truth_cameras,
    std::vector<std::array<double, 3>>* truth_points)
{
    std::vector<xjw::BATrack> tracks;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            const std::array<double, 3> truth{{
                -1.2 + 0.8 * column,
                -0.8 + 0.8 * row,
                10.0 + 0.3 * ((row + column) % 3)}};
            truth_points->push_back(truth);
            xjw::BATrack track;
            track.initialPoint = {{
                truth[0] + 0.12 * ((column % 2) ? 1.0 : -1.0),
                truth[1] + 0.09 * ((row % 2) ? -1.0 : 1.0),
                truth[2] + 0.18}};
            track.controlPointConstraints.push_back(
                {truth, 0.03, 1.0, static_cast<int>(tracks.size())});
            track.laserPlaneConstraints.push_back(
                {truth, {{0.0, 0.0, 1.0}}, 1.0, 0.0, 0});
            for (std::size_t camera_index = 0;
                 camera_index < truth_cameras.size();
                 ++camera_index)
            {
                const auto pixel = projectPoint(truth_cameras[camera_index], truth);
                track.observations.push_back(
                    {static_cast<int>(camera_index), pixel[0], pixel[1], 1.0});
            }
            tracks.push_back(std::move(track));
        }
    }
    return tracks;
}

xjw::BALaserRangeConstraint makeLaserShot(
    const std::vector<xjw::FramePinholeCamera>& truth_cameras,
    const std::array<double, 3>& truth_point)
{
    xjw::BALaserRangeConstraint shot;
    shot.cameraIndex = 2;
    shot.initialPoint = {{truth_point[0] + 0.3,
                          truth_point[1] - 0.2,
                          truth_point[2] + 0.25}};
    shot.sigmaRangeMeters = 0.03;
    shot.weight = 1.0;
    shot.leverArmCameraMeters = {{0.15, -0.05, 0.08}};
    const auto center = truth_cameras[2].cameraCenter();
    const std::array<double, 3> emitter{{
        center[0] + shot.leverArmCameraMeters[0],
        center[1] + shot.leverArmCameraMeters[1],
        center[2] + shot.leverArmCameraMeters[2]}};
    shot.observedRangeMeters = distance(truth_point, emitter);
    shot.pointMode = xjw::BALaserPointMode::Constrained;
    shot.pointPrior = truth_point;
    shot.pointPriorSqrtInformation = {{
        5.0, 0.0, 0.0,
        0.0, 5.0, 0.0,
        0.0, 0.0, 5.0}};
    shot.shotId = "plamatrix-parity-shot";
    for (int camera_index : {0, 1})
    {
        const auto pixel = projectPoint(
            truth_cameras[static_cast<std::size_t>(camera_index)], truth_point);
        shot.measuredImageObservations.push_back(
            {camera_index, pixel[0], pixel[1], 1.0});
    }
    return shot;
}

} // namespace

TEST(BundleAdjustPlaMatrixConstraintParityTest,
     GcpLidarScalePoseAndLaserRangeMatchAcrossBackends)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixCpu))
    {
        GTEST_SKIP() << "PlaMatrix backend is unavailable";
    }
    const std::vector<xjw::FramePinholeCamera> truth_cameras{
        makeConstraintCamera(-3.0, 0.0),
        makeConstraintCamera(3.0, 0.0),
        makeConstraintCamera(0.0, -2.5),
        makeConstraintCamera(0.0, 2.5)};
    auto initial_cameras = truth_cameras;
    const double delta_2[6] = {0.006, -0.008, 0.004, 0.16, -0.1, 0.14};
    const double delta_3[6] = {-0.005, 0.007, -0.003, -0.13, 0.09, -0.12};
    initial_cameras[2].applyDeltaPose(delta_2);
    initial_cameras[3].applyDeltaPose(delta_3);
    std::vector<std::array<double, 3>> truth_points;
    auto tracks = makeConstraintTracks(truth_cameras, &truth_points);

    xjw::BAOptions options;
    options.refineCameraPose = true;
    options.fixedCameraIndices = {0, 1};
    options.gaugePolicy = xjw::BAGaugePolicy::RequireExplicitGauge;
    options.enableControlPointConstraints = true;
    options.controlPointWeight = 40.0;
    options.controlPointHuberDeltaMeters = 0.0;
    options.enableLaserPlaneConstraints = true;
    options.laserPlaneWeight = 20.0;
    options.laserHuberDeltaMeters = 0.0;
    options.enableScaleBarConstraints = true;
    options.scaleBarWeight = 50.0;
    options.scaleBarHuberDeltaMeters = 0.0;
    options.scaleBarConstraints.push_back(
        {0, 1, distance(truth_points[0], truth_points[1]), 0.02, 1.0, 0});
    options.cameraPosePriors.resize(truth_cameras.size());
    for (std::size_t camera_index : {2u, 3u})
    {
        auto& prior = options.cameraPosePriors[camera_index];
        prior.enabled = true;
        prior.cameraCenter = truth_cameras[camera_index].cameraCenter();
        prior.cameraToWorldRotation =
            truth_cameras[camera_index].cameraToWorldRotation();
        prior.positionSigmaMeters = 0.08;
        prior.rotationSigmaDegrees = 1.0;
    }
    options.cameraPosePriorWeight = 20.0;
    options.cameraPosePriorHuberDelta = 0.0;
    options.cameraPlaneConstraint.enabled = true;
    options.cameraPlaneConstraint.point = {{0.0, 0.0, 0.0}};
    options.cameraPlaneConstraint.normal = {{0.0, 0.0, 1.0}};
    options.cameraPlaneConstraint.sigmaMeters = 0.1;
    options.cameraPlaneConstraint.weight = 10.0;
    options.cameraPlaneHuberDelta = 0.0;
    options.enableLaserRangeConstraints = true;
    options.laserRangeWeight = 10.0;
    options.laserRangeHuberDelta = 0.0;
    const std::array<double, 3> laser_truth{{0.35, -0.25, 9.6}};
    options.laserRangeConstraints.push_back(
        makeLaserShot(truth_cameras, laser_truth));
    options.enablePointFilter = false;
    options.maxIterations = 50;
    options.damping = 1e-3;
    options.stepTolerance = 1e-9;
    options.allowBackendFallback = false;

    auto plamatrix_options = options;
    plamatrix_options.backend = xjw::BABackend::PlaMatrixCpu;
    const auto plamatrix = xjw::BundleAdjust::optimizePoints(
        initial_cameras, tracks, plamatrix_options);
    ASSERT_TRUE(plamatrix.solutionUsable) << plamatrix.backendMessage;

    EXPECT_GT(plamatrix.controlPointConstraintCount, 0);
    EXPECT_GT(plamatrix.laserConstraintCount, 0);
    EXPECT_GT(plamatrix.scaleBarConstraintCount, 0);
    EXPECT_EQ(plamatrix.laserRangeConstraintCount, 1);

    std::vector<xjw::BABackend> backends;
    for (const auto backend : {
             xjw::BABackend::PlaMatrixCuda,
             xjw::BABackend::PlaMatrixOpenCl})
    {
        if (xjw::BundleAdjust::isBackendAvailable(backend))
        {
            backends.push_back(backend);
        }
    }
    for (const auto backend : backends)
    {
        auto backend_options = options;
        backend_options.backend = backend;
        const auto result = xjw::BundleAdjust::optimizePoints(
            initial_cameras, tracks, backend_options);
        ASSERT_TRUE(result.solutionUsable)
            << xjw::BundleAdjust::backendName(backend) << ": " << result.backendMessage;
        EXPECT_EQ(result.usedBackend, backend);
        EXPECT_FALSE(result.backendFallback);
        EXPECT_EQ(result.controlPointConstraintCount, plamatrix.controlPointConstraintCount);
        EXPECT_EQ(result.laserConstraintCount, plamatrix.laserConstraintCount);
        EXPECT_EQ(result.scaleBarConstraintCount, plamatrix.scaleBarConstraintCount);
        EXPECT_EQ(result.laserRangeConstraintCount, 1);
        EXPECT_NEAR(result.meanRmsAfter, plamatrix.meanRmsAfter, 2e-3)
            << "backend=" << xjw::BundleAdjust::backendName(backend)
            << ", iterations=" << result.points.front().iterations
            << ", accepted=" << result.plaMatrixAcceptedSteps
            << ", rejected=" << result.plaMatrixRejectedSteps
            << ", initial_cost=" << result.plaMatrixInitialCost
            << ", final_cost=" << result.plaMatrixFinalCost;
        for (std::size_t camera_index : {2u, 3u})
        {
            EXPECT_LT(distance(result.refinedCameras[camera_index].cameraCenter(),
                               plamatrix.refinedCameras[camera_index].cameraCenter()),
                      5e-3);
        }
        ASSERT_EQ(result.laserRangeShots.size(), 1u);
        ASSERT_EQ(plamatrix.laserRangeShots.size(), 1u);
        EXPECT_LT(distance(result.laserRangeShots[0].point,
                           plamatrix.laserRangeShots[0].point),
                  5e-3);
    }
}

TEST(BundleAdjustPlaMatrixConstraintJacobianTest,
     PosePriorAndLaserRangeMatchCentralDifferences)
{
    using xjw::detail::plamatrix_ba::ConstraintLinearization;
    using xjw::detail::plamatrix_ba::linearizeLaserRange;
    using xjw::detail::plamatrix_ba::linearizePosePrior;
    auto camera = makeConstraintCamera(0.4, -0.7);
    const double initial_delta[6] = {0.03, -0.02, 0.01, 0.0, 0.0, 0.0};
    camera.applyDeltaPose(initial_delta);

    xjw::BAOptions options;
    options.cameraPosePriorWeight = 3.5;
    options.cameraPosePriorHuberDelta = 0.0;
    options.laserRangeWeight = 2.5;
    options.laserRangeHuberDelta = 0.0;
    xjw::BACameraPosePrior prior;
    prior.enabled = true;
    prior.cameraCenter = {{0.3, -0.5, 0.2}};
    prior.cameraToWorldRotation = makeConstraintCamera(0.0, 0.0).cameraToWorldRotation();
    prior.positionSigmaMeters = 0.7;
    prior.rotationSigmaDegrees = 2.0;

    ConstraintLinearization pose;
    ASSERT_TRUE(linearizePosePrior(camera, prior, options, &pose));
    constexpr double epsilon = 1e-7;
    for (int parameter = 0; parameter < 6; ++parameter)
    {
        double delta[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        delta[parameter] = epsilon;
        auto plus_camera = camera;
        plus_camera.applyDeltaPose(delta);
        delta[parameter] = -epsilon;
        auto minus_camera = camera;
        minus_camera.applyDeltaPose(delta);
        ConstraintLinearization plus;
        ConstraintLinearization minus;
        ASSERT_TRUE(linearizePosePrior(plus_camera, prior, options, &plus));
        ASSERT_TRUE(linearizePosePrior(minus_camera, prior, options, &minus));
        for (int row = 0; row < pose.residualSize; ++row)
        {
            const double numerical =
                (plus.residual[static_cast<std::size_t>(row)] -
                 minus.residual[static_cast<std::size_t>(row)]) /
                (2.0 * epsilon);
            EXPECT_NEAR(
                pose.primaryJacobian[static_cast<std::size_t>(row * 9 + parameter)],
                numerical,
                2e-5);
        }
    }

    xjw::BALaserRangeConstraint shot;
    shot.cameraIndex = 0;
    shot.observedRangeMeters = 9.2;
    shot.sigmaRangeMeters = 0.4;
    shot.weight = 1.3;
    shot.leverArmCameraMeters = {{0.3, -0.15, 0.08}};
    const std::array<double, 3> point{{1.1, -0.2, 9.8}};
    ConstraintLinearization range;
    ASSERT_TRUE(linearizeLaserRange(camera, shot, point, options, &range));
    for (int parameter = 0; parameter < 6; ++parameter)
    {
        double delta[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        delta[parameter] = epsilon;
        auto plus_camera = camera;
        plus_camera.applyDeltaPose(delta);
        delta[parameter] = -epsilon;
        auto minus_camera = camera;
        minus_camera.applyDeltaPose(delta);
        ConstraintLinearization plus;
        ConstraintLinearization minus;
        ASSERT_TRUE(linearizeLaserRange(plus_camera, shot, point, options, &plus));
        ASSERT_TRUE(linearizeLaserRange(minus_camera, shot, point, options, &minus));
        const double numerical = (plus.residual[0] - minus.residual[0]) /
                                 (2.0 * epsilon);
        EXPECT_NEAR(range.primaryJacobian[static_cast<std::size_t>(parameter)],
                    numerical,
                    2e-6);
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        auto plus_point = point;
        auto minus_point = point;
        plus_point[static_cast<std::size_t>(axis)] += epsilon;
        minus_point[static_cast<std::size_t>(axis)] -= epsilon;
        ConstraintLinearization plus;
        ConstraintLinearization minus;
        ASSERT_TRUE(linearizeLaserRange(camera, shot, plus_point, options, &plus));
        ASSERT_TRUE(linearizeLaserRange(camera, shot, minus_point, options, &minus));
        const double numerical = (plus.residual[0] - minus.residual[0]) /
                                 (2.0 * epsilon);
        EXPECT_NEAR(range.pointJacobian[static_cast<std::size_t>(axis)],
                    numerical,
                    2e-6);
    }
}
