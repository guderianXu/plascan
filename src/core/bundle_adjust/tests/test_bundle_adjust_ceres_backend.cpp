#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "Camera.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{

xjw::Camera makeCamera(double cx, double cy, double cz)
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{cx, cy, cz}});
    return camera;
}

bool projectPoint(const xjw::Camera &camera, const std::array<double, 3> &point, double *u, double *v)
{
    const double world[3] = {point[0], point[1], point[2]};
    double pixel[2] = {0.0, 0.0};
    if (!camera.projectWorldPoint(world, pixel))
    {
        return false;
    }
    *u = pixel[0];
    *v = pixel[1];
    return true;
}

double distance3d(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::array<double, 3> laserOrigin(
    const xjw::Camera &camera,
    const std::array<double, 3> &leverArmCameraMeters)
{
    const std::array<double, 3> center = camera.cameraCenter();
    const std::array<double, 9> rotation = camera.cameraToWorldRotation();
    std::array<double, 3> origin = center;
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            origin[static_cast<size_t>(row)] +=
                rotation[static_cast<size_t>(row * 3 + column)] *
                leverArmCameraMeters[static_cast<size_t>(column)];
        }
    }
    return origin;
}

xjw::BATrack makeTrack(const std::vector<xjw::Camera> &cameras,
                       const std::array<double, 3> &truth,
                       const std::array<double, 3> &initial)
{
    xjw::BATrack track;
    track.initialPoint = initial;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        double u = 0.0;
        double v = 0.0;
        if (projectPoint(cameras[i], truth, &u, &v))
        {
            track.observations.push_back(
                xjw::BAObservation{static_cast<int>(i), u, v, 1.0});
        }
    }
    return track;
}

} // namespace

TEST(BundleAdjustCeresBackendTest, CeresCpuBackendOptimizesPointAndReportsBackend)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
    };
    const std::array<double, 3> truth{{0.5, -0.4, 42.0}};
    const std::array<double, 3> initial{{truth[0] + 2.0, truth[1] - 1.5, truth[2] + 4.0}};
    const xjw::BATrack track = makeTrack(cameras, truth, initial);
    ASSERT_GE(track.observations.size(), 3u);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 25;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    ASSERT_EQ(result.points.size(), 1u);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.requestedBackend, xjw::BABackend::CeresCpu);
    EXPECT_EQ(result.usedBackend, xjw::BABackend::CeresCpu);
    EXPECT_FALSE(result.usedGpu);
    EXPECT_FALSE(result.backendFallback);
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::Success);
    EXPECT_TRUE(result.solutionUsable);
    EXPECT_GT(result.ceresSuccessfulSteps, 0);
    EXPECT_LT(result.ceresFinalCost, result.ceresInitialCost);
    EXPECT_LT(result.meanRmsAfter, result.meanRmsBefore);
    EXPECT_LT(distance3d(result.points.front().point, truth), distance3d(initial, truth));
}

TEST(BundleAdjustCeresBackendTest, StatisticalLaserWeightReducesPointToPlaneDistance)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-2.0, 0.0, 0.0),
        makeCamera(2.0, 0.0, 0.0),
        makeCamera(0.0, 2.0, 0.0),
    };
    const std::array<double, 3> initial{{0.5, 0.2, 12.0}};
    xjw::BATrack track = makeTrack(cameras, initial, initial);
    track.laserPlaneConstraints.push_back(
        {{{0.0, 0.0, 10.0}}, {{0.0, 0.0, 1.0}}, 1.0, 2.0, 0});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableLaserPlaneConstraints = true;
    options.laserPlaneWeight = 40000.0;
    options.laserHuberDeltaMeters = 0.02;
    options.maxIterations = 30;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_EQ(result.laserConstraintCount, 1);
    EXPECT_GT(result.ceresSuccessfulSteps, 0);
    EXPECT_LT(result.laserRmsAfterMeters, result.laserRmsBeforeMeters);
}

TEST(BundleAdjustCeresBackendTest, StationaryNonzeroResidualIsUsable)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-2.0, 0.0, 0.0),
        makeCamera(2.0, 0.0, 0.0),
    };
    const std::array<double, 3> truth{{0.0, 0.0, 10.0}};
    xjw::BATrack track = makeTrack(cameras, truth, truth);
    ASSERT_EQ(track.observations.size(), 2U);
    track.observations[0].v += 1.0;
    track.observations[1].v -= 1.0;

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 10;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_TRUE(result.solutionUsable) << result.backendMessage;
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::Success);
    EXPECT_GT(result.ceresInitialCost, 0.0);
    EXPECT_NEAR(result.ceresFinalCost, result.ceresInitialCost, 1.0e-12);
}

TEST(BundleAdjustCeresBackendTest, InitialGrossTrackGateKeepsPathologicalTrackOutOfProblem)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-2.0, 0.0, 0.0),
        makeCamera(2.0, 0.0, 0.0),
    };
    const std::array<double, 3> truth{{0.0, 0.0, 10.0}};
    xjw::BATrack validTrack = makeTrack(cameras, truth, {{0.1, 0.0, 10.5}});
    xjw::BATrack grossTrack = makeTrack(cameras, truth, truth);
    for (xjw::BAObservation &observation : grossTrack.observations)
    {
        observation.u += 1000.0;
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxCeresInitialTrackRms = 100.0;
    options.maxIterations = 15;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {validTrack, grossTrack}, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    EXPECT_EQ(result.ceresRejectedInitialTracks, 1);
    ASSERT_EQ(result.points.size(), 2u);
    EXPECT_TRUE(result.points[0].valid);
    EXPECT_FALSE(result.points[1].valid);
}

TEST(BundleAdjustCeresBackendTest, ProgressCancellationDoesNotPublishPartialSolution)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
    };
    const std::array<double, 3> truth{{0.5, -0.4, 42.0}};
    const std::array<double, 3> initial{{truth[0] + 2.0, truth[1] - 1.5, truth[2] + 4.0}};
    const xjw::BATrack track = makeTrack(cameras, truth, initial);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 25;
    options.progressCallback = [](int, int, double, int)
    {
        return false;
    };

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    ASSERT_EQ(result.points.size(), 1u);
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::Cancelled);
    EXPECT_FALSE(result.solutionUsable);
    EXPECT_EQ(result.points.front().point, initial);
    EXPECT_EQ(result.refinedCameras.size(), cameras.size());
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        EXPECT_EQ(result.refinedCameras[i].cameraCenter(), cameras[i].cameraCenter());
        EXPECT_EQ(result.refinedCameras[i].cameraToWorldRotation(),
                  cameras[i].cameraToWorldRotation());
    }
}

TEST(BundleAdjustCeresBackendTest, ProgressNeverExceedsConfiguredIterationCount)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
    };
    const std::array<double, 3> truth{{0.5, -0.4, 42.0}};
    const std::array<double, 3> initial{{truth[0] + 2.0, truth[1] - 1.5, truth[2] + 4.0}};
    const xjw::BATrack track = makeTrack(cameras, truth, initial);

    int callbackCount = 0;
    int largestCurrentIteration = 0;
    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 1;
    options.progressCallback =
        [&callbackCount, &largestCurrentIteration](int current, int maximum, double, int)
    {
        ++callbackCount;
        largestCurrentIteration = std::max(largestCurrentIteration, current);
        EXPECT_LE(current, maximum);
        EXPECT_EQ(maximum, 1);
        return true;
    };

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_TRUE(result.solutionUsable);
    EXPECT_EQ(callbackCount, 1);
    EXPECT_EQ(largestCurrentIteration, 1);
}

TEST(BundleAdjustCeresBackendTest, CeresPointOnlyUsesDenseQrSolver)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
        makeCamera(0.0, -8.0, 0.0),
    };
    std::vector<xjw::BATrack> tracks;
    for (int i = 0; i < 16; ++i)
    {
        const double x = (static_cast<double>(i % 4) - 1.5) * 0.25;
        const double y = (static_cast<double>(i / 4) - 1.5) * 0.25;
        const std::array<double, 3> truth{{x, y, 40.0 + static_cast<double>(i % 3)}};
        tracks.push_back(makeTrack(cameras, truth, {{x + 0.5, y - 0.4, truth[2] + 1.0}}));
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 3;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);

    EXPECT_EQ(result.usedBackend, xjw::BABackend::CeresCpu);
    EXPECT_EQ(result.ceresLinearSolverName, "dense_qr_cpu");
    EXPECT_EQ(result.optimizedTracks, static_cast<int>(tracks.size()));
    EXPECT_LT(result.meanRmsAfter, result.meanRmsBefore);
}

TEST(BundleAdjustCeresBackendTest, AutoDiffPosePriorStabilizesCameraCenter)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> truthCameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
        makeCamera(0.0, 1.0, 0.0),
    };
    std::vector<xjw::Camera> initialCameras = truthCameras;
    initialCameras[1] = makeCamera(1.6, 0.2, 0.0);

    std::vector<xjw::BATrack> tracks;
    for (int index = 0; index < 36; ++index)
    {
        const std::array<double, 3> truth{{
            (static_cast<double>(index % 6) - 2.5) * 0.15,
            (static_cast<double>(index / 6) - 2.5) * 0.15,
            8.0 + static_cast<double>(index % 3) * 0.1,
        }};
        tracks.push_back(makeTrack(truthCameras, truth, truth));
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = true;
    options.fixedCameraIndices = {0};
    options.enablePointFilter = false;
    options.maxIterations = 20;
    options.cameraPosePriors.resize(initialCameras.size());
    options.cameraPosePriors[1].enabled = true;
    options.cameraPosePriors[1].cameraCenter =
        truthCameras[1].cameraCenter();
    options.cameraPosePriors[1].cameraToWorldRotation =
        truthCameras[1].cameraToWorldRotation();
    options.cameraPosePriors[1].positionSigmaMeters = 0.05;
    options.cameraPosePriors[1].rotationSigmaDegrees = 1.0;
    options.cameraPosePriorWeight = 100.0;

    const double before = distance3d(
        initialCameras[1].cameraCenter(),
        truthCameras[1].cameraCenter());
    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(
            initialCameras, tracks, options);

    ASSERT_TRUE(result.solutionUsable);
    ASSERT_EQ(result.refinedCameras.size(), initialCameras.size());
    const double after = distance3d(
        result.refinedCameras[1].cameraCenter(),
        truthCameras[1].cameraCenter());
    EXPECT_LT(after, before * 0.1);
}

TEST(BundleAdjustCeresBackendTest, CameraPlaneConstraintOnlyRemovesNormalDrift)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    std::vector<xjw::Camera> cameras{
        makeCamera(-2.0, 0.0, 0.0),
        makeCamera(2.0, 0.0, 0.0),
        makeCamera(0.0, 2.0, 1.2),
        makeCamera(0.0, -2.0, -0.8),
    };
    std::vector<xjw::BATrack> tracks;
    for (int index = 0; index < 36; ++index)
    {
        const std::array<double, 3> point{{
            (static_cast<double>(index % 6) - 2.5) * 0.12,
            (static_cast<double>(index / 6) - 2.5) * 0.12,
            12.0 + static_cast<double>(index % 3) * 0.1,
        }};
        tracks.push_back(makeTrack(cameras, point, point));
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = true;
    options.gaugePolicy = xjw::BAGaugePolicy::CallerManaged;
    options.fixedCameraIndices = {0, 1};
    options.enablePointFilter = false;
    options.maxIterations = 30;
    options.cameraPlaneConstraint.enabled = true;
    options.cameraPlaneConstraint.point = {{0.0, 0.0, 0.0}};
    options.cameraPlaneConstraint.normal = {{0.0, 0.0, 1.0}};
    options.cameraPlaneConstraint.sigmaMeters = 0.05;
    options.cameraPlaneConstraint.weight = 100.0;
    options.cameraPlaneHuberDelta = 0.0;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, tracks, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_EQ(result.refinedCameras.size(), cameras.size());
    EXPECT_LT(std::abs(result.refinedCameras[2].cameraCenter()[2]), 0.05);
    EXPECT_LT(std::abs(result.refinedCameras[3].cameraCenter()[2]), 0.05);
}

TEST(BundleAdjustCeresBackendTest, CameraLayerReferencePreservesLegitimateNonPlanarTrajectory)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    std::vector<xjw::Camera> cameras{
        makeCamera(-2.0, 0.0, 0.0),
        makeCamera(2.0, 0.0, 0.0),
        makeCamera(0.0, 2.0, 1.2),
        makeCamera(0.0, -2.0, -0.8),
    };
    std::vector<xjw::BATrack> tracks;
    for (int index = 0; index < 36; ++index)
    {
        const std::array<double, 3> point{{
            (static_cast<double>(index % 6) - 2.5) * 0.12,
            (static_cast<double>(index / 6) - 2.5) * 0.12,
            12.0 + static_cast<double>(index % 3) * 0.1,
        }};
        tracks.push_back(makeTrack(cameras, point, point));
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = true;
    options.gaugePolicy = xjw::BAGaugePolicy::CallerManaged;
    options.fixedCameraIndices = {0, 1};
    options.enablePointFilter = false;
    options.maxIterations = 30;
    options.cameraPlaneConstraint.enabled = true;
    options.cameraPlaneConstraint.point = {{0.0, 0.0, 0.0}};
    options.cameraPlaneConstraint.normal = {{0.0, 0.0, 1.0}};
    options.cameraPlaneConstraint.referenceSignedDistances = {0.0, 0.0, 1.2, -0.8};
    options.cameraPlaneConstraint.sigmaMeters = 0.05;
    options.cameraPlaneConstraint.weight = 100.0;
    options.cameraPlaneHuberDelta = 0.0;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, tracks, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_EQ(result.refinedCameras.size(), cameras.size());
    EXPECT_NEAR(result.refinedCameras[2].cameraCenter()[2], 1.2, 0.05);
    EXPECT_NEAR(result.refinedCameras[3].cameraCenter()[2], -0.8, 0.05);
}

TEST(BundleAdjustCeresBackendTest, LegacyRequestDoesNotSilentlyIgnoreCameraPlaneConstraint)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
        makeCamera(0.0, 1.0, 0.4),
    };
    std::vector<xjw::BATrack> tracks;
    for (int index = 0; index < 12; ++index)
    {
        const std::array<double, 3> point{{
            (static_cast<double>(index % 4) - 1.5) * 0.1,
            (static_cast<double>(index / 4) - 1.0) * 0.1,
            10.0,
        }};
        tracks.push_back(makeTrack(cameras, point, point));
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::LegacyCpu;
    options.refineCameraPose = true;
    options.gaugePolicy = xjw::BAGaugePolicy::CallerManaged;
    options.fixedCameraIndices = {0, 1};
    options.enablePointFilter = false;
    options.cameraPlaneConstraint.enabled = true;
    options.cameraPlaneConstraint.sigmaMeters = 0.05;
    options.cameraPlaneConstraint.weight = 100.0;
    options.cameraPlaneHuberDelta = 0.0;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, tracks, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    EXPECT_EQ(result.requestedBackend, xjw::BABackend::LegacyCpu);
    EXPECT_EQ(result.usedBackend, xjw::BABackend::CeresCpu);
    EXPECT_TRUE(result.backendFallback);
    EXPECT_LT(std::abs(result.refinedCameras[2].cameraCenter()[2]), 0.05);
}

TEST(BundleAdjustCeresBackendTest, CeresSkipsTracksThatCannotConstrainProblem)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
    };
    const std::array<double, 3> truth{{0.5, -0.4, 42.0}};
    xjw::BATrack validTrack = makeTrack(cameras, truth, {{1.0, -0.8, 44.0}});
    validTrack.observations.push_back({0, 512.0, 384.0, 0.0});
    validTrack.observations.push_back(
        {1, 512.0, 384.0, std::numeric_limits<double>::quiet_NaN()});

    xjw::BATrack nonFiniteInitial = validTrack;
    nonFiniteInitial.initialPoint = {{std::numeric_limits<double>::quiet_NaN(), 0.0, 10.0}};

    xjw::BATrack singleCameraTrack;
    singleCameraTrack.initialPoint = {{0.0, 0.0, 30.0}};
    singleCameraTrack.observations.push_back(xjw::BAObservation{0, 512.0, 384.0, 1.0});
    singleCameraTrack.observations.push_back(xjw::BAObservation{0, 513.0, 385.0, 1.0});

    const std::vector<xjw::BATrack> tracks{validTrack, nonFiniteInitial, singleCameraTrack};

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 5;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);

    ASSERT_EQ(result.points.size(), tracks.size());
    EXPECT_TRUE(result.points[0].valid);
    EXPECT_FALSE(result.points[1].valid);
    EXPECT_FALSE(result.points[2].valid);
    EXPECT_EQ(result.optimizedTracks, 1);
    EXPECT_EQ(result.observationCount, 3);
    EXPECT_EQ(result.ceresLinearSolverName, "dense_qr_cpu");
}

TEST(BundleAdjustCeresBackendTest, LargePointOnlyCeresRequestFallsBackToLegacy)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
    };
    const xjw::BATrack track = makeTrack(cameras, {{0.5, -0.4, 42.0}}, {{1.0, -0.8, 44.0}});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxCeresPointOnlyObservations = 1;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_EQ(result.requestedBackend, xjw::BABackend::CeresCpu);
    EXPECT_EQ(result.usedBackend, xjw::BABackend::LegacyCpu);
    EXPECT_TRUE(result.backendFallback);
    EXPECT_NE(result.backendMessage.find("point-only Ceres"), std::string::npos);
}

TEST(BundleAdjustCeresBackendTest, CeresCudaRequestFallsBackWhenCudaSolverIsUnavailable)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
    };
    const std::array<double, 3> truth{{0.2, 0.1, 35.0}};
    const xjw::BATrack track = makeTrack(cameras, truth, {{1.0, -1.0, 40.0}});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCuda;
    options.minCeresCudaCameras = 1;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 10;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_EQ(result.requestedBackend, xjw::BABackend::CeresCuda);
    if (result.usedBackend == xjw::BABackend::CeresCuda)
    {
        EXPECT_TRUE(result.usedGpu);
        EXPECT_FALSE(result.backendFallback);
    }
    else
    {
        EXPECT_NE(result.usedBackend, xjw::BABackend::CeresCuda);
        EXPECT_FALSE(result.usedGpu);
        EXPECT_TRUE(result.backendFallback);
        EXPECT_NE(result.backendMessage.find("CUDA"), std::string::npos);
    }
}

TEST(BundleAdjustCeresBackendTest, NativeCudaRequestFallsBackWhenBackendUnavailable)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };

    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 5.0}};
    track.observations.push_back(xjw::BAObservation{0, 320.0, 240.0, 1.0});
    track.observations.push_back(xjw::BAObservation{1, 300.0, 240.0, 1.0});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::NativeCuda;
    options.refineCameraPose = true;
    options.allowBackendFallback = true;
    options.maxIterations = 1;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_EQ(result.requestedBackend, xjw::BABackend::NativeCuda);
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::NativeCuda))
    {
        EXPECT_EQ(result.usedBackend, xjw::BABackend::LegacyCpu);
        EXPECT_TRUE(result.backendFallback);
        EXPECT_FALSE(result.backendMessage.empty());
        EXPECT_NE(result.backendMessage.find("native_cuda"), std::string::npos);
    }
}

TEST(BundleAdjustCeresBackendTest, CeresCpuReportsControlPointConstraintStats)
{
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 12.0}};
    track.observations.push_back(xjw::BAObservation{0, 512.0, 384.0, 1.0});
    track.observations.push_back(xjw::BAObservation{1, 512.0, 384.0, 1.0});

    xjw::BAControlPointConstraint constraint;
    constraint.point = {{0.0, 0.0, 10.0}};
    constraint.sigmaMeters = 0.05;
    constraint.weight = 1.0;
    track.controlPointConstraints.push_back(constraint);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableControlPointConstraints = true;
    options.controlPointHuberDeltaMeters = 10.0;
    options.maxIterations = 25;

    const std::vector<xjw::Camera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(0.0, 0.0, 0.0),
    };
    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    ASSERT_EQ(result.points.size(), 1u);
    ASSERT_TRUE(result.points.front().valid);
    EXPECT_EQ(result.controlPointConstraintCount, 1);
    EXPECT_NEAR(result.controlPointRmsBeforeMeters, 2.0, 1e-9);
    EXPECT_LT(result.controlPointRmsAfterMeters, 0.05);
    EXPECT_NEAR(result.points.front().point[2], 10.0, 0.05);
}

TEST(BundleAdjustCeresBackendTest, CeresDoesNotCountTracksWithNonFiniteRms)
{
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, -10.0}};
    track.observations.push_back(xjw::BAObservation{0, 512.0, 384.0, 1.0});
    track.observations.push_back(xjw::BAObservation{1, 512.0, 384.0, 1.0});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 3;

    const std::vector<xjw::Camera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    ASSERT_EQ(result.points.size(), 1u);
    EXPECT_FALSE(result.points.front().valid);
    EXPECT_EQ(result.optimizedTracks, 0);
    EXPECT_TRUE(std::isinf(result.meanRmsAfter));
    EXPECT_FALSE(result.solutionUsable);
}

TEST(BundleAdjustCeresBackendTest, CeresCudaAndCpuReachComparableRms)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
        makeCamera(0.0, -8.0, 0.0),
    };
    const std::array<double, 3> truth{{0.4, -0.3, 38.0}};
    const xjw::BATrack track = makeTrack(cameras, truth, {{1.5, -1.2, 43.0}});

    xjw::BAOptions cpuOptions;
    cpuOptions.backend = xjw::BABackend::CeresCpu;
    cpuOptions.refineCameraPose = false;
    cpuOptions.enablePointFilter = false;
    cpuOptions.maxIterations = 20;
    const xjw::BAResult cpu = xjw::BundleAdjust::optimizePoints(cameras, {track}, cpuOptions);

    xjw::BAOptions cudaOptions = cpuOptions;
    cudaOptions.backend = xjw::BABackend::CeresCuda;
    cudaOptions.minCeresCudaCameras = 1;
    const xjw::BAResult gpu = xjw::BundleAdjust::optimizePoints(cameras, {track}, cudaOptions);

    ASSERT_EQ(cpu.points.size(), 1u);
    ASSERT_EQ(gpu.points.size(), 1u);
    ASSERT_TRUE(cpu.points.front().valid);
    ASSERT_TRUE(gpu.points.front().valid);
    EXPECT_NEAR(gpu.meanRmsAfter, cpu.meanRmsAfter, 1e-6);
}

TEST(BundleAdjustCeresBackendTest, FreeLaserShotUsesOnlyRealMeasuredImagesAndDoesNotPolluteTrackStats)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-2.0, 0.0, 0.0),
        makeCamera(2.0, 0.0, 0.0),
    };
    const std::array<double, 3> truth{{0.25, 0.15, 10.0}};

    xjw::BALaserRangeConstraint shot;
    shot.cameraIndex = 0;
    shot.initialPoint = {{0.9, -0.6, 12.0}};
    shot.observedRangeMeters =
        distance3d(cameras.front().cameraCenter(), truth);
    shot.sigmaRangeMeters = 0.02;
    shot.pointMode = xjw::BALaserPointMode::Free;
    shot.shotId = "free-shot-001";
    shot.ephemerisTimeSeconds = 12345.25;
    shot.sourceIndex = 17;
    for (size_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex)
    {
        double u = 0.0;
        double v = 0.0;
        ASSERT_TRUE(projectPoint(cameras[cameraIndex], truth, &u, &v));
        shot.measuredImageObservations.push_back(
            xjw::BAObservation{
                static_cast<int>(cameraIndex), u, v, 1.0});
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.gaugePolicy = xjw::BAGaugePolicy::CallerManaged;
    options.enableLaserRangeConstraints = true;
    options.laserRangeConstraints = {shot};
    options.laserRangeHuberDelta = 0.0;
    options.huberDelta = 0.0;
    options.enablePointFilter = true;
    options.maxIterations = 40;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {}, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_EQ(result.solveStatus, xjw::BASolveStatus::Success);
    ASSERT_EQ(result.laserRangeShots.size(), 1u);
    const xjw::BARefinedLaserRangeShot &refined =
        result.laserRangeShots.front();
    ASSERT_TRUE(refined.valid);
    EXPECT_EQ(refined.shotId, shot.shotId);
    EXPECT_DOUBLE_EQ(refined.ephemerisTimeSeconds,
                     shot.ephemerisTimeSeconds);
    EXPECT_EQ(refined.sourceIndex, shot.sourceIndex);
    EXPECT_EQ(refined.pointMode, xjw::BALaserPointMode::Free);
    EXPECT_LT(distance3d(refined.point, truth), 1.0e-3);
    EXPECT_EQ(result.laserRangeConstraintCount, 1);
    EXPECT_LT(result.laserRangeRmsAfterMeters,
              result.laserRangeRmsBeforeMeters);
    EXPECT_LT(std::abs(refined.normalizedResidualAfter), 1.0e-3);
    EXPECT_EQ(result.totalTracks, 0);
    EXPECT_EQ(result.optimizedTracks, 0);
    EXPECT_TRUE(result.points.empty());
}

TEST(BundleAdjustCeresBackendTest, LaserRangeLeverArmUsesUpdatedCameraRotation)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const xjw::Camera camera = makeCamera(0.0, 0.0, 0.0);
    const std::vector<xjw::Camera> cameras{camera};
    const std::array<double, 3> leverArm{{1.0, 0.0, 0.0}};
    const std::array<double, 3> truthLaserOrigin{{0.0, 1.0, 0.0}};
    const std::vector<std::array<double, 3>> targetPoints{
        {{1.0, 2.0, 8.0}},
        {{-3.0, 1.0, 7.0}},
        {{2.0, -4.0, 6.0}},
        {{4.0, 3.0, 5.0}},
        {{-2.0, -3.0, 9.0}},
    };

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = true;
    options.gaugePolicy = xjw::BAGaugePolicy::CallerManaged;
    options.enableLaserRangeConstraints = true;
    options.laserRangeHuberDelta = 0.0;
    options.enablePointFilter = false;
    options.maxIterations = 80;
    options.cameraPosePriors.resize(1);
    options.cameraPosePriors.front().enabled = true;
    options.cameraPosePriors.front().cameraCenter = camera.cameraCenter();
    options.cameraPosePriors.front().cameraToWorldRotation =
        camera.cameraToWorldRotation();
    options.cameraPosePriors.front().positionSigmaMeters = 1.0e-4;
    options.cameraPosePriors.front().rotationSigmaDegrees = 1000.0;
    options.cameraPosePriorWeight = 1.0;
    options.cameraPosePriorHuberDelta = 0.0;
    for (size_t index = 0; index < targetPoints.size(); ++index)
    {
        xjw::BALaserRangeConstraint shot;
        shot.cameraIndex = 0;
        shot.initialPoint = targetPoints[index];
        shot.observedRangeMeters =
            distance3d(truthLaserOrigin, targetPoints[index]);
        shot.sigmaRangeMeters = 1.0e-3;
        shot.leverArmCameraMeters = leverArm;
        shot.pointMode = xjw::BALaserPointMode::Fixed;
        shot.shotId = "lever-" + std::to_string(index);
        shot.ephemerisTimeSeconds = static_cast<double>(index);
        shot.sourceIndex = static_cast<int>(index);
        options.laserRangeConstraints.push_back(shot);
    }

    const std::array<double, 3> originBefore =
        laserOrigin(camera, leverArm);
    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {}, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_EQ(result.refinedCameras.size(), 1u);
    const std::array<double, 3> originAfter =
        laserOrigin(result.refinedCameras.front(), leverArm);
    EXPECT_LT(distance3d(originAfter, truthLaserOrigin),
              distance3d(originBefore, truthLaserOrigin) * 0.1);
    EXPECT_LT(result.laserRangeRmsAfterMeters,
              result.laserRangeRmsBeforeMeters * 0.05);
    ASSERT_EQ(result.laserRangeShots.size(), targetPoints.size());
    for (size_t index = 0; index < result.laserRangeShots.size(); ++index)
    {
        EXPECT_TRUE(result.laserRangeShots[index].valid);
        EXPECT_EQ(result.laserRangeShots[index].sourceIndex,
                  static_cast<int>(index));
    }
}

TEST(BundleAdjustCeresBackendTest, ConstrainedLaserPointUsesFullSqrtInformationPrior)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{makeCamera(0.0, 0.0, 0.0)};
    const std::array<double, 3> truth{{0.2, -0.1, 10.0}};
    xjw::BALaserRangeConstraint shot;
    shot.cameraIndex = 0;
    shot.initialPoint = {{1.0, -0.8, 12.0}};
    shot.observedRangeMeters =
        distance3d(cameras.front().cameraCenter(), truth);
    shot.sigmaRangeMeters = 0.02;
    shot.pointMode = xjw::BALaserPointMode::Constrained;
    shot.pointPrior = truth;
    shot.pointPriorSqrtInformation = {{50.0, 5.0, 0.0,
                                       0.0, 40.0, 4.0,
                                       0.0, 0.0, 30.0}};

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.enableLaserRangeConstraints = true;
    options.laserRangeConstraints = {shot};
    options.laserRangeHuberDelta = 0.0;
    options.enablePointFilter = false;
    options.maxIterations = 40;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {}, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_EQ(result.laserRangeShots.size(), 1u);
    ASSERT_TRUE(result.laserRangeShots.front().valid);
    EXPECT_LT(distance3d(result.laserRangeShots.front().point, truth),
              1.0e-3);
    EXPECT_LT(result.laserRangeRmsAfterMeters,
              result.laserRangeRmsBeforeMeters);
}

TEST(BundleAdjustLaserRangeValidationTest, RejectsInvalidShotFieldsAndUnobservableFreePoint)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    xjw::BALaserRangeConstraint validShot;
    validShot.cameraIndex = 0;
    validShot.initialPoint = {{0.0, 0.0, 10.0}};
    validShot.observedRangeMeters = 10.0;
    validShot.sigmaRangeMeters = 0.1;
    validShot.pointMode = xjw::BALaserPointMode::Fixed;
    validShot.ephemerisTimeSeconds = 42.0;

    auto solve = [&](const xjw::BALaserRangeConstraint &shot)
    {
        xjw::BAOptions options;
        options.backend = xjw::BABackend::CeresCpu;
        options.refineCameraPose = false;
        options.enableLaserRangeConstraints = true;
        options.laserRangeConstraints = {shot};
        options.allowBackendFallback = false;
        return xjw::BundleAdjust::optimizePoints(cameras, {}, options);
    };

    xjw::BALaserRangeConstraint invalid = validShot;
    invalid.pointMode = xjw::BALaserPointMode::Unspecified;
    EXPECT_EQ(solve(invalid).solveStatus, xjw::BASolveStatus::InvalidInput);

    invalid = validShot;
    invalid.cameraIndex = -1;
    EXPECT_EQ(solve(invalid).solveStatus, xjw::BASolveStatus::InvalidInput);

    invalid = validShot;
    invalid.observedRangeMeters = 0.0;
    EXPECT_EQ(solve(invalid).solveStatus, xjw::BASolveStatus::InvalidInput);

    invalid = validShot;
    invalid.sigmaRangeMeters = 0.0;
    EXPECT_EQ(solve(invalid).solveStatus, xjw::BASolveStatus::InvalidInput);

    invalid = validShot;
    invalid.initialPoint = cameras.front().cameraCenter();
    EXPECT_EQ(solve(invalid).solveStatus, xjw::BASolveStatus::InvalidInput);

    invalid = validShot;
    invalid.leverArmCameraMeters[1] =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(solve(invalid).solveStatus, xjw::BASolveStatus::InvalidInput);

    invalid = validShot;
    invalid.ephemerisTimeSeconds =
        std::numeric_limits<double>::infinity();
    EXPECT_EQ(solve(invalid).solveStatus, xjw::BASolveStatus::InvalidInput);

    invalid = validShot;
    invalid.pointMode = xjw::BALaserPointMode::Constrained;
    invalid.pointPrior = invalid.initialPoint;
    invalid.pointPriorSqrtInformation = {{1.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0,
                                          0.0, 0.0, 1.0}};
    EXPECT_EQ(solve(invalid).solveStatus, xjw::BASolveStatus::InvalidInput);

    invalid = validShot;
    invalid.pointMode = xjw::BALaserPointMode::Free;
    double u = 0.0;
    double v = 0.0;
    ASSERT_TRUE(projectPoint(cameras.front(), invalid.initialPoint, &u, &v));
    invalid.measuredImageObservations.push_back({0, u, v, 1.0});
    EXPECT_EQ(solve(invalid).solveStatus,
              xjw::BASolveStatus::UnsupportedConfiguration);

    xjw::BAOptions disabledOptions;
    disabledOptions.backend = xjw::BABackend::CeresCpu;
    disabledOptions.refineCameraPose = false;
    disabledOptions.laserRangeConstraints = {validShot};
    disabledOptions.allowBackendFallback = false;
    EXPECT_EQ(
        xjw::BundleAdjust::optimizePoints(cameras, {}, disabledOptions).solveStatus,
        xjw::BASolveStatus::InvalidInput);
}

TEST(BundleAdjustLaserRangeBackendTest, AutoRequiresCeresAndUnsupportedBackendsDoNotIgnoreRange)
{
    xjw::BAOptions selectionOptions;
    selectionOptions.backend = xjw::BABackend::Auto;
    selectionOptions.enableLaserRangeConstraints = true;
    const xjw::BABackendDecision decision =
        xjw::BundleAdjust::decideBackendForProblem({}, selectionOptions);
    EXPECT_EQ(decision.backend, xjw::BABackend::CeresCpu);

    EXPECT_TRUE(xjw::BundleAdjust::backendCapabilities(
                    xjw::BABackend::CeresCpu)
                    .supportsLaserRangeConstraints);
    EXPECT_FALSE(xjw::BundleAdjust::backendCapabilities(
                     xjw::BABackend::LegacyCpu)
                     .supportsLaserRangeConstraints);
    EXPECT_FALSE(xjw::BundleAdjust::backendCapabilities(
                     xjw::BABackend::NativeCuda)
                     .supportsLaserRangeConstraints);

    const std::vector<xjw::Camera> cameras{makeCamera(0.0, 0.0, 0.0)};
    xjw::BALaserRangeConstraint shot;
    shot.cameraIndex = 0;
    shot.initialPoint = {{0.0, 0.0, 10.0}};
    shot.observedRangeMeters = 10.0;
    shot.sigmaRangeMeters = 0.1;
    shot.pointMode = xjw::BALaserPointMode::Fixed;

    for (const xjw::BABackend backend :
         {xjw::BABackend::LegacyCpu, xjw::BABackend::NativeCuda})
    {
        xjw::BAOptions options;
        options.backend = backend;
        options.refineCameraPose = false;
        options.enableLaserRangeConstraints = true;
        options.laserRangeConstraints = {shot};
        options.allowBackendFallback = false;
        const xjw::BAResult result =
            xjw::BundleAdjust::optimizePoints(cameras, {}, options);
        EXPECT_EQ(result.solveStatus,
                  xjw::BASolveStatus::UnsupportedConfiguration);
        EXPECT_EQ(result.usedBackend, backend);
        EXPECT_FALSE(result.solutionUsable);
    }
}

TEST(BundleAdjustLaserRangeBackendTest, AutoQualityRejectionNeverFallsBackWithoutRangeSupport)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    const std::array<double, 3> truth{{0.0, 0.0, 10.0}};
    const xjw::BATrack track = makeTrack(cameras, truth, {{0.1, 0.0, 10.5}});
    xjw::BALaserRangeConstraint shot;
    shot.cameraIndex = 0;
    shot.initialPoint = truth;
    shot.observedRangeMeters =
        distance3d(cameras.front().cameraCenter(), truth);
    shot.sigmaRangeMeters = 0.1;
    shot.pointMode = xjw::BALaserPointMode::Fixed;

    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.enableLaserRangeConstraints = true;
    options.laserRangeConstraints = {shot};
    options.minAcceptedValidTrackRatio = 1.01;
    options.maxIterations = 20;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_EQ(result.requestedBackend, xjw::BABackend::Auto);
    EXPECT_EQ(result.usedBackend, xjw::BABackend::CeresCpu);
    EXPECT_FALSE(result.backendFallback);
    EXPECT_TRUE(result.qualityGateRejected);
    EXPECT_FALSE(result.solutionUsable);
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::NumericalFailure);
    EXPECT_EQ(result.laserRangeConstraintCount, 1);
    EXPECT_NE(result.backendMessage.find("没有兼容的 legacy 回退"),
              std::string::npos);
}
