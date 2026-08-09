#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "BundleAdjustQuality.h"
#include "BundleAdjustValidation.h"
#include "Camera.h"

#include <array>
#include <cmath>
#include <limits>
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
            track.observations.push_back(xjw::BAObservation{static_cast<int>(i), u, v, 1.0});
        }
    }
    return track;
}

} // namespace

TEST(BundleAdjustQualityGateTest, AutoPointOnlyProblemUsesLegacyCpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = false;
    options.minCeresCudaCameras = 1;
    options.minCeresCudaObservations = 1;

    xjw::BAProblemStats stats;
    stats.cameraCount = 120;
    stats.trackCount = 20000;
    stats.observationCount = 160000;

    EXPECT_EQ(xjw::BundleAdjust::selectBackendForProblem(stats, options),
              xjw::BABackend::LegacyCpu);
}

TEST(BundleAdjustValidationTest, ProblemSummaryCountsOnlyUsableObservations)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };

    xjw::BATrack usable;
    usable.initialPoint = {{0.0, 0.0, 5.0}};
    usable.observations.push_back({0, 320.0, 240.0, 1.0});
    usable.observations.push_back({1, 300.0, 240.0, 0.5});
    usable.observations.push_back({0, 321.0, 240.0, 0.0});
    usable.observations.push_back(
        {1, 299.0, 240.0, std::numeric_limits<double>::quiet_NaN()});
    usable.observations.push_back({-1, 320.0, 240.0, 1.0});
    usable.observations.push_back(
        {static_cast<int>(cameras.size()), 300.0, 240.0, 1.0});

    xjw::BATrack unusable = usable;
    unusable.observations.resize(1);

    const xjw::BAProblemStats stats =
        xjw::BundleAdjust::summarizeProblem(cameras, {usable, unusable});

    EXPECT_EQ(stats.cameraCount, 2);
    EXPECT_EQ(stats.trackCount, 1);
    EXPECT_EQ(stats.observationCount, 2);
}

TEST(BundleAdjustQualityGateTest, AutoDoesNotSelectNativeCudaForPointOnlyProblem)
{
    xjw::BAProblemStats stats;
    stats.cameraCount = 100;
    stats.trackCount = 10000;
    stats.observationCount = 1000000;

    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = false;
    const auto selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);
    EXPECT_NE(selected, xjw::BABackend::NativeCuda);
}

TEST(BundleAdjustQualityGateTest, ExplicitNativeCudaFallsBackWhenControlPointsEnabled)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };

    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 5.0}};
    track.observations.push_back({0, 320.0, 240.0, 1.0});
    track.observations.push_back({1, 300.0, 240.0, 1.0});
    track.controlPointConstraints.push_back({{{0.0, 0.0, 5.0}}, 1.0, 1.0, 0});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::NativeCuda;
    options.enableControlPointConstraints = true;
    options.allowBackendFallback = true;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);
    EXPECT_EQ(result.requestedBackend, xjw::BABackend::NativeCuda);
    EXPECT_NE(result.usedBackend, xjw::BABackend::NativeCuda);
    EXPECT_TRUE(result.backendFallback);
    EXPECT_FALSE(result.backendMessage.empty());
}

TEST(BundleAdjustQualityGateTest, AutoRejectsCeresCandidateWhenQualityGateFails)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres backend is not available in this build";
    }

    const std::vector<xjw::Camera> cameras{
        makeCamera(-5.0, 0.0, 0.0),
        makeCamera(0.0, 0.0, 0.0),
        makeCamera(5.0, 0.0, 0.0),
    };
    std::vector<xjw::BATrack> tracks;
    for (int i = 0; i < 120; ++i)
    {
        const double x = (static_cast<double>(i % 12) - 5.5) * 0.25;
        const double y = (static_cast<double>(i / 12) - 4.5) * 0.25;
        const std::array<double, 3> truth{{x, y, 38.0 + static_cast<double>(i % 3)}};
        const std::array<double, 3> initial{{x + 0.8, y - 0.6, truth[2] + 2.0}};
        xjw::BATrack track = makeTrack(cameras, truth, initial);
        ASSERT_GE(track.observations.size(), 3u);
        tracks.push_back(track);
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.fixedCameraIndices = {0};
    options.enablePointFilter = false;
    options.maxIterations = 1;
    options.maxPointIterations = 1;
    options.maxCameraIterations = 1;
    options.minCeresCudaCameras = 9999;
    options.minCeresCpuObservations = 1;
    options.enableBackendQualityGate = true;
    options.compareAutoBackendWithLegacy = true;
    options.maxAcceptedRmsGrowth = 1e-12;
    options.minAcceptedValidTrackRatio = 1.01;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);

    EXPECT_EQ(result.requestedBackend, xjw::BABackend::Auto);
    EXPECT_EQ(result.usedBackend, xjw::BABackend::LegacyCpu);
    EXPECT_TRUE(result.backendFallback);
    EXPECT_TRUE(result.qualityGateRejected);
    EXPECT_TRUE(std::isfinite(result.validTrackRatio));
    EXPECT_NE(result.qualityGateMessage.find("质量门控"), std::string::npos);
}

TEST(BundleAdjustConvergenceTest, ExactLegacyProblemStopsAfterMinimumConvergenceRounds)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    std::vector<xjw::BATrack> tracks;
    for (int i = 0; i < 24; ++i)
    {
        const std::array<double, 3> point{{
            (static_cast<double>(i % 6) - 2.5) * 0.2,
            (static_cast<double>(i / 6) - 1.5) * 0.2,
            8.0 + static_cast<double>(i % 3) * 0.1,
        }};
        tracks.push_back(makeTrack(cameras, point, point));
    }

    int reportedIterations = 0;
    xjw::BAOptions options;
    options.backend = xjw::BABackend::LegacyCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.maxIterations = 20;
    options.progressCallback =
        [&reportedIterations](int iteration, int, double, int)
        {
            reportedIterations = iteration;
            return true;
        };

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);

    ASSERT_TRUE(result.solutionUsable);
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::Success);
    EXPECT_LE(reportedIterations, 3);
    EXPECT_NEAR(result.meanRmsAfter, 0.0, 1e-8);
}

TEST(BundleAdjustQualityGateTest, AdaptiveFilterUsesAbsoluteFloorAndMedianScale)
{
    const std::vector<double> pointRms{1.0, 2.0, 9.0};
    EXPECT_DOUBLE_EQ(
        xjw::detail::adaptivePointFilterThreshold(pointRms, 2.5, 3.0),
        6.0);
    EXPECT_DOUBLE_EQ(
        xjw::detail::adaptivePointFilterThreshold(pointRms, 8.0, 3.0),
        8.0);
}

TEST(BundleAdjustQualityGateTest, FinalizerRejectsTrackBehindAnyObservationCamera)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    const std::array<double, 3> truth{{0.0, 0.0, 5.0}};
    xjw::BATrack track = makeTrack(cameras, truth, truth);
    ASSERT_EQ(track.observations.size(), 2u);

    xjw::BAResult result;
    result.solveStatus = xjw::BASolveStatus::Success;
    result.solutionUsable = true;
    result.refinedCameras = cameras;
    result.points.resize(1);
    result.points.front().valid = true;
    result.points.front().point = {{0.0, 0.0, -5.0}};

    xjw::BAOptions options;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    xjw::detail::finalizeBundleAdjustResult(cameras, {track}, options, &result);

    EXPECT_FALSE(result.points.front().valid);
    EXPECT_FALSE(result.solutionUsable);
    EXPECT_EQ(result.optimizedTracks, 0);
    EXPECT_TRUE(std::isinf(result.meanRmsAfter));
}

TEST(BundleAdjustQualityGateTest, JointBaWithoutGaugeConstraintIsRejected)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    const std::array<double, 3> point{{0.0, 0.0, 5.0}};
    const xjw::BATrack track = makeTrack(cameras, point, point);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::LegacyCpu;
    options.refineCameraPose = true;
    options.gaugePolicy = xjw::BAGaugePolicy::RequireExplicitGauge;
    options.enablePointFilter = false;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_FALSE(result.solutionUsable);
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::UnsupportedConfiguration);
    EXPECT_NE(result.backendMessage.find("gauge"), std::string::npos);
}

TEST(BundleAdjustQualityGateTest, ConstraintRegressionIsRejected)
{
    std::string message;
    EXPECT_FALSE(
        xjw::detail::constraintRmsPassesQualityGate(
            12, 0.10, 0.20, 1.25, "控制点约束", &message));
    EXPECT_NE(message.find("控制点约束"), std::string::npos);

    message.clear();
    EXPECT_TRUE(
        xjw::detail::constraintRmsPassesQualityGate(
            12, 0.10, 0.12, 1.25, "控制点约束", &message));
    EXPECT_TRUE(message.empty());
}

TEST(BundleAdjustValidationTest, RejectsNonPositiveFiniteDifferenceStep)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    const std::array<double, 3> point{{0.0, 0.0, 5.0}};

    xjw::BAOptions options;
    options.backend = xjw::BABackend::LegacyCpu;
    options.refineCameraPose = false;
    options.finiteDiffEps = 0.0;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(
            cameras,
            {makeTrack(cameras, point, point)},
            options);

    EXPECT_FALSE(result.solutionUsable);
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::InvalidInput);
    EXPECT_NE(result.backendMessage.find("有限差分"), std::string::npos);
}

TEST(BundleAdjustQualityGateTest, ConstraintStatsExcludeRejectedTracksFromBothSides)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    const std::array<double, 3> visiblePoint{{0.0, 0.0, 5.0}};
    xjw::BATrack validTrack =
        makeTrack(cameras, visiblePoint, visiblePoint);
    validTrack.controlPointConstraints.push_back(
        {{{0.0, 0.0, 4.0}}, 1.0, 1.0, 0});

    xjw::BATrack rejectedTrack =
        makeTrack(
            cameras,
            visiblePoint,
            {{0.0, 0.0, -5.0}});
    rejectedTrack.controlPointConstraints.push_back(
        {{{0.0, 0.0, -8.0}}, 1.0, 1.0, 0});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::LegacyCpu;
    options.refineCameraPose = false;
    options.enableControlPointConstraints = true;
    options.enablePointFilter = false;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(
            cameras,
            {validTrack, rejectedTrack},
            options);

    ASSERT_TRUE(result.solutionUsable);
    ASSERT_EQ(result.controlPointConstraintCount, 1);
    ASSERT_EQ(result.points.size(), 2U);
    ASSERT_TRUE(result.points[0].valid);
    ASSERT_FALSE(result.points[1].valid);
    const double expected_after = std::abs(result.points[0].point[2] - 4.0);
    EXPECT_NEAR(result.controlPointRmsBeforeMeters, 1.0, 1.0e-9);
    EXPECT_NEAR(result.controlPointRmsAfterMeters, expected_after, 1.0e-9);
}

TEST(BundleAdjustValidationTest, LaserPlanesDoNotBypassAutoGaugeAnchors)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    const std::array<double, 3> point{{0.0, 0.0, 5.0}};
    xjw::BATrack track = makeTrack(cameras, point, point);
    track.laserPlaneConstraints.push_back(
        {{{0.0, 0.0, 5.0}}, {{0.0, 0.0, 1.0}}, 1.0, 0.0, 0});

    xjw::BAOptions requested;
    requested.refineCameraPose = true;
    requested.enableLaserPlaneConstraints = true;
    xjw::BAOptions normalized;
    const auto validation = xjw::detail::validateAndNormalizeBundleAdjustOptions(
        cameras, {track}, requested, &normalized);

    ASSERT_TRUE(validation.ok) << validation.message;
    EXPECT_EQ(normalized.fixedCameraIndices.size(), 2u);
}

TEST(BundleAdjustValidationTest, SinglePosePriorDoesNotClaimAbsoluteScale)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    const std::array<double, 3> point{{0.0, 0.0, 5.0}};
    xjw::BAOptions requested;
    requested.refineCameraPose = true;
    requested.cameraPosePriors.resize(cameras.size());
    requested.cameraPosePriors[0].enabled = true;
    requested.cameraPosePriors[0].cameraCenter = cameras[0].cameraCenter();

    xjw::BAOptions normalized;
    const auto validation = xjw::detail::validateAndNormalizeBundleAdjustOptions(
        cameras, {makeTrack(cameras, point, point)}, requested, &normalized);

    ASSERT_TRUE(validation.ok) << validation.message;
    EXPECT_EQ(normalized.fixedCameraIndices.size(), 2U);
}

TEST(BundleAdjustValidationTest, SingleControlPointStillRequiresRigidCameraAnchor)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0),
        makeCamera(1.0, 0.0, 0.0),
    };
    const std::array<double, 3> point{{0.0, 0.0, 5.0}};
    xjw::BATrack track = makeTrack(cameras, point, point);
    track.controlPointConstraints.push_back({point, 0.01, 1.0, 0});

    xjw::BAOptions requested;
    requested.refineCameraPose = true;
    requested.enableControlPointConstraints = true;
    xjw::BAOptions normalized;
    const auto validation = xjw::detail::validateAndNormalizeBundleAdjustOptions(
        cameras, {track}, requested, &normalized);

    ASSERT_TRUE(validation.ok) << validation.message;
    EXPECT_EQ(normalized.fixedCameraIndices.size(), 1U);
}
