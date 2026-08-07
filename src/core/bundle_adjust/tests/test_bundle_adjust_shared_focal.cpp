#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "Camera.h"

#include <array>
#include <cmath>
#include <vector>

namespace
{

xjw::Camera makeCamera(double cx, double cy, double cz, double focal)
{
    xjw::Camera camera;
    camera.setIntrinsics(focal, focal, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{cx, cy, cz}});
    return camera;
}

xjw::Camera makeCameraWithIntrinsics(double cameraX,
                                     double cameraY,
                                     double cameraZ,
                                     double focalX,
                                     double focalY,
                                     double principalX,
                                     double principalY)
{
    xjw::Camera camera = makeCamera(cameraX, cameraY, cameraZ, focalX);
    camera.setIntrinsics(focalX, focalY, principalX, principalY);
    return camera;
}

bool projectPoint(const xjw::Camera &camera,
                  const std::array<double, 3> &point,
                  double *u,
                  double *v)
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

xjw::BATrack makeTrack(const std::vector<xjw::Camera> &truthCameras,
                       const std::array<double, 3> &truth)
{
    xjw::BATrack track;
    track.initialPoint = truth;
    track.controlPointConstraints.push_back({truth, 0.01, 1.0, 0});
    for (size_t i = 0; i < truthCameras.size(); ++i)
    {
        double u = 0.0;
        double v = 0.0;
        if (projectPoint(truthCameras[i], truth, &u, &v))
        {
            track.observations.push_back({static_cast<int>(i), u, v, 1.0});
        }
    }
    return track;
}

std::vector<xjw::BATrack> makeSharedFocalTracks(const std::vector<xjw::Camera> &truthCameras)
{
    std::vector<xjw::BATrack> tracks;
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            const std::array<double, 3> truth{{
                (static_cast<double>(x) - 2.0) * 0.35,
                (static_cast<double>(y) - 2.0) * 0.25,
                8.0 + static_cast<double>((x + y) % 4) * 0.7,
            }};
            xjw::BATrack track = makeTrack(truthCameras, truth);
            if (track.observations.size() >= 3)
            {
                tracks.push_back(std::move(track));
            }
        }
    }
    return tracks;
}

std::vector<xjw::BATrack> makeWideCalibrationTracks(
    const std::vector<xjw::Camera> &truthCameras)
{
    std::vector<xjw::BATrack> tracks;
    for (int y = -6; y <= 6; ++y)
    {
        for (int x = -6; x <= 6; ++x)
        {
            const std::array<double, 3> truth{{
                static_cast<double>(x) * 0.45,
                static_cast<double>(y) * 0.35,
                7.0 + static_cast<double>((x - y + 24) % 5) * 0.55,
            }};
            xjw::BATrack track = makeTrack(truthCameras, truth);
            if (track.observations.size() >= 3)
            {
                tracks.push_back(std::move(track));
            }
        }
    }
    return tracks;
}

} // namespace

TEST(BundleAdjustSharedFocalTest, SharedFocalRefinementImprovesWrongNoCameraInitialFocal)
{
    const std::vector<xjw::Camera> truthCameras{
        makeCamera(-2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, -2.0, 0.0, 1500.0),
        makeCamera(2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, 2.0, 0.0, 1500.0),
    };
    const std::vector<xjw::Camera> initialCameras{
        makeCamera(-2.0, 0.0, 0.0, 900.0),
        makeCamera(0.0, -2.0, 0.0, 900.0),
        makeCamera(2.0, 0.0, 0.0, 900.0),
        makeCamera(0.0, 2.0, 0.0, 900.0),
    };
    const std::vector<xjw::BATrack> tracks = makeSharedFocalTracks(truthCameras);
    ASSERT_GE(tracks.size(), 20u);

    xjw::BAOptions fixedOptions;
    fixedOptions.backend = xjw::BABackend::LegacyCpu;
    fixedOptions.refineCameraPose = false;
    fixedOptions.enablePointFilter = false;
    fixedOptions.enableControlPointConstraints = true;
    fixedOptions.controlPointWeight = 100.0;
    fixedOptions.controlPointHuberDeltaMeters = 0.5;
    fixedOptions.maxIterations = 8;

    xjw::BAOptions sharedFocalOptions = fixedOptions;
    sharedFocalOptions.refineSharedFocalLength = true;
    sharedFocalOptions.minSharedFocalScale = 0.5;
    sharedFocalOptions.maxSharedFocalScale = 2.0;

    const xjw::BAResult fixed = xjw::BundleAdjust::optimizePoints(initialCameras, tracks, fixedOptions);
    const xjw::BAResult shared = xjw::BundleAdjust::optimizePoints(initialCameras, tracks, sharedFocalOptions);

    ASSERT_FALSE(shared.refinedCameras.empty());
    EXPECT_LT(shared.meanRmsAfter, fixed.meanRmsAfter * 0.75);
    EXPECT_GT(shared.refinedCameras.front().focalX(), 1200.0);
    EXPECT_LT(shared.refinedCameras.front().focalX(), 1700.0);
    EXPECT_GT(shared.refinedSharedFocalScale, 1.0);
}

TEST(BundleAdjustSharedFocalTest, CeresJointlyRefinesSharedFocalAndPoints)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    const std::vector<xjw::Camera> truthCameras{
        makeCamera(-2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, -2.0, 0.0, 1500.0),
        makeCamera(2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, 2.0, 0.0, 1500.0),
    };
    const std::vector<xjw::Camera> initialCameras{
        makeCamera(-2.0, 0.0, 0.0, 900.0),
        makeCamera(0.0, -2.0, 0.0, 900.0),
        makeCamera(2.0, 0.0, 0.0, 900.0),
        makeCamera(0.0, 2.0, 0.0, 900.0),
    };
    const std::vector<xjw::BATrack> tracks = makeSharedFocalTracks(truthCameras);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.minSharedFocalScale = 0.5;
    options.maxSharedFocalScale = 2.0;
    options.enableControlPointConstraints = true;
    options.controlPointWeight = 100.0;
    options.controlPointHuberDeltaMeters = 0.5;
    options.enablePointFilter = false;
    options.maxIterations = 30;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(initialCameras, tracks, options);

    ASSERT_EQ(result.usedBackend, xjw::BABackend::CeresCpu);
    ASSERT_TRUE(result.solutionUsable);
    ASSERT_FALSE(result.refinedCameras.empty());
    EXPECT_GT(result.refinedCameras.front().focalX(), 1200.0);
    EXPECT_LT(result.refinedCameras.front().focalX(), 1700.0);
    EXPECT_GT(result.refinedSharedFocalScale, 1.0);
    EXPECT_EQ(result.refinedIntrinsicCount, static_cast<int>(initialCameras.size()));
}

TEST(BundleAdjustSharedFocalTest, CeresUsesOneAbsoluteFocalForHeterogeneousInputs)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    const std::vector<xjw::Camera> truthCameras{
        makeCamera(-2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, -2.0, 0.0, 1500.0),
        makeCamera(2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, 2.0, 0.0, 1500.0),
    };
    const std::vector<xjw::Camera> initialCameras{
        makeCamera(-2.0, 0.0, 0.0, 800.0),
        makeCamera(0.0, -2.0, 0.0, 900.0),
        makeCamera(2.0, 0.0, 0.0, 1000.0),
        makeCamera(0.0, 2.0, 0.0, 1100.0),
    };
    const std::vector<xjw::BATrack> tracks = makeSharedFocalTracks(truthCameras);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.minSharedFocalScale = 0.5;
    options.maxSharedFocalScale = 2.0;
    options.enableControlPointConstraints = true;
    options.controlPointWeight = 100.0;
    options.controlPointHuberDeltaMeters = 0.5;
    options.enablePointFilter = false;
    options.maxIterations = 30;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(initialCameras, tracks, options);

    ASSERT_EQ(result.usedBackend, xjw::BABackend::CeresCpu);
    ASSERT_TRUE(result.solutionUsable);
    ASSERT_EQ(result.refinedCameras.size(), initialCameras.size());

    const double refinedFocal = result.refinedCameras.front().focalX();
    EXPECT_GT(refinedFocal, 1200.0);
    EXPECT_LT(refinedFocal, 1700.0);
    for (const xjw::Camera &camera : result.refinedCameras)
    {
        EXPECT_NEAR(camera.focalX(), refinedFocal, 1e-6);
    }
}

TEST(BundleAdjustSharedFocalTest, AutoBackendCanSelectCeresForSharedFocal)
{
    xjw::BAProblemStats stats;
    stats.cameraCount = 120;
    stats.trackCount = 20000;
    stats.observationCount = 1000000;

    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.refineSharedFocalLength = true;
    options.minCeresCudaCameras = 1;
    options.minCeresCudaObservations = 1;
    options.minCeresCpuObservations = 1;

    const xjw::BABackend selected =
        xjw::BundleAdjust::selectBackendForProblem(stats, options);
    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCuda))
    {
        EXPECT_EQ(selected, xjw::BABackend::CeresCuda);
    }
    else if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        EXPECT_EQ(selected, xjw::BABackend::CeresCpu);
    }
    else
    {
        EXPECT_EQ(selected, xjw::BABackend::LegacyCpu);
    }
}

TEST(BundleAdjustSharedFocalTest, CeresRefinesIndependentCalibrationGroups)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    const std::vector<xjw::Camera> truthCameras{
        makeCamera(-2.0, 0.0, 0.0, 1200.0),
        makeCamera(0.0, -2.0, 0.0, 1200.0),
        makeCamera(2.0, 0.0, 0.0, 1800.0),
        makeCamera(0.0, 2.0, 0.0, 1800.0),
    };
    const std::vector<xjw::Camera> initialCameras{
        makeCamera(-2.0, 0.0, 0.0, 900.0),
        makeCamera(0.0, -2.0, 0.0, 900.0),
        makeCamera(2.0, 0.0, 0.0, 900.0),
        makeCamera(0.0, 2.0, 0.0, 900.0),
    };
    const std::vector<xjw::BATrack> tracks = makeSharedFocalTracks(truthCameras);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.cameraCalibrationGroupIds = {0, 0, 1, 1};
    options.minSharedFocalScale = 0.5;
    options.maxSharedFocalScale = 2.5;
    options.enableControlPointConstraints = true;
    options.controlPointWeight = 100.0;
    options.controlPointHuberDeltaMeters = 0.5;
    options.enablePointFilter = false;
    options.maxIterations = 40;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(initialCameras, tracks, options);

    ASSERT_TRUE(result.solutionUsable);
    ASSERT_EQ(result.refinedCameras.size(), initialCameras.size());
    EXPECT_EQ(result.refinedCalibrationGroupCount, 2);
    EXPECT_NEAR(result.refinedCameras[0].focalX(),
                result.refinedCameras[1].focalX(),
                1e-6);
    EXPECT_NEAR(result.refinedCameras[2].focalX(),
                result.refinedCameras[3].focalX(),
                1e-6);
    EXPECT_NEAR(result.refinedCameras[0].focalX(), 1200.0, 30.0);
    EXPECT_NEAR(result.refinedCameras[2].focalX(), 1800.0, 30.0);
    EXPECT_GT(result.refinedCameras[2].focalX() -
                  result.refinedCameras[0].focalX(),
              400.0);
}

TEST(BundleAdjustSharedFocalTest, StagedSelfCalibrationReportsTwoSolveStages)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    const std::vector<xjw::Camera> truthCameras{
        makeCamera(-2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, -2.0, 0.0, 1500.0),
        makeCamera(2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, 2.0, 0.0, 1500.0),
    };
    const std::vector<xjw::Camera> initialCameras{
        makeCamera(-2.0, 0.0, 0.0, 900.0),
        makeCamera(0.0, -2.0, 0.0, 900.0),
        makeCamera(2.0, 0.0, 0.0, 900.0),
        makeCamera(0.0, 2.0, 0.0, 900.0),
    };

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.stageSharedFocalRefinement = true;
    options.enableControlPointConstraints = true;
    options.controlPointWeight = 100.0;
    options.enablePointFilter = false;
    options.maxIterations = 12;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(
            initialCameras,
            makeSharedFocalTracks(truthCameras),
            options);

    ASSERT_TRUE(result.solutionUsable);
    EXPECT_EQ(result.selfCalibrationStagesRun, 2);
}

TEST(BundleAdjustSharedFocalTest, RejectsCalibrationGroupCountMismatch)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0, 900.0),
        makeCamera(1.0, 0.0, 0.0, 900.0),
    };
    xjw::BAOptions options;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.cameraCalibrationGroupIds = {0};

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(
            cameras,
            makeSharedFocalTracks(cameras),
            options);

    EXPECT_FALSE(result.solutionUsable);
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::InvalidInput);
    EXPECT_NE(result.backendMessage.find("标定分组"), std::string::npos);
}

TEST(BundleAdjustSharedFocalTest, CeresRecoversBoundedSharedPinholeIntrinsics)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    const std::vector<xjw::Camera> truthCameras{
        makeCameraWithIntrinsics(-2.0, 0.0, 0.0, 1500.0, 1530.0, 485.0, 420.0),
        makeCameraWithIntrinsics(0.0, -2.0, 0.0, 1500.0, 1530.0, 485.0, 420.0),
        makeCameraWithIntrinsics(2.0, 0.0, 0.0, 1500.0, 1530.0, 485.0, 420.0),
        makeCameraWithIntrinsics(0.0, 2.0, 0.0, 1500.0, 1530.0, 485.0, 420.0),
    };
    const std::vector<xjw::Camera> initialCameras{
        makeCameraWithIntrinsics(-2.0, 0.0, 0.0, 1400.0, 1400.0, 512.0, 384.0),
        makeCameraWithIntrinsics(0.0, -2.0, 0.0, 1400.0, 1400.0, 512.0, 384.0),
        makeCameraWithIntrinsics(2.0, 0.0, 0.0, 1400.0, 1400.0, 512.0, 384.0),
        makeCameraWithIntrinsics(0.0, 2.0, 0.0, 1400.0, 1400.0, 512.0, 384.0),
    };

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.refineSharedFocalAspectRatio = true;
    options.refineSharedPrincipalPoint = true;
    options.minSharedFocalScale = 0.8;
    options.maxSharedFocalScale = 1.3;
    options.minSharedFocalAspectScale = 0.9;
    options.maxSharedFocalAspectScale = 1.15;
    options.maxSharedPrincipalPointOffsetFraction = 0.08;
    options.sharedPrincipalPointPriorSigmaFraction = 0.10;
    options.sharedFocalAspectPriorSigma = 0.10;
    options.enableControlPointConstraints = true;
    options.controlPointWeight = 1000.0;
    options.controlPointHuberDeltaMeters = 0.5;
    options.enablePointFilter = false;
    options.maxIterations = 60;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(
            initialCameras,
            makeSharedFocalTracks(truthCameras),
            options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_EQ(result.refinedCameras.size(), initialCameras.size());
    const xjw::Camera &camera = result.refinedCameras.front();
    EXPECT_NEAR(camera.focalX(), 1500.0, 15.0);
    EXPECT_NEAR(camera.focalY(), 1530.0, 20.0);
    EXPECT_NEAR(camera.principalX(), 485.0, 8.0);
    EXPECT_NEAR(camera.principalY(), 420.0, 8.0);
    EXPECT_LT(result.meanRmsAfter, 0.25);
    EXPECT_NEAR(result.refinedSharedPrincipalOffsetX, -27.0, 8.0);
    EXPECT_NEAR(result.refinedSharedPrincipalOffsetY, 36.0, 8.0);
}

TEST(BundleAdjustSharedFocalTest, RejectsPrincipalPointRefinementWithoutSharedFocal)
{
    const std::vector<xjw::Camera> cameras{
        makeCamera(-1.0, 0.0, 0.0, 900.0),
        makeCamera(1.0, 0.0, 0.0, 900.0),
    };
    xjw::BAOptions options;
    options.refineCameraPose = false;
    options.refineSharedPrincipalPoint = true;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(
            cameras,
            makeSharedFocalTracks(cameras),
            options);

    EXPECT_FALSE(result.solutionUsable);
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::InvalidInput);
    EXPECT_NE(result.backendMessage.find("主点优化"), std::string::npos);
}

TEST(BundleAdjustSharedFocalTest, CeresRecoversBoundedSharedBrownConradyDistortion)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres CPU backend is not available";
    }

    std::vector<xjw::Camera> truthCameras{
        makeCamera(-2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, -2.0, 0.0, 1500.0),
        makeCamera(2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, 2.0, 0.0, 1500.0),
    };
    for (xjw::Camera &camera : truthCameras)
    {
        camera.setDistortion(-0.12, 0.035, 0.08, 0.003, -0.002);
    }
    const std::vector<xjw::Camera> initialCameras{
        makeCamera(-2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, -2.0, 0.0, 1500.0),
        makeCamera(2.0, 0.0, 0.0, 1500.0),
        makeCamera(0.0, 2.0, 0.0, 1500.0),
    };

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.allowBackendFallback = false;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.minSharedFocalScale = 1.0;
    options.maxSharedFocalScale = 1.0;
    options.refineSharedRadialDistortion = true;
    options.maxSharedRadialK1Abs = 0.30;
    options.maxSharedRadialK2Abs = 0.15;
    options.maxSharedRadialK3Abs = 0.20;
    options.maxSharedTangentialP1Abs = 0.01;
    options.maxSharedTangentialP2Abs = 0.01;
    options.sharedRadialK1PriorSigma = 1.0;
    options.sharedRadialK2PriorSigma = 1.0;
    options.sharedRadialK3PriorSigma = 1.0;
    options.sharedTangentialP1PriorSigma = 1.0;
    options.sharedTangentialP2PriorSigma = 1.0;
    options.enableControlPointConstraints = true;
    options.controlPointWeight = 1000.0;
    options.enablePointFilter = false;
    options.maxIterations = 60;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(
        initialCameras, makeWideCalibrationTracks(truthCameras), options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    ASSERT_EQ(result.usedBackend, xjw::BABackend::CeresCpu) << result.backendMessage;
    EXPECT_EQ(result.selfCalibrationStagesRun, 3);
    ASSERT_FALSE(result.refinedCameras.empty());
    const xjw::Camera::Distortion distortion =
        result.refinedCameras.front().distortion();
    EXPECT_NEAR(distortion.radialK1, -0.12, 0.02);
    EXPECT_NEAR(distortion.radialK2, 0.035, 0.03);
    EXPECT_NEAR(distortion.radialK3, 0.08, 0.04);
    EXPECT_NEAR(distortion.tangentialP1, 0.003, 0.001);
    EXPECT_NEAR(distortion.tangentialP2, -0.002, 0.001);
    EXPECT_NEAR(result.refinedSharedRadialK3, 0.08, 0.04);
    EXPECT_NEAR(result.refinedSharedTangentialP1, 0.003, 0.001);
    EXPECT_NEAR(result.refinedSharedTangentialP2, -0.002, 0.001);
    EXPECT_LT(result.meanRmsAfter, 0.25);
}
