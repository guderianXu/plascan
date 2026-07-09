#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "Camera.h"

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
    EXPECT_LT(result.meanRmsAfter, result.meanRmsBefore);
    EXPECT_LT(distance3d(result.points.front().point, truth), distance3d(initial, truth));
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
    EXPECT_TRUE(std::isfinite(result.meanRmsAfter));
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
