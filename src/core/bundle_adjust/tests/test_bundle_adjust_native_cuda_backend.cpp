#include "BundleAdjust.h"
#include "Camera.h"

#include <gtest/gtest.h>

#include <vector>

namespace
{

xjw::Camera makeCamera(double cx)
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 320.0, 240.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{cx, 0.0, 0.0}});
    return camera;
}

} // namespace

TEST(NativeCudaBackendTest, ExplicitBackendReportsActiveWorkset)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::NativeCuda))
    {
        GTEST_SKIP() << "native_cuda backend is not available in this build";
    }

    std::vector<xjw::Camera> cameras{makeCamera(0.0), makeCamera(1.0), makeCamera(2.0)};
    xjw::BATrack track;
    track.initialPoint = {{0.1, 0.0, 8.0}};
    track.observations.push_back({0, 332.5, 240.0, 1.0});
    track.observations.push_back({1, 207.5, 240.0, 1.0});
    track.observations.push_back({2, 82.5, 240.0, 1.0});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::NativeCuda;
    options.refineCameraPose = true;
    options.fixedCameraIndices.push_back(0);
    options.maxIterations = 1;
    options.enablePointFilter = false;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);
    EXPECT_EQ(result.usedBackend, xjw::BABackend::NativeCuda);
    EXPECT_TRUE(result.usedGpu);
    EXPECT_EQ(result.nativeCudaActiveCameras, 3);
    EXPECT_EQ(result.nativeCudaActiveTracks, 1);
    EXPECT_EQ(result.nativeCudaActiveObservations, 3);
}

TEST(NativeCudaBackendTest, ReducesReprojectionRmsOnSyntheticGlobalProblem)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::NativeCuda))
    {
        GTEST_SKIP() << "native_cuda backend is not available in this build";
    }

    std::vector<xjw::Camera> cameras{makeCamera(0.0), makeCamera(1.0), makeCamera(2.0), makeCamera(3.0)};
    std::vector<xjw::BATrack> tracks;
    for (int i = 0; i < 40; ++i)
    {
        const double x = -0.5 + 0.025 * static_cast<double>(i);
        const double z = 8.0 + 0.02 * static_cast<double>(i);
        xjw::BATrack track;
        track.initialPoint = {{x + 0.05, 0.02, z - 0.10}};
        for (int ci = 0; ci < static_cast<int>(cameras.size()); ++ci)
        {
            const double world[3] = {x, 0.0, z};
            double pixel[2] = {0.0, 0.0};
            ASSERT_TRUE(cameras[static_cast<size_t>(ci)].projectWorldPoint(world, pixel));
            track.observations.push_back({ci, pixel[0], pixel[1], 1.0});
        }
        tracks.push_back(track);
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::NativeCuda;
    options.refineCameraPose = true;
    options.fixedCameraIndices.push_back(0);
    options.maxIterations = 5;
    options.nativeCudaMaxPcgIterations = 80;
    options.nativeCudaPcgTolerance = 1e-5;
    options.enablePointFilter = false;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);
    ASSERT_EQ(result.usedBackend, xjw::BABackend::NativeCuda);
    EXPECT_TRUE(result.usedGpu);
    EXPECT_LT(result.meanRmsAfter, result.meanRmsBefore);
    EXPECT_LT(result.meanRmsAfter, 0.25);
    EXPECT_GT(result.nativeCudaAcceptedSteps, 0);
}
