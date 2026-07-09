#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "Camera.h"

#include <array>
#include <cmath>
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
