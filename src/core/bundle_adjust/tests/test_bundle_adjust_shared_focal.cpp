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

TEST(BundleAdjustSharedFocalTest, AutoBackendKeepsSharedFocalOnLegacySolver)
{
    xjw::BAProblemStats stats;
    stats.cameraCount = 120;
    stats.trackCount = 20000;
    stats.observationCount = 1000000;

    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.refineSharedFocalLength = true;
    options.minNativeCudaCameras = 1;
    options.minNativeCudaObservations = 1;
    options.minCeresCudaCameras = 1;
    options.minCeresCudaObservations = 1;
    options.minCeresCpuObservations = 1;

    EXPECT_EQ(xjw::BundleAdjust::selectBackendForProblem(stats, options),
              xjw::BABackend::LegacyCpu);
}
