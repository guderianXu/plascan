#include "BundleAdjust.h"
#include "Camera.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
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

std::vector<xjw::Camera> makeCameras(int count)
{
    std::vector<xjw::Camera> cameras;
    cameras.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        const double t = (static_cast<double>(i) / std::max(1, count - 1) - 0.5) * 18.0;
        cameras.push_back(makeCamera(t, std::sin(i * 0.45) * 3.0, 0.0));
    }
    return cameras;
}

std::vector<xjw::BATrack> makeTracks(const std::vector<xjw::Camera> &cameras,
                                     int trackCount,
                                     int viewsPerTrack)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> xy(-4.0, 4.0);
    std::uniform_real_distribution<double> z(32.0, 52.0);
    std::normal_distribution<double> initNoise(0.0, 0.30);
    std::normal_distribution<double> imageNoise(0.0, 0.05);

    std::vector<xjw::BATrack> tracks;
    tracks.reserve(static_cast<size_t>(trackCount));
    for (int i = 0; i < trackCount; ++i)
    {
        const std::array<double, 3> truth{{xy(rng), xy(rng), z(rng)}};
        xjw::BATrack track;
        track.initialPoint = {{truth[0] + initNoise(rng),
                               truth[1] + initNoise(rng),
                               truth[2] + initNoise(rng)}};
        const int start = i % static_cast<int>(cameras.size());
        for (int k = 0; k < viewsPerTrack; ++k)
        {
            const int ci = (start + k * 3) % static_cast<int>(cameras.size());
            double u = 0.0;
            double v = 0.0;
            if (projectPoint(cameras[static_cast<size_t>(ci)], truth, &u, &v))
            {
                track.observations.push_back(xjw::BAObservation{ci,
                                                                u + imageNoise(rng),
                                                                v + imageNoise(rng),
                                                                1.0});
            }
        }
        if (track.observations.size() >= 2)
        {
            tracks.push_back(track);
        }
    }
    return tracks;
}

std::string sanitizeField(std::string value)
{
    std::replace(value.begin(), value.end(), ',', ';');
    std::replace(value.begin(), value.end(), '\n', ' ');
    std::replace(value.begin(), value.end(), '\r', ' ');
    return value;
}

std::vector<std::string> splitBackends(const std::string &raw)
{
    std::vector<std::string> names;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
                       return std::isspace(ch) != 0;
                   }),
                   item.end());
        if (!item.empty())
        {
            names.push_back(item);
        }
    }
    return names;
}

bool shouldRunBackend(const std::vector<std::string> &requested, const std::string &name)
{
    return std::find(requested.begin(), requested.end(), name) != requested.end();
}

void runCase(const char *name,
             xjw::BABackend backend,
             const std::vector<xjw::Camera> &cameras,
             const std::vector<xjw::BATrack> &tracks,
             int threads,
             int iterations,
             bool refinePose)
{
    xjw::BAOptions options;
    options.backend = backend;
    options.numThreads = threads;
    options.maxIterations = iterations;
    options.refineCameraPose = refinePose;
    options.enablePointFilter = false;
    options.minCeresCudaCameras = 1;
    options.minCeresCudaObservations = 1;
    options.minCeresCpuObservations = 1;
    options.allowBackendFallback = true;
    options.enableBackendQualityGate = true;
    options.compareAutoBackendWithLegacy = true;
    if (refinePose)
    {
        // 固定首相机，避免相机位姿参与时的整体 gauge 漂移影响基准稳定性。
        options.fixedCameraIndices.push_back(0);
    }

    const auto t0 = std::chrono::steady_clock::now();
    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    std::cout << name
              << ",requested=" << xjw::BundleAdjust::backendName(result.requestedBackend)
              << ",used=" << xjw::BundleAdjust::backendName(result.usedBackend)
              << ",gpu=" << (result.usedGpu ? "true" : "false")
              << ",fallback=" << (result.backendFallback ? "true" : "false")
              << ",solver=" << result.ceresLinearSolverName
              << ",observations=" << result.observationCount
              << ",tracks=" << result.totalTracks
              << ",optimized=" << result.optimizedTracks
              << ",valid_ratio=" << result.validTrackRatio
              << ",rms_before=" << result.meanRmsBefore
              << ",rms_after=" << result.meanRmsAfter
              << ",quality_rejected=" << (result.qualityGateRejected ? "true" : "false")
              << ",backend_reason=" << sanitizeField(result.backendSelectionReason)
              << ",quality_message=" << sanitizeField(result.qualityGateMessage)
              << ",setup_seconds=" << result.setupSeconds
              << ",solve_seconds=" << result.solveSeconds
              << ",total_seconds=" << result.totalSeconds
              << ",seconds=" << seconds
              << "\n";
    std::cout << std::flush;
}

} // namespace

int main(int argc, char **argv)
{
    const int cameraCount = argc > 1 ? std::max(2, std::atoi(argv[1])) : 80;
    const int trackCount = argc > 2 ? std::max(1, std::atoi(argv[2])) : 3000;
    const int viewsPerTrack = argc > 3 ? std::max(2, std::atoi(argv[3])) : 8;
    const int iterations = argc > 4 ? std::max(1, std::atoi(argv[4])) : 8;
    const int threads = argc > 5 ? std::max(1, std::atoi(argv[5])) : 32;
    const bool refinePose = argc > 6 ? std::atoi(argv[6]) != 0 : false;
    const std::vector<std::string> requestedBackends =
        splitBackends(argc > 7 ? argv[7] : "legacy_cpu,ceres_cpu,ceres_cuda,auto");

    const auto cameras = makeCameras(cameraCount);
    const auto tracks = makeTracks(cameras, trackCount, viewsPerTrack);
    std::cout << "dataset,cameras=" << cameras.size()
              << ",tracks=" << tracks.size()
              << ",views_per_track=" << viewsPerTrack
              << ",iterations=" << iterations
              << ",threads=" << threads
              << ",refine_pose=" << (refinePose ? "true" : "false")
              << "\n";
    if (shouldRunBackend(requestedBackends, "legacy_cpu"))
    {
        runCase("legacy_cpu", xjw::BABackend::LegacyCpu, cameras, tracks, threads, iterations, refinePose);
    }
    if (shouldRunBackend(requestedBackends, "ceres_cpu"))
    {
        runCase("ceres_cpu", xjw::BABackend::CeresCpu, cameras, tracks, threads, iterations, refinePose);
    }
    if (shouldRunBackend(requestedBackends, "ceres_cuda"))
    {
        runCase("ceres_cuda", xjw::BABackend::CeresCuda, cameras, tracks, threads, iterations, refinePose);
    }
    if (shouldRunBackend(requestedBackends, "auto"))
    {
        runCase("auto", xjw::BABackend::Auto, cameras, tracks, threads, iterations, refinePose);
    }
    return 0;
}
