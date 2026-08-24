#include "RegularGrid3D.h"
#include "VisualHullFieldEvaluator.h"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{

    xjw::mesh::VisualHullView makeView(double centerX, const std::array<double, 3>& origin = {0.0, 0.0, 0.0})
    {
        xjw::mesh::VisualHullView view;
        view.camera.setIntrinsics(60.0, 61.0, 32.0, 32.0);
        view.camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
                            {origin[0] + centerX, origin[1], origin[2] - 3.0});
        view.camera.setDistortion(0.002, -0.0001, 0.0, 0.0002, -0.0001);
        view.silhouetteMask = cv::Mat::zeros(64, 64, CV_8UC1);
        cv::circle(view.silhouetteMask, cv::Point(32, 32), 15, cv::Scalar(255), cv::FILLED);
        view.depthMap = cv::Mat(64, 64, CV_32FC1, cv::Scalar(3.0f));
        return view;
    }

    std::vector<xjw::mesh::VisualHullView> makeViews(const std::array<double, 3>& origin = {0.0, 0.0, 0.0})
    {
        return {makeView(-0.15, origin), makeView(0.15, origin)};
    }

    xjw::mesh::detail::RegularGrid3D makeGrid(const std::array<float, 3>& origin = {0.0f, 0.0f, 0.0f})
    {
        return {{origin[0] - 0.65f, origin[1] - 0.65f, origin[2] - 0.4f},
                {origin[0] + 0.65f, origin[1] + 0.65f, origin[2] + 0.4f},
                {12, 12, 10}};
    }

    void expectBackendParity(xjw::mesh::VisualHullComputeBackend backend,
                             bool continuous,
                             const std::array<float, 3>& origin = {0.0f, 0.0f, 0.0f})
    {
        const std::array<double, 3> camera_origin{
            static_cast<double>(origin[0]), static_cast<double>(origin[1]), static_cast<double>(origin[2])};
        const std::vector<xjw::mesh::VisualHullView> views = makeViews(camera_origin);
        const xjw::mesh::detail::RegularGrid3D grid = makeGrid(origin);
        xjw::mesh::VisualHullConfig cpu_config;
        cpu_config.minimumVisibleViews = 2;
        cpu_config.allowedSilhouetteViolations = 0;
        cpu_config.enableDepthFreeSpaceCarving = true;
        cpu_config.minimumDepthFreeSpaceViolations = 2;
        cpu_config.relativeDepthTolerance = 0.01f;
        cpu_config.closeVolumeBoundary = false;
        cpu_config.useContinuousSilhouetteField = continuous;
        cpu_config.computeBackend = xjw::mesh::VisualHullComputeBackend::Cpu;

        std::vector<float> cpu_field;
        std::string error;
        ASSERT_TRUE(
            xjw::mesh::detail::evaluateVisualHullFieldGrid(views, cpu_config, grid, &cpu_field, nullptr, &error))
            << error;

        xjw::mesh::VisualHullConfig device_config = cpu_config;
        device_config.computeBackend = backend;
        device_config.gpuSlabDepth = 3;
        int callback_count = 0;
        xjw::mesh::VisualHullExecutionInfo callback_info;
        device_config.executionInfoFn = [&](const xjw::mesh::VisualHullExecutionInfo& info)
        {
            ++callback_count;
            callback_info = info;
        };
        std::vector<float> device_field;
        xjw::mesh::VisualHullComputeBackend used_backend = xjw::mesh::VisualHullComputeBackend::Cpu;
        xjw::mesh::VisualHullExecutionInfo execution_info;
        ASSERT_TRUE(xjw::mesh::detail::evaluateVisualHullFieldGrid(
            views, device_config, grid, &device_field, &used_backend, &error, &execution_info))
            << error;
        EXPECT_EQ(used_backend, backend);
        EXPECT_EQ(execution_info.requestedBackend, backend);
        EXPECT_EQ(execution_info.actualBackend, backend);
        EXPECT_GE(execution_info.actualDeviceIndex, 0);
        EXPECT_FALSE(execution_info.usedFallback);
        EXPECT_TRUE(execution_info.fallbackReason.empty());
        EXPECT_EQ(callback_count, 1);
        EXPECT_EQ(callback_info.actualBackend, backend);
        EXPECT_EQ(callback_info.actualDeviceIndex, execution_info.actualDeviceIndex);
        ASSERT_EQ(device_field.size(), cpu_field.size());

        for (std::size_t index = 0; index < cpu_field.size(); ++index)
        {
            if (!continuous)
            {
                EXPECT_EQ(device_field[index], cpu_field[index]) << index;
                continue;
            }
            EXPECT_NEAR(device_field[index], cpu_field[index], 2.0e-3f) << index;
            if (std::abs(cpu_field[index]) > 5.0e-3f)
            {
                EXPECT_EQ(device_field[index] < 0.0f, cpu_field[index] < 0.0f) << index;
            }
        }
    }

} // namespace

TEST(RegularGrid3DTest, UsesCellResolutionAndXMajorSampleIndexing)
{
    const xjw::mesh::detail::RegularGrid3D grid{{-1.0f, -2.0f, -3.0f}, {1.0f, 2.0f, 3.0f}, {2, 4, 6}};
    ASSERT_TRUE(grid.isValid());
    EXPECT_EQ(grid.sampleCount(), 3U * 5U * 7U);
    EXPECT_EQ(grid.linearIndex(2, 1, 1), 20U);
    EXPECT_FLOAT_EQ(grid.spacing(0), 1.0f);
    EXPECT_EQ(grid.samplePosition(2, 4, 6), (std::array<float, 3>{1.0f, 2.0f, 3.0f}));
    EXPECT_TRUE(grid.isBoundarySample(2, 2, 2));
    EXPECT_FALSE(grid.isBoundarySample(1, 2, 2));
}

TEST(VisualHullFieldBackendTest, AutoCpuFallbackIsObservable)
{
    xjw::mesh::VisualHullConfig config;
    config.minimumVisibleViews = 1;
    config.computeBackend = xjw::mesh::VisualHullComputeBackend::Auto;
    config.computeDeviceIndex = std::numeric_limits<int>::max();
    bool reported_fallback_progress = false;
    config.progressFn = [&](const std::string& message, float)
    { reported_fallback_progress = reported_fallback_progress || message.find("回退 CPU") != std::string::npos; };
    int callback_count = 0;
    xjw::mesh::VisualHullExecutionInfo callback_info;
    config.executionInfoFn = [&](const xjw::mesh::VisualHullExecutionInfo& info)
    {
        ++callback_count;
        callback_info = info;
    };

    std::vector<float> field;
    std::string error;
    xjw::mesh::VisualHullComputeBackend used_backend = xjw::mesh::VisualHullComputeBackend::Auto;
    xjw::mesh::VisualHullExecutionInfo execution_info;
    ASSERT_TRUE(xjw::mesh::detail::evaluateVisualHullFieldGrid(
        makeViews(), config, makeGrid(), &field, &used_backend, &error, &execution_info))
        << error;
    EXPECT_EQ(used_backend, xjw::mesh::VisualHullComputeBackend::Cpu);
    EXPECT_EQ(execution_info.requestedBackend, xjw::mesh::VisualHullComputeBackend::Auto);
    EXPECT_EQ(execution_info.actualBackend, xjw::mesh::VisualHullComputeBackend::Cpu);
    EXPECT_EQ(execution_info.actualDeviceIndex, -1);
    EXPECT_TRUE(execution_info.usedFallback);
    EXPECT_NE(execution_info.fallbackReason.find("CUDA"), std::string::npos);
    EXPECT_NE(execution_info.fallbackReason.find("OpenCL"), std::string::npos);
    EXPECT_TRUE(reported_fallback_progress);
    EXPECT_EQ(callback_count, 1);
    EXPECT_EQ(callback_info.fallbackReason, execution_info.fallbackReason);
}

TEST(VisualHullFieldBackendTest, ExplicitInvalidDeviceDoesNotFallback)
{
    const xjw::mesh::VisualHullComputeBackend candidates[] = {xjw::mesh::VisualHullComputeBackend::Cuda,
                                                              xjw::mesh::VisualHullComputeBackend::OpenCL};
    for (const xjw::mesh::VisualHullComputeBackend backend : candidates)
    {
        xjw::mesh::VisualHullConfig config;
        config.minimumVisibleViews = 1;
        config.computeBackend = backend;
        config.computeDeviceIndex = std::numeric_limits<int>::max();
        std::vector<float> field;
        std::string error;
        EXPECT_FALSE(
            xjw::mesh::detail::evaluateVisualHullFieldGrid(makeViews(), config, makeGrid(), &field, nullptr, &error));
        EXPECT_NE(error.find("unavailable"), std::string::npos);
    }
}

TEST(VisualHullFieldBackendTest, DeviceBackendsMatchCpuField)
{
    const xjw::mesh::VisualHullComputeBackend candidates[] = {xjw::mesh::VisualHullComputeBackend::Cuda,
                                                              xjw::mesh::VisualHullComputeBackend::OpenCL};
    int evaluated_backends = 0;
    for (const xjw::mesh::VisualHullComputeBackend backend : candidates)
    {
        if (!xjw::mesh::detail::isVisualHullFieldBackendAvailable(backend))
        {
            continue;
        }
        expectBackendParity(backend, false);
        expectBackendParity(backend, true);
        ++evaluated_backends;
    }
    if (evaluated_backends == 0)
    {
        GTEST_SKIP() << "no CUDA or OpenCL device is available";
    }
}

TEST(VisualHullFieldBackendTest, LargeWorldCoordinatesMatchCpuField)
{
    const std::array<float, 3> large_origin{10000000.0f, -7000000.0f, 3000000.0f};
    const xjw::mesh::VisualHullComputeBackend candidates[] = {xjw::mesh::VisualHullComputeBackend::Cuda,
                                                              xjw::mesh::VisualHullComputeBackend::OpenCL};
    int evaluated_backends = 0;
    for (const xjw::mesh::VisualHullComputeBackend backend : candidates)
    {
        if (!xjw::mesh::detail::isVisualHullFieldBackendAvailable(backend))
        {
            continue;
        }
        expectBackendParity(backend, false, large_origin);
        expectBackendParity(backend, true, large_origin);
        ++evaluated_backends;
    }
    if (evaluated_backends == 0)
    {
        GTEST_SKIP() << "no CUDA or OpenCL device is available";
    }
}

TEST(VisualHullFieldBackendTest, DeviceCancellationIsObservedBetweenSlabs)
{
    const xjw::mesh::VisualHullComputeBackend candidates[] = {xjw::mesh::VisualHullComputeBackend::Cuda,
                                                              xjw::mesh::VisualHullComputeBackend::OpenCL};
    int evaluated_backends = 0;
    for (const xjw::mesh::VisualHullComputeBackend backend : candidates)
    {
        if (!xjw::mesh::detail::isVisualHullFieldBackendAvailable(backend))
        {
            continue;
        }
        xjw::mesh::VisualHullConfig config;
        config.minimumVisibleViews = 1;
        config.computeBackend = backend;
        config.gpuSlabDepth = 2;
        int cancellation_checks = 0;
        config.isCancelled = [&]() { return ++cancellation_checks >= 4; };
        std::vector<float> field;
        std::string error;
        EXPECT_FALSE(
            xjw::mesh::detail::evaluateVisualHullFieldGrid(makeViews(), config, makeGrid(), &field, nullptr, &error));
        EXPECT_NE(error.find("cancelled"), std::string::npos);
        EXPECT_TRUE(field.empty());
        ++evaluated_backends;
    }
    if (evaluated_backends == 0)
    {
        GTEST_SKIP() << "no CUDA or OpenCL device is available";
    }
}
