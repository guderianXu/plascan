#include "DepthComputeScheduler.h"
#include "GpuDeviceLease.h"
#include "PatchMatchCUDA.h"

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDateTime>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace
{

using xjw::mvs::DepthComputeBackend;
using xjw::mvs::DepthComputeScheduler;
using xjw::mvs::DepthComputeWorker;
using xjw::mvs::DepthFrameTask;
using xjw::mvs::GpuDeviceDescriptor;
using xjw::mvs::GpuDeviceLeaseSet;
using xjw::mvs::buildDepthComputeWorkerPool;
using xjw::mvs::fallbackGpuPhysicalIdentity;
using xjw::mvs::isUsableOpenClPatchMatchDevice;
using xjw::mvs::resolveDepthComputeBackend;
using xjw::mvs::resolvePatchMatchEstimatorBackend;

TEST(DepthComputeSchedulerTest, ReturnsHighestPriorityFrameFirst)
{
    DepthComputeScheduler scheduler({{1, 10.0f}, {2, 30.0f}, {3, 20.0f}});
    const DepthComputeWorker worker{DepthComputeBackend::Cuda, 1};

    ASSERT_EQ(scheduler.takeNext(worker), 2);
    ASSERT_EQ(scheduler.takeNext(worker), 3);
    ASSERT_EQ(scheduler.takeNext(worker), 1);
    EXPECT_FALSE(scheduler.takeNext(worker).has_value());
    EXPECT_EQ(worker.id(), "CUDA:1");
}

TEST(DepthComputeSchedulerTest, SharesEachFrameOnceAcrossConcurrentWorkers)
{
    std::vector<DepthFrameTask> tasks;
    for (int index = 0; index < 100; ++index)
    {
        tasks.push_back({index, static_cast<float>(index)});
    }
    DepthComputeScheduler scheduler(std::move(tasks));
    std::vector<int> visits(100, 0);
    std::mutex visits_mutex;

    auto run_worker = [&scheduler, &visits, &visits_mutex](DepthComputeWorker worker)
    {
        while (const std::optional<int> task = scheduler.takeNext(worker))
        {
            {
                std::lock_guard<std::mutex> lock(visits_mutex);
                ++visits[static_cast<std::size_t>(*task)];
            }
            scheduler.complete(worker, std::chrono::milliseconds(1), true);
        }
    };

    std::thread cuda_worker(run_worker,
                            DepthComputeWorker{DepthComputeBackend::Cuda, 0});
    std::thread cpu_worker(run_worker,
                           DepthComputeWorker{DepthComputeBackend::Cpu, 0});
    cuda_worker.join();
    cpu_worker.join();

    for (int count : visits)
    {
        EXPECT_EQ(count, 1);
    }
    EXPECT_EQ(scheduler.pendingTaskCount(), 0U);

    int completed_tasks = 0;
    for (const auto &[worker_id, stats] : scheduler.workerStats())
    {
        EXPECT_FALSE(worker_id.empty());
        completed_tasks += stats.completedTasks;
    }
    EXPECT_EQ(completed_tasks, 100);
}

TEST(DepthComputeSchedulerTest, UsesStableOpenClWorkerName)
{
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 2};
    EXPECT_EQ(opencl_worker.id(), "OpenCL:2");
}

TEST(DepthComputeSchedulerTest, BuildsPreparationSlotsForCallerProvidedMixedPool)
{
    const std::vector<DepthComputeWorker> physical_workers = {
        {DepthComputeBackend::Cuda, 0},
        {DepthComputeBackend::OpenCl, 1}};

    const std::vector<DepthComputeWorker> workers = buildDepthComputeWorkerPool(
        physical_workers, 1, 3, 4);

    ASSERT_EQ(workers.size(), 3U);
    EXPECT_EQ(workers[0].id(), "CUDA:0");
    EXPECT_EQ(workers[1].id(), "OpenCL:1");
    EXPECT_EQ(workers[2].id(), "OpenCL:1");
}

TEST(DepthComputeSchedulerTest, AutomaticBackendUsesStrictAcceleratorPriority)
{
    EXPECT_EQ(resolveDepthComputeBackend(std::nullopt, true, true),
              DepthComputeBackend::Cuda);
    EXPECT_EQ(resolveDepthComputeBackend(std::nullopt, false, true),
              DepthComputeBackend::OpenCl);
    EXPECT_EQ(resolveDepthComputeBackend(std::nullopt, false, false),
              DepthComputeBackend::Cpu);
}

TEST(DepthComputeSchedulerTest, LegacyAutoAccelerationToggleForcesCpu)
{
    EXPECT_EQ(resolveDepthComputeBackend(std::nullopt, true, true, false),
              DepthComputeBackend::Cpu);
    EXPECT_EQ(resolveDepthComputeBackend(
                  DepthComputeBackend::OpenCl, true, false, false),
              DepthComputeBackend::OpenCl);
}

TEST(PatchMatchBackendSelectionTest, OpenClRequiresAvailableDeviceAndOnlineCompiler)
{
    EXPECT_TRUE(isUsableOpenClPatchMatchDevice(true, true, true, true));
    EXPECT_FALSE(isUsableOpenClPatchMatchDevice(true, false, true, true));
    EXPECT_FALSE(isUsableOpenClPatchMatchDevice(true, true, true, false));
    EXPECT_FALSE(isUsableOpenClPatchMatchDevice(false, true, true, true));
    EXPECT_FALSE(isUsableOpenClPatchMatchDevice(true, true, false, true));
}

TEST(PatchMatchBackendSelectionTest, EnumeratedOpenClDevicesAreUsable)
{
    const auto devices = xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    EXPECT_EQ(xjw::mvs::PatchMatchDepthEstimator::isOpenClAvailable(),
              !devices.empty());
    for (const xjw::mvs::OpenClDeviceInfo &device : devices)
    {
        EXPECT_TRUE(device.isGpu);
        EXPECT_TRUE(device.available);
        EXPECT_TRUE(device.compilerAvailable);
    }
}

TEST(PatchMatchBackendSelectionTest, LegacyUseCudaFalseKeepsAutoEstimatorOnCpu)
{
    EXPECT_EQ(resolvePatchMatchEstimatorBackend(
                  xjw::mvs::PatchMatchBackend::Auto, false, true, true),
              xjw::mvs::PatchMatchBackend::Cpu);
}

TEST(PatchMatchBackendSelectionTest, ExplicitBackendOverridesLegacyAutoToggle)
{
    EXPECT_EQ(resolvePatchMatchEstimatorBackend(
                  xjw::mvs::PatchMatchBackend::Cuda, false, false, true),
              xjw::mvs::PatchMatchBackend::Cuda);
    EXPECT_EQ(resolvePatchMatchEstimatorBackend(
                  xjw::mvs::PatchMatchBackend::OpenCl, false, true, false),
              xjw::mvs::PatchMatchBackend::OpenCl);
}

TEST(PatchMatchBackendSelectionTest, EnabledAutoUsesCudaThenOpenClThenCpu)
{
    EXPECT_EQ(resolvePatchMatchEstimatorBackend(
                  xjw::mvs::PatchMatchBackend::Auto, true, true, true),
              xjw::mvs::PatchMatchBackend::Cuda);
    EXPECT_EQ(resolvePatchMatchEstimatorBackend(
                  xjw::mvs::PatchMatchBackend::Auto, true, false, true),
              xjw::mvs::PatchMatchBackend::OpenCl);
    EXPECT_EQ(resolvePatchMatchEstimatorBackend(
                  xjw::mvs::PatchMatchBackend::Auto, true, false, false),
              xjw::mvs::PatchMatchBackend::Cpu);
}

TEST(DepthComputeSchedulerTest, ExplicitBackendIsNeverSubstituted)
{
    EXPECT_EQ(resolveDepthComputeBackend(DepthComputeBackend::Cuda, false, true),
              DepthComputeBackend::Cuda);
    EXPECT_EQ(resolveDepthComputeBackend(DepthComputeBackend::OpenCl, true, false),
              DepthComputeBackend::OpenCl);
    EXPECT_EQ(resolveDepthComputeBackend(DepthComputeBackend::Cpu, true, true),
              DepthComputeBackend::Cpu);
}

TEST(PatchMatchOpenClPreparationTest, RejectsUnknownDeviceIndexWithDiagnostic)
{
    std::string error;
    EXPECT_FALSE(xjw::mvs::PatchMatchDepthEstimator::prepareOpenClDevice(
        1000000, &error));
    EXPECT_FALSE(error.empty());
}

TEST(GpuDeviceLeaseTest, PreventsConcurrentProcessLeaseForSamePhysicalDevice)
{
    const std::string identity = "test-gpu-" + std::to_string(
        QCoreApplication::applicationPid()) + "-" + std::to_string(
        QDateTime::currentMSecsSinceEpoch());
    const std::vector<GpuDeviceDescriptor> devices = {{identity, "Test GPU"}};

    GpuDeviceLeaseSet first;
    QString error;
    ASSERT_TRUE(first.acquire(devices, &error)) << error.toStdString();

    GpuDeviceLeaseSet second;
    EXPECT_FALSE(second.acquire(devices, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("Test GPU")));
}

TEST(GpuDeviceLeaseTest, BusyDeviceDoesNotPreventLeasingAnotherDevice)
{
    const std::string prefix = "test-gpu-pool-" + std::to_string(
        QCoreApplication::applicationPid()) + "-" + std::to_string(
        QDateTime::currentMSecsSinceEpoch());
    const GpuDeviceDescriptor busy_device{prefix + "-busy", "Busy Test GPU"};
    const GpuDeviceDescriptor free_device{prefix + "-free", "Free Test GPU"};

    GpuDeviceLeaseSet occupied;
    QString error;
    ASSERT_TRUE(occupied.acquire({busy_device}, &error)) << error.toStdString();

    GpuDeviceLeaseSet busy_candidate;
    EXPECT_FALSE(busy_candidate.acquire({busy_device}, &error));

    GpuDeviceLeaseSet free_candidate;
    EXPECT_TRUE(free_candidate.acquire({free_device}, &error))
        << error.toStdString();
}

TEST(GpuDeviceLeaseTest, NormalizesFallbackIdentityDeterministically)
{
    EXPECT_EQ(fallbackGpuPhysicalIdentity("Advanced Micro Devices, Inc.",
                                          "AMD Radeon(TM) Graphics",
                                          0),
              fallbackGpuPhysicalIdentity("Advanced Micro Devices, Inc.",
                                          "AMD Radeon(TM) Graphics",
                                          0));
}

TEST(DepthComputeSchedulerTest, AddsOpenClPreparationLaneWhenCudaIsUnavailable)
{
    const std::vector<DepthComputeWorker> workers = buildDepthComputeWorkerPool(
        {{DepthComputeBackend::OpenCl, 0}}, 0, 2, 4);

    ASSERT_EQ(workers.size(), 2U);
    EXPECT_EQ(workers[0].id(), "OpenCL:0");
    EXPECT_EQ(workers[1].id(), "OpenCL:0");
}

TEST(DepthComputeSchedulerTest, CapsWorkersWithoutDroppingEarlierPhysicalDevices)
{
    const std::vector<DepthComputeWorker> workers = buildDepthComputeWorkerPool(
        {{DepthComputeBackend::Cuda, 0},
         {DepthComputeBackend::Cuda, 1},
         {DepthComputeBackend::OpenCl, 2}},
        4,
        2,
        3);

    ASSERT_EQ(workers.size(), 3U);
    EXPECT_EQ(workers[0].id(), "CUDA:0");
    EXPECT_EQ(workers[1].id(), "CUDA:1");
    EXPECT_EQ(workers[2].id(), "OpenCL:2");
}

} // namespace
