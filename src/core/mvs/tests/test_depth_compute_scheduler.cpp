#include "DepthComputeScheduler.h"
#include "GpuDeviceLease.h"

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

TEST(DepthComputeSchedulerTest, SharesEachFrameOnceAcrossHeterogeneousWorkers)
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

TEST(DepthComputeSchedulerTest, CapsMixedOpenClBackendAtOnePreparationWorker)
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
