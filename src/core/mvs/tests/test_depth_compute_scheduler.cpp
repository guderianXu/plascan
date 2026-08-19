#include "DepthComputeScheduler.h"
#include "GpuDeviceLease.h"
#include "PatchMatchCUDA.h"

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDateTime>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace
{

using xjw::mvs::DepthComputeBackend;
using xjw::mvs::DepthComputeSchedulingPolicy;
using xjw::mvs::DepthComputeScheduler;
using xjw::mvs::DepthComputeWorker;
using xjw::mvs::DepthTaskCompletionResult;
using xjw::mvs::DepthFrameTask;
using xjw::mvs::DepthTaskClaim;
using xjw::mvs::DepthTaskClaimStatus;
using xjw::mvs::GpuDeviceDescriptor;
using xjw::mvs::GpuDeviceLeaseSet;
using xjw::mvs::buildDepthComputeWorkerPool;
using xjw::mvs::depthComputeWorkerFromId;
using xjw::mvs::fallbackGpuPhysicalIdentity;
using xjw::mvs::isNvidiaOpenClVendor;
using xjw::mvs::isUsableOpenClPatchMatchDevice;
using xjw::mvs::normalizedGpuDeviceName;
using xjw::mvs::recommendedOpenClFullFrameFloorPerDevice;
using xjw::mvs::resolveDepthComputeBackend;
using xjw::mvs::resolvePatchMatchEstimatorBackend;
using xjw::mvs::shouldSkipUnstableOpenClCudaAlias;

TEST(DepthComputeSchedulerTest, ReturnsHighestPriorityFrameFirst)
{
    DepthComputeScheduler scheduler({{1, 10.0f}, {2, 30.0f}, {3, 20.0f}});
    const DepthComputeWorker worker{DepthComputeBackend::Cuda, 1};

    DepthTaskClaim claim = scheduler.claimNext(worker);
    ASSERT_EQ(claim.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(claim.viewIndex, 2);
    claim = scheduler.claimNext(worker);
    ASSERT_EQ(claim.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(claim.viewIndex, 3);
    claim = scheduler.claimNext(worker);
    ASSERT_EQ(claim.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(claim.viewIndex, 1);
    EXPECT_EQ(scheduler.claimNext(worker).status, DepthTaskClaimStatus::Exhausted);
    EXPECT_EQ(worker.id(), "CUDA:1");
}

TEST(DepthComputeSchedulerTest, OpenClFloorScalesWithPhysicalCudaDevices)
{
    EXPECT_EQ(recommendedOpenClFullFrameFloorPerDevice(true, 63, 1, 1), 0);
    EXPECT_EQ(recommendedOpenClFullFrameFloorPerDevice(true, 64, 1, 1), 1);
    EXPECT_EQ(recommendedOpenClFullFrameFloorPerDevice(true, 127, 2, 1), 0);
    EXPECT_EQ(recommendedOpenClFullFrameFloorPerDevice(true, 128, 2, 1), 1);
    EXPECT_EQ(recommendedOpenClFullFrameFloorPerDevice(false, 100, 1, 1), 0);
    EXPECT_EQ(recommendedOpenClFullFrameFloorPerDevice(true, 100, 1, 0), 0);
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
        while (true)
        {
            const DepthTaskClaim claim = scheduler.claimNext(worker);
            if (claim.status != DepthTaskClaimStatus::Task)
            {
                break;
            }
            {
                std::lock_guard<std::mutex> lock(visits_mutex);
                ++visits[static_cast<std::size_t>(claim.viewIndex)];
            }
            scheduler.complete(
                worker, claim.viewIndex, std::chrono::milliseconds(1), true);
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

TEST(DepthComputeSchedulerTest, DuplicateSlotsOnlyClaimOneCalibrationFrame)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 2.0f}, {2, 1.0f}},
        true,
        {cuda_worker, opencl_worker});

    const DepthTaskClaim first_cuda_claim = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(first_cuda_claim.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(first_cuda_claim.viewIndex, 1);
    EXPECT_TRUE(first_cuda_claim.calibrationProbe);
    EXPECT_EQ(scheduler.claimNext(cuda_worker).status,
              DepthTaskClaimStatus::Retry);

    const DepthTaskClaim opencl_claim = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(opencl_claim.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(opencl_claim.viewIndex, 2);
    EXPECT_TRUE(opencl_claim.calibrationProbe);
}

TEST(DepthComputeSchedulerTest, ReportsFastestSuccessfulAlternativeBackendEma)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker second_cuda_worker{DepthComputeBackend::Cuda, 1};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 2};
    DepthComputeScheduler scheduler(
        {{1, 4.0f}, {2, 3.0f}, {3, 2.0f}, {4, 1.0f}},
        true,
        {cuda_worker, second_cuda_worker, opencl_worker});

    EXPECT_FALSE(
        scheduler.fastestSuccessfulAlternativeBackendEmaMilliseconds(
            opencl_worker).has_value());

    const DepthTaskClaim cuda_claim = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        cuda_worker, cuda_claim.viewIndex, std::chrono::milliseconds(40), true);

    const auto opencl_alternative =
        scheduler.fastestSuccessfulAlternativeBackendEmaMilliseconds(
            opencl_worker);
    ASSERT_TRUE(opencl_alternative.has_value());
    EXPECT_DOUBLE_EQ(*opencl_alternative, 40.0);
    EXPECT_FALSE(
        scheduler.fastestSuccessfulAlternativeBackendEmaMilliseconds(
            second_cuda_worker).has_value());
}

TEST(DepthComputeSchedulerTest, AtomicallyRequeuesUnprofitableCalibrationProbe)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 4.0f}, {2, 3.0f}, {3, 2.0f}, {4, 1.0f}},
        true,
        {cuda_worker, opencl_worker});

    const DepthTaskClaim cuda_claim = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        cuda_worker, cuda_claim.viewIndex, std::chrono::milliseconds(40), true);

    const DepthTaskClaim opencl_probe = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(opencl_probe.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(opencl_probe.calibrationProbe);
    const DepthTaskClaim cuda_followup = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_followup.status, DepthTaskClaimStatus::Task);
    EXPECT_FALSE(scheduler.tryRejectUnprofitableCalibrationProbe(
        opencl_worker,
        opencl_probe.viewIndex,
        39.0,
        std::chrono::milliseconds(50)).has_value());

    const auto alternative = scheduler.tryRejectUnprofitableCalibrationProbe(
        opencl_worker,
        opencl_probe.viewIndex,
        40.0,
        std::chrono::milliseconds(51));
    ASSERT_TRUE(alternative.has_value());
    EXPECT_DOUBLE_EQ(*alternative, 40.0);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Retire);

    // If the alternative fails immediately after the atomic handoff, it is
    // now the last healthy backend and therefore remains available to drain
    // the already-committed retry.
    const DepthTaskCompletionResult cuda_failure = scheduler.complete(
        cuda_worker,
        cuda_followup.viewIndex,
        std::chrono::milliseconds(1),
        false);
    EXPECT_FALSE(cuda_failure.retryScheduled);
    EXPECT_FALSE(cuda_failure.workerRetired);

    const DepthTaskClaim retry = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(retry.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(retry.viewIndex, opencl_probe.viewIndex);
    EXPECT_FALSE(scheduler.complete(
        opencl_worker,
        opencl_probe.viewIndex,
        std::chrono::milliseconds(52),
        false).accepted);

    const auto stats = scheduler.workerStats();
    EXPECT_EQ(stats.at(opencl_worker.id()).completedTasks, 1);
    EXPECT_EQ(stats.at(opencl_worker.id()).failedTasks, 1);
    EXPECT_DOUBLE_EQ(stats.at(opencl_worker.id()).elapsedMilliseconds, 51.0);
    EXPECT_DOUBLE_EQ(stats.at(opencl_worker.id()).emaElapsedMilliseconds, 0.0);
}

TEST(DepthComputeSchedulerTest, GuaranteedOpenClFrameRunsPastCoarseGate)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeSchedulingPolicy policy;
    policy.guaranteedOpenClFullFramesPerDevice = 1;
    policy.maximumOpenClInFlightTasksPerDevice = 1;
    DepthComputeScheduler scheduler(
        {{1, 8.0f},
         {2, 7.0f},
         {3, 6.0f},
         {4, 5.0f},
         {5, 4.0f},
         {6, 3.0f},
         {7, 2.0f},
         {8, 1.0f}},
        true,
        {cuda_worker, opencl_worker, opencl_worker},
        policy);

    const DepthTaskClaim cuda_calibration = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_calibration.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        cuda_worker,
        cuda_calibration.viewIndex,
        std::chrono::milliseconds(10),
        true).accepted);

    const DepthTaskClaim opencl_calibration = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(opencl_calibration.status, DepthTaskClaimStatus::Task);
    EXPECT_TRUE(opencl_calibration.calibrationProbe);
    EXPECT_TRUE(opencl_calibration.requiresFullFrame);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Retry);
    EXPECT_FALSE(scheduler.tryRejectUnprofitableCalibrationProbe(
        opencl_worker,
        opencl_calibration.viewIndex,
        100.0,
        std::chrono::milliseconds(101)).has_value());

    // Let CUDA drain enough of the queue while the protected OpenCL frame is
    // running. The successful full frame satisfies the floor, after which the
    // normal tail-benefit rule prevents a second slow assignment.
    for (int completed = 0; completed < 4; ++completed)
    {
        const DepthTaskClaim cuda_claim = scheduler.claimNext(cuda_worker);
        ASSERT_EQ(cuda_claim.status, DepthTaskClaimStatus::Task);
        ASSERT_TRUE(scheduler.complete(
            cuda_worker,
            cuda_claim.viewIndex,
            std::chrono::milliseconds(10),
            true).accepted);
    }
    ASSERT_TRUE(scheduler.complete(
        opencl_worker,
        opencl_calibration.viewIndex,
        std::chrono::milliseconds(100),
        true).accepted);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Retry);
}

TEST(DepthComputeSchedulerTest, FailedGuaranteedOpenClFrameRetriesOnceOnCuda)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeSchedulingPolicy policy;
    policy.guaranteedOpenClFullFramesPerDevice = 1;
    policy.maximumOpenClInFlightTasksPerDevice = 1;
    DepthComputeScheduler scheduler(
        {{1, 4.0f}, {2, 3.0f}, {3, 2.0f}, {4, 1.0f}},
        true,
        {cuda_worker, opencl_worker},
        policy);

    const DepthTaskClaim cuda_calibration = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_calibration.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        cuda_worker,
        cuda_calibration.viewIndex,
        std::chrono::milliseconds(10),
        true).accepted);

    const DepthTaskClaim protected_opencl = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(protected_opencl.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(protected_opencl.requiresFullFrame);
    const DepthTaskCompletionResult failed_opencl = scheduler.complete(
        opencl_worker,
        protected_opencl.viewIndex,
        std::chrono::milliseconds(100),
        false);
    EXPECT_TRUE(failed_opencl.accepted);
    EXPECT_TRUE(failed_opencl.retryScheduled);
    EXPECT_TRUE(failed_opencl.workerRetired);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Retire);

    const DepthTaskClaim cuda_retry = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_retry.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(cuda_retry.viewIndex, protected_opencl.viewIndex);
    const DepthTaskCompletionResult retry_completion = scheduler.complete(
        cuda_worker,
        cuda_retry.viewIndex,
        std::chrono::milliseconds(10),
        true);
    EXPECT_TRUE(retry_completion.accepted);
    EXPECT_FALSE(retry_completion.retryScheduled);
}

TEST(DepthComputeSchedulerTest, GuaranteesOneFrameForEachPhysicalOpenClDevice)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker first_opencl{DepthComputeBackend::OpenCl, 1};
    const DepthComputeWorker second_opencl{DepthComputeBackend::OpenCl, 2};
    DepthComputeSchedulingPolicy policy;
    policy.guaranteedOpenClFullFramesPerDevice = 1;
    policy.maximumOpenClInFlightTasksPerDevice = 1;
    DepthComputeScheduler scheduler(
        {{1, 6.0f},
         {2, 5.0f},
         {3, 4.0f},
         {4, 3.0f},
         {5, 2.0f},
         {6, 1.0f}},
        true,
        {cuda_worker, first_opencl, first_opencl, second_opencl, second_opencl},
        policy);

    const DepthTaskClaim cuda_calibration = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_calibration.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        cuda_worker,
        cuda_calibration.viewIndex,
        std::chrono::milliseconds(10),
        true).accepted);

    const DepthTaskClaim first_claim = scheduler.claimNext(first_opencl);
    ASSERT_EQ(first_claim.status, DepthTaskClaimStatus::Task);
    EXPECT_TRUE(first_claim.requiresFullFrame);
    EXPECT_EQ(scheduler.claimNext(first_opencl).status,
              DepthTaskClaimStatus::Retry);

    const DepthTaskClaim second_claim = scheduler.claimNext(second_opencl);
    ASSERT_EQ(second_claim.status, DepthTaskClaimStatus::Task);
    EXPECT_TRUE(second_claim.requiresFullFrame);
    EXPECT_NE(second_claim.viewIndex, first_claim.viewIndex);
    EXPECT_EQ(scheduler.claimNext(second_opencl).status,
              DepthTaskClaimStatus::Retry);
}

TEST(DepthComputeSchedulerTest, ConcurrentOpenClSlotsKeepOnePhysicalFrameInFlight)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeSchedulingPolicy policy;
    policy.guaranteedOpenClFullFramesPerDevice = 1;
    policy.maximumOpenClInFlightTasksPerDevice = 1;
    std::vector<DepthFrameTask> tasks;
    for (int index = 0; index < 10; ++index)
    {
        tasks.push_back({index, static_cast<float>(10 - index)});
    }
    DepthComputeScheduler scheduler(
        std::move(tasks),
        true,
        {cuda_worker, opencl_worker, opencl_worker},
        policy);

    constexpr int contender_count = 8;
    std::atomic<bool> start{false};
    std::vector<DepthTaskClaimStatus> statuses(contender_count);
    std::vector<std::thread> contenders;
    contenders.reserve(contender_count);
    for (int index = 0; index < contender_count; ++index)
    {
        contenders.emplace_back([&scheduler,
                                 &opencl_worker,
                                 &start,
                                 &statuses,
                                 index]
        {
            while (!start.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }
            statuses[static_cast<std::size_t>(index)] =
                scheduler.claimNext(opencl_worker).status;
        });
    }
    start.store(true, std::memory_order_release);
    for (std::thread &contender : contenders)
    {
        contender.join();
    }

    EXPECT_EQ(std::count(statuses.begin(),
                         statuses.end(),
                         DepthTaskClaimStatus::Task),
              1);
    EXPECT_EQ(std::count(statuses.begin(),
                         statuses.end(),
                         DepthTaskClaimStatus::Retry),
              contender_count - 1);
}

TEST(DepthComputeSchedulerTest, BoundedOpenClPipelineAllowsTwoPhysicalFramesInFlight)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeSchedulingPolicy policy;
    policy.maximumOpenClInFlightTasksPerDevice = 2;
    DepthComputeScheduler scheduler(
        {{1, 4.0f}, {2, 3.0f}, {3, 2.0f}, {4, 1.0f}},
        false,
        {cuda_worker, opencl_worker, opencl_worker},
        policy);

    const DepthTaskClaim first = scheduler.claimNext(opencl_worker);
    const DepthTaskClaim second = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(first.status, DepthTaskClaimStatus::Task);
    ASSERT_EQ(second.status, DepthTaskClaimStatus::Task);
    EXPECT_NE(first.viewIndex, second.viewIndex);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Retry);

    ASSERT_TRUE(scheduler.complete(
        opencl_worker,
        first.viewIndex,
        std::chrono::milliseconds(10),
        true).accepted);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Task);
}

TEST(DepthComputeSchedulerTest, FastOpenClMayClaimNormalFrameAfterFloor)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeSchedulingPolicy policy;
    policy.guaranteedOpenClFullFramesPerDevice = 1;
    policy.maximumOpenClInFlightTasksPerDevice = 1;
    DepthComputeScheduler scheduler(
        {{1, 6.0f},
         {2, 5.0f},
         {3, 4.0f},
         {4, 3.0f},
         {5, 2.0f},
         {6, 1.0f}},
        true,
        {cuda_worker, opencl_worker},
        policy);

    const DepthTaskClaim cuda_calibration = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_calibration.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        cuda_worker,
        cuda_calibration.viewIndex,
        std::chrono::milliseconds(100),
        true).accepted);
    const DepthTaskClaim protected_opencl = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(protected_opencl.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(protected_opencl.requiresFullFrame);
    ASSERT_TRUE(scheduler.complete(
        opencl_worker,
        protected_opencl.viewIndex,
        std::chrono::milliseconds(50),
        true).accepted);

    const DepthTaskClaim normal_opencl = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(normal_opencl.status, DepthTaskClaimStatus::Task);
    EXPECT_FALSE(normal_opencl.calibrationProbe);
    EXPECT_FALSE(normal_opencl.requiresFullFrame);
}

TEST(DepthComputeSchedulerTest, KeepsProbeWhenAlternativeRetiresBeforeAtomicGate)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 5.0f}, {2, 4.0f}, {3, 3.0f}, {4, 2.0f}, {5, 1.0f}},
        true,
        {cuda_worker, opencl_worker});

    const DepthTaskClaim cuda_calibration = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_calibration.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        cuda_worker,
        cuda_calibration.viewIndex,
        std::chrono::milliseconds(40),
        true);

    const DepthTaskClaim opencl_probe = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(opencl_probe.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(opencl_probe.calibrationProbe);

    const DepthTaskClaim cuda_failure = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_failure.status, DepthTaskClaimStatus::Task);
    const DepthTaskCompletionResult cuda_completion = scheduler.complete(
        cuda_worker,
        cuda_failure.viewIndex,
        std::chrono::milliseconds(1),
        false);
    ASSERT_TRUE(cuda_completion.retryScheduled);
    ASSERT_TRUE(cuda_completion.workerRetired);

    EXPECT_FALSE(scheduler.tryRejectUnprofitableCalibrationProbe(
        opencl_worker,
        opencl_probe.viewIndex,
        100.0,
        std::chrono::milliseconds(110)).has_value());
    const DepthTaskCompletionResult opencl_completion = scheduler.complete(
        opencl_worker,
        opencl_probe.viewIndex,
        std::chrono::milliseconds(120),
        true);
    EXPECT_TRUE(opencl_completion.accepted);
    EXPECT_FALSE(opencl_completion.retryScheduled);

    const DepthTaskClaim retry = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(retry.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(retry.viewIndex, cuda_failure.viewIndex);
}

TEST(DepthComputeSchedulerTest, RetryWaitObservesAlreadyCompletedStateChange)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 3.0f}, {2, 2.0f}, {3, 1.0f}},
        true,
        {cuda_worker, opencl_worker});

    const DepthTaskClaim first_claim = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(first_claim.status, DepthTaskClaimStatus::Task);
    const DepthTaskClaim retry_claim = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(retry_claim.status, DepthTaskClaimStatus::Retry);

    scheduler.complete(
        cuda_worker, first_claim.viewIndex, std::chrono::milliseconds(10), true);
    EXPECT_TRUE(scheduler.waitForStateChange(
        retry_claim.revision, std::chrono::milliseconds(10)));
    EXPECT_EQ(scheduler.claimNext(cuda_worker).status,
              DepthTaskClaimStatus::Task);
}

TEST(DepthComputeSchedulerTest, PausesSlowWorkerWhenFastWorkerCanClearQueueFirst)
{
    const DepthComputeWorker fast_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker slow_worker{DepthComputeBackend::OpenCl, 1};
    std::vector<DepthFrameTask> tasks;
    for (int index = 0; index < 8; ++index)
    {
        tasks.push_back({index, static_cast<float>(8 - index)});
    }
    DepthComputeScheduler scheduler(
        std::move(tasks), true, {fast_worker, slow_worker});

    const DepthTaskClaim fast_claim = scheduler.claimNext(fast_worker);
    ASSERT_EQ(fast_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        fast_worker, fast_claim.viewIndex, std::chrono::milliseconds(10), true);
    const DepthTaskClaim slow_claim = scheduler.claimNext(slow_worker);
    ASSERT_EQ(slow_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        slow_worker, slow_claim.viewIndex, std::chrono::milliseconds(100), true);

    EXPECT_EQ(scheduler.claimNext(slow_worker).status,
              DepthTaskClaimStatus::Retry);
    EXPECT_EQ(scheduler.claimNext(slow_worker).status,
              DepthTaskClaimStatus::Retry);
    EXPECT_EQ(scheduler.claimNext(fast_worker).status,
              DepthTaskClaimStatus::Task);
    EXPECT_EQ(scheduler.pendingTaskCount(), 5U);
}

TEST(DepthComputeSchedulerTest, KeepsSlowWorkerWhenLongQueueBenefitsFromIt)
{
    const DepthComputeWorker fast_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker slow_worker{DepthComputeBackend::OpenCl, 1};
    std::vector<DepthFrameTask> tasks;
    for (int index = 0; index < 13; ++index)
    {
        tasks.push_back({index, static_cast<float>(13 - index)});
    }
    DepthComputeScheduler scheduler(
        std::move(tasks), true, {fast_worker, slow_worker});

    const DepthTaskClaim fast_claim = scheduler.claimNext(fast_worker);
    ASSERT_EQ(fast_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        fast_worker, fast_claim.viewIndex, std::chrono::milliseconds(10), true);
    const DepthTaskClaim slow_claim = scheduler.claimNext(slow_worker);
    ASSERT_EQ(slow_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        slow_worker, slow_claim.viewIndex, std::chrono::milliseconds(100), true);

    EXPECT_EQ(scheduler.claimNext(slow_worker).status,
              DepthTaskClaimStatus::Task);
    EXPECT_EQ(scheduler.pendingTaskCount(), 10U);
}

TEST(DepthComputeSchedulerTest, FastInFlightTaskCanMakeSlowWorkerProfitable)
{
    const DepthComputeWorker fast_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker slow_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 4.0f}, {2, 3.0f}, {3, 2.0f}, {4, 1.0f}},
        true,
        {fast_worker, slow_worker});

    const DepthTaskClaim fast_claim = scheduler.claimNext(fast_worker);
    ASSERT_EQ(fast_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        fast_worker, fast_claim.viewIndex, std::chrono::milliseconds(10), true);
    const DepthTaskClaim slow_claim = scheduler.claimNext(slow_worker);
    ASSERT_EQ(slow_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        slow_worker, slow_claim.viewIndex, std::chrono::milliseconds(15), true);

    ASSERT_EQ(scheduler.claimNext(fast_worker).status,
              DepthTaskClaimStatus::Task);
    EXPECT_EQ(scheduler.claimNext(slow_worker).status,
              DepthTaskClaimStatus::Task);
}

TEST(DepthComputeSchedulerTest, SlowInFlightTaskPreventsSecondTailClaim)
{
    const DepthComputeWorker fast_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker slow_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 5.0f}, {2, 4.0f}, {3, 3.0f}, {4, 2.0f}, {5, 1.0f}},
        true,
        {fast_worker, slow_worker});

    const DepthTaskClaim fast_claim = scheduler.claimNext(fast_worker);
    ASSERT_EQ(fast_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        fast_worker, fast_claim.viewIndex, std::chrono::milliseconds(10), true);
    const DepthTaskClaim slow_claim = scheduler.claimNext(slow_worker);
    ASSERT_EQ(slow_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        slow_worker, slow_claim.viewIndex, std::chrono::milliseconds(15), true);

    ASSERT_EQ(scheduler.claimNext(slow_worker).status,
              DepthTaskClaimStatus::Task);
    EXPECT_EQ(scheduler.claimNext(slow_worker).status,
              DepthTaskClaimStatus::Retry);
}

TEST(DepthComputeSchedulerTest, TailEstimateUsesPhysicalCudaSlotThroughput)
{
    const DepthComputeWorker cuda_slot{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 6.0f},
         {2, 5.0f},
         {3, 4.0f},
         {4, 3.0f},
         {5, 2.0f},
         {6, 1.0f}},
        true,
        {cuda_slot, cuda_slot, opencl_worker});

    const DepthTaskClaim cuda_calibration = scheduler.claimNext(cuda_slot);
    ASSERT_EQ(cuda_calibration.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        cuda_slot,
        cuda_calibration.viewIndex,
        std::chrono::milliseconds(100),
        true).accepted);
    const DepthTaskClaim opencl_calibration = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(opencl_calibration.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        opencl_worker,
        opencl_calibration.viewIndex,
        std::chrono::milliseconds(250),
        true).accepted);

    const DepthTaskClaim first_saturated = scheduler.claimNext(cuda_slot);
    const DepthTaskClaim second_saturated = scheduler.claimNext(cuda_slot);
    ASSERT_EQ(first_saturated.status, DepthTaskClaimStatus::Task);
    ASSERT_EQ(second_saturated.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        cuda_slot,
        first_saturated.viewIndex,
        std::chrono::milliseconds(100),
        true).accepted);
    const DepthTaskClaim replacement = scheduler.claimNext(cuda_slot);
    ASSERT_EQ(replacement.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        cuda_slot,
        second_saturated.viewIndex,
        std::chrono::milliseconds(200),
        true).accepted);
    const auto physical_cuda_service =
        scheduler.fastestSuccessfulAlternativeBackendEmaMilliseconds(
            opencl_worker);
    ASSERT_TRUE(physical_cuda_service.has_value());
    EXPECT_DOUBLE_EQ(*physical_cuda_service, 100.0);

    // The second saturated completion took 200 ms with two frames in flight,
    // establishing 100 ms physical service. Host-frame EMA is 135 ms here and
    // would incorrectly give the 250 ms OpenCL worker the final queued task.
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Retry);
}

TEST(DepthComputeSchedulerTest, DuplicateHostSlotsShareOnePhysicalWorkerSample)
{
    const DepthComputeWorker opencl_slot{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 2.0f}, {2, 1.0f}},
        true,
        {opencl_slot, opencl_slot});

    const DepthTaskClaim first_claim = scheduler.claimNext(opencl_slot);
    ASSERT_EQ(first_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        opencl_slot, first_claim.viewIndex, std::chrono::milliseconds(80), true);
    const DepthTaskClaim second_claim = scheduler.claimNext(opencl_slot);
    ASSERT_EQ(second_claim.status, DepthTaskClaimStatus::Task);
    EXPECT_FALSE(second_claim.calibrationProbe);
    scheduler.complete(
        opencl_slot, second_claim.viewIndex, std::chrono::milliseconds(40), true);

    const auto stats = scheduler.workerStats();
    ASSERT_EQ(stats.size(), 1U);
    const auto stats_it = stats.find(opencl_slot.id());
    ASSERT_NE(stats_it, stats.end());
    EXPECT_EQ(stats_it->second.completedTasks, 2);
    EXPECT_EQ(stats_it->second.successfulTasks, 2);
    EXPECT_DOUBLE_EQ(stats_it->second.elapsedMilliseconds, 120.0);
    EXPECT_DOUBLE_EQ(stats_it->second.emaElapsedMilliseconds, 66.0);
}

TEST(DepthComputeSchedulerTest, FewerTasksThanDevicesUsesLegacySharedQueue)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 1.0f}}, true, {cuda_worker, opencl_worker});

    EXPECT_EQ(scheduler.claimNext(cuda_worker).status,
              DepthTaskClaimStatus::Task);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Exhausted);
}

TEST(DepthComputeSchedulerTest, LegacySchedulingNeverStopsSlowWorkerAtTail)
{
    const DepthComputeWorker fast_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker slow_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler({{1, 3.0f}, {2, 2.0f}, {3, 1.0f}});

    const DepthTaskClaim fast_claim = scheduler.claimNext(fast_worker);
    ASSERT_EQ(fast_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        fast_worker, fast_claim.viewIndex, std::chrono::milliseconds(10), true);
    const DepthTaskClaim slow_claim = scheduler.claimNext(slow_worker);
    ASSERT_EQ(slow_claim.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        slow_worker, slow_claim.viewIndex, std::chrono::milliseconds(100), true);

    EXPECT_EQ(scheduler.claimNext(slow_worker).status,
              DepthTaskClaimStatus::Task);
}

TEST(DepthComputeSchedulerTest, LegacyFailureIsFinalWithoutRetiringWorker)
{
    const DepthComputeWorker worker{DepthComputeBackend::Cuda, 0};
    DepthComputeScheduler scheduler({{1, 2.0f}, {2, 1.0f}});

    const DepthTaskClaim failed_claim = scheduler.claimNext(worker);
    ASSERT_EQ(failed_claim.status, DepthTaskClaimStatus::Task);
    const auto failure = scheduler.complete(
        worker,
        failed_claim.viewIndex,
        std::chrono::milliseconds(1),
        false);
    EXPECT_TRUE(failure.accepted);
    EXPECT_FALSE(failure.retryScheduled);
    EXPECT_FALSE(failure.workerRetired);

    const DepthTaskClaim next_claim = scheduler.claimNext(worker);
    ASSERT_EQ(next_claim.status, DepthTaskClaimStatus::Task);
    EXPECT_NE(next_claim.viewIndex, failed_claim.viewIndex);
    EXPECT_EQ(scheduler.pendingTaskCount(), 0U);
}

TEST(DepthComputeSchedulerTest, RejectsCompletionFromWorkerThatDoesNotOwnView)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler({{1, 1.0f}});

    const DepthTaskClaim claim = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(claim.status, DepthTaskClaimStatus::Task);
    const auto rejected = scheduler.complete(
        opencl_worker,
        claim.viewIndex,
        std::chrono::milliseconds(1),
        true);
    EXPECT_FALSE(rejected.accepted);

    const auto accepted = scheduler.complete(
        cuda_worker,
        claim.viewIndex,
        std::chrono::milliseconds(1),
        true);
    EXPECT_TRUE(accepted.accepted);
    EXPECT_EQ(scheduler.workerStats().at(cuda_worker.id()).completedTasks, 1);
}

TEST(DepthComputeSchedulerTest, ConcurrentHybridWorkersClaimEveryFrameExactlyOnce)
{
    std::vector<DepthFrameTask> tasks;
    for (int index = 0; index < 100; ++index)
    {
        tasks.push_back({index, static_cast<float>(index)});
    }

    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        std::move(tasks), true, {cuda_worker, opencl_worker});
    std::vector<int> visits(100, 0);
    std::mutex visits_mutex;

    const auto run_worker = [&scheduler, &visits, &visits_mutex](
                                const DepthComputeWorker worker,
                                const std::chrono::milliseconds elapsed)
    {
        while (true)
        {
            const DepthTaskClaim claim = scheduler.claimNext(worker);
            if (claim.status == DepthTaskClaimStatus::Retry)
            {
                scheduler.waitForStateChange(
                    claim.revision, std::chrono::milliseconds(2));
                continue;
            }
            if (claim.status != DepthTaskClaimStatus::Task)
            {
                break;
            }
            {
                std::lock_guard<std::mutex> lock(visits_mutex);
                ++visits[static_cast<std::size_t>(claim.viewIndex)];
            }
            scheduler.complete(worker, claim.viewIndex, elapsed, true);
        }
    };

    std::thread cuda_slot_1(run_worker, cuda_worker, std::chrono::milliseconds(10));
    std::thread cuda_slot_2(run_worker, cuda_worker, std::chrono::milliseconds(10));
    std::thread opencl_slot_1(
        run_worker, opencl_worker, std::chrono::milliseconds(80));
    std::thread opencl_slot_2(
        run_worker, opencl_worker, std::chrono::milliseconds(80));
    cuda_slot_1.join();
    cuda_slot_2.join();
    opencl_slot_1.join();
    opencl_slot_2.join();

    for (const int count : visits)
    {
        EXPECT_EQ(count, 1);
    }
    EXPECT_EQ(scheduler.pendingTaskCount(), 0U);
}

TEST(DepthComputeSchedulerTest, ConcurrentFailureRetriesExactlyOnceWithoutDroppingFrame)
{
    std::vector<DepthFrameTask> tasks;
    for (int index = 0; index < 50; ++index)
    {
        tasks.push_back({index, static_cast<float>(index)});
    }

    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        std::move(tasks), true, {cuda_worker, opencl_worker});
    const DepthTaskClaim failed_claim = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(failed_claim.status, DepthTaskClaimStatus::Task);

    std::vector<int> attempts(50, 0);
    std::vector<int> successes(50, 0);
    std::mutex visits_mutex;
    attempts[static_cast<std::size_t>(failed_claim.viewIndex)] = 1;
    std::atomic<bool> start{false};

    std::thread cuda_thread([&]
    {
        while (!start.load())
        {
            std::this_thread::yield();
        }
        while (true)
        {
            const DepthTaskClaim claim = scheduler.claimNext(cuda_worker);
            if (claim.status == DepthTaskClaimStatus::Retry)
            {
                scheduler.waitForStateChange(
                    claim.revision, std::chrono::milliseconds(2));
                continue;
            }
            if (claim.status != DepthTaskClaimStatus::Task)
            {
                break;
            }
            {
                std::lock_guard<std::mutex> lock(visits_mutex);
                ++attempts[static_cast<std::size_t>(claim.viewIndex)];
                ++successes[static_cast<std::size_t>(claim.viewIndex)];
            }
            scheduler.complete(
                cuda_worker,
                claim.viewIndex,
                std::chrono::milliseconds(10),
                true);
        }
    });
    std::thread failure_thread([&]
    {
        while (!start.load())
        {
            std::this_thread::yield();
        }
        const auto completion = scheduler.complete(
            opencl_worker,
            failed_claim.viewIndex,
            std::chrono::milliseconds(1),
            false);
        EXPECT_TRUE(completion.retryScheduled);
        EXPECT_TRUE(completion.workerRetired);
    });

    start = true;
    cuda_thread.join();
    failure_thread.join();

    for (int view_index = 0; view_index < 50; ++view_index)
    {
        EXPECT_EQ(successes[static_cast<std::size_t>(view_index)], 1);
        EXPECT_EQ(attempts[static_cast<std::size_t>(view_index)],
                  view_index == failed_claim.viewIndex ? 2 : 1);
    }
    EXPECT_EQ(scheduler.pendingTaskCount(), 0U);
}

TEST(DepthComputeSchedulerTest, FailedCalibrationRetiresWorkerWithoutTrainingEma)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 5.0f}, {2, 4.0f}, {3, 3.0f}, {4, 2.0f}, {5, 1.0f}},
        true,
        {cuda_worker, opencl_worker});

    const DepthTaskClaim failed_claim = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(failed_claim.status, DepthTaskClaimStatus::Task);
    const auto failure = scheduler.complete(
        opencl_worker,
        failed_claim.viewIndex,
        std::chrono::milliseconds(1),
        false);
    EXPECT_TRUE(failure.accepted);
    EXPECT_TRUE(failure.retryScheduled);
    EXPECT_TRUE(failure.workerRetired);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Retire);

    bool retried_failed_view = false;
    while (true)
    {
        const DepthTaskClaim claim = scheduler.claimNext(cuda_worker);
        if (claim.status == DepthTaskClaimStatus::Retry)
        {
            scheduler.waitForStateChange(
                claim.revision, std::chrono::milliseconds(2));
            continue;
        }
        if (claim.status != DepthTaskClaimStatus::Task)
        {
            break;
        }
        retried_failed_view = retried_failed_view ||
                              claim.viewIndex == failed_claim.viewIndex;
        scheduler.complete(
            cuda_worker,
            claim.viewIndex,
            std::chrono::milliseconds(10),
            true);
    }

    EXPECT_TRUE(retried_failed_view);
    const auto stats = scheduler.workerStats();
    const auto failed_stats_it = stats.find(opencl_worker.id());
    ASSERT_NE(failed_stats_it, stats.end());
    EXPECT_EQ(failed_stats_it->second.completedTasks, 1);
    EXPECT_EQ(failed_stats_it->second.successfulTasks, 0);
    EXPECT_EQ(failed_stats_it->second.failedTasks, 1);
    EXPECT_DOUBLE_EQ(failed_stats_it->second.emaElapsedMilliseconds, 0.0);
    EXPECT_EQ(scheduler.pendingTaskCount(), 0U);
}

TEST(DepthComputeSchedulerTest, FailureAfterCalibrationRetiresAndRetriesOnce)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 5.0f}, {2, 4.0f}, {3, 3.0f}, {4, 2.0f}, {5, 1.0f}},
        true,
        {cuda_worker, opencl_worker});

    const DepthTaskClaim cuda_calibration = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(cuda_calibration.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        cuda_worker,
        cuda_calibration.viewIndex,
        std::chrono::milliseconds(10),
        true);
    const DepthTaskClaim opencl_calibration = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(opencl_calibration.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        opencl_worker,
        opencl_calibration.viewIndex,
        std::chrono::milliseconds(12),
        true);

    const DepthTaskClaim failed_claim = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(failed_claim.status, DepthTaskClaimStatus::Task);
    const auto failure = scheduler.complete(
        opencl_worker,
        failed_claim.viewIndex,
        std::chrono::milliseconds(2),
        false);
    EXPECT_TRUE(failure.retryScheduled);
    EXPECT_TRUE(failure.workerRetired);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Retire);

    const DepthTaskClaim retry_claim = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(retry_claim.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(retry_claim.viewIndex, failed_claim.viewIndex);
    scheduler.complete(
        cuda_worker,
        retry_claim.viewIndex,
        std::chrono::milliseconds(10),
        true);

    const auto stats = scheduler.workerStats();
    EXPECT_EQ(stats.at(opencl_worker.id()).successfulTasks, 1);
    EXPECT_EQ(stats.at(opencl_worker.id()).failedTasks, 1);
    EXPECT_DOUBLE_EQ(stats.at(cuda_worker.id()).emaElapsedMilliseconds, 10.0);
}

TEST(DepthComputeSchedulerTest, RetryMustCrossBackendNotOnlyPhysicalDevice)
{
    const DepthComputeWorker failed_cuda{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker other_cuda{DepthComputeBackend::Cuda, 1};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 2};
    DepthComputeScheduler scheduler(
        {{1, 6.0f}, {2, 5.0f}, {3, 4.0f},
         {4, 3.0f}, {5, 2.0f}, {6, 1.0f}},
        true,
        {failed_cuda, other_cuda, opencl_worker});

    const DepthTaskClaim failed_claim = scheduler.claimNext(failed_cuda);
    ASSERT_EQ(failed_claim.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        failed_cuda,
        failed_claim.viewIndex,
        std::chrono::milliseconds(1),
        false).retryScheduled);

    const DepthTaskClaim same_backend_claim = scheduler.claimNext(other_cuda);
    ASSERT_EQ(same_backend_claim.status, DepthTaskClaimStatus::Task);
    EXPECT_NE(same_backend_claim.viewIndex, failed_claim.viewIndex);

    const DepthTaskClaim cross_backend_claim = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(cross_backend_claim.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(cross_backend_claim.viewIndex, failed_claim.viewIndex);
}

TEST(DepthComputeSchedulerTest, RetriedFrameIsNeverQueuedForAThirdAttempt)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 4.0f}, {2, 3.0f}, {3, 2.0f}, {4, 1.0f}},
        true,
        {cuda_worker, opencl_worker});

    const DepthTaskClaim first_attempt = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(first_attempt.status, DepthTaskClaimStatus::Task);
    const auto first_failure = scheduler.complete(
        cuda_worker,
        first_attempt.viewIndex,
        std::chrono::milliseconds(1),
        false);
    ASSERT_TRUE(first_failure.retryScheduled);
    ASSERT_TRUE(first_failure.workerRetired);

    const DepthTaskClaim second_attempt = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(second_attempt.status, DepthTaskClaimStatus::Task);
    ASSERT_EQ(second_attempt.viewIndex, first_attempt.viewIndex);
    const auto second_failure = scheduler.complete(
        opencl_worker,
        second_attempt.viewIndex,
        std::chrono::milliseconds(1),
        false);
    EXPECT_TRUE(second_failure.accepted);
    EXPECT_FALSE(second_failure.retryScheduled);
    EXPECT_FALSE(second_failure.workerRetired);

    const DepthTaskClaim next_frame = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(next_frame.status, DepthTaskClaimStatus::Task);
    EXPECT_NE(next_frame.viewIndex, first_attempt.viewIndex);
    EXPECT_EQ(scheduler.pendingTaskCount(), 2U);
}

TEST(DepthComputeSchedulerTest, LastHealthyBackendKeepsDrainingAfterFinalFailure)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    DepthComputeScheduler scheduler(
        {{1, 5.0f}, {2, 4.0f}, {3, 3.0f}, {4, 2.0f}, {5, 1.0f}},
        true,
        {cuda_worker, opencl_worker});

    const DepthTaskClaim opencl_failure = scheduler.claimNext(opencl_worker);
    ASSERT_EQ(opencl_failure.status, DepthTaskClaimStatus::Task);
    ASSERT_TRUE(scheduler.complete(
        opencl_worker,
        opencl_failure.viewIndex,
        std::chrono::milliseconds(1),
        false).retryScheduled);

    const DepthTaskClaim retry = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(retry.status, DepthTaskClaimStatus::Task);
    ASSERT_EQ(retry.viewIndex, opencl_failure.viewIndex);
    scheduler.complete(
        cuda_worker, retry.viewIndex, std::chrono::milliseconds(10), true);

    const DepthTaskClaim final_failure = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(final_failure.status, DepthTaskClaimStatus::Task);
    const auto completion = scheduler.complete(
        cuda_worker,
        final_failure.viewIndex,
        std::chrono::milliseconds(1),
        false);
    EXPECT_FALSE(completion.retryScheduled);
    EXPECT_FALSE(completion.workerRetired);

    const DepthTaskClaim remaining = scheduler.claimNext(cuda_worker);
    ASSERT_EQ(remaining.status, DepthTaskClaimStatus::Task);
    EXPECT_NE(remaining.viewIndex, final_failure.viewIndex);
}

TEST(DepthComputeSchedulerTest, TailPausedBackendReactivatesForFailedFastFrame)
{
    const DepthComputeWorker fast_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker slow_worker{DepthComputeBackend::OpenCl, 1};
    std::vector<DepthFrameTask> tasks;
    for (int index = 0; index < 8; ++index)
    {
        tasks.push_back({index, static_cast<float>(8 - index)});
    }
    DepthComputeScheduler scheduler(
        std::move(tasks), true, {fast_worker, slow_worker});

    const DepthTaskClaim fast_calibration = scheduler.claimNext(fast_worker);
    ASSERT_EQ(fast_calibration.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        fast_worker,
        fast_calibration.viewIndex,
        std::chrono::milliseconds(10),
        true);
    const DepthTaskClaim slow_calibration = scheduler.claimNext(slow_worker);
    ASSERT_EQ(slow_calibration.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        slow_worker,
        slow_calibration.viewIndex,
        std::chrono::milliseconds(100),
        true);
    ASSERT_EQ(scheduler.claimNext(slow_worker).status,
              DepthTaskClaimStatus::Retry);

    const DepthTaskClaim failed_fast_claim = scheduler.claimNext(fast_worker);
    ASSERT_EQ(failed_fast_claim.status, DepthTaskClaimStatus::Task);
    const auto failure = scheduler.complete(
        fast_worker,
        failed_fast_claim.viewIndex,
        std::chrono::milliseconds(1),
        false);
    ASSERT_TRUE(failure.retryScheduled);
    ASSERT_TRUE(failure.workerRetired);

    const DepthTaskClaim slow_retry = scheduler.claimNext(slow_worker);
    ASSERT_EQ(slow_retry.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(slow_retry.viewIndex, failed_fast_claim.viewIndex);
}

TEST(DepthComputeSchedulerTest, SlowCrossBackendWorkerDrainsCudaExcludedTailRetry)
{
    const DepthComputeWorker failed_cuda{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker healthy_cuda{DepthComputeBackend::Cuda, 1};
    const DepthComputeWorker slow_opencl{DepthComputeBackend::OpenCl, 2};
    DepthComputeScheduler scheduler(
        {{1, 6.0f}, {2, 5.0f}, {3, 4.0f},
         {4, 3.0f}, {5, 2.0f}, {6, 1.0f}},
        true,
        {failed_cuda, healthy_cuda, slow_opencl});

    const DepthTaskClaim failed_cuda_calibration =
        scheduler.claimNext(failed_cuda);
    ASSERT_EQ(failed_cuda_calibration.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        failed_cuda,
        failed_cuda_calibration.viewIndex,
        std::chrono::milliseconds(10),
        true);

    const DepthTaskClaim healthy_cuda_calibration =
        scheduler.claimNext(healthy_cuda);
    ASSERT_EQ(healthy_cuda_calibration.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        healthy_cuda,
        healthy_cuda_calibration.viewIndex,
        std::chrono::milliseconds(10),
        true);

    const DepthTaskClaim opencl_calibration = scheduler.claimNext(slow_opencl);
    ASSERT_EQ(opencl_calibration.status, DepthTaskClaimStatus::Task);
    scheduler.complete(
        slow_opencl,
        opencl_calibration.viewIndex,
        std::chrono::milliseconds(100),
        true);

    const DepthTaskClaim failed_claim = scheduler.claimNext(failed_cuda);
    ASSERT_EQ(failed_claim.status, DepthTaskClaimStatus::Task);
    const auto failure = scheduler.complete(
        failed_cuda,
        failed_claim.viewIndex,
        std::chrono::milliseconds(1),
        false);
    ASSERT_TRUE(failure.retryScheduled);
    ASSERT_TRUE(failure.workerRetired);

    for (int expected_view : {5, 6})
    {
        const DepthTaskClaim normal_claim = scheduler.claimNext(healthy_cuda);
        ASSERT_EQ(normal_claim.status, DepthTaskClaimStatus::Task);
        EXPECT_EQ(normal_claim.viewIndex, expected_view);
        scheduler.complete(
            healthy_cuda,
            normal_claim.viewIndex,
            std::chrono::milliseconds(10),
            true);
    }
    EXPECT_EQ(scheduler.claimNext(healthy_cuda).status,
              DepthTaskClaimStatus::Retry);
    ASSERT_EQ(scheduler.pendingTaskCount(), 1U);

    const DepthTaskClaim retry = scheduler.claimNext(slow_opencl);
    ASSERT_EQ(retry.status, DepthTaskClaimStatus::Task);
    EXPECT_EQ(retry.viewIndex, failed_claim.viewIndex);
    scheduler.complete(
        slow_opencl,
        retry.viewIndex,
        std::chrono::milliseconds(100),
        true);
    EXPECT_EQ(scheduler.pendingTaskCount(), 0U);
}

TEST(DepthComputeSchedulerTest, UnlistedWorkerCannotConsumeHybridTasks)
{
    const DepthComputeWorker cuda_worker{DepthComputeBackend::Cuda, 0};
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 1};
    const DepthComputeWorker unlisted_worker{DepthComputeBackend::Cpu, 0};
    DepthComputeScheduler scheduler(
        {{1, 2.0f}, {2, 1.0f}},
        true,
        {cuda_worker, opencl_worker});

    EXPECT_EQ(scheduler.claimNext(unlisted_worker).status,
              DepthTaskClaimStatus::Retire);
    EXPECT_EQ(scheduler.pendingTaskCount(), 2U);
    EXPECT_EQ(scheduler.claimNext(cuda_worker).status,
              DepthTaskClaimStatus::Task);
    EXPECT_EQ(scheduler.claimNext(opencl_worker).status,
              DepthTaskClaimStatus::Task);
    EXPECT_EQ(scheduler.pendingTaskCount(), 0U);
}

TEST(DepthComputeSchedulerTest, UsesStableOpenClWorkerName)
{
    const DepthComputeWorker opencl_worker{DepthComputeBackend::OpenCl, 2};
    EXPECT_EQ(opencl_worker.id(), "OpenCL:2");
}

TEST(DepthComputeSchedulerTest, ParsesStableWorkerIdentifiers)
{
    const auto cuda_worker = depthComputeWorkerFromId("CUDA:2");
    ASSERT_TRUE(cuda_worker.has_value());
    EXPECT_EQ(cuda_worker->backend, DepthComputeBackend::Cuda);
    EXPECT_EQ(cuda_worker->deviceIndex, 2);

    const auto opencl_worker = depthComputeWorkerFromId("OpenCL:1");
    ASSERT_TRUE(opencl_worker.has_value());
    EXPECT_EQ(opencl_worker->backend, DepthComputeBackend::OpenCl);
    EXPECT_EQ(opencl_worker->deviceIndex, 1);

    const auto cpu_worker = depthComputeWorkerFromId("CPU:0");
    ASSERT_TRUE(cpu_worker.has_value());
    EXPECT_EQ(cpu_worker->backend, DepthComputeBackend::Cpu);
    EXPECT_EQ(cpu_worker->deviceIndex, 0);
}

TEST(DepthComputeSchedulerTest, RejectsMalformedWorkerIdentifiers)
{
    EXPECT_FALSE(depthComputeWorkerFromId("Vulkan:0").has_value());
    EXPECT_FALSE(depthComputeWorkerFromId("CUDA:-1").has_value());
    EXPECT_FALSE(depthComputeWorkerFromId("OpenCL:").has_value());
    EXPECT_FALSE(depthComputeWorkerFromId("CPU:0-extra").has_value());
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
        for (int cuda_index = 0;
             cuda_index < xjw::mvs::PatchMatchDepthEstimator::cudaDeviceCount();
             ++cuda_index)
        {
            if (normalizedGpuDeviceName(device.name) != normalizedGpuDeviceName(
                    xjw::mvs::PatchMatchDepthEstimator::cudaDeviceName(cuda_index)))
            {
                continue;
            }
            EXPECT_EQ(device.physicalDeviceIdentity,
                      xjw::mvs::PatchMatchDepthEstimator::cudaDeviceIdentity(
                          cuda_index));
        }
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

TEST(GpuDeviceLeaseTest, ConservativelySkipsUnstableNvidiaOpenClCudaAlias)
{
    EXPECT_TRUE(shouldSkipUnstableOpenClCudaAlias(
        "NVIDIA Corporation", "name:nvidiageforcertx5080:0", true));
    EXPECT_FALSE(shouldSkipUnstableOpenClCudaAlias(
        "NVIDIA Corporation", "pci:0000:01:00", true));
    EXPECT_FALSE(shouldSkipUnstableOpenClCudaAlias(
        "Advanced Micro Devices, Inc.", "name:amdradeon:1", true));
    EXPECT_FALSE(shouldSkipUnstableOpenClCudaAlias(
        "NVIDIA Corporation", "name:nvidiageforcertx5080:0", false));
}

TEST(GpuDeviceLeaseTest, IdentifiesNvidiaOpenClVendorsForAutoModeDeduplication)
{
    EXPECT_TRUE(isNvidiaOpenClVendor("NVIDIA Corporation"));
    EXPECT_TRUE(isNvidiaOpenClVendor("  nViDiA  "));
    EXPECT_FALSE(isNvidiaOpenClVendor("Advanced Micro Devices, Inc."));
    EXPECT_FALSE(isNvidiaOpenClVendor("Intel(R) Corporation"));
}

TEST(GpuDeviceLeaseTest, NormalizesCrossApiDeviceNames)
{
    EXPECT_EQ(normalizedGpuDeviceName("NVIDIA GeForce RTX 5080"),
              normalizedGpuDeviceName("NVIDIA GeForce RTX-5080"));
    EXPECT_NE(normalizedGpuDeviceName("NVIDIA GeForce RTX 5080"),
              normalizedGpuDeviceName("AMD Radeon(TM) Graphics"));
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
