#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace xjw
{
namespace mvs
{

enum class DepthComputeBackend
{
    Cpu,
    Cuda,
    OpenCl
};

const char *depthComputeBackendName(DepthComputeBackend backend);

/// Resolves one frame-compute backend. A missing requested backend means Auto:
/// when automatic acceleration is enabled it uses CUDA -> OpenCL -> CPU, and
/// otherwise it selects CPU. An explicit backend is never replaced here.
DepthComputeBackend resolveDepthComputeBackend(
    std::optional<DepthComputeBackend> requestedBackend,
    bool cudaAvailable,
    bool openClAvailable,
    bool automaticAccelerationEnabled = true);

struct DepthComputeWorker
{
    DepthComputeBackend backend = DepthComputeBackend::Cpu;
    int deviceIndex = -1;

    std::string id() const;
};

/// Parses the stable worker identifiers produced by DepthComputeWorker::id().
/// Unknown backends, malformed indices, and negative device indices are rejected.
std::optional<DepthComputeWorker> depthComputeWorkerFromId(std::string_view workerId);

/// Builds frame-level host workers while keeping one worker for every physical
/// accelerator before adding at most one preparation lane for each device.
std::vector<DepthComputeWorker> buildDepthComputeWorkerPool(
    const std::vector<DepthComputeWorker> &physicalWorkers,
    int cudaHostSlotCount,
    int openClHostSlotCount,
    std::size_t maximumWorkerCount);

struct DepthFrameTask
{
    int viewIndex = -1;
    float priority = 0.0f;
};

struct DepthComputeWorkerStats
{
    int completedTasks = 0;
    int successfulTasks = 0;
    int failedTasks = 0;
    double elapsedMilliseconds = 0.0;
    double emaElapsedMilliseconds = 0.0;
};

enum class DepthTaskClaimStatus
{
    /// One frame was removed from the shared queue and is owned by this caller.
    Task,
    /// The worker remains available but is waiting for calibration, a more
    /// profitable tail assignment, or an in-flight retry decision.
    Retry,
    /// The physical worker failed while another backend remained available.
    Retire,
    /// No queued frames remain.
    Exhausted
};

struct DepthTaskClaim
{
    DepthTaskClaimStatus status = DepthTaskClaimStatus::Exhausted;
    int viewIndex = -1;
    std::uint64_t revision = 0;
    /// True only for the first task used to establish this physical worker's
    /// throughput. Callers may use a cheaper probe path before committing to
    /// the complete frame.
    bool calibrationProbe = false;
    /// This task is part of the bounded OpenCL contribution floor and must run
    /// through the complete depth pyramid instead of being rejected after the
    /// coarse calibration level.
    bool requiresFullFrame = false;
};

struct DepthComputeSchedulingPolicy
{
    /// Per physical OpenCL device. A value of one lets a sufficiently large
    /// hybrid batch measure and retain one useful complete iGPU frame.
    int guaranteedOpenClFullFramesPerDevice = 0;
    /// Duplicate host preparation slots share one physical OpenCL queue. Keep
    /// their frame backlog bounded so a slow iGPU cannot create a long tail.
    /// Zero leaves legacy callers unlimited; the hybrid generator explicitly
    /// selects one for its bounded iGPU lane.
    int maximumOpenClInFlightTasksPerDevice = 0;
};

/// Returns a bounded per-device OpenCL contribution floor for a hybrid batch.
/// The threshold scales with physical CUDA throughput so adding CUDA devices
/// cannot turn the iGPU contribution into an avoidable queue tail.
int recommendedOpenClFullFrameFloorPerDevice(
    bool benefitAwareScheduling,
    std::size_t pendingTaskCount,
    int physicalCudaDeviceCount,
    int physicalOpenClDeviceCount);

struct DepthTaskCompletionResult
{
    /// False means that the view was not owned by this physical worker. No
    /// scheduler state or statistics are changed for a rejected completion.
    bool accepted = false;
    /// A failed first attempt was returned to the queue for another backend.
    bool retryScheduled = false;
    /// This worker failed while a healthy alternative backend remained.
    bool workerRetired = false;
};

class DepthComputeScheduler
{
public:
    explicit DepthComputeScheduler(
        std::vector<DepthFrameTask> tasks,
        bool enableBenefitAwareScheduling = false,
        std::vector<DepthComputeWorker> participatingWorkers = {},
        DepthComputeSchedulingPolicy policy = {});

    DepthTaskClaim claimNext(const DepthComputeWorker &worker);
    /// Waits without a lost-wakeup race after a Retry claim. The caller passes
    /// the revision carried by that claim and still supplies a short timeout so
    /// external cancellation can be observed promptly.
    bool waitForStateChange(std::uint64_t observedRevision,
                            std::chrono::milliseconds maximumWait);
    /// Completes the exact frame returned by claimNext(). In heterogeneous
    /// benefit-aware mode, a first failure can be requeued once for a different
    /// backend; retryScheduled tells the caller not to publish a final failure yet.
    DepthTaskCompletionResult complete(
        const DepthComputeWorker &worker,
        int viewIndex,
        std::chrono::duration<double, std::milli> elapsed,
        bool success);

    std::size_t pendingTaskCount() const;
    std::unordered_map<std::string, DepthComputeWorkerStats> workerStats() const;
    /// Returns the fastest successful physical-device service EMA reported by
    /// a healthy alternative backend, falling back to whole-frame latency
    /// until saturated host-slot samples exist.
    std::optional<double> fastestSuccessfulAlternativeBackendEmaMilliseconds(
        const DepthComputeWorker &worker) const;
    /// Atomically rejects a still-owned first calibration probe only when a
    /// healthy, successfully calibrated alternative backend remains faster.
    /// A returned EMA means the frame was requeued for that backend and the
    /// current physical worker was retired; nullopt means the caller must keep
    /// computing the current pyramid.
    std::optional<double> tryRejectUnprofitableCalibrationProbe(
        const DepthComputeWorker &worker,
        int viewIndex,
        double coarseLevelElapsedMilliseconds,
        std::chrono::duration<double, std::milli> probeElapsed);

private:
    struct PendingTask
    {
        int viewIndex = -1;
        int retryCount = 0;
        std::optional<DepthComputeBackend> excludedBackend;
    };

    struct InFlightTask
    {
        std::string workerId;
        DepthComputeBackend backend = DepthComputeBackend::Cpu;
        int retryCount = 0;
        bool calibrationProbe = false;
        bool requiresFullFrame = false;
    };

    struct WorkerSchedulingState
    {
        int assignedTasks = 0;
        int inFlightTasks = 0;
        bool calibrationInFlight = false;
        bool calibrationSucceeded = false;
        bool pausedForBenefit = false;
        bool retirementPending = false;
        bool failureRetired = false;
        int successfulGuaranteedFullFrames = 0;
        int guaranteedFullFramesInFlight = 0;
        bool discardedFirstSaturatedCompletion = false;
        int physicalServiceSamples = 0;
        double emaPhysicalServiceMilliseconds = 0.0;
    };

    using PendingTaskIterator = std::deque<PendingTask>::iterator;

    PendingTaskIterator findEligibleTask(const DepthComputeWorker &worker);
    PendingTaskIterator findPendingCrossBackendRetry(
        const DepthComputeWorker &worker);
    DepthTaskClaim assignNextTask(const DepthComputeWorker &worker,
                                  WorkerSchedulingState &state,
                                  PendingTaskIterator taskIt,
                                  bool calibrationProbe = false,
                                  bool requiresFullFrame = false);
    bool needsGuaranteedFullFrame(const DepthComputeWorker &worker,
                                  const WorkerSchedulingState &state) const;
    bool isParticipatingWorker(const std::string &workerId) const;
    bool hasHealthyAlternativeBackend(const DepthComputeWorker &worker) const;
    void reactivateAlternativeBackends(const DepthComputeWorker &worker);
    bool shouldReserveForCalibration() const;
    bool shouldPauseAtQueueTail(const std::string &workerId) const;
    std::optional<double>
    fastestSuccessfulAlternativeBackendEmaMillisecondsLocked(
        const DepthComputeWorker &worker) const;
    void advanceRevision();

    mutable std::mutex _mutex;
    std::condition_variable _stateChanged;
    std::deque<PendingTask> _pendingTasks;
    std::unordered_map<int, InFlightTask> _inFlightTasks;
    std::unordered_map<std::string, DepthComputeWorkerStats> _workerStats;
    std::unordered_map<std::string, WorkerSchedulingState> _workerSchedulingStates;
    std::vector<std::string> _participatingWorkerIds;
    DepthComputeSchedulingPolicy _policy;
    bool _benefitAwareSchedulingEnabled = false;
    std::uint64_t _revision = 0;
};

} // namespace mvs
} // namespace xjw
