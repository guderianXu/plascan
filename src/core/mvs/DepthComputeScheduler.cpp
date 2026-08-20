#include "DepthComputeScheduler.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <utility>

namespace xjw
{
namespace mvs
{

namespace
{

constexpr std::size_t kMinimumOpenClContributionFramesPerCudaDevice = 64;

} // namespace

int recommendedOpenClFullFrameFloorPerDevice(
    bool benefitAwareScheduling,
    std::size_t pendingTaskCount,
    int physicalCudaDeviceCount,
    int physicalOpenClDeviceCount)
{
    if (!benefitAwareScheduling || physicalOpenClDeviceCount <= 0)
    {
        return 0;
    }

    const std::size_t cuda_device_count = static_cast<std::size_t>(
        std::max(1, physicalCudaDeviceCount));
    const std::size_t minimum_batch_size =
        kMinimumOpenClContributionFramesPerCudaDevice * cuda_device_count;
    return pendingTaskCount >= minimum_batch_size ? 1 : 0;
}

const char *depthComputeBackendName(DepthComputeBackend backend)
{
    switch (backend)
    {
    case DepthComputeBackend::Cpu:
        return "CPU";
    case DepthComputeBackend::Cuda:
        return "CUDA";
    case DepthComputeBackend::OpenCl:
        return "OpenCL";
    }
    return "Unknown";
}

DepthComputeBackend resolveDepthComputeBackend(
    std::optional<DepthComputeBackend> requestedBackend,
    bool cudaAvailable,
    bool openClAvailable)
{
    if (requestedBackend.has_value())
    {
        return *requestedBackend;
    }
    if (cudaAvailable)
    {
        return DepthComputeBackend::Cuda;
    }
    if (openClAvailable)
    {
        return DepthComputeBackend::OpenCl;
    }
    return DepthComputeBackend::Cpu;
}

std::string DepthComputeWorker::id() const
{
    std::string result = depthComputeBackendName(backend);
    if (deviceIndex >= 0)
    {
        result += ":" + std::to_string(deviceIndex);
    }
    return result;
}

std::optional<DepthComputeWorker> depthComputeWorkerFromId(
    std::string_view workerId)
{
    const auto parse_backend = [workerId](std::string_view backend_name,
                                           DepthComputeBackend backend)
        -> std::optional<DepthComputeWorker>
    {
        if (workerId == backend_name)
        {
            return DepthComputeWorker{backend, -1};
        }
        if (workerId.size() <= backend_name.size() + 1 ||
            workerId.substr(0, backend_name.size()) != backend_name ||
            workerId[backend_name.size()] != ':')
        {
            return std::nullopt;
        }

        const std::string_view index_text = workerId.substr(backend_name.size() + 1);
        int device_index = -1;
        const auto [end, error] = std::from_chars(
            index_text.data(), index_text.data() + index_text.size(), device_index);
        if (error != std::errc{} || end != index_text.data() + index_text.size() ||
            device_index < 0)
        {
            return std::nullopt;
        }
        return DepthComputeWorker{backend, device_index};
    };

    if (const auto worker = parse_backend("CUDA", DepthComputeBackend::Cuda))
    {
        return worker;
    }
    if (const auto worker = parse_backend("OpenCL", DepthComputeBackend::OpenCl))
    {
        return worker;
    }
    return parse_backend("CPU", DepthComputeBackend::Cpu);
}

std::vector<DepthComputeWorker> buildDepthComputeWorkerPool(
    const std::vector<DepthComputeWorker> &physicalWorkers,
    int cudaHostSlotCount,
    int openClHostSlotCount,
    std::size_t maximumWorkerCount)
{
    if (maximumWorkerCount == 0)
    {
        return {};
    }

    std::vector<DepthComputeWorker> workers;
    workers.reserve(std::min(maximumWorkerCount, physicalWorkers.size()));
    for (const DepthComputeWorker &worker : physicalWorkers)
    {
        if (workers.size() >= maximumWorkerCount)
        {
            return workers;
        }
        workers.push_back(worker);
    }

    const auto add_host_slots = [&](DepthComputeBackend backend, int requestedCount)
    {
        std::vector<DepthComputeWorker> candidates;
        for (const DepthComputeWorker &worker : physicalWorkers)
        {
            if (worker.backend == backend)
            {
                candidates.push_back(worker);
            }
        }
        if (candidates.empty())
        {
            return;
        }

        std::size_t current_count = static_cast<std::size_t>(std::count_if(
            workers.begin(), workers.end(), [backend](const DepthComputeWorker &worker)
            {
                return worker.backend == backend;
            }));
        const std::size_t maximum_pipeline_count = candidates.size() * 2;
        const std::size_t target_count = std::min(
            maximum_pipeline_count,
            std::max(
                current_count,
                static_cast<std::size_t>(std::max(0, requestedCount))));
        std::size_t candidate_index = 0;
        while (current_count < target_count && workers.size() < maximumWorkerCount)
        {
            workers.push_back(candidates[candidate_index % candidates.size()]);
            ++candidate_index;
            ++current_count;
        }
    };

    add_host_slots(DepthComputeBackend::Cuda, cudaHostSlotCount);
    add_host_slots(DepthComputeBackend::OpenCl, openClHostSlotCount);
    return workers;
}

DepthComputeScheduler::DepthComputeScheduler(
    std::vector<DepthFrameTask> tasks,
    bool enableBenefitAwareScheduling,
    std::vector<DepthComputeWorker> participatingWorkers,
    DepthComputeSchedulingPolicy policy)
    : _policy(policy)
{
    _policy.guaranteedOpenClFullFramesPerDevice = std::max(
        0, _policy.guaranteedOpenClFullFramesPerDevice);
    _policy.maximumOpenClInFlightTasksPerDevice = std::max(
        0, _policy.maximumOpenClInFlightTasksPerDevice);
    std::stable_sort(tasks.begin(), tasks.end(), [](const DepthFrameTask &left,
                                                     const DepthFrameTask &right)
    {
        return left.priority > right.priority;
    });
    for (const DepthFrameTask &task : tasks)
    {
        if (task.viewIndex >= 0)
        {
            _pendingTasks.push_back({task.viewIndex, 0, std::nullopt});
        }
    }

    for (const DepthComputeWorker &worker : participatingWorkers)
    {
        const std::string worker_id = worker.id();
        if (std::find(_participatingWorkerIds.begin(),
                      _participatingWorkerIds.end(),
                      worker_id) != _participatingWorkerIds.end())
        {
            continue;
        }
        _participatingWorkerIds.push_back(worker_id);
    }
    _benefitAwareSchedulingEnabled = enableBenefitAwareScheduling &&
                                     _participatingWorkerIds.size() > 1 &&
                                     _pendingTasks.size() >=
                                         _participatingWorkerIds.size();
    for (const std::string &worker_id : _participatingWorkerIds)
    {
        _workerSchedulingStates.try_emplace(worker_id);
    }
}

DepthTaskClaim DepthComputeScheduler::claimNext(const DepthComputeWorker &worker)
{
    std::lock_guard<std::mutex> lock(_mutex);
    const std::string worker_id = worker.id();
    _workerStats.try_emplace(worker_id);

    if (!_benefitAwareSchedulingEnabled)
    {
        WorkerSchedulingState &state = _workerSchedulingStates[worker_id];
        if (worker.backend == DepthComputeBackend::OpenCl &&
            _policy.maximumOpenClInFlightTasksPerDevice > 0 &&
            state.inFlightTasks >=
                _policy.maximumOpenClInFlightTasksPerDevice)
        {
            return {DepthTaskClaimStatus::Retry, -1, _revision};
        }
        if (_pendingTasks.empty())
        {
            return {DepthTaskClaimStatus::Exhausted, -1, _revision};
        }
        return assignNextTask(worker, state, _pendingTasks.begin());
    }

    auto state_it = _workerSchedulingStates.find(worker_id);
    if (state_it == _workerSchedulingStates.end() ||
        state_it->second.failureRetired)
    {
        return {DepthTaskClaimStatus::Retire, -1, _revision};
    }

    WorkerSchedulingState &state = state_it->second;
    if (worker.backend == DepthComputeBackend::OpenCl &&
        _policy.maximumOpenClInFlightTasksPerDevice > 0 &&
        state.inFlightTasks >= _policy.maximumOpenClInFlightTasksPerDevice)
    {
        return {DepthTaskClaimStatus::Retry, -1, _revision};
    }
    if (state.retirementPending)
    {
        if (state.inFlightTasks > 0)
        {
            return {DepthTaskClaimStatus::Retry, -1, _revision};
        }
        const PendingTaskIterator recovery_it =
            findPendingCrossBackendRetry(worker);
        if (recovery_it != _pendingTasks.end())
        {
            return assignNextTask(worker, state, recovery_it);
        }
        state.retirementPending = false;
        state.failureRetired = true;
        advanceRevision();
        return {DepthTaskClaimStatus::Retire, -1, _revision};
    }
    if (_pendingTasks.empty())
    {
        return {_inFlightTasks.empty() ? DepthTaskClaimStatus::Exhausted
                                       : DepthTaskClaimStatus::Retry,
                -1,
                _revision};
    }

    const PendingTaskIterator task_it = findEligibleTask(worker);
    if (task_it == _pendingTasks.end())
    {
        return {DepthTaskClaimStatus::Retry, -1, _revision};
    }

    if (!state.calibrationSucceeded)
    {
        if (state.calibrationInFlight)
        {
            return {DepthTaskClaimStatus::Retry, -1, _revision};
        }
        state.calibrationInFlight = true;
        return assignNextTask(
            worker,
            state,
            task_it,
            true,
            needsGuaranteedFullFrame(worker, state));
    }

    if (needsGuaranteedFullFrame(worker, state))
    {
        state.pausedForBenefit = false;
        return assignNextTask(worker, state, task_it, false, true);
    }

    if (shouldReserveForCalibration())
    {
        return {DepthTaskClaimStatus::Retry, -1, _revision};
    }
    if (shouldPauseAtQueueTail(worker_id))
    {
        if (!state.pausedForBenefit)
        {
            state.pausedForBenefit = true;
            advanceRevision();
        }
        return {DepthTaskClaimStatus::Retry, -1, _revision};
    }
    state.pausedForBenefit = false;
    return assignNextTask(worker, state, task_it);
}

bool DepthComputeScheduler::waitForStateChange(
    std::uint64_t observedRevision,
    std::chrono::milliseconds maximumWait)
{
    std::unique_lock<std::mutex> lock(_mutex);
    return _stateChanged.wait_for(lock, maximumWait, [this, observedRevision]
    {
        return _revision != observedRevision;
    });
}

DepthTaskCompletionResult DepthComputeScheduler::complete(
    const DepthComputeWorker &worker,
    int viewIndex,
    std::chrono::duration<double, std::milli> elapsed,
    bool success)
{
    std::lock_guard<std::mutex> lock(_mutex);
    const std::string worker_id = worker.id();
    const auto task_it = _inFlightTasks.find(viewIndex);
    if (task_it == _inFlightTasks.end() ||
        task_it->second.workerId != worker_id)
    {
        return {};
    }

    const InFlightTask completed_task = task_it->second;
    _inFlightTasks.erase(task_it);
    DepthTaskCompletionResult result;
    result.accepted = true;

    DepthComputeWorkerStats &stats = _workerStats[worker_id];
    WorkerSchedulingState &state = _workerSchedulingStates[worker_id];
    ++stats.completedTasks;
    stats.successfulTasks += success ? 1 : 0;
    stats.failedTasks += success ? 0 : 1;
    const double elapsed_milliseconds = elapsed.count();
    stats.elapsedMilliseconds += elapsed_milliseconds;

    if (success && std::isfinite(elapsed_milliseconds) &&
        elapsed_milliseconds > 0.0)
    {
        constexpr double duration_ema_alpha = 0.35;
        if (stats.emaElapsedMilliseconds <= 0.0)
        {
            stats.emaElapsedMilliseconds = elapsed_milliseconds;
        }
        else
        {
            stats.emaElapsedMilliseconds =
                duration_ema_alpha * elapsed_milliseconds +
                (1.0 - duration_ema_alpha) * stats.emaElapsedMilliseconds;
        }

        if (state.inFlightTasks > 1)
        {
            if (!state.discardedFirstSaturatedCompletion)
            {
                // The first completion after filling a second host slot often
                // contains no queue wait. It cannot establish serialized GPU
                // service time, so wait for the next saturated completion.
                state.discardedFirstSaturatedCompletion = true;
            }
            else
            {
                const double physical_service_sample =
                    elapsed_milliseconds /
                    static_cast<double>(state.inFlightTasks);
                if (state.emaPhysicalServiceMilliseconds <= 0.0)
                {
                    state.emaPhysicalServiceMilliseconds =
                        physical_service_sample;
                }
                else
                {
                    state.emaPhysicalServiceMilliseconds =
                        duration_ema_alpha * physical_service_sample +
                        (1.0 - duration_ema_alpha) *
                            state.emaPhysicalServiceMilliseconds;
                }
                ++state.physicalServiceSamples;
            }
        }
    }

    state.inFlightTasks = std::max(0, state.inFlightTasks - 1);
    if (completed_task.requiresFullFrame)
    {
        state.guaranteedFullFramesInFlight = std::max(
            0, state.guaranteedFullFramesInFlight - 1);
        if (success)
        {
            ++state.successfulGuaranteedFullFrames;
        }
    }
    if (_benefitAwareSchedulingEnabled && isParticipatingWorker(worker_id))
    {
        if (!state.calibrationSucceeded)
        {
            state.calibrationInFlight = false;
            if (success && stats.emaElapsedMilliseconds > 0.0)
            {
                state.calibrationSucceeded = true;
            }
        }

        if (!success && hasHealthyAlternativeBackend(worker))
        {
            state.pausedForBenefit = false;
            reactivateAlternativeBackends(worker);
            if (completed_task.retryCount == 0)
            {
                _pendingTasks.push_front(
                    {viewIndex, 1, completed_task.backend});
                result.retryScheduled = true;
            }
            state.retirementPending =
                findPendingCrossBackendRetry(worker) != _pendingTasks.end();
            state.failureRetired = !state.retirementPending;
            result.workerRetired = state.failureRetired;
        }
    }
    advanceRevision();
    return result;
}

DepthComputeScheduler::PendingTaskIterator
DepthComputeScheduler::findEligibleTask(const DepthComputeWorker &worker)
{
    return std::find_if(
        _pendingTasks.begin(), _pendingTasks.end(), [&worker](const PendingTask &task)
        {
            return !task.excludedBackend.has_value() ||
                   *task.excludedBackend != worker.backend;
        });
}

DepthComputeScheduler::PendingTaskIterator
DepthComputeScheduler::findPendingCrossBackendRetry(
    const DepthComputeWorker &worker)
{
    return std::find_if(
        _pendingTasks.begin(), _pendingTasks.end(), [&worker](const PendingTask &task)
        {
            return task.retryCount > 0 && task.excludedBackend.has_value() &&
                   *task.excludedBackend != worker.backend;
        });
}

DepthTaskClaim DepthComputeScheduler::assignNextTask(
    const DepthComputeWorker &worker,
    WorkerSchedulingState &state,
    PendingTaskIterator taskIt,
    bool calibrationProbe,
    bool requiresFullFrame)
{
    const PendingTask task = *taskIt;
    _pendingTasks.erase(taskIt);
    ++state.assignedTasks;
    ++state.inFlightTasks;
    state.guaranteedFullFramesInFlight += requiresFullFrame ? 1 : 0;
    _inFlightTasks.insert_or_assign(
        task.viewIndex,
        InFlightTask{
            worker.id(),
            worker.backend,
            task.retryCount,
            calibrationProbe,
            requiresFullFrame});
    advanceRevision();
    return {
        DepthTaskClaimStatus::Task,
        task.viewIndex,
        _revision,
        calibrationProbe,
        requiresFullFrame};
}

bool DepthComputeScheduler::needsGuaranteedFullFrame(
    const DepthComputeWorker &worker,
    const WorkerSchedulingState &state) const
{
    return _benefitAwareSchedulingEnabled &&
           worker.backend == DepthComputeBackend::OpenCl &&
           state.successfulGuaranteedFullFrames +
                   state.guaranteedFullFramesInFlight <
               _policy.guaranteedOpenClFullFramesPerDevice;
}

bool DepthComputeScheduler::isParticipatingWorker(
    const std::string &workerId) const
{
    return std::find(_participatingWorkerIds.begin(),
                     _participatingWorkerIds.end(),
                     workerId) != _participatingWorkerIds.end();
}

bool DepthComputeScheduler::hasHealthyAlternativeBackend(
    const DepthComputeWorker &worker) const
{
    return std::any_of(
        _participatingWorkerIds.begin(),
        _participatingWorkerIds.end(),
        [this, &worker](const std::string &participant_id)
        {
            if (participant_id == worker.id())
            {
                return false;
            }
            const auto participant = depthComputeWorkerFromId(participant_id);
            const auto state_it = _workerSchedulingStates.find(participant_id);
            return participant.has_value() &&
                   participant->backend != worker.backend &&
                   state_it != _workerSchedulingStates.end() &&
                   !state_it->second.retirementPending &&
                   !state_it->second.failureRetired;
        });
}

void DepthComputeScheduler::reactivateAlternativeBackends(
    const DepthComputeWorker &worker)
{
    for (const std::string &participant_id : _participatingWorkerIds)
    {
        const auto participant = depthComputeWorkerFromId(participant_id);
        auto state_it = _workerSchedulingStates.find(participant_id);
        if (participant.has_value() &&
            participant->backend != worker.backend &&
            state_it != _workerSchedulingStates.end() &&
            !state_it->second.retirementPending &&
            !state_it->second.failureRetired)
        {
            state_it->second.pausedForBenefit = false;
        }
    }
}

bool DepthComputeScheduler::shouldReserveForCalibration() const
{
    std::size_t required_calibration_tasks = 0;
    for (const std::string &participant_id : _participatingWorkerIds)
    {
        const auto state_it = _workerSchedulingStates.find(participant_id);
        if (state_it != _workerSchedulingStates.end() &&
            !state_it->second.failureRetired &&
            !state_it->second.retirementPending)
        {
            const auto participant = depthComputeWorkerFromId(participant_id);
            const bool has_eligible_task = participant.has_value() &&
                std::any_of(
                    _pendingTasks.begin(),
                    _pendingTasks.end(),
                    [&participant](const PendingTask &task)
                    {
                        return !task.excludedBackend.has_value() ||
                               *task.excludedBackend != participant->backend;
                    });
            if (!has_eligible_task)
            {
                continue;
            }

            int required_tasks = 0;
            if (!state_it->second.calibrationSucceeded &&
                !state_it->second.calibrationInFlight)
            {
                required_tasks = 1;
            }
            if (participant->backend == DepthComputeBackend::OpenCl)
            {
                required_tasks = std::max(
                    required_tasks,
                    _policy.guaranteedOpenClFullFramesPerDevice -
                        state_it->second.successfulGuaranteedFullFrames -
                        state_it->second.guaranteedFullFramesInFlight);
            }
            required_calibration_tasks += static_cast<std::size_t>(
                std::max(0, required_tasks));
        }
    }
    return _pendingTasks.size() <= required_calibration_tasks;
}

bool DepthComputeScheduler::shouldPauseAtQueueTail(
    const std::string &workerId) const
{
    const auto worker_state_it = _workerSchedulingStates.find(workerId);
    const auto worker_stats_it = _workerStats.find(workerId);
    if (worker_state_it == _workerSchedulingStates.end() ||
        worker_stats_it == _workerStats.end() ||
        !worker_state_it->second.calibrationSucceeded ||
        worker_stats_it->second.emaElapsedMilliseconds <= 0.0)
    {
        return false;
    }

    double fastest_elapsed_milliseconds = std::numeric_limits<double>::infinity();
    int fastest_in_flight_tasks = 0;
    std::string fastest_worker_id;
    for (const std::string &participant_id : _participatingWorkerIds)
    {
        const auto state_it = _workerSchedulingStates.find(participant_id);
        const auto stats_it = _workerStats.find(participant_id);
        if (state_it == _workerSchedulingStates.end() ||
            state_it->second.failureRetired ||
            state_it->second.retirementPending)
        {
            continue;
        }
        if (!state_it->second.calibrationSucceeded ||
            stats_it == _workerStats.end() ||
            stats_it->second.successfulTasks == 0 ||
            stats_it->second.emaElapsedMilliseconds <= 0.0)
        {
            return false;
        }
        const double physical_service_milliseconds =
            state_it->second.physicalServiceSamples > 0
            ? state_it->second.emaPhysicalServiceMilliseconds
            : stats_it->second.emaElapsedMilliseconds;
        if (physical_service_milliseconds < fastest_elapsed_milliseconds)
        {
            // Duplicate host slots overlap preparation around one serialized
            // physical GPU queue. Prefer service samples observed while more
            // than one slot was actually in flight; never infer doubled
            // throughput merely because a second slot has just been claimed.
            fastest_elapsed_milliseconds = physical_service_milliseconds;
            fastest_in_flight_tasks = state_it->second.inFlightTasks;
            fastest_worker_id = participant_id;
        }
    }

    if (fastest_worker_id.empty() || fastest_worker_id == workerId)
    {
        return false;
    }

    const auto candidate_worker = depthComputeWorkerFromId(workerId);
    const auto fastest_worker = depthComputeWorkerFromId(fastest_worker_id);
    if (!candidate_worker.has_value() || !fastest_worker.has_value())
    {
        return false;
    }

    const auto is_eligible_for_backend = [](const PendingTask &task,
                                             DepthComputeBackend backend)
    {
        return !task.excludedBackend.has_value() ||
               *task.excludedBackend != backend;
    };
    const std::size_t fastest_eligible_pending_tasks =
        static_cast<std::size_t>(std::count_if(
            _pendingTasks.begin(),
            _pendingTasks.end(),
            [&is_eligible_for_backend, &fastest_worker](const PendingTask &task)
            {
                return is_eligible_for_backend(task, fastest_worker->backend);
            }));
    const bool candidate_must_drain_exclusive_task = std::any_of(
        _pendingTasks.begin(),
        _pendingTasks.end(),
        [&is_eligible_for_backend,
         &candidate_worker,
         &fastest_worker](const PendingTask &task)
        {
            return is_eligible_for_backend(task, candidate_worker->backend) &&
                   !is_eligible_for_backend(task, fastest_worker->backend);
        });
    if (candidate_must_drain_exclusive_task)
    {
        return false;
    }

    const double candidate_service_milliseconds =
        worker_state_it->second.physicalServiceSamples > 0
        ? worker_state_it->second.emaPhysicalServiceMilliseconds
        : worker_stats_it->second.emaElapsedMilliseconds;
    const double candidate_finish_milliseconds =
        static_cast<double>(worker_state_it->second.inFlightTasks + 1) *
        candidate_service_milliseconds;
    const double fastest_queue_clear_milliseconds =
        static_cast<double>(fastest_in_flight_tasks +
                            static_cast<int>(fastest_eligible_pending_tasks)) *
        fastest_elapsed_milliseconds;
    return candidate_finish_milliseconds >= fastest_queue_clear_milliseconds;
}

void DepthComputeScheduler::advanceRevision()
{
    ++_revision;
    _stateChanged.notify_all();
}

std::size_t DepthComputeScheduler::pendingTaskCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _pendingTasks.size();
}

std::unordered_map<std::string, DepthComputeWorkerStats>
DepthComputeScheduler::workerStats() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _workerStats;
}

std::optional<double>
DepthComputeScheduler::fastestSuccessfulAlternativeBackendEmaMilliseconds(
    const DepthComputeWorker &worker) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return fastestSuccessfulAlternativeBackendEmaMillisecondsLocked(worker);
}

std::optional<double>
DepthComputeScheduler::tryRejectUnprofitableCalibrationProbe(
    const DepthComputeWorker &worker,
    int viewIndex,
    double coarseLevelElapsedMilliseconds,
    std::chrono::duration<double, std::milli> probeElapsed)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_benefitAwareSchedulingEnabled ||
        !std::isfinite(coarseLevelElapsedMilliseconds))
    {
        return std::nullopt;
    }

    const std::string worker_id = worker.id();
    const auto task_it = _inFlightTasks.find(viewIndex);
    const auto state_it = _workerSchedulingStates.find(worker_id);
    if (task_it == _inFlightTasks.end() ||
        task_it->second.workerId != worker_id ||
        !task_it->second.calibrationProbe ||
        task_it->second.requiresFullFrame ||
        task_it->second.retryCount != 0 ||
        state_it == _workerSchedulingStates.end() ||
        state_it->second.failureRetired ||
        state_it->second.retirementPending)
    {
        return std::nullopt;
    }

    const std::optional<double> alternative_elapsed =
        fastestSuccessfulAlternativeBackendEmaMillisecondsLocked(worker);
    if (!alternative_elapsed.has_value() ||
        coarseLevelElapsedMilliseconds < *alternative_elapsed)
    {
        return std::nullopt;
    }

    // Commit the retry insertion before changing ownership so a deque-growth
    // exception leaves the calibration probe's scheduler state untouched.
    _pendingTasks.push_front(
        {viewIndex, task_it->second.retryCount + 1, worker.backend});
    _inFlightTasks.erase(task_it);

    DepthComputeWorkerStats &stats = _workerStats[worker_id];
    ++stats.completedTasks;
    ++stats.failedTasks;
    const double probe_elapsed_milliseconds = probeElapsed.count();
    if (std::isfinite(probe_elapsed_milliseconds) &&
        probe_elapsed_milliseconds > 0.0)
    {
        stats.elapsedMilliseconds += probe_elapsed_milliseconds;
    }

    WorkerSchedulingState &state = state_it->second;
    state.inFlightTasks = std::max(0, state.inFlightTasks - 1);
    state.calibrationInFlight = false;
    state.calibrationSucceeded = false;
    state.pausedForBenefit = false;
    state.retirementPending = false;
    state.failureRetired = true;
    reactivateAlternativeBackends(worker);
    advanceRevision();
    return alternative_elapsed;
}

std::optional<double>
DepthComputeScheduler::fastestSuccessfulAlternativeBackendEmaMillisecondsLocked(
    const DepthComputeWorker &worker) const
{
    double fastest_elapsed_milliseconds =
        std::numeric_limits<double>::infinity();
    for (const std::string &participant_id : _participatingWorkerIds)
    {
        const auto participant = depthComputeWorkerFromId(participant_id);
        const auto state_it = _workerSchedulingStates.find(participant_id);
        const auto stats_it = _workerStats.find(participant_id);
        if (!participant.has_value() ||
            participant->backend == worker.backend ||
            state_it == _workerSchedulingStates.end() ||
            state_it->second.retirementPending ||
            state_it->second.failureRetired ||
            stats_it == _workerStats.end() ||
            stats_it->second.successfulTasks <= 0 ||
            !std::isfinite(stats_it->second.emaElapsedMilliseconds) ||
            stats_it->second.emaElapsedMilliseconds <= 0.0)
        {
            continue;
        }
        const double effective_elapsed_milliseconds =
            state_it->second.physicalServiceSamples > 0 &&
                    std::isfinite(
                        state_it->second.emaPhysicalServiceMilliseconds) &&
                    state_it->second.emaPhysicalServiceMilliseconds > 0.0
                ? state_it->second.emaPhysicalServiceMilliseconds
                : stats_it->second.emaElapsedMilliseconds;
        fastest_elapsed_milliseconds = std::min(
            fastest_elapsed_milliseconds,
            effective_elapsed_milliseconds);
    }
    if (!std::isfinite(fastest_elapsed_milliseconds))
    {
        return std::nullopt;
    }
    return fastest_elapsed_milliseconds;
}

} // namespace mvs
} // namespace xjw
