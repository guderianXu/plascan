#include "TaskScheduler.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <limits>
#include <set>
#include <utility>

namespace xjw::task_runtime
{

    struct TaskScheduler::RunRecord
    {
        TaskRunSnapshot snapshot;
        std::shared_ptr<TaskControlToken> control = std::make_shared<TaskControlToken>();
        bool resourcesHeld = false;
    };

    struct TaskScheduler::ResourceUsage
    {
        struct ProjectUse
        {
            int readers = 0;
            bool writer = false;
        };

        int cpuSlots = 0;
        std::unordered_map<std::string, int> acceleratorSlots;
        std::unordered_map<std::string, ProjectUse> projects;
    };

    namespace
    {

        constexpr auto taskOrderLess = [](const auto* lhs, const auto* rhs)
        {
            if (lhs->snapshot.definition.priority != rhs->snapshot.definition.priority)
            {
                return lhs->snapshot.definition.priority > rhs->snapshot.definition.priority;
            }
            return lhs->snapshot.queueSequence < rhs->snapshot.queueSequence;
        };

        bool isPendingState(TaskState state)
        {
            return state == TaskState::Queued || state == TaskState::Blocked;
        }

    } // namespace

    TaskScheduler::TaskScheduler(TaskSchedulerLimits limits)
        : _limits(std::move(limits)), _resourceUsage(std::make_unique<ResourceUsage>())
    {
        _limits.cpuSlots = std::max(1, _limits.cpuSlots);
        _workers.reserve(static_cast<std::size_t>(_limits.cpuSlots));
        for (int index = 0; index < _limits.cpuSlots; ++index)
        {
            _workers.emplace_back(&TaskScheduler::workerLoop, this);
        }
    }

    TaskScheduler::~TaskScheduler()
    {
        shutdown();
    }

    void TaskScheduler::registerExecutor(const std::string& kind, std::shared_ptr<ITaskExecutor> executor)
    {
        std::vector<TaskEvent> events;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (kind.empty() || !executor)
            {
                return;
            }
            _executors[kind] = std::move(executor);
            refreshBlockedStatesLocked(&events);
            _stateChanged.notify_all();
        }
        publishEvents(events);
    }

    void TaskScheduler::setProjectEpochGuard(std::shared_ptr<IProjectEpochGuard> guard)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _projectEpochGuard = std::move(guard);
    }

    TaskSubmitResult TaskScheduler::submit(TaskDefinition definition)
    {
        std::vector<TaskDefinition> definitions;
        definitions.push_back(std::move(definition));
        return submitBatch(std::move(definitions));
    }

    TaskSubmitResult TaskScheduler::submitBatch(std::vector<TaskDefinition> definitions)
    {
        TaskSubmitResult result;
        std::vector<TaskEvent> events;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_stopping)
            {
                result.error = "scheduler_stopping";
                return result;
            }
            if (definitions.empty())
            {
                result.error = "empty_batch";
                return result;
            }

            std::set<TaskId> batch_ids;
            for (const TaskDefinition& definition : definitions)
            {
                if (definition.taskId.empty() || definition.kind.empty())
                {
                    result.error = "task_id_and_kind_required";
                    return result;
                }
                if (_taskRuns.contains(definition.taskId) || !batch_ids.insert(definition.taskId).second)
                {
                    result.error = "duplicate_task_id:" + definition.taskId;
                    return result;
                }
                if (definition.resources.cpuSlots < 0 || definition.resources.acceleratorSlots < 0)
                {
                    result.error = "negative_resource_request:" + definition.taskId;
                    return result;
                }
            }
            if (!dependenciesValidLocked(definitions, &result.error) ||
                !dependenciesAcyclicLocked(definitions, &result.error))
            {
                return result;
            }

            const auto now = std::chrono::system_clock::now();
            for (TaskDefinition& definition : definitions)
            {
                auto record = std::make_unique<RunRecord>();
                record->snapshot.runId = nextRunIdLocked(definition.taskId);
                record->snapshot.definition = std::move(definition);
                record->snapshot.queueSequence = _nextQueueSequence++;
                record->snapshot.submittedAt = now;
                const RunId run_id = record->snapshot.runId;
                const TaskId task_id = record->snapshot.definition.taskId;
                _taskRuns[task_id] = run_id;
                result.runIds.push_back(run_id);
                events.push_back({TaskEventKind::Submitted, run_id, record->snapshot.revision, record->snapshot.state});
                _runs.emplace(run_id, std::move(record));
            }
            refreshBlockedStatesLocked(&events);
            result.accepted = true;
            _stateChanged.notify_all();
        }
        publishEvents(events);
        return result;
    }

    TaskSubmitResult TaskScheduler::restore(std::vector<TaskRunSnapshot> snapshots)
    {
        TaskSubmitResult result;
        std::vector<TaskEvent> events;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_stopping)
            {
                result.error = "scheduler_stopping";
                return result;
            }
            std::set<RunId> restored_run_ids;
            std::set<TaskId> restored_task_ids;
            std::vector<TaskDefinition> restored_definitions;
            restored_definitions.reserve(snapshots.size());
            for (const TaskRunSnapshot& snapshot : snapshots)
            {
                if (snapshot.runId.empty() || snapshot.definition.taskId.empty() || snapshot.definition.kind.empty())
                {
                    result.error = "invalid_restored_snapshot";
                    return result;
                }
                if (_runs.contains(snapshot.runId) || _taskRuns.contains(snapshot.definition.taskId) ||
                    !restored_run_ids.insert(snapshot.runId).second ||
                    !restored_task_ids.insert(snapshot.definition.taskId).second)
                {
                    result.error = "duplicate_restored_run:" + snapshot.runId;
                    return result;
                }
                restored_definitions.push_back(snapshot.definition);
            }
            if (!dependenciesValidLocked(restored_definitions, &result.error) ||
                !dependenciesAcyclicLocked(restored_definitions, &result.error))
            {
                return result;
            }

            for (TaskRunSnapshot& snapshot : snapshots)
            {
                if (snapshot.state == TaskState::Running || snapshot.state == TaskState::PauseRequested ||
                    snapshot.state == TaskState::CancelRequested)
                {
                    snapshot.state = TaskState::Interrupted;
                    snapshot.blockedReason = "process_interrupted";
                    ++snapshot.revision;
                }
                auto record = std::make_unique<RunRecord>();
                record->snapshot = std::move(snapshot);
                if (record->snapshot.state == TaskState::Paused)
                {
                    TaskControlSource(*record->control).requestPause();
                }
                _nextQueueSequence = std::max(_nextQueueSequence, record->snapshot.queueSequence + 1);
                const RunId run_id = record->snapshot.runId;
                const TaskId task_id = record->snapshot.definition.taskId;
                _taskRuns[task_id] = run_id;
                result.runIds.push_back(run_id);
                events.push_back({TaskEventKind::Submitted, run_id, record->snapshot.revision, record->snapshot.state});
                _runs.emplace(run_id, std::move(record));
            }
            refreshBlockedStatesLocked(&events);
            result.accepted = true;
            _stateChanged.notify_all();
        }
        publishEvents(events);
        return result;
    }

    TaskCommandResult TaskScheduler::requestPause(const RunId& runId, std::optional<std::uint64_t> expectedRevision)
    {
        std::vector<TaskEvent> events;
        TaskCommandResult result;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto found = _runs.find(runId);
            if (found == _runs.end())
            {
                return commandErrorLocked("run_not_found");
            }
            RunRecord& record = *found->second;
            if (!revisionMatches(record, expectedRevision))
            {
                return commandErrorLocked("revision_conflict");
            }
            if (!record.snapshot.definition.capabilities.canPause)
            {
                return commandErrorLocked("pause_not_supported");
            }
            if (record.snapshot.state == TaskState::Paused || record.snapshot.state == TaskState::PauseRequested)
            {
                result = commandSnapshotLocked(record);
            }
            else if (isPendingState(record.snapshot.state))
            {
                TaskControlSource(*record.control).requestPause();
                transitionLocked(record, TaskState::Paused, TaskEventKind::StateChanged, &events);
                result = commandSnapshotLocked(record);
            }
            else if (record.snapshot.state == TaskState::Running)
            {
                TaskControlSource(*record.control).requestPause();
                transitionLocked(record, TaskState::PauseRequested, TaskEventKind::StateChanged, &events);
                result = commandSnapshotLocked(record);
            }
            else
            {
                return commandErrorLocked("state_not_pausable");
            }
            _stateChanged.notify_all();
        }
        publishEvents(events);
        return result;
    }

    TaskCommandResult TaskScheduler::resume(const RunId& runId, std::optional<std::uint64_t> expectedRevision)
    {
        std::vector<TaskEvent> events;
        TaskCommandResult result;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto found = _runs.find(runId);
            if (found == _runs.end())
            {
                return commandErrorLocked("run_not_found");
            }
            RunRecord& record = *found->second;
            if (!revisionMatches(record, expectedRevision))
            {
                return commandErrorLocked("revision_conflict");
            }
            if (record.snapshot.state != TaskState::Paused && record.snapshot.state != TaskState::Interrupted)
            {
                return commandErrorLocked("state_not_resumable");
            }
            if (record.snapshot.state == TaskState::Interrupted)
            {
                if (!record.snapshot.definition.capabilities.canCheckpoint || !record.snapshot.checkpoint)
                {
                    return commandErrorLocked("checkpoint_required");
                }
                ++record.snapshot.attemptId;
            }
            TaskControlSource(*record.control).resume();
            transitionLocked(record, TaskState::Queued, TaskEventKind::StateChanged, &events);
            refreshBlockedStatesLocked(&events);
            result = commandSnapshotLocked(record);
            _stateChanged.notify_all();
        }
        publishEvents(events);
        return result;
    }

    TaskCommandResult TaskScheduler::requestCancel(const RunId& runId, std::optional<std::uint64_t> expectedRevision)
    {
        std::vector<TaskEvent> events;
        TaskCommandResult result;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto found = _runs.find(runId);
            if (found == _runs.end())
            {
                return commandErrorLocked("run_not_found");
            }
            RunRecord& record = *found->second;
            if (!revisionMatches(record, expectedRevision))
            {
                return commandErrorLocked("revision_conflict");
            }
            if (!record.snapshot.definition.capabilities.canCancel)
            {
                return commandErrorLocked("cancel_not_supported");
            }
            if (record.snapshot.state == TaskState::Cancelled || record.snapshot.state == TaskState::CancelRequested)
            {
                return commandSnapshotLocked(record);
            }
            if (isTerminalTaskState(record.snapshot.state))
            {
                return commandErrorLocked("state_not_cancellable");
            }
            TaskControlSource(*record.control).requestCancellation();
            const TaskState next_state =
                record.snapshot.state == TaskState::Running || record.snapshot.state == TaskState::PauseRequested
                    ? TaskState::CancelRequested
                    : TaskState::Cancelled;
            transitionLocked(record, next_state, TaskEventKind::StateChanged, &events);
            if (next_state == TaskState::Cancelled)
            {
                record.snapshot.finishedAt = std::chrono::system_clock::now();
            }
            refreshBlockedStatesLocked(&events);
            result = commandSnapshotLocked(record);
            _stateChanged.notify_all();
        }
        publishEvents(events);
        return result;
    }

    TaskCommandResult
    TaskScheduler::setPriority(const RunId& runId, int priority, std::optional<std::uint64_t> expectedRevision)
    {
        std::vector<TaskEvent> events;
        TaskCommandResult result;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto found = _runs.find(runId);
            if (found == _runs.end())
            {
                return commandErrorLocked("run_not_found");
            }
            RunRecord& record = *found->second;
            if (!revisionMatches(record, expectedRevision))
            {
                return commandErrorLocked("revision_conflict");
            }
            if (!isQueueMutable(record) || !record.snapshot.definition.capabilities.canReorder)
            {
                return commandErrorLocked("reorder_not_supported_in_state");
            }
            if (record.snapshot.definition.priority != priority)
            {
                record.snapshot.definition.priority = priority;
                ++record.snapshot.revision;
                events.push_back(
                    {TaskEventKind::PriorityChanged, runId, record.snapshot.revision, record.snapshot.state});
            }
            result = commandSnapshotLocked(record);
            _stateChanged.notify_all();
        }
        publishEvents(events);
        return result;
    }

    TaskCommandResult TaskScheduler::moveBefore(const RunId& runId,
                                                const RunId& referenceRunId,
                                                std::optional<std::uint64_t> expectedRevision)
    {
        std::vector<TaskEvent> events;
        TaskCommandResult result;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto found = _runs.find(runId);
            const auto reference_found = _runs.find(referenceRunId);
            if (found == _runs.end() || reference_found == _runs.end())
            {
                return commandErrorLocked("run_not_found");
            }
            RunRecord& record = *found->second;
            RunRecord& reference = *reference_found->second;
            if (!revisionMatches(record, expectedRevision))
            {
                return commandErrorLocked("revision_conflict");
            }
            if (!isQueueMutable(record) || !isQueueMutable(reference) ||
                !record.snapshot.definition.capabilities.canReorder)
            {
                return commandErrorLocked("reorder_not_supported_in_state");
            }
            reorderRelativeLocked(record, reference, true, &events);
            result = commandSnapshotLocked(record);
            _stateChanged.notify_all();
        }
        publishEvents(events);
        return result;
    }

    TaskCommandResult TaskScheduler::moveAfter(const RunId& runId,
                                               const RunId& referenceRunId,
                                               std::optional<std::uint64_t> expectedRevision)
    {
        std::vector<TaskEvent> events;
        TaskCommandResult result;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto found = _runs.find(runId);
            const auto reference_found = _runs.find(referenceRunId);
            if (found == _runs.end() || reference_found == _runs.end())
            {
                return commandErrorLocked("run_not_found");
            }
            RunRecord& record = *found->second;
            RunRecord& reference = *reference_found->second;
            if (!revisionMatches(record, expectedRevision))
            {
                return commandErrorLocked("revision_conflict");
            }
            if (!isQueueMutable(record) || !isQueueMutable(reference) ||
                !record.snapshot.definition.capabilities.canReorder)
            {
                return commandErrorLocked("reorder_not_supported_in_state");
            }
            reorderRelativeLocked(record, reference, false, &events);
            result = commandSnapshotLocked(record);
            _stateChanged.notify_all();
        }
        publishEvents(events);
        return result;
    }

    std::optional<TaskRunSnapshot> TaskScheduler::snapshot(const RunId& runId) const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        const auto found = _runs.find(runId);
        if (found == _runs.end())
        {
            return std::nullopt;
        }
        return found->second->snapshot;
    }

    std::vector<TaskRunSnapshot> TaskScheduler::snapshots() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<const RunRecord*> ordered;
        ordered.reserve(_runs.size());
        for (const auto& [run_id, record] : _runs)
        {
            (void)run_id;
            ordered.push_back(record.get());
        }
        std::sort(ordered.begin(), ordered.end(), taskOrderLess);

        std::vector<TaskRunSnapshot> result;
        result.reserve(ordered.size());
        for (const RunRecord* record : ordered)
        {
            result.push_back(record->snapshot);
        }
        return result;
    }

    std::size_t TaskScheduler::clearTerminalRuns()
    {
        std::vector<TaskEvent> events;
        std::size_t removed = 0;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            std::set<TaskId> dependencies_of_active_runs;
            for (const auto& [run_id, record] : _runs)
            {
                (void)run_id;
                if (!isTerminalTaskState(record->snapshot.state))
                {
                    dependencies_of_active_runs.insert(record->snapshot.definition.dependencies.begin(),
                                                       record->snapshot.definition.dependencies.end());
                }
            }
            for (auto iterator = _runs.begin(); iterator != _runs.end();)
            {
                const RunRecord& record = *iterator->second;
                if (!isTerminalTaskState(record.snapshot.state) ||
                    dependencies_of_active_runs.contains(record.snapshot.definition.taskId))
                {
                    ++iterator;
                    continue;
                }
                events.push_back(
                    {TaskEventKind::Removed, record.snapshot.runId, record.snapshot.revision, record.snapshot.state});
                const auto task_run = _taskRuns.find(record.snapshot.definition.taskId);
                if (task_run != _taskRuns.end() && task_run->second == record.snapshot.runId)
                {
                    _taskRuns.erase(task_run);
                }
                iterator = _runs.erase(iterator);
                ++removed;
            }
            _stateChanged.notify_all();
        }
        publishEvents(events);
        return removed;
    }

    bool TaskScheduler::waitForState(const RunId& runId, TaskState state, std::chrono::milliseconds timeout) const
    {
        std::unique_lock<std::mutex> lock(_mutex);
        return _stateChanged.wait_for(lock,
                                      timeout,
                                      [this, &runId, state]
                                      {
                                          const auto found = _runs.find(runId);
                                          return found != _runs.end() && found->second->snapshot.state == state;
                                      });
    }

    std::uint64_t TaskScheduler::subscribe(EventListener listener)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        const std::uint64_t id = _nextSubscriptionId++;
        _listeners.emplace(id, std::move(listener));
        return id;
    }

    void TaskScheduler::unsubscribe(std::uint64_t subscriptionId)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _listeners.erase(subscriptionId);
    }

    void TaskScheduler::shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_stopping)
            {
                return;
            }
            _stopping = true;
            for (auto& [run_id, record] : _runs)
            {
                (void)run_id;
                if (!isTerminalTaskState(record->snapshot.state))
                {
                    TaskControlSource(*record->control).requestCancellation();
                }
            }
            _stateChanged.notify_all();
        }
        for (std::thread& worker : _workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        _workers.clear();
    }

    TaskCommandResult TaskScheduler::commandErrorLocked(const std::string& error) const
    {
        return {false, error, std::nullopt};
    }

    TaskCommandResult TaskScheduler::commandSnapshotLocked(const RunRecord& record) const
    {
        return {true, {}, record.snapshot};
    }

    bool TaskScheduler::revisionMatches(const RunRecord& record, std::optional<std::uint64_t> expectedRevision) const
    {
        return !expectedRevision || *expectedRevision == record.snapshot.revision;
    }

    bool TaskScheduler::isQueueMutable(const RunRecord& record) const
    {
        return isPendingState(record.snapshot.state);
    }

    void TaskScheduler::refreshBlockedStatesLocked(std::vector<TaskEvent>* events)
    {
        bool propagated_failure = false;
        do
        {
            propagated_failure = false;
            for (auto& [run_id, record_ptr] : _runs)
            {
                (void)run_id;
                RunRecord& record = *record_ptr;
                if (!isPendingState(record.snapshot.state))
                {
                    continue;
                }

                std::string blocked_reason;
                std::string failed_dependency;
                for (const TaskId& dependency_id : record.snapshot.definition.dependencies)
                {
                    const auto dependency_run = _taskRuns.find(dependency_id);
                    if (dependency_run == _taskRuns.end())
                    {
                        blocked_reason = "missing_dependency:" + dependency_id;
                        break;
                    }
                    const TaskState dependency_state = _runs.at(dependency_run->second)->snapshot.state;
                    if (dependency_state == TaskState::Failed || dependency_state == TaskState::Cancelled)
                    {
                        failed_dependency = dependency_id;
                        break;
                    }
                    if (dependency_state != TaskState::Succeeded)
                    {
                        blocked_reason = "waiting_dependency:" + dependency_id;
                        break;
                    }
                }
                if (!failed_dependency.empty())
                {
                    record.snapshot.blockedReason.clear();
                    record.snapshot.error =
                        TaskError{"dependency_failed", "dependency did not succeed:" + failed_dependency, false};
                    record.snapshot.finishedAt = std::chrono::system_clock::now();
                    transitionLocked(record, TaskState::Failed, TaskEventKind::StateChanged, events);
                    propagated_failure = true;
                    continue;
                }
                if (blocked_reason.empty() && !_executors.contains(record.snapshot.definition.kind))
                {
                    blocked_reason = "executor_unavailable:" + record.snapshot.definition.kind;
                }
                if (blocked_reason.empty())
                {
                    resourcesAvailableLocked(
                        record.snapshot.definition.resources, record.snapshot.definition.projectKey, &blocked_reason);
                }

                const TaskState desired_state = blocked_reason.empty() ? TaskState::Queued : TaskState::Blocked;
                const bool reason_changed = record.snapshot.blockedReason != blocked_reason;
                record.snapshot.blockedReason = blocked_reason;
                if (record.snapshot.state != desired_state)
                {
                    transitionLocked(record, desired_state, TaskEventKind::StateChanged, events);
                }
                else if (reason_changed)
                {
                    ++record.snapshot.revision;
                    events->push_back(
                        {TaskEventKind::StateChanged, record.snapshot.runId, record.snapshot.revision, desired_state});
                }
            }
        } while (propagated_failure);
    }

    TaskScheduler::RunRecord* TaskScheduler::selectRunnableLocked()
    {
        std::vector<RunRecord*> candidates;
        for (auto& [run_id, record] : _runs)
        {
            (void)run_id;
            if (record->snapshot.state == TaskState::Queued)
            {
                std::string reason;
                if (resourcesAvailableLocked(
                        record->snapshot.definition.resources, record->snapshot.definition.projectKey, &reason))
                {
                    candidates.push_back(record.get());
                }
            }
        }
        if (candidates.empty())
        {
            return nullptr;
        }
        return *std::min_element(candidates.begin(), candidates.end(), taskOrderLess);
    }

    bool TaskScheduler::resourcesAvailableLocked(const TaskResourceRequest& request,
                                                 const std::string& projectKey,
                                                 std::string* reason) const
    {
        const int requested_cpu = std::max(0, request.cpuSlots);
        if (_resourceUsage->cpuSlots + requested_cpu > _limits.cpuSlots)
        {
            if (reason)
            {
                *reason = "waiting_resource:cpu";
            }
            return false;
        }
        if (request.acceleratorSlots > 0)
        {
            const auto budget = _limits.acceleratorSlots.find(request.accelerator);
            const auto used = _resourceUsage->acceleratorSlots.find(request.accelerator);
            const int available = budget == _limits.acceleratorSlots.end() ? 0 : budget->second;
            const int in_use = used == _resourceUsage->acceleratorSlots.end() ? 0 : used->second;
            if (in_use + request.acceleratorSlots > available)
            {
                if (reason)
                {
                    *reason = "waiting_resource:accelerator:" + request.accelerator;
                }
                return false;
            }
        }
        if (!projectKey.empty() && request.projectAccess != ProjectAccess::None)
        {
            const auto found = _resourceUsage->projects.find(projectKey);
            const ResourceUsage::ProjectUse use =
                found == _resourceUsage->projects.end() ? ResourceUsage::ProjectUse{} : found->second;
            const bool conflict =
                request.projectAccess == ProjectAccess::Write ? use.writer || use.readers > 0 : use.writer;
            if (conflict)
            {
                if (reason)
                {
                    *reason = "waiting_resource:project:" + projectKey;
                }
                return false;
            }
        }
        if (reason)
        {
            reason->clear();
        }
        return true;
    }

    void TaskScheduler::acquireResourcesLocked(const TaskDefinition& definition)
    {
        const TaskResourceRequest& request = definition.resources;
        _resourceUsage->cpuSlots += std::max(0, request.cpuSlots);
        if (request.acceleratorSlots > 0)
        {
            _resourceUsage->acceleratorSlots[request.accelerator] += request.acceleratorSlots;
        }
        if (!definition.projectKey.empty())
        {
            ResourceUsage::ProjectUse& use = _resourceUsage->projects[definition.projectKey];
            if (request.projectAccess == ProjectAccess::Write)
            {
                use.writer = true;
            }
            else if (request.projectAccess == ProjectAccess::Read)
            {
                ++use.readers;
            }
        }
    }

    void TaskScheduler::releaseResourcesLocked(const TaskDefinition& definition)
    {
        const TaskResourceRequest& request = definition.resources;
        _resourceUsage->cpuSlots = std::max(0, _resourceUsage->cpuSlots - std::max(0, request.cpuSlots));
        if (request.acceleratorSlots > 0)
        {
            int& used = _resourceUsage->acceleratorSlots[request.accelerator];
            used = std::max(0, used - request.acceleratorSlots);
        }
        if (!definition.projectKey.empty())
        {
            auto found = _resourceUsage->projects.find(definition.projectKey);
            if (found != _resourceUsage->projects.end())
            {
                if (request.projectAccess == ProjectAccess::Write)
                {
                    found->second.writer = false;
                }
                else if (request.projectAccess == ProjectAccess::Read)
                {
                    found->second.readers = std::max(0, found->second.readers - 1);
                }
                if (!found->second.writer && found->second.readers == 0)
                {
                    _resourceUsage->projects.erase(found);
                }
            }
        }
    }

    bool TaskScheduler::dependenciesValidLocked(const std::vector<TaskDefinition>& definitions,
                                                std::string* error) const
    {
        std::set<TaskId> available;
        for (const auto& [task_id, run_id] : _taskRuns)
        {
            (void)run_id;
            available.insert(task_id);
        }
        for (const TaskDefinition& definition : definitions)
        {
            available.insert(definition.taskId);
        }
        for (const TaskDefinition& definition : definitions)
        {
            for (const TaskId& dependency : definition.dependencies)
            {
                if (!available.contains(dependency))
                {
                    if (error)
                    {
                        *error = "missing_dependency:" + dependency;
                    }
                    return false;
                }
                if (dependency == definition.taskId)
                {
                    if (error)
                    {
                        *error = "dependency_cycle:" + dependency;
                    }
                    return false;
                }
            }
        }
        return true;
    }

    bool TaskScheduler::dependenciesAcyclicLocked(const std::vector<TaskDefinition>& definitions,
                                                  std::string* error) const
    {
        std::unordered_map<TaskId, std::vector<TaskId>> graph;
        for (const auto& [run_id, record] : _runs)
        {
            (void)run_id;
            graph[record->snapshot.definition.taskId] = record->snapshot.definition.dependencies;
        }
        for (const TaskDefinition& definition : definitions)
        {
            graph[definition.taskId] = definition.dependencies;
        }

        std::unordered_map<TaskId, int> colors;
        std::function<bool(const TaskId&)> visit = [&](const TaskId& task_id)
        {
            int& color = colors[task_id];
            if (color == 1)
            {
                return false;
            }
            if (color == 2)
            {
                return true;
            }
            color = 1;
            const auto found = graph.find(task_id);
            if (found != graph.end())
            {
                for (const TaskId& dependency : found->second)
                {
                    if (!visit(dependency))
                    {
                        return false;
                    }
                }
            }
            color = 2;
            return true;
        };

        for (const auto& [task_id, dependencies] : graph)
        {
            (void)dependencies;
            if (!visit(task_id))
            {
                if (error)
                {
                    *error = "dependency_cycle:" + task_id;
                }
                return false;
            }
        }
        return true;
    }

    RunId TaskScheduler::nextRunIdLocked(const TaskId& taskId)
    {
        return taskId + "-run-" + std::to_string(_nextRunSequence++);
    }

    void TaskScheduler::transitionLocked(RunRecord& record,
                                         TaskState state,
                                         TaskEventKind kind,
                                         std::vector<TaskEvent>* events)
    {
        if (record.snapshot.state == state)
        {
            return;
        }
        if (!canTransitionTaskState(record.snapshot.state, state))
        {
            return;
        }
        record.snapshot.state = state;
        ++record.snapshot.revision;
        events->push_back({kind, record.snapshot.runId, record.snapshot.revision, state});
    }

    void TaskScheduler::reorderRelativeLocked(RunRecord& record,
                                              RunRecord& reference,
                                              bool before,
                                              std::vector<TaskEvent>* events)
    {
        if (&record == &reference)
        {
            return;
        }
        std::vector<RunRecord*> ordered;
        for (auto& [run_id, candidate] : _runs)
        {
            (void)run_id;
            if (isQueueMutable(*candidate))
            {
                ordered.push_back(candidate.get());
            }
        }
        std::sort(ordered.begin(), ordered.end(), taskOrderLess);
        ordered.erase(std::remove(ordered.begin(), ordered.end(), &record), ordered.end());
        const auto reference_position = std::find(ordered.begin(), ordered.end(), &reference);
        if (reference_position == ordered.end())
        {
            return;
        }
        record.snapshot.definition.priority = reference.snapshot.definition.priority;
        ordered.insert(before ? reference_position : std::next(reference_position), &record);
        for (RunRecord* candidate : ordered)
        {
            candidate->snapshot.queueSequence = _nextQueueSequence++;
        }
        ++record.snapshot.revision;
        events->push_back(
            {TaskEventKind::PriorityChanged, record.snapshot.runId, record.snapshot.revision, record.snapshot.state});
    }

    void TaskScheduler::workerLoop()
    {
        while (true)
        {
            RunId run_id;
            std::vector<TaskEvent> events;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _stateChanged.wait(lock,
                                   [this]
                                   {
                                       if (_stopping)
                                       {
                                           return true;
                                       }
                                       return selectRunnableLocked() != nullptr;
                                   });
                if (_stopping)
                {
                    return;
                }
                RunRecord* record = selectRunnableLocked();
                if (!record)
                {
                    continue;
                }
                acquireResourcesLocked(record->snapshot.definition);
                record->resourcesHeld = true;
                record->snapshot.blockedReason.clear();
                record->snapshot.startedAt = std::chrono::system_clock::now();
                transitionLocked(*record, TaskState::Running, TaskEventKind::StateChanged, &events);
                refreshBlockedStatesLocked(&events);
                run_id = record->snapshot.runId;
            }
            publishEvents(events);
            executeRun(run_id);
        }
    }

    void TaskScheduler::executeRun(const RunId& runId)
    {
        TaskDefinition definition;
        AttemptId attempt_id = 1;
        std::optional<TaskCheckpoint> checkpoint;
        std::shared_ptr<TaskControlToken> control;
        std::shared_ptr<ITaskExecutor> executor;
        std::shared_ptr<IProjectEpochGuard> epoch_guard;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto found = _runs.find(runId);
            if (found == _runs.end())
            {
                return;
            }
            RunRecord& record = *found->second;
            definition = record.snapshot.definition;
            attempt_id = record.snapshot.attemptId;
            checkpoint = record.snapshot.checkpoint;
            control = record.control;
            executor = _executors[definition.kind];
            epoch_guard = _projectEpochGuard;
        }

        TaskExecutionContext context{
            runId,
            attempt_id,
            *control,
            checkpoint,
            [this, runId](const TaskProgress& progress)
            {
                TaskEvent event;
                bool changed = false;
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    const auto found = _runs.find(runId);
                    if (found != _runs.end() && (found->second->snapshot.state == TaskState::Running ||
                                                 found->second->snapshot.state == TaskState::PauseRequested ||
                                                 found->second->snapshot.state == TaskState::CancelRequested))
                    {
                        found->second->snapshot.progress = progress;
                        ++found->second->snapshot.revision;
                        event = {TaskEventKind::ProgressChanged,
                                 runId,
                                 found->second->snapshot.revision,
                                 found->second->snapshot.state};
                        changed = true;
                        _stateChanged.notify_all();
                    }
                }
                if (changed)
                {
                    publishEvents({event});
                }
            },
            [this, runId](const TaskCheckpoint& saved_checkpoint)
            {
                TaskEvent event;
                bool changed = false;
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    const auto found = _runs.find(runId);
                    if (found != _runs.end() && found->second->snapshot.definition.capabilities.canCheckpoint)
                    {
                        found->second->snapshot.checkpoint = saved_checkpoint;
                        ++found->second->snapshot.revision;
                        event = {TaskEventKind::CheckpointSaved,
                                 runId,
                                 found->second->snapshot.revision,
                                 found->second->snapshot.state};
                        changed = true;
                    }
                }
                if (changed)
                {
                    publishEvents({event});
                }
            },
            [definition, epoch_guard] { return !epoch_guard || epoch_guard->isCurrent(definition); }};

        TaskExecutionOutcome outcome;
        try
        {
            outcome = executor->execute(definition, context);
        }
        catch (const std::exception& exception)
        {
            outcome.status = TaskExecutionStatus::Failed;
            outcome.error = TaskError{"executor_exception", exception.what(), false};
        }
        catch (...)
        {
            outcome.status = TaskExecutionStatus::Failed;
            outcome.error = TaskError{"executor_exception", "unknown exception", false};
        }

        std::vector<TaskEvent> events;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            const auto found = _runs.find(runId);
            if (found == _runs.end())
            {
                return;
            }
            RunRecord& record = *found->second;
            if (record.resourcesHeld)
            {
                releaseResourcesLocked(record.snapshot.definition);
                record.resourcesHeld = false;
            }
            if (outcome.checkpoint)
            {
                record.snapshot.checkpoint = outcome.checkpoint;
            }

            TaskState final_state = TaskState::Failed;
            if (control->isCancellationRequested() || outcome.status == TaskExecutionStatus::Cancelled)
            {
                final_state = TaskState::Cancelled;
            }
            else if (outcome.status == TaskExecutionStatus::Paused)
            {
                if (record.snapshot.definition.capabilities.canPause &&
                    (!record.snapshot.definition.capabilities.canCheckpoint || record.snapshot.checkpoint))
                {
                    final_state = TaskState::Paused;
                }
                else
                {
                    outcome.error =
                        TaskError{"pause_checkpoint_missing", "executor paused without a required checkpoint", false};
                }
            }
            else if (outcome.status == TaskExecutionStatus::Succeeded)
            {
                const bool generation_current =
                    !_projectEpochGuard || _projectEpochGuard->isCurrent(record.snapshot.definition);
                if (generation_current)
                {
                    final_state = TaskState::Succeeded;
                }
                else
                {
                    outcome.error = TaskError{
                        "stale_project_generation", "task result belongs to an obsolete project session", false};
                }
            }
            record.snapshot.result = outcome.result;
            record.snapshot.error = outcome.error;
            if (final_state == TaskState::Succeeded)
            {
                record.snapshot.checkpoint.reset();
            }
            transitionLocked(record, final_state, TaskEventKind::StateChanged, &events);
            if (isTerminalTaskState(final_state))
            {
                record.snapshot.finishedAt = std::chrono::system_clock::now();
            }
            refreshBlockedStatesLocked(&events);
            _stateChanged.notify_all();
        }
        publishEvents(events);
    }

    void TaskScheduler::publishEvents(const std::vector<TaskEvent>& events) const
    {
        if (events.empty())
        {
            return;
        }
        std::vector<EventListener> listeners;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            listeners.reserve(_listeners.size());
            for (const auto& [id, listener] : _listeners)
            {
                (void)id;
                listeners.push_back(listener);
            }
        }
        for (const TaskEvent& event : events)
        {
            for (const EventListener& listener : listeners)
            {
                if (listener)
                {
                    listener(event);
                }
            }
        }
    }

} // namespace xjw::task_runtime
