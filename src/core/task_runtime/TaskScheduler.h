#pragma once

#include "TaskControl.h"
#include "TaskTypes.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace xjw::task_runtime
{

    class IProjectEpochGuard
    {
    public:
        virtual ~IProjectEpochGuard() = default;
        virtual bool isCurrent(const TaskDefinition& definition) const = 0;
    };

    enum class TaskExecutionStatus
    {
        Succeeded,
        Failed,
        Paused,
        Cancelled
    };

    struct TaskExecutionOutcome
    {
        TaskExecutionStatus status = TaskExecutionStatus::Failed;
        std::optional<TaskCheckpoint> checkpoint;
        std::optional<TaskResult> result;
        std::optional<TaskError> error;
    };

    struct TaskExecutionContext
    {
        RunId runId;
        AttemptId attemptId = 1;
        const TaskControlToken& control;
        std::optional<TaskCheckpoint> checkpoint;
        std::function<void(const TaskProgress&)> reportProgress;
        std::function<void(const TaskCheckpoint&)> saveCheckpoint;
        std::function<bool()> isProjectEpochCurrent;
    };

    class ITaskExecutor
    {
    public:
        virtual ~ITaskExecutor() = default;
        virtual TaskExecutionOutcome execute(const TaskDefinition& definition, TaskExecutionContext& context) = 0;
    };

    struct TaskSchedulerLimits
    {
        int cpuSlots = 1;
        std::unordered_map<std::string, int> acceleratorSlots;
    };

    struct TaskCommandResult
    {
        bool accepted = false;
        std::string error;
        std::optional<TaskRunSnapshot> snapshot;
    };

    struct TaskSubmitResult
    {
        bool accepted = false;
        std::string error;
        std::vector<RunId> runIds;
    };

    class TaskScheduler final
    {
    public:
        using EventListener = std::function<void(const TaskEvent&)>;

        explicit TaskScheduler(TaskSchedulerLimits limits = {});
        ~TaskScheduler();
        TaskScheduler(const TaskScheduler&) = delete;
        TaskScheduler& operator=(const TaskScheduler&) = delete;

        void registerExecutor(const std::string& kind, std::shared_ptr<ITaskExecutor> executor);
        void setProjectEpochGuard(std::shared_ptr<IProjectEpochGuard> guard);

        TaskSubmitResult submit(TaskDefinition definition);
        TaskSubmitResult submitBatch(std::vector<TaskDefinition> definitions);
        TaskSubmitResult restore(std::vector<TaskRunSnapshot> snapshots);
        TaskCommandResult requestPause(const RunId& runId,
                                       std::optional<std::uint64_t> expectedRevision = std::nullopt);
        TaskCommandResult resume(const RunId& runId, std::optional<std::uint64_t> expectedRevision = std::nullopt);
        TaskCommandResult requestCancel(const RunId& runId,
                                        std::optional<std::uint64_t> expectedRevision = std::nullopt);
        TaskCommandResult
        setPriority(const RunId& runId, int priority, std::optional<std::uint64_t> expectedRevision = std::nullopt);
        TaskCommandResult moveBefore(const RunId& runId,
                                     const RunId& referenceRunId,
                                     std::optional<std::uint64_t> expectedRevision = std::nullopt);
        TaskCommandResult moveAfter(const RunId& runId,
                                    const RunId& referenceRunId,
                                    std::optional<std::uint64_t> expectedRevision = std::nullopt);

        std::optional<TaskRunSnapshot> snapshot(const RunId& runId) const;
        std::vector<TaskRunSnapshot> snapshots() const;
        std::size_t clearTerminalRuns();
        bool waitForState(const RunId& runId, TaskState state, std::chrono::milliseconds timeout) const;

        std::uint64_t subscribe(EventListener listener);
        void unsubscribe(std::uint64_t subscriptionId);
        void shutdown();

    private:
        struct RunRecord;
        struct ResourceUsage;

        TaskCommandResult commandErrorLocked(const std::string& error) const;
        TaskCommandResult commandSnapshotLocked(const RunRecord& record) const;
        bool revisionMatches(const RunRecord& record, std::optional<std::uint64_t> expectedRevision) const;
        bool isQueueMutable(const RunRecord& record) const;
        void refreshBlockedStatesLocked(std::vector<TaskEvent>* events);
        RunRecord* selectRunnableLocked();
        bool resourcesAvailableLocked(const TaskResourceRequest& request,
                                      const std::string& projectKey,
                                      std::string* reason) const;
        void acquireResourcesLocked(const TaskDefinition& definition);
        void releaseResourcesLocked(const TaskDefinition& definition);
        bool dependenciesValidLocked(const std::vector<TaskDefinition>& definitions, std::string* error) const;
        bool dependenciesAcyclicLocked(const std::vector<TaskDefinition>& definitions, std::string* error) const;
        RunId nextRunIdLocked(const TaskId& taskId);
        void transitionLocked(RunRecord& record, TaskState state, TaskEventKind kind, std::vector<TaskEvent>* events);
        void
        reorderRelativeLocked(RunRecord& record, RunRecord& reference, bool before, std::vector<TaskEvent>* events);
        void workerLoop();
        void executeRun(const RunId& runId);
        void publishEvents(const std::vector<TaskEvent>& events) const;

        TaskSchedulerLimits _limits;
        mutable std::mutex _mutex;
        mutable std::condition_variable _stateChanged;
        bool _stopping = false;
        std::uint64_t _nextRunSequence = 1;
        std::uint64_t _nextQueueSequence = 1;
        std::uint64_t _nextSubscriptionId = 1;
        std::unordered_map<RunId, std::unique_ptr<RunRecord>> _runs;
        std::unordered_map<TaskId, RunId> _taskRuns;
        std::unordered_map<std::string, std::shared_ptr<ITaskExecutor>> _executors;
        std::unordered_map<std::uint64_t, EventListener> _listeners;
        std::shared_ptr<IProjectEpochGuard> _projectEpochGuard;
        std::unique_ptr<ResourceUsage> _resourceUsage;
        std::vector<std::thread> _workers;
    };

} // namespace xjw::task_runtime
