#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xjw::task_runtime
{

    using TaskId = std::string;
    using RunId = std::string;
    using AttemptId = std::uint32_t;

    enum class TaskState
    {
        Queued,
        Blocked,
        Running,
        PauseRequested,
        Paused,
        CancelRequested,
        Succeeded,
        Failed,
        Cancelled,
        Interrupted
    };

    enum class ProjectAccess
    {
        None,
        Read,
        Write
    };

    struct TaskCapabilities
    {
        bool canPause = false;
        bool canCheckpoint = false;
        bool canReorder = true;
        bool canCancel = true;
    };

    struct TaskResourceRequest
    {
        int cpuSlots = 1;
        std::string accelerator;
        int acceleratorSlots = 0;
        ProjectAccess projectAccess = ProjectAccess::None;
    };

    struct TaskDefinition
    {
        TaskId taskId;
        std::string kind;
        std::string displayName;
        std::string projectKey;
        std::string chunkId;
        std::uint64_t projectGeneration = 0;
        int priority = 0;
        std::vector<TaskId> dependencies;
        TaskCapabilities capabilities;
        TaskResourceRequest resources;
        std::string payload;
    };

    struct TaskProgress
    {
        std::string stage;
        std::uint64_t completedUnits = 0;
        std::uint64_t totalUnits = 0;
        double fraction = 0.0;
    };

    struct TaskCheckpoint
    {
        std::uint32_t schemaVersion = 1;
        std::string location;
        std::string inputSignature;
        std::uint64_t completedUnits = 0;
        std::uint64_t totalUnits = 0;
    };

    struct TaskResult
    {
        std::string summary;
        std::string outputLocation;
    };

    struct TaskError
    {
        std::string code;
        std::string message;
        bool retryable = false;
    };

    struct TaskRunSnapshot
    {
        RunId runId;
        AttemptId attemptId = 1;
        TaskDefinition definition;
        TaskState state = TaskState::Queued;
        std::uint64_t revision = 1;
        std::uint64_t queueSequence = 0;
        std::string blockedReason;
        TaskProgress progress;
        std::optional<TaskCheckpoint> checkpoint;
        std::optional<TaskResult> result;
        std::optional<TaskError> error;
        std::chrono::system_clock::time_point submittedAt;
        std::optional<std::chrono::system_clock::time_point> startedAt;
        std::optional<std::chrono::system_clock::time_point> finishedAt;
    };

    enum class TaskEventKind
    {
        Submitted,
        StateChanged,
        ProgressChanged,
        CheckpointSaved,
        PriorityChanged,
        Removed
    };

    struct TaskEvent
    {
        TaskEventKind kind = TaskEventKind::StateChanged;
        RunId runId;
        std::uint64_t revision = 0;
        TaskState state = TaskState::Queued;
    };

    const char* taskStateName(TaskState state);
    std::optional<TaskState> taskStateFromName(const std::string& name);
    bool isTerminalTaskState(TaskState state);
    bool canTransitionTaskState(TaskState from, TaskState to);

} // namespace xjw::task_runtime
