#include "TaskTypes.h"

#include <array>
#include <utility>

namespace xjw::task_runtime
{
    namespace
    {

        using Transition = std::pair<TaskState, TaskState>;

        constexpr std::array<Transition, 27> kAllowedTransitions{{
            {TaskState::Queued, TaskState::Blocked},
            {TaskState::Queued, TaskState::Running},
            {TaskState::Queued, TaskState::Paused},
            {TaskState::Queued, TaskState::Failed},
            {TaskState::Queued, TaskState::Cancelled},
            {TaskState::Blocked, TaskState::Queued},
            {TaskState::Blocked, TaskState::Paused},
            {TaskState::Blocked, TaskState::Failed},
            {TaskState::Blocked, TaskState::Cancelled},
            {TaskState::Running, TaskState::PauseRequested},
            {TaskState::Running, TaskState::CancelRequested},
            {TaskState::Running, TaskState::Succeeded},
            {TaskState::Running, TaskState::Failed},
            {TaskState::Running, TaskState::Cancelled},
            {TaskState::Running, TaskState::Paused},
            {TaskState::PauseRequested, TaskState::Paused},
            {TaskState::PauseRequested, TaskState::CancelRequested},
            {TaskState::PauseRequested, TaskState::Succeeded},
            {TaskState::PauseRequested, TaskState::Failed},
            {TaskState::PauseRequested, TaskState::Cancelled},
            {TaskState::Paused, TaskState::Queued},
            {TaskState::Paused, TaskState::Cancelled},
            {TaskState::CancelRequested, TaskState::Cancelled},
            {TaskState::CancelRequested, TaskState::Succeeded},
            {TaskState::CancelRequested, TaskState::Failed},
            {TaskState::Interrupted, TaskState::Queued},
            {TaskState::Interrupted, TaskState::Cancelled},
        }};

    } // namespace

    const char* taskStateName(TaskState state)
    {
        switch (state)
        {
        case TaskState::Queued:
            return "queued";
        case TaskState::Blocked:
            return "blocked";
        case TaskState::Running:
            return "running";
        case TaskState::PauseRequested:
            return "pause_requested";
        case TaskState::Paused:
            return "paused";
        case TaskState::CancelRequested:
            return "cancel_requested";
        case TaskState::Succeeded:
            return "succeeded";
        case TaskState::Failed:
            return "failed";
        case TaskState::Cancelled:
            return "cancelled";
        case TaskState::Interrupted:
            return "interrupted";
        }
        return "unknown";
    }

    std::optional<TaskState> taskStateFromName(const std::string& name)
    {
        constexpr std::array<TaskState, 10> states{TaskState::Queued,
                                                   TaskState::Blocked,
                                                   TaskState::Running,
                                                   TaskState::PauseRequested,
                                                   TaskState::Paused,
                                                   TaskState::CancelRequested,
                                                   TaskState::Succeeded,
                                                   TaskState::Failed,
                                                   TaskState::Cancelled,
                                                   TaskState::Interrupted};
        for (const TaskState state : states)
        {
            if (name == taskStateName(state))
            {
                return state;
            }
        }
        return std::nullopt;
    }

    bool isTerminalTaskState(TaskState state)
    {
        return state == TaskState::Succeeded || state == TaskState::Failed || state == TaskState::Cancelled;
    }

    bool canTransitionTaskState(TaskState from, TaskState to)
    {
        if (from == to)
        {
            return true;
        }
        for (const Transition& transition : kAllowedTransitions)
        {
            if (transition.first == from && transition.second == to)
            {
                return true;
            }
        }
        return false;
    }

} // namespace xjw::task_runtime
