#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace xjw::task_runtime
{
    enum class WorkflowStatus
    {
        Succeeded,
        Failed,
        Cancelled
    };

    struct WorkflowError
    {
        std::string code;
        std::string message;
        std::string stage;
        std::string path;
    };

    struct WorkflowOutcome
    {
        WorkflowStatus status = WorkflowStatus::Failed;
        WorkflowError error;

        bool succeeded() const noexcept
        {
            return status == WorkflowStatus::Succeeded;
        }
    };

    struct WorkflowProgress
    {
        std::string stage;
        double ratio = 0.0;
    };

    // Cancellation only: algorithms do not claim checkpoint/pause support.
    struct WorkflowControl
    {
        std::shared_ptr<std::atomic_bool> cancellation;
        std::function<bool()> cancellationCheck;
        std::function<void(const WorkflowProgress&)> progress;

        bool isCancelled() const
        {
            return (cancellation && cancellation->load(std::memory_order_relaxed)) ||
                   (cancellationCheck && cancellationCheck());
        }

        void reportProgress(std::string stage, double ratio) const
        {
            if (progress)
            {
                progress({std::move(stage), std::isfinite(ratio) ? std::clamp(ratio, 0.0, 1.0) : 0.0});
            }
        }
    };

} // namespace xjw::task_runtime
