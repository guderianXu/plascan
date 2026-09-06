#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string_view>

namespace xjw::task_runtime
{

    enum class TaskControlDecision
    {
        Continue,
        Pause,
        Cancel
    };

    class TaskControlToken final
    {
    public:
        bool isCancellationRequested() const;
        bool isPauseRequested() const;
        TaskControlDecision pollAtSafePoint(std::string_view safePoint) const;

        // Compatibility helper for legacy executors. New scheduler executors should
        // return a Paused outcome so their worker/resource leases can be released.
        TaskControlDecision waitIfPaused(std::string_view safePoint) const;

    private:
        friend class TaskControlSource;

        void requestPause();
        void resume();
        void requestCancellation();

        std::atomic_bool _pauseRequested{false};
        std::atomic_bool _cancellationRequested{false};
        mutable std::mutex _mutex;
        mutable std::condition_variable _stateChanged;
    };

    class TaskControlSource final
    {
    public:
        explicit TaskControlSource(TaskControlToken& token);

        void requestPause() const;
        void resume() const;
        void requestCancellation() const;

    private:
        TaskControlToken* _token = nullptr;
    };

} // namespace xjw::task_runtime
