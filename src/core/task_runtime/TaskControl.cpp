#include "TaskControl.h"

namespace xjw::task_runtime
{

    bool TaskControlToken::isCancellationRequested() const
    {
        return _cancellationRequested.load(std::memory_order_acquire);
    }

    bool TaskControlToken::isPauseRequested() const
    {
        return _pauseRequested.load(std::memory_order_acquire);
    }

    TaskControlDecision TaskControlToken::pollAtSafePoint(std::string_view) const
    {
        if (isCancellationRequested())
        {
            return TaskControlDecision::Cancel;
        }
        if (isPauseRequested())
        {
            return TaskControlDecision::Pause;
        }
        return TaskControlDecision::Continue;
    }

    TaskControlDecision TaskControlToken::waitIfPaused(std::string_view safePoint) const
    {
        TaskControlDecision decision = pollAtSafePoint(safePoint);
        if (decision != TaskControlDecision::Pause)
        {
            return decision;
        }

        std::unique_lock<std::mutex> lock(_mutex);
        _stateChanged.wait(lock,
                           [this] {
                               return !_pauseRequested.load(std::memory_order_acquire) ||
                                      _cancellationRequested.load(std::memory_order_acquire);
                           });
        return isCancellationRequested() ? TaskControlDecision::Cancel : TaskControlDecision::Continue;
    }

    void TaskControlToken::requestPause()
    {
        _pauseRequested.store(true, std::memory_order_release);
    }

    void TaskControlToken::resume()
    {
        _pauseRequested.store(false, std::memory_order_release);
        _stateChanged.notify_all();
    }

    void TaskControlToken::requestCancellation()
    {
        _cancellationRequested.store(true, std::memory_order_release);
        _stateChanged.notify_all();
    }

    TaskControlSource::TaskControlSource(TaskControlToken& token) : _token(&token)
    {
    }

    void TaskControlSource::requestPause() const
    {
        _token->requestPause();
    }

    void TaskControlSource::resume() const
    {
        _token->resume();
    }

    void TaskControlSource::requestCancellation() const
    {
        _token->requestCancellation();
    }

} // namespace xjw::task_runtime
