#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace xjw::common::concurrency
{

class WorkerFailureState
{
public:
    WorkerFailureState() = default;

    WorkerFailureState(const WorkerFailureState &) = delete;
    WorkerFailureState &operator=(const WorkerFailureState &) = delete;

    std::stop_token stopToken() const noexcept
    {
        return _stopSource.get_token();
    }

    bool stopRequested() const noexcept
    {
        return _stopSource.stop_requested();
    }

    void requestStop() noexcept
    {
        _stopSource.request_stop();
    }

    void capture(std::exception_ptr failure) noexcept
    {
        if (!failure)
        {
            return;
        }

        lockFailure();
        if (!_failure)
        {
            _failure = std::move(failure);
        }
        unlockFailure();
        _stopSource.request_stop();
    }

    void captureCurrentException() noexcept
    {
        capture(std::current_exception());
    }

    bool hasFailure() const noexcept
    {
        lockFailure();
        const bool result = static_cast<bool>(_failure);
        unlockFailure();
        return result;
    }

    void rethrowIfFailed() const
    {
        std::exception_ptr failure;
        lockFailure();
        failure = _failure;
        unlockFailure();
        if (failure)
        {
            std::rethrow_exception(failure);
        }
    }

private:
    void lockFailure() const noexcept
    {
        while (_failureLock.test_and_set(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }

    void unlockFailure() const noexcept
    {
        _failureLock.clear(std::memory_order_release);
    }

    std::stop_source _stopSource;
    mutable std::atomic_flag _failureLock = ATOMIC_FLAG_INIT;
    std::exception_ptr _failure;
};

// The group must be owned by the thread that launches and joins its workers.
// Its destructor requests cooperative cancellation and joins every worker.
class SafeWorkerGroup
{
public:
    using StartHook = std::function<void(std::size_t)>;

    explicit SafeWorkerGroup(std::size_t expected_worker_count = 0,
                             StartHook start_hook = {})
        : _startHook(std::move(start_hook))
    {
        _workers.reserve(expected_worker_count);
    }

    ~SafeWorkerGroup()
    {
        requestStop();
        joinNoThrow();
    }

    SafeWorkerGroup(const SafeWorkerGroup &) = delete;
    SafeWorkerGroup &operator=(const SafeWorkerGroup &) = delete;
    SafeWorkerGroup(SafeWorkerGroup &&) = delete;
    SafeWorkerGroup &operator=(SafeWorkerGroup &&) = delete;

    template <typename Fn>
    bool launch(Fn &&worker)
    {
        if (_failureState.stopRequested())
        {
            return false;
        }

        const std::size_t worker_index = _nextWorkerIndex;
        try
        {
            if (_startHook)
            {
                _startHook(worker_index);
            }

            _workers.emplace_back(
                [this, task = std::forward<Fn>(worker)](std::stop_token) mutable
                {
                    try
                    {
                        if constexpr (std::is_invocable_v<decltype(task) &, std::stop_token>)
                        {
                            std::invoke(task, _failureState.stopToken());
                        }
                        else
                        {
                            static_assert(std::is_invocable_v<decltype(task) &>,
                                          "worker must be invocable with zero arguments or a stop_token");
                            std::invoke(task);
                        }
                    }
                    catch (...)
                    {
                        try
                        {
                            _failureState.captureCurrentException();
                        }
                        catch (...)
                        {
                            _failureState.requestStop();
                        }
                    }
                });
            ++_nextWorkerIndex;
            return true;
        }
        catch (...)
        {
            try
            {
                _failureState.captureCurrentException();
            }
            catch (...)
            {
                _failureState.requestStop();
                throw;
            }
            return false;
        }
    }

    void requestStop() noexcept
    {
        _failureState.requestStop();
    }

    bool stopRequested() const noexcept
    {
        return _failureState.stopRequested();
    }

    std::stop_token stopToken() const noexcept
    {
        return _failureState.stopToken();
    }

    std::size_t launchedCount() const noexcept
    {
        return _workers.size();
    }

    void wait()
    {
        for (std::jthread &worker : _workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        _workers.clear();
    }

    void waitAndRethrow()
    {
        wait();
        _failureState.rethrowIfFailed();
    }

private:
    void joinNoThrow() noexcept
    {
        for (std::jthread &worker : _workers)
        {
            if (!worker.joinable())
            {
                continue;
            }
            try
            {
                worker.join();
            }
            catch (...)
            {
                // SafeWorkerGroup is owned by the thread that launches workers.
                // Joining from that owner cannot self-join; this guard keeps the
                // destructor non-throwing if the runtime reports another error.
            }
        }
    }

    WorkerFailureState _failureState;
    StartHook _startHook;
    std::vector<std::jthread> _workers;
    std::size_t _nextWorkerIndex = 0;
};

// Copies the callable once per worker. References captured by that callable
// still identify shared state and must be synchronized by the caller.
template <typename Fn>
void runWorkerGroup(std::size_t worker_count,
                    Fn &&worker,
                    SafeWorkerGroup::StartHook start_hook = {})
{
    using Worker = std::decay_t<Fn>;
    static_assert(std::is_copy_constructible_v<Worker>,
                  "runWorkerGroup copies the callable once per worker");

    struct WorkerInvocation
    {
        Worker task;
        std::size_t index = 0;

        void operator()(std::stop_token stop_token)
        {
            if constexpr (std::is_invocable_v<Worker &, std::size_t, std::stop_token>)
            {
                std::invoke(task, index, stop_token);
            }
            else if constexpr (std::is_invocable_v<Worker &, std::stop_token>)
            {
                std::invoke(task, stop_token);
            }
            else if constexpr (std::is_invocable_v<Worker &, std::size_t>)
            {
                std::invoke(task, index);
            }
            else
            {
                static_assert(std::is_invocable_v<Worker &>,
                              "worker has an unsupported signature");
                std::invoke(task);
            }
        }
    };

    // Prepare every callable copy before the first thread starts. A throwing
    // copy therefore cannot race a worker failure or strand launched threads.
    std::vector<WorkerInvocation> invocations;
    invocations.reserve(worker_count);
    for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index)
    {
        invocations.push_back(WorkerInvocation{worker, worker_index});
    }

    SafeWorkerGroup group(worker_count, std::move(start_hook));
    for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index)
    {
        if (group.stopRequested())
        {
            break;
        }
        if (!group.launch(std::move(invocations[worker_index])))
        {
            break;
        }
    }
    group.waitAndRethrow();
}

/**
 * @brief 使用动态任务分配并行遍历 `[0, item_count)`。
 *
 * 每个索引只会执行一次。调用方应让不同索引写入互不重叠的结果槽位，并在
 * 本函数返回后统一提交对共享容器的结构性修改。
 */
template <typename Fn>
void parallelForIndices(std::size_t item_count,
                        std::size_t maximum_worker_count,
                        Fn &&task)
{
    if (item_count == 0)
    {
        return;
    }

    const std::size_t worker_count = std::min(
        item_count,
        std::max<std::size_t>(1, maximum_worker_count));
    if (worker_count == 1)
    {
        for (std::size_t index = 0; index < item_count; ++index)
        {
            std::invoke(task, index);
        }
        return;
    }

    std::atomic<std::size_t> next_index{0};
    runWorkerGroup(
        worker_count,
        [&](std::stop_token stop_token)
        {
            while (!stop_token.stop_requested())
            {
                const std::size_t index = next_index.fetch_add(
                    1, std::memory_order_relaxed);
                if (index >= item_count)
                {
                    break;
                }
                std::invoke(task, index);
            }
        });
}

} // namespace xjw::common::concurrency
