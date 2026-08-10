#include "concurrency/SafeWorkerGroup.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace
{

using xjw::common::concurrency::SafeWorkerGroup;
using xjw::common::concurrency::WorkerFailureState;
using xjw::common::concurrency::parallelForIndices;
using xjw::common::concurrency::runWorkerGroup;

TEST(SafeWorkerGroupTest, ParallelForIndicesVisitsEveryItemExactlyOnce)
{
    std::vector<std::atomic_int> visits(257);
    parallelForIndices(visits.size(), 8, [&](std::size_t index)
    {
        ++visits[index];
    });

    for (const std::atomic_int &visit : visits)
    {
        EXPECT_EQ(visit.load(), 1);
    }
}

TEST(SafeWorkerGroupTest, ParallelForIndicesSupportsSerialAndEmptyRanges)
{
    int sum = 0;
    parallelForIndices(6, 1, [&](std::size_t index)
    {
        sum += static_cast<int>(index);
    });
    parallelForIndices(0, 16, [&](std::size_t)
    {
        ADD_FAILURE() << "empty range must not invoke task";
    });

    EXPECT_EQ(sum, 15);
}

TEST(SafeWorkerGroupTest, WorkerExceptionIsRethrownAfterEveryWorkerJoins)
{
    std::atomic_int started{0};
    std::atomic_int finished{0};

    try
    {
        runWorkerGroup(4, [&started, &finished](std::size_t workerIndex,
                                                std::stop_token stopToken)
        {
            ++started;
            while (started.load() < 4 && !stopToken.stop_requested())
            {
                std::this_thread::yield();
            }
            if (workerIndex == 0)
            {
                if (stopToken.stop_requested())
                {
                    return;
                }
                throw std::runtime_error("worker sentinel");
            }
            while (!stopToken.stop_requested())
            {
                std::this_thread::yield();
            }
            ++finished;
        });
        FAIL() << "worker failure was not rethrown";
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_EQ(std::string(error.what()), "worker sentinel");
    }

    EXPECT_EQ(finished.load(), 3);
}

TEST(SafeWorkerGroupTest, PartialCreationFailureStopsAndJoinsCreatedWorkers)
{
    std::atomic_int started{0};
    std::atomic_int finished{0};
    std::atomic_int start_hook_calls{0};
    SafeWorkerGroup group(
        4,
        [&start_hook_calls](std::size_t workerIndex)
        {
            ++start_hook_calls;
            if (workerIndex == 2)
            {
                throw std::runtime_error("thread factory sentinel");
            }
        });

    for (int index = 0; index < 4; ++index)
    {
        const bool launched = group.launch([&](std::stop_token stopToken)
        {
            ++started;
            while (!stopToken.stop_requested())
            {
                std::this_thread::yield();
            }
            ++finished;
        });
        if (!launched)
        {
            break;
        }
    }

    EXPECT_EQ(group.launchedCount(), 2U);
    EXPECT_EQ(start_hook_calls.load(), 3);
    EXPECT_TRUE(group.stopRequested());
    EXPECT_FALSE(group.launch([] {}));
    try
    {
        group.waitAndRethrow();
        FAIL() << "thread creation failure was not rethrown";
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_EQ(std::string(error.what()), "thread factory sentinel");
    }
    EXPECT_EQ(started.load(), 2);
    EXPECT_EQ(finished.load(), 2);
}

TEST(SafeWorkerGroupTest, DestructorRequestsStopAndJoinsWorkers)
{
    std::atomic_bool started{false};
    std::atomic_bool finished{false};
    {
        SafeWorkerGroup group(1);
        ASSERT_TRUE(group.launch([&](std::stop_token stopToken)
        {
            started = true;
            while (!stopToken.stop_requested())
            {
                std::this_thread::yield();
            }
            finished = true;
        }));
        while (!started.load())
        {
            std::this_thread::yield();
        }
    }

    EXPECT_TRUE(finished.load());
}

TEST(SafeWorkerGroupTest, SharedFailureStatePreservesFirstException)
{
    WorkerFailureState failureState;
    failureState.capture(std::make_exception_ptr(std::runtime_error("first")));
    failureState.capture(std::make_exception_ptr(std::runtime_error("second")));

    EXPECT_TRUE(failureState.stopRequested());
    try
    {
        failureState.rethrowIfFailed();
        FAIL() << "captured failure was not rethrown";
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_EQ(std::string(error.what()), "first");
    }
}

} // namespace
