#include "FeaturePreparationQueue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{

    std::vector<xjw::matchphotos::FeaturePreparationRequest> makeRequests(int count)
    {
        std::vector<xjw::matchphotos::FeaturePreparationRequest> requests;
        requests.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index)
        {
            xjw::matchphotos::FeaturePreparationRequest request;
            request.index = index;
            request.imagePath = QStringLiteral("image_%1.png").arg(index);
            requests.push_back(std::move(request));
        }
        return requests;
    }

} // namespace

TEST(FeaturePreparationQueueTest, PreservesOrderAndBoundsBufferedImages)
{
    std::atomic_bool cancel{false};
    xjw::matchphotos::FeaturePreparationQueue queue(makeRequests(8),
                                                    2,
                                                    &cancel,
                                                    [](const xjw::matchphotos::FeaturePreparationRequest& request)
                                                    {
                                                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                                                        xjw::matchphotos::PreparedFeatureImage prepared;
                                                        prepared.index = request.index;
                                                        prepared.imagePath = request.imagePath;
                                                        return prepared;
                                                    });

    std::vector<int> consumed;
    xjw::matchphotos::PreparedFeatureImage prepared;
    while (queue.take(&prepared))
    {
        consumed.push_back(prepared.index);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    ASSERT_EQ(consumed.size(), 8U);
    for (int index = 0; index < 8; ++index)
    {
        EXPECT_EQ(consumed[static_cast<std::size_t>(index)], index);
    }
    EXPECT_LE(queue.peakBufferedCount(), 2);
    EXPECT_GE(queue.peakBufferedCount(), 1);
}

TEST(FeaturePreparationQueueTest, StopsPreparingAfterCancellation)
{
    std::atomic_bool cancel{false};
    std::atomic_int preparedCount{0};
    xjw::matchphotos::FeaturePreparationQueue queue(
        makeRequests(40),
        2,
        &cancel,
        [&preparedCount](const xjw::matchphotos::FeaturePreparationRequest& request)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ++preparedCount;
            xjw::matchphotos::PreparedFeatureImage prepared;
            prepared.index = request.index;
            prepared.imagePath = request.imagePath;
            return prepared;
        });

    xjw::matchphotos::PreparedFeatureImage first;
    ASSERT_TRUE(queue.take(&first));
    cancel.store(true);

    xjw::matchphotos::PreparedFeatureImage ignored;
    while (queue.take(&ignored))
    {
    }

    EXPECT_LT(preparedCount.load(), 40);
}

TEST(FeaturePreparationQueueTest, ConvertsPreparationExceptionToOrderedError)
{
    std::atomic_bool cancel{false};
    xjw::matchphotos::FeaturePreparationQueue queue(makeRequests(2),
                                                    1,
                                                    &cancel,
                                                    [](const xjw::matchphotos::FeaturePreparationRequest& request)
                                                    {
                                                        if (request.index == 0)
                                                        {
                                                            throw std::runtime_error("injected preparation failure");
                                                        }
                                                        xjw::matchphotos::PreparedFeatureImage prepared;
                                                        prepared.index = request.index;
                                                        return prepared;
                                                    });

    xjw::matchphotos::PreparedFeatureImage failed;
    ASSERT_TRUE(queue.take(&failed));
    EXPECT_EQ(failed.index, 0);
    EXPECT_EQ(failed.errorMessage, QStringLiteral("injected preparation failure"));

    xjw::matchphotos::PreparedFeatureImage succeeded;
    ASSERT_TRUE(queue.take(&succeeded));
    EXPECT_EQ(succeeded.index, 1);
}

TEST(FeaturePreparationQueueTest, ParallelWorkersStillDeliverRequestsInOrder)
{
    std::atomic_bool cancel{false};
    std::atomic_int started{0};
    std::atomic_int active{0};
    std::atomic_int peakActive{0};
    xjw::matchphotos::FeaturePreparationQueue queue(
        makeRequests(9),
        4,
        &cancel,
        [&](const xjw::matchphotos::FeaturePreparationRequest& request)
        {
            const int current = active.fetch_add(1) + 1;
            int peak = peakActive.load();
            while (peak < current && !peakActive.compare_exchange_weak(peak, current))
            {
            }
            started.fetch_add(1);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            while (started.load() < 3 && std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::yield();
            }
            active.fetch_sub(1);

            xjw::matchphotos::PreparedFeatureImage prepared;
            prepared.index = request.index;
            prepared.imagePath = request.imagePath;
            return prepared;
        },
        3);

    EXPECT_EQ(queue.workerCount(), 3);
    std::vector<int> consumed;
    xjw::matchphotos::PreparedFeatureImage prepared;
    while (queue.take(&prepared))
    {
        consumed.push_back(prepared.index);
    }

    ASSERT_EQ(consumed.size(), 9U);
    for (int index = 0; index < 9; ++index)
    {
        EXPECT_EQ(consumed[static_cast<std::size_t>(index)], index);
    }
    EXPECT_GE(peakActive.load(), 2);
    EXPECT_LE(queue.peakBufferedCount(), 4);
}

TEST(FeaturePreparationQueueTest, MultipleConsumersReceiveEveryPreparedImageOnce)
{
    constexpr int imageCount = 24;
    std::atomic_bool cancel{false};
    xjw::matchphotos::FeaturePreparationQueue queue(
        makeRequests(imageCount),
        4,
        &cancel,
        [](const xjw::matchphotos::FeaturePreparationRequest& request)
        {
            xjw::matchphotos::PreparedFeatureImage prepared;
            prepared.index = request.index;
            prepared.imagePath = request.imagePath;
            return prepared;
        },
        4);

    std::vector<int> seen(static_cast<std::size_t>(imageCount), 0);
    std::mutex seenMutex;
    std::vector<std::thread> consumers;
    for (int worker = 0; worker < 2; ++worker)
    {
        consumers.emplace_back(
            [&]()
            {
                xjw::matchphotos::PreparedFeatureImage prepared;
                while (queue.take(&prepared))
                {
                    std::lock_guard lock(seenMutex);
                    ++seen[static_cast<std::size_t>(prepared.index)];
                }
            });
    }
    for (std::thread& consumer : consumers)
    {
        consumer.join();
    }

    for (const int count : seen)
    {
        EXPECT_EQ(count, 1);
    }
}
