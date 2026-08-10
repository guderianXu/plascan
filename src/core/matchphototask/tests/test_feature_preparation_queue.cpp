#include "FeaturePreparationQueue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
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
    xjw::matchphotos::FeaturePreparationQueue queue(
        makeRequests(8),
        2,
        &cancel,
        [](const xjw::matchphotos::FeaturePreparationRequest &request)
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
        [&preparedCount](const xjw::matchphotos::FeaturePreparationRequest &request)
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
    xjw::matchphotos::FeaturePreparationQueue queue(
        makeRequests(2),
        1,
        &cancel,
        [](const xjw::matchphotos::FeaturePreparationRequest &request)
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
    EXPECT_EQ(failed.errorMessage,
              QStringLiteral("injected preparation failure"));

    xjw::matchphotos::PreparedFeatureImage succeeded;
    ASSERT_TRUE(queue.take(&succeeded));
    EXPECT_EQ(succeeded.index, 1);
}
