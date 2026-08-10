#include "DepthMemoryPolicy.h"
#include "MvsImageCache.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

using namespace std::chrono_literals;
using xjw::mvs::DepthConsistencyMemoryEstimate;
using xjw::mvs::MvsImageCache;
using xjw::mvs::MvsImageCacheStrategy;
using xjw::mvs::MvsImageFrame;
using xjw::mvs::MvsImageMemoryFrame;

MvsImageCache::Loader deterministicLoader(std::atomic_int *loadCount)
{
    return [loadCount](int frameIndex,
                       const std::atomic_bool *,
                       MvsImageFrame *frame,
                       std::string *)
    {
        ++(*loadCount);
        frame->gray = cv::Mat(3, 4, CV_8UC1, cv::Scalar(frameIndex + 1));
        frame->preparedGray = frame->gray;
        frame->validMask = cv::Mat(3, 4, CV_8UC1, cv::Scalar(255));
        return true;
    };
}

TEST(MvsImageMemoryPolicyTest, SharedAllocationIsCountedOnlyOnce)
{
    const std::vector<xjw::mvs::MvsMemoryAllocationEstimate> allocations{
        {10, 4096},
        {10, 4096},
        {11, 2048}};

    EXPECT_EQ(xjw::mvs::estimateUniqueMvsAllocationBytes(allocations), 6144U);

    MvsImageFrame frame;
    frame.gray = cv::Mat(4, 4, CV_8UC1, cv::Scalar(1));
    frame.preparedGray = frame.gray;
    frame.validMask = cv::Mat(4, 4, CV_8UC1, cv::Scalar(255));
    EXPECT_EQ(frame.residentBytes(), 32U);
}

TEST(MvsImageMemoryPolicyTest, FiveHundredViewsSelectBoundedWorkingSet)
{
    const std::vector<MvsImageMemoryFrame> frames(
        500, MvsImageMemoryFrame{4000, 3000, true});
    DepthConsistencyMemoryEstimate depth;
    depth.largestFramePixels = 12'000'000;
    depth.residentFrameBytes = 156'000'000'000ULL;
    depth.consistencySnapshotBytes = 24'000'000'000ULL;
    depth.retainedEvidenceBytes = 84'000'000'000ULL;
    depth.transientFrameBytes = 1'488'000'000ULL;

    const auto decision = xjw::mvs::decideMvsPipelineMemoryPolicy(
        frames,
        depth,
        true,
        8,
        4,
        512ULL * 1024ULL * 1024ULL,
        4ULL * depth.transientFrameBytes,
        32ULL * 1024ULL * 1024ULL * 1024ULL,
        30ULL * 1024ULL * 1024ULL * 1024ULL,
        0.60f,
        2ULL * 1024ULL * 1024ULL * 1024ULL);

    EXPECT_EQ(decision.imageStrategy, MvsImageCacheStrategy::Bounded);
    EXPECT_EQ(decision.minimumImageCacheCapacity, 36U);
    EXPECT_EQ(decision.imageCacheCapacity, 36U);
    EXPECT_LT(decision.imageCacheCapacity, frames.size());
    EXPECT_FALSE(decision.retainAllDepthFrames);
}

TEST(MvsImageMemoryPolicyTest, RejectsBudgetBelowMinimumLeasedWorkingSet)
{
    const std::vector<MvsImageMemoryFrame> frames(
        20, MvsImageMemoryFrame{4096, 4096, false});
    DepthConsistencyMemoryEstimate depth;
    depth.largestFramePixels = 4096ULL * 4096ULL;
    depth.transientFrameBytes = 512ULL * 1024ULL * 1024ULL;

    const auto decision = xjw::mvs::decideMvsPipelineMemoryPolicy(
        frames,
        depth,
        true,
        4,
        2,
        256ULL * 1024ULL * 1024ULL,
        1024ULL * 1024ULL * 1024ULL,
        2ULL * 1024ULL * 1024ULL * 1024ULL,
        1024ULL * 1024ULL * 1024ULL,
        0.60f,
        512ULL * 1024ULL * 1024ULL);

    EXPECT_EQ(decision.imageStrategy, MvsImageCacheStrategy::Insufficient);
    EXPECT_LT(decision.imageCacheCapacity, decision.minimumImageCacheCapacity);
    EXPECT_GT(decision.requiredBytes, decision.availableBytes);
}

TEST(MvsImageMemoryPolicyTest, ForcedResidentDepthRejectsStreamingOnlyBudget)
{
    const std::vector<MvsImageMemoryFrame> frames(
        50, MvsImageMemoryFrame{4000, 3000, true});
    DepthConsistencyMemoryEstimate depth;
    depth.largestFramePixels = 12'000'000;
    depth.residentFrameBytes = 10ULL * 1024ULL * 1024ULL * 1024ULL;
    depth.consistencySnapshotBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    depth.retainedEvidenceBytes = 3ULL * 1024ULL * 1024ULL * 1024ULL;
    depth.transientFrameBytes = 512ULL * 1024ULL * 1024ULL;

    const auto decision = xjw::mvs::decideMvsPipelineMemoryPolicy(
        frames,
        depth,
        false,
        4,
        2,
        256ULL * 1024ULL * 1024ULL,
        512ULL * 1024ULL * 1024ULL,
        8ULL * 1024ULL * 1024ULL * 1024ULL,
        7ULL * 1024ULL * 1024ULL * 1024ULL,
        0.60f,
        1024ULL * 1024ULL * 1024ULL);

    EXPECT_EQ(decision.imageStrategy, MvsImageCacheStrategy::Insufficient);
    EXPECT_TRUE(decision.retainAllDepthFrames);
    EXPECT_EQ(decision.requiredBytes,
              decision.estimate.boundedImageBytes
                  + decision.estimate.depthResidentBytes
                  + decision.estimate.saveQueueBytes
                  + decision.estimate.backendStagingBytes);
    EXPECT_GT(decision.requiredBytes, decision.availableBytes);
}

TEST(MvsImageMemoryPolicyTest, VisibilityPeakIsCommonToEagerAndBoundedPlans)
{
    const std::vector<MvsImageMemoryFrame> frames(
        64, MvsImageMemoryFrame{640, 480, true});
    DepthConsistencyMemoryEstimate depth;
    depth.largestFramePixels = 640U * 480U;
    depth.residentFrameBytes = 64ULL * 1024ULL * 1024ULL;
    depth.transientFrameBytes = 8ULL * 1024ULL * 1024ULL;
    const auto visibility = xjw::mvs::estimateMvsVisibilityGraphMemory(
        frames.size(), 1000, 32, 16, 4);

    const auto decision = xjw::mvs::decideMvsPipelineMemoryPolicy(
        frames,
        depth,
        true,
        4,
        2,
        3ULL * 1024ULL * 1024ULL,
        5ULL * 1024ULL * 1024ULL,
        0,
        0,
        0.60f,
        0,
        visibility);

    EXPECT_EQ(decision.estimate.visibility.totalBytes, visibility.totalBytes);
    EXPECT_EQ(
        decision.estimate.eagerRequiredBytes,
        decision.estimate.eagerImageBytes
            + decision.estimate.depthResidentBytes
            + decision.estimate.saveQueueBytes
            + decision.estimate.backendStagingBytes
            + visibility.totalBytes);
    EXPECT_EQ(
        decision.estimate.boundedRequiredBytes,
        decision.estimate.boundedImageBytes
            + decision.estimate.depthStreamingBytes
            + decision.estimate.saveQueueBytes
            + decision.estimate.backendStagingBytes
            + visibility.totalBytes);
}

TEST(MvsImageCacheTest, MemoryDecisionDoesNotInvokePixelDecoder)
{
    std::atomic_int loadCount{0};
    const std::vector<MvsImageMemoryFrame> frames(
        3, MvsImageMemoryFrame{640, 480, true});
    const auto decision = xjw::mvs::decideMvsPipelineMemoryPolicy(
        frames,
        {},
        true,
        1,
        1,
        0,
        0,
        0,
        0,
        0.60f,
        0);
    MvsImageCache cache(
        frames.size(), decision.imageCacheCapacity, deterministicLoader(&loadCount));

    EXPECT_EQ(loadCount.load(), 0);
    std::atomic_bool cancelled{false};
    auto lease = cache.acquire(0, &cancelled);
    ASSERT_TRUE(lease);
    EXPECT_EQ(loadCount.load(), 1);
}

TEST(MvsImageCacheTest, EagerAndBoundedProvidersReturnIdenticalPixels)
{
    std::atomic_int eagerLoads{0};
    std::atomic_int boundedLoads{0};
    MvsImageCache eager(3, 3, deterministicLoader(&eagerLoads));
    MvsImageCache bounded(3, 1, deterministicLoader(&boundedLoads));
    std::atomic_bool cancelled{false};
    ASSERT_TRUE(eager.preloadAll(2, &cancelled));

    for (int index = 0; index < 3; ++index)
    {
        auto eagerLease = eager.acquire(index, &cancelled);
        auto boundedLease = bounded.acquire(index, &cancelled);
        ASSERT_TRUE(eagerLease);
        ASSERT_TRUE(boundedLease);
        EXPECT_EQ(cv::countNonZero(
                      eagerLease->gray != boundedLease->gray),
                  0);
    }
}

TEST(MvsImageCacheTest, LeaseCanOutliveCacheOwner)
{
    std::atomic_int loadCount{0};
    MvsImageCache::ImageLease lease;
    {
        auto cache = std::make_unique<MvsImageCache>(
            1, 1, deterministicLoader(&loadCount));
        std::atomic_bool cancelled{false};
        lease = cache->acquire(0, &cancelled);
        ASSERT_TRUE(lease);
        cache.reset();
    }

    ASSERT_TRUE(lease);
    EXPECT_EQ(lease->gray.at<std::uint8_t>(0, 0), 1);
    lease.reset();
    EXPECT_FALSE(lease);
}

TEST(MvsImageCacheTest, LoaderExceptionReleasesReservedCapacity)
{
    std::atomic_int loadCount{0};
    MvsImageCache cache(
        2,
        1,
        [&loadCount](int frameIndex,
                     const std::atomic_bool *,
                     MvsImageFrame *frame,
                     std::string *)
        {
            if (++loadCount == 1)
            {
                throw std::runtime_error("loader sentinel");
            }
            frame->gray = cv::Mat(2, 2, CV_8UC1, cv::Scalar(frameIndex + 1));
            frame->preparedGray = frame->gray;
            return true;
        });
    std::atomic_bool cancelled{false};

    std::string error;
    auto failed = cache.acquire(0, &cancelled, &error);
    EXPECT_FALSE(failed);
    EXPECT_NE(error.find("loader sentinel"), std::string::npos);
    EXPECT_EQ(cache.residentCount(), 0U);

    auto recovered = cache.acquire(1, &cancelled, &error);
    ASSERT_TRUE(recovered);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(recovered->gray.at<std::uint8_t>(0, 0), 2);
}

TEST(MvsImageCacheTest, ConcurrentAcquireUsesSingleFlightLoad)
{
    std::atomic_int loadCount{0};
    std::mutex loaderMutex;
    std::condition_variable loaderCondition;
    bool loaderStarted = false;
    bool releaseLoader = false;
    MvsImageCache cache(
        1,
        1,
        [&](int,
            const std::atomic_bool *,
            MvsImageFrame *frame,
            std::string *)
        {
            ++loadCount;
            std::unique_lock<std::mutex> lock(loaderMutex);
            loaderStarted = true;
            loaderCondition.notify_all();
            loaderCondition.wait(lock, [&releaseLoader]()
            {
                return releaseLoader;
            });
            frame->gray = cv::Mat(2, 2, CV_8UC1, cv::Scalar(7));
            frame->preparedGray = frame->gray;
            return true;
        });
    std::atomic_bool cancelled{false};
    std::vector<std::future<int>> futures;
    for (int index = 0; index < 8; ++index)
    {
        futures.push_back(std::async(std::launch::async, [&cache, &cancelled]()
        {
            auto lease = cache.acquire(0, &cancelled);
            return lease ? lease->gray.at<std::uint8_t>(0, 0) : -1;
        }));
    }

    bool startedInTime = false;
    {
        std::unique_lock<std::mutex> lock(loaderMutex);
        startedInTime = loaderCondition.wait_for(lock, 1s, [&loaderStarted]()
        {
            return loaderStarted;
        });
        releaseLoader = true;
    }
    loaderCondition.notify_all();
    ASSERT_TRUE(startedInTime);
    for (auto &future : futures)
    {
        EXPECT_EQ(future.get(), 7);
    }
    EXPECT_EQ(loadCount.load(), 1);
    EXPECT_EQ(cache.loadAttemptCount(0), 1U);
}

TEST(MvsImageCacheTest, PinnedLeaseCannotBeEvicted)
{
    std::atomic_int loadCount{0};
    MvsImageCache cache(3, 2, deterministicLoader(&loadCount));
    std::atomic_bool cancelled{false};
    auto pinned = cache.acquire(0, &cancelled);
    ASSERT_TRUE(pinned);
    {
        auto temporary = cache.acquire(1, &cancelled);
        ASSERT_TRUE(temporary);
    }
    {
        auto replacement = cache.acquire(2, &cancelled);
        ASSERT_TRUE(replacement);
    }

    EXPECT_TRUE(cache.contains(0));
    EXPECT_FALSE(cache.contains(1));
    EXPECT_TRUE(cache.contains(2));
}

TEST(MvsImageCacheTest, CancelledCapacityWaitCompletesWithinOneSecond)
{
    std::atomic_int loadCount{0};
    MvsImageCache cache(2, 1, deterministicLoader(&loadCount));
    std::atomic_bool cancelled{false};
    auto pinned = cache.acquire(0, &cancelled);
    ASSERT_TRUE(pinned);

    const auto started = std::chrono::steady_clock::now();
    auto waiter = std::async(std::launch::async, [&cache, &cancelled]()
    {
        std::string error;
        auto lease = cache.acquire(1, &cancelled, &error);
        return std::make_pair(static_cast<bool>(lease), error);
    });
    std::this_thread::sleep_for(100ms);
    cancelled.store(true, std::memory_order_relaxed);
    ASSERT_EQ(waiter.wait_for(800ms), std::future_status::ready);
    const auto [acquired, error] = waiter.get();

    EXPECT_FALSE(acquired);
    EXPECT_NE(error.find("cancelled"), std::string::npos);
    EXPECT_LT(std::chrono::steady_clock::now() - started, 1s);
    EXPECT_EQ(cache.loadAttemptCount(1), 0U);
}

} // namespace
