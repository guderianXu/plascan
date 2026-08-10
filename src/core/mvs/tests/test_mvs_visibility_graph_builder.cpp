#include "MvsVisibilityGraphBuilder.h"
#include "DepthMemoryPolicy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <limits>
#include <numeric>
#include <vector>

namespace
{

using namespace std::chrono_literals;

using xjw::mvs::MvsVisibilityCandidatePair;
using xjw::mvs::MvsVisibilityGraph;
using xjw::mvs::MvsVisibilityGraphBuildOptions;
using xjw::mvs::MvsVisibilityGraphBuilder;
using xjw::mvs::MvsVisibilityNeighbor;

std::vector<std::vector<std::size_t>> fullyVisible(std::size_t viewCount,
                                                   std::size_t pointCount)
{
    std::vector<std::vector<std::size_t>> visible(viewCount);
    for (auto &indices : visible)
    {
        indices.resize(pointCount);
        std::iota(indices.begin(), indices.end(), std::size_t{0});
    }
    return visible;
}

const MvsVisibilityNeighbor *findNeighbor(const MvsVisibilityGraph &graph,
                                          int referenceViewIndex,
                                          int peerViewIndex)
{
    const auto &neighbors = graph.neighborsByView[static_cast<std::size_t>(referenceViewIndex)];
    const auto it = std::find_if(neighbors.begin(), neighbors.end(), [peerViewIndex](const auto &neighbor)
    {
        return neighbor.viewIndex == peerViewIndex;
    });
    return it == neighbors.end() ? nullptr : &*it;
}

void expectSymmetric(const MvsVisibilityGraph &graph)
{
    for (std::size_t referenceViewIndex = 0;
         referenceViewIndex < graph.neighborsByView.size();
         ++referenceViewIndex)
    {
        for (const MvsVisibilityNeighbor &neighbor : graph.neighborsByView[referenceViewIndex])
        {
            const MvsVisibilityNeighbor *reverse = findNeighbor(
                graph, neighbor.viewIndex, static_cast<int>(referenceViewIndex));
            ASSERT_NE(reverse, nullptr);
            EXPECT_EQ(reverse->sharedTrackCount, neighbor.sharedTrackCount);
        }
    }
}

TEST(MvsVisibilityGraphBuilderTest, ThirtyThreeFullyCoVisibleViewsKeepEveryReference)
{
    constexpr std::size_t viewCount = 33;
    constexpr std::size_t pointCount = 96;
    MvsVisibilityGraphBuildOptions options;
    options.workerCount = 4;

    const MvsVisibilityGraph graph = MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
        pointCount, fullyVisible(viewCount, pointCount), options);

    ASSERT_EQ(graph.neighborsByView.size(), viewCount);
    for (const auto &neighbors : graph.neighborsByView)
    {
        ASSERT_FALSE(neighbors.empty());
        for (const MvsVisibilityNeighbor &neighbor : neighbors)
        {
            EXPECT_EQ(neighbor.sharedTrackCount, static_cast<int>(pointCount));
        }
    }
    expectSymmetric(graph);
}

TEST(MvsVisibilityGraphBuilderTest, SixtyFourFullyCoVisibleViewsKeepEveryReference)
{
    constexpr std::size_t viewCount = 64;
    constexpr std::size_t pointCount = 80;
    MvsVisibilityGraphBuildOptions options;
    options.workerCount = 8;

    const MvsVisibilityGraph graph = MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
        pointCount, fullyVisible(viewCount, pointCount), options);

    for (const auto &neighbors : graph.neighborsByView)
    {
        ASSERT_FALSE(neighbors.empty());
        for (const MvsVisibilityNeighbor &neighbor : neighbors)
        {
            EXPECT_EQ(neighbor.sharedTrackCount, static_cast<int>(pointCount));
        }
    }
    expectSymmetric(graph);
}

TEST(MvsVisibilityGraphBuilderTest, CountsAreExactAndAdjacencyIsSymmetric)
{
    std::vector<std::vector<std::size_t>> visible{
        {0, 1, 2, 4},
        {1, 2, 3, 4},
        {2, 4, 5},
        {0, 5}
    };
    const MvsVisibilityGraph graph = MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
        6, std::move(visible));

    ASSERT_NE(findNeighbor(graph, 0, 1), nullptr);
    EXPECT_EQ(findNeighbor(graph, 0, 1)->sharedTrackCount, 3);
    ASSERT_NE(findNeighbor(graph, 0, 2), nullptr);
    EXPECT_EQ(findNeighbor(graph, 0, 2)->sharedTrackCount, 2);
    ASSERT_NE(findNeighbor(graph, 0, 3), nullptr);
    EXPECT_EQ(findNeighbor(graph, 0, 3)->sharedTrackCount, 1);
    expectSymmetric(graph);
}

TEST(MvsVisibilityGraphBuilderTest, RequiredVerifiedPairRemainsCandidateWithoutSharedTrack)
{
    MvsVisibilityGraphBuildOptions options;
    options.requiredPairs.push_back(MvsVisibilityCandidatePair{0, 2});
    const MvsVisibilityGraph graph = MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
        2,
        {{0}, {0}, {1}},
        options);

    const MvsVisibilityNeighbor *required = findNeighbor(graph, 0, 2);
    ASSERT_NE(required, nullptr);
    EXPECT_EQ(required->sharedTrackCount, 0);
    expectSymmetric(graph);
}

TEST(MvsVisibilityGraphBuilderTest, OneAndEightWorkersProduceIdenticalGraph)
{
    constexpr std::size_t viewCount = 64;
    constexpr std::size_t pointCount = 257;
    std::vector<std::vector<std::size_t>> visible(viewCount);
    for (std::size_t viewIndex = 0; viewIndex < viewCount; ++viewIndex)
    {
        for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
        {
            if ((viewIndex * 11U + pointIndex * 7U) % 9U != 0U)
            {
                visible[viewIndex].push_back(pointIndex);
            }
        }
    }

    MvsVisibilityGraphBuildOptions serialOptions;
    serialOptions.workerCount = 1;
    MvsVisibilityGraphBuildOptions parallelOptions = serialOptions;
    parallelOptions.workerCount = 8;
    const MvsVisibilityGraph serial = MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
        pointCount, visible, serialOptions);
    const MvsVisibilityGraph parallel = MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
        pointCount, std::move(visible), parallelOptions);

    EXPECT_EQ(parallel.visiblePointIndicesByView, serial.visiblePointIndicesByView);
    EXPECT_EQ(parallel.visibilityBits, serial.visibilityBits);
    EXPECT_EQ(parallel.neighborsByView, serial.neighborsByView);
    EXPECT_EQ(parallel.statistics.candidatePairCount, serial.statistics.candidatePairCount);
}

TEST(MvsVisibilityGraphBuilderTest, FiveThousandViewsUseSparsePairCounters)
{
    constexpr std::size_t viewCount = 5000;
    constexpr std::size_t pointCount = 48;
    MvsVisibilityGraphBuildOptions options;
    options.workerCount = 8;
    options.maximumSampledPeersPerView = 16;

    const MvsVisibilityGraph graph = MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
        pointCount, fullyVisible(viewCount, pointCount), options);

    const std::size_t denseCounterEntries = viewCount * viewCount;
    EXPECT_LE(graph.statistics.pairCounterEntryCount,
              viewCount * static_cast<std::size_t>(options.maximumSampledPeersPerView));
    EXPECT_LT(graph.statistics.pairCounterEntryCount, denseCounterEntries / 100U);
    EXPECT_LT(graph.statistics.estimatedPairStorageBytes,
              denseCounterEntries * sizeof(int));
    for (const auto &neighbors : graph.neighborsByView)
    {
        EXPECT_FALSE(neighbors.empty());
    }
    expectSymmetric(graph);
}

TEST(MvsVisibilityGraphBuilderTest, GlobalLargeViewSetNeverExpandsSmallPointSetsQuadratically)
{
    constexpr std::size_t viewCount = 64;
    constexpr std::size_t pointCount = 64;
    std::vector<std::vector<std::size_t>> visible(viewCount);
    for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
    {
        for (std::size_t offset = 0; offset < 32; ++offset)
        {
            visible[(pointIndex + offset) % viewCount].push_back(pointIndex);
        }
    }

    MvsVisibilityGraphBuildOptions options;
    options.fullPairVisibilityLimit = 1000;
    options.maximumSampledPeersPerView = 4;
    options.requiredPairs.push_back({0, 32});
    const MvsVisibilityGraph graph = MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
        pointCount, std::move(visible), options);

    const std::size_t strictPairUpperBound =
        viewCount * static_cast<std::size_t>(options.maximumSampledPeersPerView)
        + options.requiredPairs.size();
    EXPECT_LE(graph.statistics.candidatePairCount, strictPairUpperBound);
    ASSERT_NE(findNeighbor(graph, 0, 32), nullptr);
    EXPECT_EQ(findNeighbor(graph, 0, 32)->sharedTrackCount, 0);
}

TEST(MvsVisibilityGraphBuilderTest, OrbitalShortlistCoversLeftRightAndSafeAngles)
{
    constexpr int viewCount = 64;
    constexpr double twoPi = 6.28318530717958647692;
    std::vector<xjw::mvs::CameraView> views(static_cast<std::size_t>(viewCount));
    for (int viewIndex = 0; viewIndex < viewCount; ++viewIndex)
    {
        const double angle = twoPi * static_cast<double>(viewIndex)
            / static_cast<double>(viewCount);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const std::array<double, 9> cameraToWorld{
            sine, 0.0, -cosine,
            -cosine, 0.0, -sine,
            0.0, 1.0, 0.0};
        views[static_cast<std::size_t>(viewIndex)].camera.setPose(
            cameraToWorld, {10.0 * cosine, 10.0 * sine, 0.0});
        views[static_cast<std::size_t>(viewIndex)].camera.setIntrinsics(
            500.0, 500.0, 500.0, 500.0);
        views[static_cast<std::size_t>(viewIndex)].imageWidth = 1000;
        views[static_cast<std::size_t>(viewIndex)].imageHeight = 1000;
    }
    xjw::mvs::SparseCloud sparse;
    sparse.minPt = {-1.0f, -1.0f, -1.0f};
    sparse.maxPt = {1.0f, 1.0f, 1.0f};
    sparse.points.assign(128, {0.0f, 0.0f, 0.0f});

    const auto firstShortlist = MvsVisibilityGraphBuilder::buildGeometryPeerShortlist(
        views, sparse, 8);
    const auto secondShortlist = MvsVisibilityGraphBuilder::buildGeometryPeerShortlist(
        views, sparse, 8);
    EXPECT_EQ(firstShortlist, secondShortlist);

    MvsVisibilityGraphBuildOptions options;
    options.maximumSampledPeersPerView = 16;
    const MvsVisibilityGraph graph = MvsVisibilityGraphBuilder::build(
        views, sparse, options);

    for (int viewIndex = 0; viewIndex < viewCount; ++viewIndex)
    {
        bool hasLeftSector = false;
        bool hasRightSector = false;
        bool hasSafeLeftAngle = false;
        bool hasSafeRightAngle = false;
        for (const MvsVisibilityNeighbor &neighbor :
             graph.neighborsByView[static_cast<std::size_t>(viewIndex)])
        {
            int signedStep = (neighbor.viewIndex - viewIndex + viewCount) % viewCount;
            if (signedStep > viewCount / 2)
            {
                signedStep -= viewCount;
            }
            const double angleDegrees = std::abs(signedStep) * 360.0 / viewCount;
            hasLeftSector = hasLeftSector || signedStep < 0;
            hasRightSector = hasRightSector || signedStep > 0;
            hasSafeLeftAngle = hasSafeLeftAngle
                || (signedStep < 0 && angleDegrees >= 20.0 && angleDegrees <= 70.0);
            hasSafeRightAngle = hasSafeRightAngle
                || (signedStep > 0 && angleDegrees >= 20.0 && angleDegrees <= 70.0);
        }
        EXPECT_TRUE(hasLeftSector) << "view=" << viewIndex;
        EXPECT_TRUE(hasRightSector) << "view=" << viewIndex;
        EXPECT_TRUE(hasSafeLeftAngle) << "view=" << viewIndex;
        EXPECT_TRUE(hasSafeRightAngle) << "view=" << viewIndex;
    }
    EXPECT_LE(graph.statistics.candidatePairCount, views.size() * 16U);
}

TEST(MvsVisibilityGraphBuilderTest, CancellationReturnsWithoutExceptionOrPartialGraph)
{
    std::atomic_bool cancelled{true};
    MvsVisibilityGraphBuildOptions options;
    options.cancelFlag = &cancelled;

    MvsVisibilityGraph graph;
    EXPECT_NO_THROW(
        graph = MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
            1024, fullyVisible(64, 1024), options));
    EXPECT_TRUE(graph.cancelled);
    EXPECT_TRUE(graph.visibilityBits.empty());
    EXPECT_TRUE(graph.neighborsByView.empty());
}

TEST(MvsVisibilityGraphBuilderTest, RunningBuildCancelsAtCooperativeCheckpoint)
{
    constexpr std::size_t viewCount = 256;
    constexpr std::size_t pointCount = 65536;
    std::atomic_bool cancelled{false};
    std::atomic_bool checkpointReported{false};
    std::promise<void> checkpointReachedPromise;
    std::future<void> checkpointReached = checkpointReachedPromise.get_future();
    std::promise<void> releaseCheckpointPromise;
    const std::shared_future<void> releaseCheckpoint =
        releaseCheckpointPromise.get_future().share();

    MvsVisibilityGraphBuildOptions options;
    options.workerCount = 1;
    options.cancelFlag = &cancelled;
    options.cooperativeCheckpointHook = [&]()
    {
        if (!checkpointReported.exchange(true, std::memory_order_relaxed))
        {
            checkpointReachedPromise.set_value();
            releaseCheckpoint.wait();
        }
    };

    auto buildFuture = std::async(
        std::launch::async,
        [options, visible = std::vector<std::vector<std::size_t>>(viewCount)]() mutable
        {
            return MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
                pointCount, std::move(visible), options);
        });

    const std::future_status checkpointStatus = checkpointReached.wait_for(2s);
    cancelled.store(true, std::memory_order_relaxed);
    releaseCheckpointPromise.set_value();
    EXPECT_EQ(checkpointStatus, std::future_status::ready);
    ASSERT_EQ(buildFuture.wait_for(2s), std::future_status::ready);

    const MvsVisibilityGraph graph = buildFuture.get();
    EXPECT_TRUE(graph.cancelled);
    EXPECT_TRUE(graph.visiblePointIndicesByView.empty());
    EXPECT_TRUE(graph.visibilityBits.empty());
    EXPECT_TRUE(graph.neighborsByView.empty());
}

TEST(MvsVisibilityMemoryEstimateTest, IncludesEveryVisibilityPeakCategory)
{
    const auto estimate = xjw::mvs::estimateMvsVisibilityGraphMemory(
        64, 100, 1000, 16, 3);

    EXPECT_FALSE(estimate.saturated);
    EXPECT_EQ(estimate.candidatePairUpperBound, 64U * 16U + 3U);
    EXPECT_GT(estimate.visibilityBitsetBytes, 0U);
    EXPECT_GT(estimate.visibleIndexBytes, 0U);
    EXPECT_GT(estimate.pairBytes, 0U);
    EXPECT_GT(estimate.nominatedPeerBytes, 0U);
    EXPECT_GT(estimate.adjacencyBytes, 0U);
    EXPECT_GT(
        estimate.totalBytes,
        estimate.visibilityBitsetBytes
            + estimate.visibleIndexBytes
            + estimate.pairBytes
            + estimate.nominatedPeerBytes
            + estimate.adjacencyBytes);
}

TEST(MvsVisibilityMemoryEstimateTest, ArithmeticSaturatesInsteadOfWrapping)
{
    const auto estimate = xjw::mvs::estimateMvsVisibilityGraphMemory(
        std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::size_t>::max(),
        32,
        std::numeric_limits<int>::max(),
        std::numeric_limits<std::size_t>::max());

    EXPECT_TRUE(estimate.saturated);
    EXPECT_EQ(estimate.totalBytes, std::numeric_limits<std::uint64_t>::max());
}

} // namespace
