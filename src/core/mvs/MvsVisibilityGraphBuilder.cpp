#include "MvsVisibilityGraphBuilder.h"

#include "MvsViewSelection.h"
#include "concurrency/SafeWorkerGroup.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace xjw::mvs
{
namespace
{

using xjw::common::concurrency::runWorkerGroup;

std::uint64_t pairKey(int firstViewIndex, int secondViewIndex)
{
    const auto first = static_cast<std::uint32_t>(std::min(firstViewIndex, secondViewIndex));
    const auto second = static_cast<std::uint32_t>(std::max(firstViewIndex, secondViewIndex));
    return (static_cast<std::uint64_t>(first) << 32U) | static_cast<std::uint64_t>(second);
}

std::pair<int, int> decodePairKey(std::uint64_t key)
{
    return {
        static_cast<int>(static_cast<std::uint32_t>(key >> 32U)),
        static_cast<int>(static_cast<std::uint32_t>(key))
    };
}

template <typename Fn>
void parallelFor(std::size_t itemCount,
                 int requestedWorkerCount,
                 const std::atomic_bool *cancelFlag,
                 Fn &&fn)
{
    if (itemCount == 0 || (cancelFlag && cancelFlag->load(std::memory_order_relaxed)))
    {
        return;
    }

    const std::size_t workerCount = std::min(
        itemCount,
        static_cast<std::size_t>(std::max(1, requestedWorkerCount)));
    if (workerCount == 1)
    {
        for (std::size_t index = 0; index < itemCount; ++index)
        {
            if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
            {
                break;
            }
            fn(index);
        }
        return;
    }

    std::atomic_size_t nextIndex{0};
    runWorkerGroup(workerCount, [&](std::stop_token stopToken)
    {
        while (!stopToken.stop_requested())
        {
            if (cancelFlag && cancelFlag->load(std::memory_order_relaxed))
            {
                break;
            }
            const std::size_t index = nextIndex.fetch_add(1, std::memory_order_relaxed);
            if (index >= itemCount)
            {
                break;
            }
            fn(index);
        }
    });
}

bool cancellationRequested(const MvsVisibilityGraphBuildOptions &options) noexcept
{
    return options.cancelFlag
        && options.cancelFlag->load(std::memory_order_relaxed);
}

bool cooperativeCheckpoint(const MvsVisibilityGraphBuildOptions &options)
{
    if (options.cooperativeCheckpointHook)
    {
        options.cooperativeCheckpointHook();
    }
    return cancellationRequested(options);
}

MvsVisibilityGraph cancelledGraph() noexcept
{
    MvsVisibilityGraph graph;
    graph.cancelled = true;
    return graph;
}

bool visibilityBit(const std::vector<std::uint64_t> &bits,
                   std::size_t wordCount,
                   int viewIndex,
                   std::size_t pointIndex) noexcept
{
    if (viewIndex < 0 || wordCount == 0)
    {
        return false;
    }
    const std::size_t offset = static_cast<std::size_t>(viewIndex) * wordCount + pointIndex / 64U;
    return offset < bits.size() && (bits[offset] & (std::uint64_t{1} << (pointIndex % 64U))) != 0;
}

std::uint64_t phaseSeed(std::size_t pointIndex, int viewIndex) noexcept
{
    std::uint64_t value = static_cast<std::uint64_t>(pointIndex) + 0x9e3779b97f4a7c15ULL;
    value ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(viewIndex))
             + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::size_t saturatedAdd(std::size_t lhs, std::size_t rhs) noexcept
{
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    return rhs > maximum - lhs ? maximum : lhs + rhs;
}

std::size_t saturatedMultiply(std::size_t lhs, std::size_t rhs) noexcept
{
    if (lhs == 0 || rhs == 0)
    {
        return 0;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    return lhs > maximum / rhs ? maximum : lhs * rhs;
}

int effectiveFullPairLimit(const MvsVisibilityGraphBuildOptions &options) noexcept
{
    return std::clamp(
        options.fullPairVisibilityLimit,
        2,
        kMvsVisibilityFullPairViewLimit);
}

std::array<double, 3> sparseSceneCenter(const SparseCloud &sparse) noexcept
{
    std::array<double, 3> center{};
    for (std::size_t axis = 0; axis < center.size(); ++axis)
    {
        const double minimum = static_cast<double>(sparse.minPt[axis]);
        const double maximum = static_cast<double>(sparse.maxPt[axis]);
        center[axis] = std::isfinite(minimum) && std::isfinite(maximum)
            ? minimum + (maximum - minimum) * 0.5
            : 0.0;
    }
    return center;
}

} // namespace

bool MvsVisibilityGraph::isVisible(int viewIndex, std::size_t pointIndex) const noexcept
{
    if (viewIndex < 0
        || viewIndex >= static_cast<int>(visiblePointIndicesByView.size())
        || pointIndex >= pointCount)
    {
        return false;
    }
    return visibilityBit(
        visibilityBits, statistics.visibilityWordCount, viewIndex, pointIndex);
}

MvsVisibilityGraph MvsVisibilityGraphBuilder::build(
    const std::vector<CameraView> &views,
    const SparseCloud &sparse,
    const MvsVisibilityGraphBuildOptions &options)
{
    if (cancellationRequested(options))
    {
        return cancelledGraph();
    }

    std::vector<std::vector<std::size_t>> visiblePointIndicesByView(views.size());
    parallelFor(views.size(), options.workerCount, options.cancelFlag, [&](std::size_t viewIndex)
    {
        auto &visible = visiblePointIndicesByView[viewIndex];
        for (std::size_t pointIndex = 0; pointIndex < sparse.points.size(); ++pointIndex)
        {
            if ((pointIndex & 255U) == 0U && cooperativeCheckpoint(options))
            {
                return;
            }
            if (isMvsSparsePointVisibleInView(views[viewIndex], sparse.points[pointIndex]))
            {
                visible.push_back(pointIndex);
            }
        }
    });

    if (cancellationRequested(options))
    {
        return cancelledGraph();
    }

    MvsVisibilityGraphBuildOptions resolvedOptions = options;
    if (views.size() > static_cast<std::size_t>(effectiveFullPairLimit(options))
        && resolvedOptions.geometryPreferredPeersByView.empty())
    {
        const int sampledPeerLimit = std::max(
            1, resolvedOptions.maximumSampledPeersPerView);
        const int geometryPeerBudget = std::min(8, sampledPeerLimit / 2);
        resolvedOptions.geometryPreferredPeersByView = buildGeometryPeerShortlist(
            views, sparse, geometryPeerBudget);
    }

    return buildFromVisiblePointIndices(
        sparse.points.size(), std::move(visiblePointIndicesByView), resolvedOptions);
}

MvsVisibilityGraph MvsVisibilityGraphBuilder::buildFromVisiblePointIndices(
    std::size_t pointCount,
    std::vector<std::vector<std::size_t>> visiblePointIndicesByView,
    const MvsVisibilityGraphBuildOptions &options)
{
    if (cancellationRequested(options))
    {
        return cancelledGraph();
    }

    const std::size_t viewCount = visiblePointIndicesByView.size();
    if (viewCount > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("MVS visibility graph has too many views");
    }

    MvsVisibilityGraph graph;
    graph.pointCount = pointCount;
    graph.statistics.visibilityWordCount =
        pointCount / 64U + (pointCount % 64U == 0 ? 0U : 1U);
    graph.visiblePointIndicesByView = std::move(visiblePointIndicesByView);
    graph.neighborsByView.resize(viewCount);

    const std::size_t wordCount = graph.statistics.visibilityWordCount;
    if (wordCount > 0 && viewCount > std::numeric_limits<std::size_t>::max() / wordCount)
    {
        throw std::length_error("MVS visibility bitset size overflows size_t");
    }
    graph.visibilityBits.assign(viewCount * wordCount, 0);

    parallelFor(viewCount, options.workerCount, options.cancelFlag, [&](std::size_t viewIndex)
    {
        auto &visible = graph.visiblePointIndicesByView[viewIndex];
        std::sort(visible.begin(), visible.end());
        visible.erase(std::unique(visible.begin(), visible.end()), visible.end());
        const std::size_t viewOffset = viewIndex * wordCount;
        for (std::size_t visibleIndex = 0; visibleIndex < visible.size(); ++visibleIndex)
        {
            if ((visibleIndex & 255U) == 0U && cooperativeCheckpoint(options))
            {
                return;
            }
            const std::size_t pointIndex = visible[visibleIndex];
            if (pointIndex >= pointCount)
            {
                throw std::invalid_argument("MVS visibility point index is out of range");
            }
            graph.visibilityBits[viewOffset + pointIndex / 64U] |=
                std::uint64_t{1} << (pointIndex % 64U);
        }
    });

    if (cancellationRequested(options))
    {
        return cancelledGraph();
    }

    std::unordered_set<std::uint64_t> requiredPairKeys;
    requiredPairKeys.reserve(options.requiredPairs.size());
    std::unordered_set<std::uint64_t> candidatePairKeys;
    const std::size_t sampledPeerLimit = viewCount > 0
        ? std::min(
              viewCount - 1U,
              static_cast<std::size_t>(
                  std::max(1, options.maximumSampledPeersPerView)))
        : 0;
    const std::size_t sampledPairUpperBound = saturatedAdd(
        options.requiredPairs.size(),
        saturatedMultiply(viewCount, sampledPeerLimit));
    candidatePairKeys.reserve(sampledPairUpperBound);
    for (std::size_t requiredPairIndex = 0;
         requiredPairIndex < options.requiredPairs.size();
         ++requiredPairIndex)
    {
        if ((requiredPairIndex & 255U) == 0U && cooperativeCheckpoint(options))
        {
            return cancelledGraph();
        }
        const MvsVisibilityCandidatePair &pair = options.requiredPairs[requiredPairIndex];
        if (pair.firstViewIndex < 0
            || pair.secondViewIndex < 0
            || pair.firstViewIndex == pair.secondViewIndex
            || pair.firstViewIndex >= static_cast<int>(viewCount)
            || pair.secondViewIndex >= static_cast<int>(viewCount))
        {
            continue;
        }
        const std::uint64_t key = pairKey(pair.firstViewIndex, pair.secondViewIndex);
        requiredPairKeys.insert(key);
        candidatePairKeys.insert(key);
    }

    const std::size_t fullPairLimit = static_cast<std::size_t>(
        effectiveFullPairLimit(options));
    // The complete-pair mode is a property of the global problem size. For
    // V > 32, even a point seen by only a few cameras must go through bounded
    // nominations, otherwise many such points can reconstruct the O(V^2) set.
    const bool buildCompletePairs = viewCount <= fullPairLimit;
    std::vector<std::vector<int>> nominatedPeersByView(viewCount);
    for (auto &peers : nominatedPeersByView)
    {
        peers.reserve(sampledPeerLimit);
    }

    auto nominatePair = [&](int referenceViewIndex, int peerViewIndex)
    {
        if (referenceViewIndex == peerViewIndex
            || referenceViewIndex < 0
            || peerViewIndex < 0)
        {
            return;
        }
        auto &nominated = nominatedPeersByView[static_cast<std::size_t>(referenceViewIndex)];
        if (nominated.size() >= sampledPeerLimit
            || std::find(nominated.begin(), nominated.end(), peerViewIndex) != nominated.end())
        {
            return;
        }
        nominated.push_back(peerViewIndex);
        candidatePairKeys.insert(pairKey(referenceViewIndex, peerViewIndex));
    };

    if (!buildCompletePairs)
    {
        const std::size_t preferredViewCount = std::min(
            viewCount, options.geometryPreferredPeersByView.size());
        const std::size_t geometryNominationLimit = sampledPeerLimit > 1
            ? std::min<std::size_t>(8, sampledPeerLimit / 2U)
            : 0;
        for (std::size_t viewIndex = 0; viewIndex < preferredViewCount; ++viewIndex)
        {
            std::size_t nominatedGeometryPeers = 0;
            for (int peerViewIndex : options.geometryPreferredPeersByView[viewIndex])
            {
                if (peerViewIndex < 0
                    || peerViewIndex >= static_cast<int>(viewCount)
                    || peerViewIndex == static_cast<int>(viewIndex))
                {
                    continue;
                }
                const std::size_t before = nominatedPeersByView[viewIndex].size();
                nominatePair(static_cast<int>(viewIndex), peerViewIndex);
                if (nominatedPeersByView[viewIndex].size() != before)
                {
                    ++nominatedGeometryPeers;
                }
                if (nominatedGeometryPeers >= geometryNominationLimit)
                {
                    break;
                }
            }
        }
    }

    std::vector<int> visibleViews;
    visibleViews.reserve(viewCount);
    for (std::size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
    {
        if (cooperativeCheckpoint(options))
        {
            return cancelledGraph();
        }
        visibleViews.clear();
        for (std::size_t viewIndex = 0; viewIndex < viewCount; ++viewIndex)
        {
            if ((viewIndex & 255U) == 0U && cooperativeCheckpoint(options))
            {
                return cancelledGraph();
            }
            if (visibilityBit(graph.visibilityBits,
                              wordCount,
                              static_cast<int>(viewIndex),
                              pointIndex))
            {
                visibleViews.push_back(static_cast<int>(viewIndex));
            }
        }

        if (visibleViews.size() < 2)
        {
            continue;
        }
        if (buildCompletePairs)
        {
            for (std::size_t first = 0; first < visibleViews.size(); ++first)
            {
                for (std::size_t second = first + 1; second < visibleViews.size(); ++second)
                {
                    candidatePairKeys.insert(pairKey(visibleViews[first], visibleViews[second]));
                }
            }
            continue;
        }

        for (std::size_t position = 0; position < visibleViews.size(); ++position)
        {
            const int referenceViewIndex = visibleViews[position];
            if (nominatedPeersByView[static_cast<std::size_t>(referenceViewIndex)].size()
                >= sampledPeerLimit)
            {
                continue;
            }

            if (position > 0)
            {
                nominatePair(referenceViewIndex, visibleViews[position - 1]);
            }
            if (position + 1 < visibleViews.size())
            {
                nominatePair(referenceViewIndex, visibleViews[position + 1]);
            }

            const std::size_t peerCount = visibleViews.size() - 1U;
            const std::size_t phaseOffset = 1U
                + static_cast<std::size_t>(phaseSeed(pointIndex, referenceViewIndex) % peerCount);
            const int phasePeer = visibleViews[(position + phaseOffset) % visibleViews.size()];
            nominatePair(referenceViewIndex, phasePeer);
        }
    }

    std::vector<std::uint64_t> sortedPairKeys(candidatePairKeys.begin(), candidatePairKeys.end());
    std::sort(sortedPairKeys.begin(), sortedPairKeys.end());
    std::vector<int> sharedTrackCounts(sortedPairKeys.size(), 0);
    parallelFor(
        sortedPairKeys.size(),
        options.workerCount,
        options.cancelFlag,
        [&](std::size_t pairIndex)
    {
        const auto [firstViewIndex, secondViewIndex] = decodePairKey(sortedPairKeys[pairIndex]);
        const std::size_t firstOffset = static_cast<std::size_t>(firstViewIndex) * wordCount;
        const std::size_t secondOffset = static_cast<std::size_t>(secondViewIndex) * wordCount;
        std::uint64_t sharedTrackCount = 0;
        for (std::size_t word = 0; word < wordCount; ++word)
        {
            if ((word & 255U) == 0U && cooperativeCheckpoint(options))
            {
                return;
            }
            sharedTrackCount += static_cast<std::uint64_t>(
                std::popcount(graph.visibilityBits[firstOffset + word]
                              & graph.visibilityBits[secondOffset + word]));
        }
        sharedTrackCounts[pairIndex] = static_cast<int>(std::min<std::uint64_t>(
            sharedTrackCount,
            static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
    });

    if (cancellationRequested(options))
    {
        return cancelledGraph();
    }

    for (std::size_t pairIndex = 0; pairIndex < sortedPairKeys.size(); ++pairIndex)
    {
        if ((pairIndex & 255U) == 0U && cooperativeCheckpoint(options))
        {
            return cancelledGraph();
        }
        const std::uint64_t key = sortedPairKeys[pairIndex];
        const int sharedTrackCount = sharedTrackCounts[pairIndex];
        if (sharedTrackCount <= 0 && requiredPairKeys.find(key) == requiredPairKeys.end())
        {
            continue;
        }
        const auto [firstViewIndex, secondViewIndex] = decodePairKey(key);
        graph.neighborsByView[static_cast<std::size_t>(firstViewIndex)].push_back(
            {secondViewIndex, sharedTrackCount});
        graph.neighborsByView[static_cast<std::size_t>(secondViewIndex)].push_back(
            {firstViewIndex, sharedTrackCount});
    }
    for (auto &neighbors : graph.neighborsByView)
    {
        std::sort(neighbors.begin(), neighbors.end(), [](const auto &lhs, const auto &rhs)
        {
            return lhs.viewIndex < rhs.viewIndex;
        });
    }

    graph.statistics.candidatePairCount = sortedPairKeys.size();
    graph.statistics.pairCounterEntryCount = sharedTrackCounts.size();
    for (const auto &neighbors : graph.neighborsByView)
    {
        graph.statistics.adjacencyEntryCount += neighbors.size();
    }
    graph.statistics.visibilityStorageBytes = saturatedMultiply(
        graph.visibilityBits.size(), sizeof(std::uint64_t));
    graph.statistics.estimatedPairStorageBytes = saturatedAdd(
        saturatedMultiply(sortedPairKeys.size(), sizeof(std::uint64_t) + sizeof(int)),
        saturatedMultiply(graph.statistics.adjacencyEntryCount, sizeof(MvsVisibilityNeighbor)));
    return graph;
}

std::vector<std::vector<int>> MvsVisibilityGraphBuilder::buildGeometryPeerShortlist(
    const std::vector<CameraView> &views,
    const SparseCloud &sparse,
    int maximumPeersPerView)
{
    std::vector<std::vector<int>> peersByView(views.size());
    if (views.size() < 2 || maximumPeersPerView <= 0)
    {
        return peersByView;
    }

    const std::size_t peerLimit = std::min(
        views.size() - 1U,
        static_cast<std::size_t>(maximumPeersPerView));
    const std::array<double, 3> sceneCenter = sparseSceneCenter(sparse);
    std::array<double, 3> minimumCenter{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> maximumCenter{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    for (const CameraView &view : views)
    {
        const std::array<double, 3> center = view.camera.cameraCenter();
        for (std::size_t axis = 0; axis < center.size(); ++axis)
        {
            if (!std::isfinite(center[axis]))
            {
                continue;
            }
            minimumCenter[axis] = std::min(minimumCenter[axis], center[axis]);
            maximumCenter[axis] = std::max(maximumCenter[axis], center[axis]);
        }
    }

    std::size_t droppedAxis = 0;
    double smallestRange = std::numeric_limits<double>::infinity();
    for (std::size_t axis = 0; axis < minimumCenter.size(); ++axis)
    {
        const double range = maximumCenter[axis] - minimumCenter[axis];
        if (std::isfinite(range) && range < smallestRange)
        {
            smallestRange = range;
            droppedAxis = axis;
        }
    }
    const std::size_t firstAxis = droppedAxis == 0 ? 1 : 0;
    const std::size_t secondAxis = droppedAxis == 2 ? 1 : 2;

    struct AngularView
    {
        double angle = 0.0;
        int viewIndex = -1;
    };
    std::vector<AngularView> angularViews;
    angularViews.reserve(views.size());
    constexpr double kMinimumProjectedRadiusSquared = 1.0e-18;
    constexpr double kTwoPi = 6.28318530717958647692;
    for (std::size_t viewIndex = 0; viewIndex < views.size(); ++viewIndex)
    {
        const std::array<double, 3> center = views[viewIndex].camera.cameraCenter();
        const double first = center[firstAxis] - sceneCenter[firstAxis];
        const double second = center[secondAxis] - sceneCenter[secondAxis];
        const double radiusSquared = first * first + second * second;
        const double fallbackAngle = kTwoPi
            * static_cast<double>(viewIndex) / static_cast<double>(views.size());
        const double angle = std::isfinite(radiusSquared)
            && radiusSquared > kMinimumProjectedRadiusSquared
            ? std::atan2(second, first)
            : fallbackAngle;
        angularViews.push_back({angle, static_cast<int>(viewIndex)});
    }
    std::sort(angularViews.begin(), angularViews.end(), [](const auto &left, const auto &right)
    {
        if (left.angle != right.angle)
        {
            return left.angle < right.angle;
        }
        return left.viewIndex < right.viewIndex;
    });

    std::vector<std::size_t> angularPositionByView(views.size());
    for (std::size_t position = 0; position < angularViews.size(); ++position)
    {
        angularPositionByView[static_cast<std::size_t>(angularViews[position].viewIndex)] = position;
    }

    std::vector<std::size_t> angularSteps{1U};
    for (std::size_t divisor : {16U, 8U, 6U, 4U})
    {
        const std::size_t step = std::max<std::size_t>(
            2U, (views.size() + divisor / 2U) / divisor);
        if (step < views.size() && std::find(angularSteps.begin(), angularSteps.end(), step)
                                      == angularSteps.end())
        {
            angularSteps.push_back(step);
        }
    }

    for (std::size_t viewIndex = 0; viewIndex < views.size(); ++viewIndex)
    {
        auto &peers = peersByView[viewIndex];
        peers.reserve(peerLimit);
        const std::size_t position = angularPositionByView[viewIndex];
        for (std::size_t step : angularSteps)
        {
            for (int direction : {-1, 1})
            {
                const std::size_t peerPosition = direction < 0
                    ? (position + views.size() - step % views.size()) % views.size()
                    : (position + step) % views.size();
                const int peerViewIndex = angularViews[peerPosition].viewIndex;
                if (peerViewIndex != static_cast<int>(viewIndex)
                    && std::find(peers.begin(), peers.end(), peerViewIndex) == peers.end())
                {
                    peers.push_back(peerViewIndex);
                }
                if (peers.size() >= peerLimit)
                {
                    break;
                }
            }
            if (peers.size() >= peerLimit)
            {
                break;
            }
        }
    }
    return peersByView;
}

} // namespace xjw::mvs
