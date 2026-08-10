#pragma once

#include "MvsTypes.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace xjw::mvs
{

inline constexpr int kMvsVisibilityFullPairViewLimit = 32;
inline constexpr int kMvsVisibilityMaximumSampledPeersPerView = 32;

struct MvsVisibilityNeighbor
{
    int viewIndex = -1;
    int sharedTrackCount = 0;

    bool operator==(const MvsVisibilityNeighbor &) const = default;
};

struct MvsVisibilityCandidatePair
{
    int firstViewIndex = -1;
    int secondViewIndex = -1;
};

struct MvsVisibilityGraphBuildOptions
{
    int workerCount = 1;
    int fullPairVisibilityLimit = kMvsVisibilityFullPairViewLimit;
    int maximumSampledPeersPerView = kMvsVisibilityMaximumSampledPeersPerView;
    std::vector<MvsVisibilityCandidatePair> requiredPairs;
    std::vector<std::vector<int>> geometryPreferredPeersByView;
    const std::atomic_bool *cancelFlag = nullptr;
    /// Optional diagnostics/test hook invoked at cooperative checkpoints.
    /// It runs on the calling worker and must be thread-safe and non-throwing.
    std::function<void()> cooperativeCheckpointHook;
};

struct MvsVisibilityGraphBuildStatistics
{
    std::size_t visibilityWordCount = 0;
    std::size_t candidatePairCount = 0;
    std::size_t pairCounterEntryCount = 0;
    std::size_t adjacencyEntryCount = 0;
    std::size_t visibilityStorageBytes = 0;
    std::size_t estimatedPairStorageBytes = 0;
};

struct MvsVisibilityGraph
{
    std::vector<std::vector<std::size_t>> visiblePointIndicesByView;
    std::vector<std::vector<MvsVisibilityNeighbor>> neighborsByView;
    std::vector<std::uint64_t> visibilityBits;
    std::size_t pointCount = 0;
    bool cancelled = false;
    MvsVisibilityGraphBuildStatistics statistics;

    bool isVisible(int viewIndex, std::size_t pointIndex) const noexcept;
};

class MvsVisibilityGraphBuilder
{
public:
    static MvsVisibilityGraph build(
        const std::vector<CameraView> &views,
        const SparseCloud &sparse,
        const MvsVisibilityGraphBuildOptions &options = {});

    static MvsVisibilityGraph buildFromVisiblePointIndices(
        std::size_t pointCount,
        std::vector<std::vector<std::size_t>> visiblePointIndicesByView,
        const MvsVisibilityGraphBuildOptions &options = {});

    /// Produces a deterministic, bounded angular shortlist around the sparse
    /// scene center. Peers are ordered in balanced left/right orbital sectors.
    static std::vector<std::vector<int>> buildGeometryPeerShortlist(
        const std::vector<CameraView> &views,
        const SparseCloud &sparse,
        int maximumPeersPerView);
};

} // namespace xjw::mvs
