#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace xjw
{
namespace mvs
{

struct DepthMemoryFrameSize
{
    int width = 0;
    int height = 0;
};

struct DepthConsistencyMemoryEstimate
{
    uint64_t totalPixels = 0;
    uint64_t largestFramePixels = 0;
    uint64_t residentFrameBytes = 0;
    uint64_t consistencySnapshotBytes = 0;
    uint64_t retainedEvidenceBytes = 0;
    uint64_t intermediatePyramidBytes = 0;
    uint64_t transientFrameBytes = 0;
    uint64_t peakBytes = 0;
};

struct DepthMemoryPolicyDecision
{
    bool retainAllFrames = false;
    uint64_t budgetBytes = 0;
    uint64_t reserveBytes = 0;
    DepthConsistencyMemoryEstimate estimate;
};

enum class MvsImageCacheStrategy
{
    Eager,
    Bounded,
    Insufficient
};

struct MvsImageMemoryFrame
{
    int width = 0;
    int height = 0;
    bool preparedSharesGray = true;
};

/// A logical allocation used by the planner. Repeated allocation identifiers
/// describe cv::Mat headers that share one backing allocation.
struct MvsMemoryAllocationEstimate
{
    std::uint64_t allocationId = 0;
    std::uint64_t bytes = 0;
};

/// Conservative peak-memory estimate for the sparse visibility graph.
/// Every field uses saturating arithmetic; `saturated` means the requested
/// graph cannot be represented by the planner's byte counters.
struct MvsVisibilityMemoryEstimate
{
    std::uint64_t visibilityBitsetBytes = 0;
    std::uint64_t visibleIndexBytes = 0;
    std::uint64_t pairBytes = 0;
    std::uint64_t nominatedPeerBytes = 0;
    std::uint64_t adjacencyBytes = 0;
    std::uint64_t candidatePairUpperBound = 0;
    std::uint64_t totalBytes = 0;
    bool saturated = false;
};

struct MvsPipelineMemoryEstimate
{
    std::uint64_t grayBytes = 0;
    std::uint64_t preparedBytes = 0;
    std::uint64_t maskBytes = 0;
    std::uint64_t eagerImageBytes = 0;
    std::uint64_t boundedImageBytes = 0;
    std::uint64_t largestImageFrameBytes = 0;
    std::uint64_t depthResidentBytes = 0;
    std::uint64_t depthStreamingBytes = 0;
    std::uint64_t saveQueueBytes = 0;
    std::uint64_t backendStagingBytes = 0;
    MvsVisibilityMemoryEstimate visibility;
    std::uint64_t eagerRequiredBytes = 0;
    std::uint64_t boundedRequiredBytes = 0;
};

struct MvsPipelineMemoryPolicyDecision
{
    MvsImageCacheStrategy imageStrategy = MvsImageCacheStrategy::Bounded;
    std::size_t imageCacheCapacity = 0;
    std::size_t minimumImageCacheCapacity = 0;
    bool retainAllDepthFrames = false;
    bool memorySnapshotAvailable = false;
    std::uint64_t budgetBytes = 0;
    std::uint64_t reserveBytes = 0;
    std::uint64_t availableBytes = 0;
    std::uint64_t requiredBytes = 0;
    MvsPipelineMemoryEstimate estimate;
};

DepthConsistencyMemoryEstimate estimateDepthConsistencyMemory(
    std::span<const DepthMemoryFrameSize> frameSizes,
    int maximumSourceViews,
    bool adaptiveGeometryEvidence,
    bool retainIntermediatePyramidLevels);

DepthMemoryPolicyDecision decideDepthMemoryPolicy(
    std::span<const DepthMemoryFrameSize> frameSizes,
    int maximumSourceViews,
    bool adaptiveGeometryEvidence,
    bool retainIntermediatePyramidLevels,
    uint64_t totalPhysicalBytes,
    uint64_t availablePhysicalBytes,
    float maximumRamFraction,
    uint64_t minimumFreeBytes);

uint64_t calculateDepthSaveQueueBudgetBytes(
    uint64_t totalPhysicalBytes,
    uint64_t availablePhysicalBytes,
    float maximumRamFraction,
    uint64_t minimumFreeBytes,
    uint64_t transientFrameBytes,
    std::size_t concurrentFrameWorkers,
    uint64_t producerWorkingSetReserveBytes);

std::uint64_t estimateUniqueMvsAllocationBytes(
    std::span<const MvsMemoryAllocationEstimate> allocations);

MvsVisibilityMemoryEstimate estimateMvsVisibilityGraphMemory(
    std::size_t viewCount,
    std::size_t pointCount,
    int fullPairVisibilityLimit,
    int maximumSampledPeersPerView,
    std::size_t requiredPairCount);

MvsPipelineMemoryPolicyDecision decideMvsPipelineMemoryPolicy(
    std::span<const MvsImageMemoryFrame> imageFrames,
    const DepthConsistencyMemoryEstimate &depthEstimate,
    bool allowDepthStreaming,
    int maximumSourceViews,
    std::size_t concurrentFrameWorkers,
    std::uint64_t saveQueueBytes,
    std::uint64_t backendStagingBytes,
    std::uint64_t totalPhysicalBytes,
    std::uint64_t availablePhysicalBytes,
    float maximumRamFraction,
    std::uint64_t minimumFreeBytes,
    const MvsVisibilityMemoryEstimate &visibilityEstimate = {});

std::string_view mvsImageCacheStrategyName(
    MvsImageCacheStrategy strategy) noexcept;

} // namespace mvs
} // namespace xjw
