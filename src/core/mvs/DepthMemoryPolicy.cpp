#include "DepthMemoryPolicy.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>

namespace xjw
{
namespace mvs
{
namespace
{

// These values model concurrently live cv::Mat payloads in the in-memory
// consistency path. They intentionally include masks and repair scratch space,
// rather than just the depth/confidence pair used by the old estimate.
constexpr uint64_t kResidentFrameBytesPerPixel = 26;
constexpr uint64_t kSnapshotBytesPerPixel = 4;
constexpr uint64_t kAdaptiveSnapshotBytesPerPixel = 4;
constexpr uint64_t kRetainedEvidenceBytesPerPixel = 14;
constexpr uint64_t kAdaptiveEvidenceBytesPerPixel = 12;
constexpr uint64_t kIntermediatePyramidBytesPerPixel = 11;
constexpr uint64_t kTransientBaseBytesPerPixel = 92;
constexpr uint64_t kSourceProjectionBytesPerPixel = 4;
constexpr uint64_t kAllocatorOverheadDivisor = 8; // 12.5% for Mat rows/allocator fragmentation.
constexpr double kMinimumDynamicReserveFraction = 0.20;

uint64_t saturatingAdd(uint64_t left, uint64_t right)
{
    if (right > std::numeric_limits<uint64_t>::max() - left)
    {
        return std::numeric_limits<uint64_t>::max();
    }
    return left + right;
}

uint64_t saturatingMultiply(uint64_t left, uint64_t right)
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
    {
        return std::numeric_limits<uint64_t>::max();
    }
    return left * right;
}

uint64_t addAllocatorOverhead(uint64_t bytes)
{
    return saturatingAdd(bytes, bytes / kAllocatorOverheadDivisor);
}

uint64_t validPixelCount(const DepthMemoryFrameSize &frameSize)
{
    if (frameSize.width <= 0 || frameSize.height <= 0)
    {
        return 0;
    }
    return saturatingMultiply(static_cast<uint64_t>(frameSize.width),
                              static_cast<uint64_t>(frameSize.height));
}

uint64_t saturatingAddTracked(uint64_t left,
                              uint64_t right,
                              bool *saturated)
{
    if (right > std::numeric_limits<uint64_t>::max() - left)
    {
        if (saturated)
        {
            *saturated = true;
        }
        return std::numeric_limits<uint64_t>::max();
    }
    return left + right;
}

uint64_t saturatingMultiplyTracked(uint64_t left,
                                   uint64_t right,
                                   bool *saturated)
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
    {
        if (saturated)
        {
            *saturated = true;
        }
        return std::numeric_limits<uint64_t>::max();
    }
    return left * right;
}

uint64_t validPixelCount(const MvsImageMemoryFrame &frameSize)
{
    if (frameSize.width <= 0 || frameSize.height <= 0)
    {
        return 0;
    }
    return saturatingMultiply(static_cast<uint64_t>(frameSize.width),
                              static_cast<uint64_t>(frameSize.height));
}

uint64_t depthResidentBytes(const DepthConsistencyMemoryEstimate &estimate)
{
    uint64_t bytes = estimate.residentFrameBytes;
    bytes = saturatingAdd(bytes, estimate.consistencySnapshotBytes);
    bytes = saturatingAdd(bytes, estimate.retainedEvidenceBytes);
    return saturatingAdd(bytes, estimate.intermediatePyramidBytes);
}

uint64_t workingImageCapacity(std::size_t viewCount,
                              std::size_t concurrentFrameWorkers,
                              int maximumSourceViews)
{
    if (viewCount == 0)
    {
        return 0;
    }
    const uint64_t workers = static_cast<uint64_t>(
        std::max<std::size_t>(1, concurrentFrameWorkers));
    const uint64_t viewsPerWorker = static_cast<uint64_t>(
        std::max(0, maximumSourceViews)) + 1;
    return std::min<uint64_t>(
        static_cast<uint64_t>(viewCount),
        saturatingMultiply(workers, viewsPerWorker));
}

} // namespace

DepthConsistencyMemoryEstimate estimateDepthConsistencyMemory(
    std::span<const DepthMemoryFrameSize> frameSizes,
    int maximumSourceViews,
    bool adaptiveGeometryEvidence,
    bool retainIntermediatePyramidLevels)
{
    DepthConsistencyMemoryEstimate result;
    for (const DepthMemoryFrameSize &frameSize : frameSizes)
    {
        const uint64_t pixels = validPixelCount(frameSize);
        result.totalPixels = saturatingAdd(result.totalPixels, pixels);
        result.largestFramePixels = std::max(result.largestFramePixels, pixels);
    }

    result.residentFrameBytes = saturatingMultiply(
        result.totalPixels, kResidentFrameBytesPerPixel);
    result.consistencySnapshotBytes = saturatingMultiply(
        result.totalPixels,
        kSnapshotBytesPerPixel +
            (adaptiveGeometryEvidence ? kAdaptiveSnapshotBytesPerPixel : 0));
    result.retainedEvidenceBytes = saturatingMultiply(
        result.totalPixels,
        kRetainedEvidenceBytesPerPixel +
            (adaptiveGeometryEvidence ? kAdaptiveEvidenceBytesPerPixel : 0));
    if (retainIntermediatePyramidLevels)
    {
        result.intermediatePyramidBytes = saturatingMultiply(
            result.totalPixels, kIntermediatePyramidBytesPerPixel);
    }

    const uint64_t sourceCount = static_cast<uint64_t>(
        std::clamp(maximumSourceViews, 1, 16));
    const uint64_t transientBytesPerPixel = saturatingAdd(
        kTransientBaseBytesPerPixel,
        saturatingMultiply(sourceCount, kSourceProjectionBytesPerPixel));
    result.transientFrameBytes = saturatingMultiply(
        result.largestFramePixels, transientBytesPerPixel);

    uint64_t peakBytes = result.residentFrameBytes;
    peakBytes = saturatingAdd(peakBytes, result.consistencySnapshotBytes);
    peakBytes = saturatingAdd(peakBytes, result.retainedEvidenceBytes);
    peakBytes = saturatingAdd(peakBytes, result.intermediatePyramidBytes);
    peakBytes = saturatingAdd(peakBytes, result.transientFrameBytes);
    result.peakBytes = addAllocatorOverhead(peakBytes);
    return result;
}

DepthMemoryPolicyDecision decideDepthMemoryPolicy(
    std::span<const DepthMemoryFrameSize> frameSizes,
    int maximumSourceViews,
    bool adaptiveGeometryEvidence,
    bool retainIntermediatePyramidLevels,
    uint64_t totalPhysicalBytes,
    uint64_t availablePhysicalBytes,
    float maximumRamFraction,
    uint64_t minimumFreeBytes)
{
    DepthMemoryPolicyDecision decision;
    decision.estimate = estimateDepthConsistencyMemory(
        frameSizes,
        maximumSourceViews,
        adaptiveGeometryEvidence,
        retainIntermediatePyramidLevels);

    if (totalPhysicalBytes == 0 || availablePhysicalBytes == 0 ||
        decision.estimate.totalPixels == 0)
    {
        return decision;
    }

    const uint64_t proportionalReserve = static_cast<uint64_t>(
        static_cast<double>(totalPhysicalBytes) * kMinimumDynamicReserveFraction);
    const uint64_t transientReserve = saturatingMultiply(
        decision.estimate.transientFrameBytes, 2);
    decision.reserveBytes = std::max({minimumFreeBytes,
                                      proportionalReserve,
                                      transientReserve});

    const double fraction = std::clamp(
        static_cast<double>(maximumRamFraction), 0.10, 0.90);
    const uint64_t totalBudget = static_cast<uint64_t>(
        static_cast<double>(totalPhysicalBytes) * fraction);
    const uint64_t availableBudget = availablePhysicalBytes > decision.reserveBytes
        ? availablePhysicalBytes - decision.reserveBytes
        : 0;
    decision.budgetBytes = std::min(totalBudget, availableBudget);
    decision.retainAllFrames = decision.estimate.peakBytes <= decision.budgetBytes;
    return decision;
}

uint64_t calculateDepthSaveQueueBudgetBytes(
    uint64_t totalPhysicalBytes,
    uint64_t availablePhysicalBytes,
    float maximumRamFraction,
    uint64_t minimumFreeBytes,
    uint64_t transientFrameBytes,
    std::size_t concurrentFrameWorkers,
    uint64_t producerWorkingSetReserveBytes)
{
    if (totalPhysicalBytes == 0 || availablePhysicalBytes == 0)
    {
        return 0;
    }

    const uint64_t proportionalReserve = static_cast<uint64_t>(
        static_cast<double>(totalPhysicalBytes) * kMinimumDynamicReserveFraction);
    const uint64_t transientReserveBytes = saturatingMultiply(
        transientFrameBytes,
        static_cast<uint64_t>(std::max<std::size_t>(1, concurrentFrameWorkers)));
    const uint64_t reserveBytes = std::max({minimumFreeBytes,
                                            proportionalReserve,
                                            transientReserveBytes,
                                            producerWorkingSetReserveBytes});
    const double fraction = std::clamp(
        static_cast<double>(maximumRamFraction), 0.10, 0.90);
    const uint64_t totalBudget = static_cast<uint64_t>(
        static_cast<double>(totalPhysicalBytes) * fraction);
    const uint64_t availableBudget = availablePhysicalBytes > reserveBytes
        ? availablePhysicalBytes - reserveBytes
        : 0;
    return std::min(totalBudget, availableBudget);
}

uint64_t estimateUniqueMvsAllocationBytes(
    std::span<const MvsMemoryAllocationEstimate> allocations)
{
    std::unordered_map<uint64_t, uint64_t> uniqueAllocations;
    uniqueAllocations.reserve(allocations.size());
    for (const MvsMemoryAllocationEstimate &allocation : allocations)
    {
        auto [entry, inserted] = uniqueAllocations.emplace(
            allocation.allocationId, allocation.bytes);
        if (!inserted)
        {
            entry->second = std::max(entry->second, allocation.bytes);
        }
    }

    uint64_t bytes = 0;
    for (const auto &[allocationId, allocationBytes] : uniqueAllocations)
    {
        static_cast<void>(allocationId);
        bytes = saturatingAdd(bytes, allocationBytes);
    }
    return bytes;
}

MvsVisibilityMemoryEstimate estimateMvsVisibilityGraphMemory(
    std::size_t viewCount,
    std::size_t pointCount,
    int fullPairVisibilityLimit,
    int maximumSampledPeersPerView,
    std::size_t requiredPairCount)
{
    MvsVisibilityMemoryEstimate estimate;
    bool saturated = false;
    const uint64_t views = static_cast<uint64_t>(viewCount);
    const uint64_t points = static_cast<uint64_t>(pointCount);
    const uint64_t visibilityWords = points / 64U + (points % 64U == 0 ? 0U : 1U);
    uint64_t completePairCount = 0;
    if (views >= 2)
    {
        const uint64_t firstFactor = views % 2U == 0 ? views / 2U : views;
        const uint64_t secondFactor = views % 2U == 0
            ? views - 1U
            : (views - 1U) / 2U;
        completePairCount = saturatingMultiplyTracked(
            firstFactor, secondFactor, &saturated);
    }
    const uint64_t validRequiredPairCount = std::min<uint64_t>(
        static_cast<uint64_t>(requiredPairCount), completePairCount);
    const uint64_t peerLimit = views > 0
        ? std::min<uint64_t>(
              views - 1U,
              static_cast<uint64_t>(std::max(1, maximumSampledPeersPerView)))
        : 0;
    const uint64_t effectiveFullPairLimit = static_cast<uint64_t>(
        std::clamp(fullPairVisibilityLimit, 2, 32));
    if (views <= effectiveFullPairLimit)
    {
        estimate.candidatePairUpperBound = completePairCount;
    }
    else
    {
        const uint64_t sampledPairCount = saturatingMultiplyTracked(
            views, peerLimit, &saturated);
        estimate.candidatePairUpperBound = std::min(
            completePairCount,
            saturatingAddTracked(
                sampledPairCount, validRequiredPairCount, &saturated));
    }

    // Visible indices use the all-points-visible upper bound because
    // projection has not run yet. The fixed per-view allowance also covers
    // the destination FrameMvsCache headers that coexist during the move.
    estimate.visibilityBitsetBytes = saturatingMultiplyTracked(
        saturatingMultiplyTracked(views, visibilityWords, &saturated),
        sizeof(std::uint64_t),
        &saturated);
    constexpr uint64_t kDestinationVisibilityStateBytesPerView = 192U;
    estimate.visibleIndexBytes = saturatingAddTracked(
        saturatingMultiplyTracked(
            saturatingMultiplyTracked(
                saturatingMultiplyTracked(views, points, &saturated),
                2U,
                &saturated),
            sizeof(std::size_t),
            &saturated),
        saturatingMultiplyTracked(
            views,
            sizeof(std::vector<std::size_t>)
                + kDestinationVisibilityStateBytesPerView,
            &saturated),
        &saturated);

    // Account for both the geometry shortlist and the live nominated-peer
    // table. Both contain at most V*K integers plus one vector header per view.
    const uint64_t onePeerTableBytes = saturatingAddTracked(
        saturatingMultiplyTracked(
            saturatingMultiplyTracked(views, peerLimit, &saturated),
            sizeof(int),
            &saturated),
        saturatingMultiplyTracked(views, sizeof(std::vector<int>), &saturated),
        &saturated);
    estimate.nominatedPeerBytes = saturatingMultiplyTracked(
        onePeerTableBytes, 2U, &saturated);

    // Candidate/required unordered-set nodes and buckets coexist with the
    // sorted key and exact-count arrays. The per-entry constants deliberately
    // exceed the standard containers' payload sizes to cover node overhead.
    constexpr uint64_t kCandidatePairWorkingBytes = 64U;
    constexpr uint64_t kRequiredPairWorkingBytes = 40U;
    estimate.pairBytes = saturatingAddTracked(
        saturatingMultiplyTracked(
            estimate.candidatePairUpperBound,
            kCandidatePairWorkingBytes,
            &saturated),
        saturatingMultiplyTracked(
            validRequiredPairCount,
            kRequiredPairWorkingBytes,
            &saturated),
        &saturated);

    // Every retained undirected candidate can contribute two adjacency
    // entries; another factor of two covers vector growth capacity. Vector
    // headers are resident even for empty neighbor lists.
    estimate.adjacencyBytes = saturatingAddTracked(
        saturatingMultiplyTracked(
            saturatingMultiplyTracked(
                estimate.candidatePairUpperBound, 4U, &saturated),
            sizeof(std::pair<int, int>),
            &saturated),
        saturatingMultiplyTracked(
            saturatingMultiplyTracked(views, 2U, &saturated),
            sizeof(std::vector<std::pair<int, int>>),
            &saturated),
        &saturated);

    estimate.totalBytes = estimate.visibilityBitsetBytes;
    estimate.totalBytes = saturatingAddTracked(
        estimate.totalBytes, estimate.visibleIndexBytes, &saturated);
    estimate.totalBytes = saturatingAddTracked(
        estimate.totalBytes, estimate.pairBytes, &saturated);
    estimate.totalBytes = saturatingAddTracked(
        estimate.totalBytes, estimate.nominatedPeerBytes, &saturated);
    estimate.totalBytes = saturatingAddTracked(
        estimate.totalBytes, estimate.adjacencyBytes, &saturated);
    estimate.totalBytes = saturatingAddTracked(
        estimate.totalBytes,
        estimate.totalBytes / kAllocatorOverheadDivisor,
        &saturated);
    estimate.saturated = saturated;
    return estimate;
}

MvsPipelineMemoryPolicyDecision decideMvsPipelineMemoryPolicy(
    std::span<const MvsImageMemoryFrame> imageFrames,
    const DepthConsistencyMemoryEstimate &depthEstimate,
    bool allowDepthStreaming,
    int maximumSourceViews,
    std::size_t concurrentFrameWorkers,
    uint64_t saveQueueBytes,
    uint64_t backendStagingBytes,
    uint64_t totalPhysicalBytes,
    uint64_t availablePhysicalBytes,
    float maximumRamFraction,
    uint64_t minimumFreeBytes,
    const MvsVisibilityMemoryEstimate &visibilityEstimate)
{
    MvsPipelineMemoryPolicyDecision decision;
    MvsPipelineMemoryEstimate &estimate = decision.estimate;

    std::vector<MvsMemoryAllocationEstimate> imageAllocations;
    imageAllocations.reserve(imageFrames.size() * 3);
    for (std::size_t index = 0; index < imageFrames.size(); ++index)
    {
        const MvsImageMemoryFrame &frame = imageFrames[index];
        const uint64_t pixels = validPixelCount(frame);
        const uint64_t grayAllocationId = static_cast<uint64_t>(index) * 3;
        const uint64_t preparedAllocationId = frame.preparedSharesGray
            ? grayAllocationId
            : grayAllocationId + 1;
        const uint64_t maskAllocationId = grayAllocationId + 2;
        imageAllocations.push_back({grayAllocationId, pixels});
        imageAllocations.push_back({preparedAllocationId, pixels});
        imageAllocations.push_back({maskAllocationId, pixels});

        estimate.grayBytes = saturatingAdd(estimate.grayBytes, pixels);
        if (!frame.preparedSharesGray)
        {
            estimate.preparedBytes = saturatingAdd(
                estimate.preparedBytes, pixels);
        }
        estimate.maskBytes = saturatingAdd(estimate.maskBytes, pixels);
        const uint64_t frameBytes = saturatingMultiply(
            pixels, frame.preparedSharesGray ? 2 : 3);
        estimate.largestImageFrameBytes = std::max(
            estimate.largestImageFrameBytes, frameBytes);
    }

    estimate.eagerImageBytes = estimateUniqueMvsAllocationBytes(
        imageAllocations);
    estimate.depthResidentBytes = depthResidentBytes(depthEstimate);
    const uint64_t activeWorkers = static_cast<uint64_t>(
        std::max<std::size_t>(1, concurrentFrameWorkers));
    estimate.depthStreamingBytes = saturatingMultiply(
        saturatingMultiply(
            depthEstimate.largestFramePixels, kResidentFrameBytesPerPixel),
        activeWorkers);
    estimate.saveQueueBytes = saveQueueBytes;
    estimate.backendStagingBytes = backendStagingBytes;
    estimate.visibility = visibilityEstimate;

    decision.minimumImageCacheCapacity = static_cast<std::size_t>(
        workingImageCapacity(
            imageFrames.size(), concurrentFrameWorkers, maximumSourceViews));
    estimate.boundedImageBytes = saturatingMultiply(
        estimate.largestImageFrameBytes,
        static_cast<uint64_t>(decision.minimumImageCacheCapacity));

    const uint64_t commonBytes = saturatingAdd(
        saturatingAdd(saveQueueBytes, backendStagingBytes),
        visibilityEstimate.totalBytes);
    estimate.eagerRequiredBytes = saturatingAdd(
        saturatingAdd(estimate.eagerImageBytes, estimate.depthResidentBytes),
        commonBytes);
    estimate.boundedRequiredBytes = saturatingAdd(
        saturatingAdd(estimate.boundedImageBytes, estimate.depthStreamingBytes),
        commonBytes);

    if (visibilityEstimate.saturated)
    {
        decision.imageStrategy = MvsImageCacheStrategy::Insufficient;
        decision.requiredBytes = std::numeric_limits<uint64_t>::max();
        return decision;
    }

    if (imageFrames.empty())
    {
        decision.imageStrategy = MvsImageCacheStrategy::Eager;
        decision.retainAllDepthFrames = true;
        decision.requiredBytes = estimate.eagerRequiredBytes;
        return decision;
    }

    decision.memorySnapshotAvailable =
        totalPhysicalBytes > 0 && availablePhysicalBytes > 0;
    if (!decision.memorySnapshotAvailable)
    {
        decision.imageStrategy = MvsImageCacheStrategy::Bounded;
        decision.imageCacheCapacity = decision.minimumImageCacheCapacity;
        decision.retainAllDepthFrames = !allowDepthStreaming;
        decision.requiredBytes = allowDepthStreaming
            ? estimate.boundedRequiredBytes
            : saturatingAdd(
                  saturatingAdd(estimate.boundedImageBytes,
                                estimate.depthResidentBytes),
                  commonBytes);
        return decision;
    }

    const uint64_t proportionalReserve = static_cast<uint64_t>(
        static_cast<double>(totalPhysicalBytes) * kMinimumDynamicReserveFraction);
    decision.reserveBytes = std::max(minimumFreeBytes, proportionalReserve);
    const double fraction = std::clamp(
        static_cast<double>(maximumRamFraction), 0.10, 0.90);
    const uint64_t totalBudget = static_cast<uint64_t>(
        static_cast<double>(totalPhysicalBytes) * fraction);
    const uint64_t availableBudget = availablePhysicalBytes > decision.reserveBytes
        ? availablePhysicalBytes - decision.reserveBytes
        : 0;
    decision.budgetBytes = std::min(totalBudget, availableBudget);
    decision.availableBytes = decision.budgetBytes;

    if (estimate.eagerRequiredBytes <= decision.budgetBytes)
    {
        decision.imageStrategy = MvsImageCacheStrategy::Eager;
        decision.imageCacheCapacity = imageFrames.size();
        decision.retainAllDepthFrames = true;
        decision.requiredBytes = estimate.eagerRequiredBytes;
        return decision;
    }

    const uint64_t boundedResidentDepthRequired = saturatingAdd(
        saturatingAdd(estimate.boundedImageBytes, estimate.depthResidentBytes),
        commonBytes);
    if (boundedResidentDepthRequired <= decision.budgetBytes)
    {
        decision.imageStrategy = MvsImageCacheStrategy::Bounded;
        decision.imageCacheCapacity = decision.minimumImageCacheCapacity;
        decision.retainAllDepthFrames = true;
        decision.requiredBytes = boundedResidentDepthRequired;
        return decision;
    }

    if (!allowDepthStreaming)
    {
        decision.imageStrategy = MvsImageCacheStrategy::Insufficient;
        decision.retainAllDepthFrames = true;
        decision.requiredBytes = boundedResidentDepthRequired;
        const uint64_t fixedBytes = saturatingAdd(
            estimate.depthResidentBytes, commonBytes);
        const uint64_t cacheBudget = decision.budgetBytes > fixedBytes
            ? decision.budgetBytes - fixedBytes
            : 0;
        decision.imageCacheCapacity = estimate.largestImageFrameBytes > 0
            ? static_cast<std::size_t>(std::min<uint64_t>(
                  imageFrames.size(),
                  cacheBudget / estimate.largestImageFrameBytes))
            : imageFrames.size();
        return decision;
    }

    decision.requiredBytes = estimate.boundedRequiredBytes;
    const uint64_t boundedFixedBytes = saturatingAdd(
        estimate.depthStreamingBytes, commonBytes);
    const uint64_t cacheBudget = decision.budgetBytes > boundedFixedBytes
        ? decision.budgetBytes - boundedFixedBytes
        : 0;
    const std::size_t capacityByBudget = estimate.largestImageFrameBytes > 0
        ? static_cast<std::size_t>(std::min<uint64_t>(
              imageFrames.size(), cacheBudget / estimate.largestImageFrameBytes))
        : imageFrames.size();
    if (capacityByBudget < decision.minimumImageCacheCapacity)
    {
        decision.imageStrategy = MvsImageCacheStrategy::Insufficient;
        decision.imageCacheCapacity = capacityByBudget;
        return decision;
    }

    decision.imageStrategy = MvsImageCacheStrategy::Bounded;
    decision.imageCacheCapacity = decision.minimumImageCacheCapacity;
    decision.retainAllDepthFrames = false;
    return decision;
}

std::string_view mvsImageCacheStrategyName(
    MvsImageCacheStrategy strategy) noexcept
{
    switch (strategy)
    {
    case MvsImageCacheStrategy::Eager:
        return "eager";
    case MvsImageCacheStrategy::Bounded:
        return "bounded";
    case MvsImageCacheStrategy::Insufficient:
        return "insufficient";
    }
    return "unknown";
}

} // namespace mvs
} // namespace xjw
