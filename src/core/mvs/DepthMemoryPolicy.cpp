#include "DepthMemoryPolicy.h"

#include <algorithm>
#include <limits>

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

} // namespace mvs
} // namespace xjw
