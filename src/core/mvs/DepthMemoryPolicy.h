#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

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

} // namespace mvs
} // namespace xjw
