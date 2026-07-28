#pragma once

#include "DepthVisibilityHistogram.h"

#include <array>
#include <cstdint>
#include <vector>

namespace xjw::mesh
{

struct AdaptiveTsdfOctreeOptions
{
    int maximumMergeLevel = 8;
    float minimumMergeAbsoluteField = 0.55f;
    float maximumMergeFieldRange = 0.15f;
};

struct AdaptiveTsdfOctreeNode
{
    std::array<int, 3> origin{};
    int size = 1;
    int level = 0;
    float value = 1.0f;
    float observationWeight = 0.0f;
    std::uint16_t geometrySourceMask = 0;
    std::uint32_t activeSampleCount = 0;
    std::uint32_t supportedSampleCount = 0;
    DepthVisibilityHistogramSummary histogram;
    std::array<int, 6> faceNeighbors{{-1, -1, -1, -1, -1, -1}};
};

struct AdaptiveTsdfOctreeStatistics
{
    std::uint64_t inputActiveSampleCount = 0;
    std::uint64_t mergedNodeCount = 0;
    std::uint64_t balanceSplitCount = 0;
    int maximumLevel = 0;
    bool twoToOneBalanced = false;
};

struct AdaptiveTsdfOctreeResult
{
    std::array<int, 3> dimensions{};
    std::vector<AdaptiveTsdfOctreeNode> leaves;
    AdaptiveTsdfOctreeStatistics statistics;
};

class AdaptiveTsdfOctree
{
public:
    static AdaptiveTsdfOctreeResult build(
        const std::array<int, 3> &dimensions,
        const std::vector<float> &field,
        const std::vector<float> &observationWeight,
        const std::vector<std::uint16_t> &geometrySourceMask,
        const std::vector<std::uint8_t> &active,
        const std::vector<std::uint8_t> &supported,
        const std::vector<DepthVisibilityHistogram> &histograms,
        const AdaptiveTsdfOctreeOptions &options);

    static bool isTwoToOneBalanced(
        const AdaptiveTsdfOctreeResult &octree);
};

} // namespace xjw::mesh
