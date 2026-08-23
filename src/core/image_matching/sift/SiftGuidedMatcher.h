#pragma once

/**
 * @file SiftGuidedMatcher.h
 * @brief 在已估计双视几何约束下补充 SIFT 对应。
 */

#include "../FeatureSet.h"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace xjw::image_matching
{

    struct SiftGuidedMatch
    {
        int index0 = -1;
        int index1 = -1;
        float confidence = 0.0f;
        float forwardRatio = 1.0f;
        float reverseRatio = 1.0f;
        float symmetricResidualPixels = -1.0f;
    };

    struct SiftGuidedMatchOptions
    {
        double epipolarThresholdPixels = 3.0;
        float maximumDescriptorRatio = 0.95f;
        // 保留该字段以兼容已有配置；空间索引会在整个极线带候选集中计算精确的前两名。
        int descriptorNeighbors = 8;
        int maximumAdditionalMatches = 5000;
        int spatialCellSizePixels = 64;
        int cancellationCheckInterval = 256;
        std::function<bool()> shouldCancel;
    };

    struct SiftGuidedMatchDiagnostics
    {
        std::uint64_t spatialCandidates = 0;
        std::uint64_t descriptorComparisons = 0;
    };

    struct SiftGuidedMatchResult
    {
        std::vector<SiftGuidedMatch> matches;
        SiftGuidedMatchDiagnostics diagnostics;
        bool canceled = false;
    };

    SiftGuidedMatchResult findGuidedSiftMatchesDetailed(const FeatureSet& features0,
                                                        const FeatureSet& features1,
                                                        const std::array<double, 9>& fundamental,
                                                        const std::vector<int>& existingFeatureIds0,
                                                        const std::vector<int>& existingFeatureIds1,
                                                        const SiftGuidedMatchOptions& options = {});

    std::vector<SiftGuidedMatch> findGuidedSiftMatches(const FeatureSet& features0,
                                                       const FeatureSet& features1,
                                                       const std::array<double, 9>& fundamental,
                                                       const std::vector<int>& existingFeatureIds0,
                                                       const std::vector<int>& existingFeatureIds1,
                                                       const SiftGuidedMatchOptions& options = {});

} // namespace xjw::image_matching
