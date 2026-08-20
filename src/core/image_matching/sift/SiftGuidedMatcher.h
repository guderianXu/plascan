#pragma once

/**
 * @file SiftGuidedMatcher.h
 * @brief 在已估计双视几何约束下补充 SIFT 对应。
 */

#include "../FeatureSet.h"

#include <array>
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
        int descriptorNeighbors = 8;
        int maximumAdditionalMatches = 5000;
    };

    std::vector<SiftGuidedMatch> findGuidedSiftMatches(const FeatureSet& features0,
                                                       const FeatureSet& features1,
                                                       const std::array<double, 9>& fundamental,
                                                       const std::vector<int>& existingFeatureIds0,
                                                       const std::vector<int>& existingFeatureIds1,
                                                       const SiftGuidedMatchOptions& options = {});

} // namespace xjw::image_matching
