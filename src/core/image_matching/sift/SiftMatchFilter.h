#pragma once

/**
 * @file SiftMatchFilter.h
 * @brief 将 SIFT 双向最近邻结果转换为稳定的一一对应。
 */

#include "../MatchResult.h"

#include <vector>

namespace xjw::image_matching
{

    struct SiftNearestMatch
    {
        int index = -1;
        float similarity = 0.0f;
        float ambiguity = 1.0f;
    };

    struct SiftMatchFilterOptions
    {
        float confidenceThreshold = 0.15f;
        float maximumRatio = 0.98f;
        float minimumAdaptiveRatio = 0.78f;
        bool adaptiveRatio = true;
        int sparseCandidateCount = 64;
    };

    /// 保留双向一致的对应，并按当前像对的 ratio 分布自适应抑制歧义描述子。
    MatchResult filterSiftMutualMatches(const std::vector<SiftNearestMatch>& forward,
                                        const std::vector<SiftNearestMatch>& reverse,
                                        const SiftMatchFilterOptions& options = {});

} // namespace xjw::image_matching
