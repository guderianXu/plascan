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

    /**
     * @brief 保留双向一致且达到统一置信度门限的 SIFT 匹配。
     *
     * 初始匹配保留互为最近邻的候选，并以宽松歧义门限删除明显重复描述子；
     * 后续再由 USAC 和引导重匹配处理摄影测量场景中的重复纹理。
     */
    MatchResult filterSiftMutualMatches(const std::vector<SiftNearestMatch>& forward,
                                        const std::vector<SiftNearestMatch>& reverse,
                                        float confidenceThreshold,
                                        float maximumAmbiguity = 0.98f);

} // namespace xjw::image_matching
