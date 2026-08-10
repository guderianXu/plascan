#pragma once

/**
 * @file CudaSiftMatchFilter.h
 * @brief 将 CUDA SIFT 双向最近邻结果转换为稳定的一一对应。
 */

#include "../MatchResult.h"

#include <vector>

namespace xjw::image_matching
{

struct CudaSiftNearestMatch
{
    int index = -1;
    float similarity = 0.0f;
    float ambiguity = 1.0f;
};

/**
 * @brief 保留双向一致且达到统一置信度门限的 SIFT 匹配。
 *
 * 双向最近邻已经提供一一一致性约束，matchThreshold 直接解释为归一化
 * SIFT 描述子相似度下限。ambiguity 不再二次压低置信度，避免大分辨率
 * 航片中的重复纹理在 USAC 几何验证前被过度删除。
 */
MatchResult filterCudaSiftMutualMatches(
    const std::vector<CudaSiftNearestMatch> &forward,
    const std::vector<CudaSiftNearestMatch> &reverse,
    float confidenceThreshold);

} // namespace xjw::image_matching
