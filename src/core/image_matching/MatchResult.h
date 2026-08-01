#pragma once

/**
 * @file MatchResult.h
 * @brief 匹配器原始输出；仅在算法执行和几何验证之间短暂存在。
 */

#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

namespace xjw::image_matching
{

struct MatchResult
{
    std::vector<int> matches0; ///< image0 特征到 image1 特征的索引，未匹配为 -1。
    std::vector<int> matches1; ///< image1 特征到 image0 特征的反向索引，未匹配为 -1。
    std::vector<float> matchingScores0; ///< 与 matches0 同长的 LightGlue 置信度。
    std::vector<float> matchingScores1; ///< 与 matches1 同长的反向置信度。
    std::vector<cv::DMatch> cvMatches; ///< 互检后的稀疏对应，供几何验证使用。
    int numMatches = 0; ///< cvMatches 的稳定计数，序列化前会再次校验。
    std::string sourceAlgorithm; ///< 运行诊断名称，不作为持久化缓存键。

    /// 没有任何互检对应时返回 true。
    bool empty() const;

    /// 从双向索引和置信度重建唯一 cv::DMatch 列表。
    void buildCvMatchesFromIndices();
};

} // namespace xjw::image_matching
