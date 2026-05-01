#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace xjw::feature_extractors
{
struct FeatureData;
}

namespace xjw::feature_match
{

/**
 * @brief 通用特征匹配结果接口。
 *
 * 统一表示 SuperGlue、LightGlue 以及传统匹配器（ORB/SIFT/BF 等）的匹配输出，
 * 供 SfM 流水线、可视化模块等下游统一消费。
 *
 * 字段约定：
 *  - matches0[i]       : 图像0第 i 个关键点在图像1中的匹配索引，-1 表示无匹配。
 *  - matches1[j]       : 图像1第 j 个关键点在图像0中的匹配索引，-1 表示无匹配。
 *  - matchingScores0/1 : 对应关键点的匹配置信度 [0,1]，传统算法由距离换算。
 *  - cvMatches         : OpenCV DMatch 格式，供 OpenCV 滤波、可视化接口使用。
 *  - numMatches        : 有效匹配数（等于 cvMatches.size()）。
 *  - sourceAlgorithm   : 小写算法名，如 "superglue"/"lightglue"/"orb"/"sift"。
 */
struct MatchResult
{
    // ------------------------------------------------------------------
    // 索引格式（DL 匹配器原生输出）
    // ------------------------------------------------------------------
    std::vector<int> matches0;           ///< 图像0关键点 -> 图像1索引（-1=无匹配）
    std::vector<int> matches1;           ///< 图像1关键点 -> 图像0索引（-1=无匹配）
    std::vector<float> matchingScores0;  ///< 图像0各匹配点置信度
    std::vector<float> matchingScores1;  ///< 图像1各匹配点置信度

    // ------------------------------------------------------------------
    // OpenCV 兼容格式（传统匹配器原生输出，可由索引格式生成）
    // ------------------------------------------------------------------
    std::vector<cv::DMatch> cvMatches;   ///< DMatch 列表

    // ------------------------------------------------------------------
    // 元信息
    // ------------------------------------------------------------------
    int numMatches = 0;                  ///< 有效匹配数
    std::string sourceAlgorithm;         ///< 来源算法名（小写）

    MatchResult() = default;

    /// @brief 是否为无效/空结果
    bool empty() const { return numMatches == 0; }

    // ------------------------------------------------------------------
    // 辅助方法
    // ------------------------------------------------------------------

    /**
     * @brief 由 matches0 / matchingScores0 填充 cvMatches。
     *
     * 适用于 SuperGlue、LightGlue 等输出索引数组的 DL 匹配器。
     * 调用后 cvMatches 与 numMatches 将被重建，已有值会被覆盖。
     */
    void buildCvMatchesFromIndices();

    /**
     * @brief 由 cvMatches 反向填充 matches0 / matches1 / matchingScores0/1。
     *
     * 适用于传统匹配器，将 DMatch 结果归一化到统一格式。
     * distance 以 1/(1+d) 映射为置信度。
     *
     * @param numKeypoints0  图像0关键点总数，用于分配 matches0 数组。
     * @param numKeypoints1  图像1关键点总数，用于分配 matches1 数组。
     */
    void buildIndicesFromCvMatches(int numKeypoints0, int numKeypoints1);

    // ------------------------------------------------------------------
    // 工厂方法
    // ------------------------------------------------------------------

    /**
     * @brief 从 DMatch 列表构建完整 MatchResult（传统匹配器适配）。
     *
     * @param matches        cv::DMatch 列表。
     * @param numKeypoints0  图像0关键点总数。
     * @param numKeypoints1  图像1关键点总数。
     * @param algoName       算法名，如 "orb"/"sift"。
     */
    static MatchResult fromCvMatches(const std::vector<cv::DMatch> &matches,
                                     int numKeypoints0,
                                     int numKeypoints1,
                                     const std::string &algoName = "");

    /**
     * @brief 从 matches0/matches1 索引数组构建（DL 匹配器适配）。
     *
     * @param matches0  图像0关键点到图像1的索引映射（-1=无匹配）。
     * @param matches1  图像1关键点到图像0的索引映射（-1=无匹配）。
     * @param scores0   图像0各关键点匹配得分。
     * @param scores1   图像1各关键点匹配得分。
     * @param algoName  算法名，如 "superglue"/"lightglue"。
     */
    static MatchResult fromIndices(const std::vector<int> &matches0,
                                   const std::vector<int> &matches1,
                                   const std::vector<float> &scores0,
                                   const std::vector<float> &scores1,
                                   const std::string &algoName = "");
};

// ---------------------------------------------------------------------------

/**
 * @brief 通用特征匹配器抽象接口。
 *
 * 所有具体匹配器（SuperGlue、LightGlue、传统 BF/FLANN 等）均应实现此接口，
 * 以便在流水线中无感知地切换匹配算法。
 *
 * 输入：一对 FeatureData（由特征提取器产生）。
 * 输出：统一的 MatchResult。
 *
 * 典型用法：
 * @code
 *     std::unique_ptr<IFeatureMatcher> matcher = createMatcher(algoName, config);
 *     MatchResult result = matcher->match(feat0, feat1);
 * @endcode
 */
class IFeatureMatcher
{
public:
    virtual ~IFeatureMatcher() = default;

    /**
     * @brief 对两张图像的特征数据执行匹配。
     *
     * @param feat0  图像0的特征数据（keypoints + descriptors）。
     * @param feat1  图像1的特征数据（keypoints + descriptors）。
     * @return       统一格式的匹配结果，包含有效匹配数与 cvMatches。
     */
    virtual MatchResult match(const xjw::feature_extractors::FeatureData &feat0,
                              const xjw::feature_extractors::FeatureData &feat1) = 0;

    /// @brief 返回匹配器的算法名称（小写），如 "superglue"/"lightglue"/"orb"。
    virtual std::string algorithmName() const = 0;
};

} // namespace xjw::feature_match