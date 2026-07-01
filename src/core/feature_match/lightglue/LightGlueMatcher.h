#pragma once

/**
 * @file    LightGlueMatcher.h
 * @brief   LightGlue 特征匹配封装类
 *
 * 实现 IFeatureMatcher 接口，支持两种调用方式：
 *   1. match(FeatureData, FeatureData)  —— 通用接口，直接对接项目已有 .sp 特征数据
 *   2. match(cv::Mat, cv::Mat)          —— 使用内置 SP 模型从原始图像端到端匹配
 *   3. extractNative(cv::Mat)           —— 使用内置 SP 模型提取特征
 *
 * 模型路径通过 LightGlueConfig 指定：
 *   - DISK/ALIKED/SIFT: resources/models/lightglue_{disk,aliked,sift}_{cpu,cuda}.torchscript
 *   - SuperPoint compatibility: resources/models/lightglue_matcher_{cpu,cuda}.torchscript
 *
 * SuperPoint 提取器模型路径可选配置，仅用于原始图像端到端匹配。
 *
 * 依赖： LibTorch、OpenCV
 */

#include "../match.h"

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace xjw::feature_match
{

/**
 * @brief LightGlue 匹配器配置。
 */
struct LightGlueConfig
{
    /// LightGlue TorchScript 匹配器模型路径（描述子维度必须与模型匹配）
    std::string matcherModelPath;
    /// 包含 SP 提取器的 TorchScript 模型路径（可选，仅 matchImages / extractNative 需要）
    std::string spModelPath;
    /// 是否使用 CUDA，默认 true（不可用时自动回退 CPU）
    bool useCuda = true;
    /// mutual-NN 置信度閘值（越高越严格，默认 0.2，与 SuperGlue 对齐）
    float scoreThreshold = 0.2f;
};

/**
 * @brief LightGlue 匹配器。
 *
 * 支持两个工作模式：
 *  - 通用模式: match(FeatureData, FeatureData)
 *    传入已有特征（关键点像素坐标 + [N,D] 描述子；DISK/ALIKED 为 128 维，
 *    SuperPoint 为 256 维），内部构造 [1,N,2] 坐标张量和 [1,N,D] 描述子张量送入 LightGlue。
 *  - 原始图像模式: matchImages(cv::Mat, cv::Mat)
 *    需要额外配置 SP 模型路径，从 BGR 图像直接匹配。
 */
class LightGlueMatcher : public IFeatureMatcher
{
public:
    /**
     * @brief 构造并加载 LightGlue 匹配器模型。
     * @throws std::runtime_error 加载失败
     */
    explicit LightGlueMatcher(const LightGlueConfig &config);

    // ========== IFeatureMatcher 接口 ==========

    /**
     * @brief 通用匹配接口：从 keypoints/descriptors 构造匹配输入张量。
     *
     * FeatureData::imageWidth/Height 用于提供图像尺寸信息（如果为 0 则尝试从坐标范围推断）。
     */
    xjw::feature_match::MatchResult match(
        const xjw::feature_extractors::FeatureData &feat0,
        const xjw::feature_extractors::FeatureData &feat1) override;

    std::string algorithmName() const override { return "lightglue"; }

    // ========== 内置 SP 模型接口 ==========

    /**
     * @brief 使用内置 SP 模型提取单幅图像特征。
     *        需要 LightGlueConfig::spModelPath 已设置。
     */
    xjw::feature_extractors::FeatureData extractNative(const cv::Mat &img) const;

    /**
     * @brief 从两幅 BGR 图像端到端匹配。
     *        需要 LightGlueConfig::spModelPath 已设置。
     */
    xjw::feature_match::MatchResult matchImages(const cv::Mat &img0, const cv::Mat &img1);

    bool isLoaded() const { return _matcherLoaded; }

private:
    /**
     * @brief 从内部 tensor scores 提取 mutual-NN 匹配对和置信度。
     * @return vector<{idx0, score}>，其中 idx1 = m0[idx0]
     */
    std::vector<std::pair<int64_t, float>> _filterScores(
        const torch::Tensor &scores,
        int64_t expectedN0,
        int64_t expectedN1) const;

    /**
     * @brief 将 BGR Mat 转为归一化 float32 Tensor [1,3,H,W] 并移至设备。
     */
    torch::Tensor _matToTensor(const cv::Mat &bgr) const;

    mutable torch::jit::script::Module _lgModel; ///< LightGlue 匹配器模型
    mutable torch::jit::script::Module _spModel; ///< SuperPoint 提取器模型（可选）
    torch::Device      _torchDevice;
    LightGlueConfig    _config;
    bool               _matcherLoaded = false;
    bool               _spLoaded = false;
};

} // namespace xjw::feature_match
