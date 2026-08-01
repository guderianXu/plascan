#pragma once

/**
 * @file MatchGeometryVerifier.h
 * @brief 原始匹配的鲁棒两视几何验证和逐点残差计算。
 */

#include "FeatureSet.h"
#include "ImageMatchTypes.h"
#include "MatchResult.h"

#include <array>
#include <vector>

namespace xjw::image_matching
{

struct MatchGeometryOptions
{
    GeometryModel model = GeometryModel::Fundamental; ///< 两视鲁棒估计使用的几何模型。
    double reprojectionThresholdPixels = 1.5; ///< USAC/RANSAC 像素内点门限。
    double confidence = 0.9999; ///< 随机采样达到有效模型的目标置信度。
    int maximumIterations = 10000; ///< 防止弱纹理像对无限采样的迭代上限。
    int minimumInliers = 20; ///< 像对进入连接点网络所需的最少内点。
    int randomSeed = 0; ///< 固定随机种子，保证临界像对可复现。
};

struct MatchGeometryResult
{
    bool modelEstimated = false;
    bool passed = false;
    GeometryModel model = GeometryModel::None;
    std::array<double, 9> matrix{};
    std::vector<bool> inlierMask; ///< 顺序与 MatchResult::cvMatches 完全一致。
    std::vector<float> residualPixels; ///< 非法观测为 -1；外点也保留真实残差。
    int validInputCount = 0;
    int inlierCount = 0;
};

class MatchGeometryVerifier
{
public:
    /**
     * @brief 验证原始匹配并计算逐点几何残差。
     *
     * 返回数组始终与输入 cvMatches 等长；非法索引和无法估计模型的情况不会
     * 改变对应次序，便于写入 `.pimatch` 后由 GUI 精确显示外点原因。
     */
    static MatchGeometryResult verify(const MatchResult &matches,
                                      const FeatureSet &features0,
                                      const FeatureSet &features1,
                                      const MatchGeometryOptions &options);
};

} // namespace xjw::image_matching
