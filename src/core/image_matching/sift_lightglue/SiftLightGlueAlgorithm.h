#pragma once

/**
 * @file SiftLightGlueAlgorithm.h
 * @brief CUDA SIFT + TensorRT LightGlue 的组合算法实现。
 */

#include "ImageMatchingAlgorithm.h"
#include "lightglue/TensorRtLightGlueMatcher.h"

#include <memory>

namespace xjw::image_matching
{

/// 持久化与 CLI 共用的稳定算法 ID，不能包含设备或精度等运行参数。
inline constexpr const char *kSiftLightGlueAlgorithmId = "sift_lightglue";

/// 算法输出语义变化时递增；门限变化由 configFingerprint 区分。
inline constexpr std::uint32_t kSiftLightGlueAlgorithmVersion = 1;

/**
 * @brief 可复用特征模式下的 CUDA SIFT + TensorRT LightGlue 算法。
 *
 * extract() 可以在多个候选像对之间复用单幅影像特征；matcher 延迟创建，避免
 * 只做候选规划或缓存命中时加载 TensorRT engine 和占用显存。
 */
class SiftLightGlueAlgorithm final : public IImageMatchingAlgorithm
{
public:
    explicit SiftLightGlueAlgorithm(ImageMatchingRuntimeConfig config);
    ~SiftLightGlueAlgorithm() override;

    ImageMatchingAlgorithmDescriptor descriptor() const override;
    FeatureSet extract(const ImageFeatureInput &input) const override;
    MatchResult matchFeatures(const FeatureSet &features0,
                              const FeatureSet &features1) override;

private:
    void ensureMatcher();

    ImageMatchingRuntimeConfig _config;
    std::unique_ptr<TensorRtLightGlueMatcher> _matcher;
};

/// 由 ImageMatchingRegistry 的一次性内建注册流程调用。
void registerSiftLightGlueAlgorithm();

} // namespace xjw::image_matching
