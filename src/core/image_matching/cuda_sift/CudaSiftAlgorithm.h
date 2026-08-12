#pragma once

/**
 * @file CudaSiftAlgorithm.h
 * @brief CUDA SIFT 特征提取与全量 GPU 描述子匹配算法。
 */

#include "../ImageMatchingAlgorithm.h"

namespace xjw::image_matching
{

inline constexpr const char *kCudaSiftAlgorithmId = "cuda_sift";
inline constexpr std::uint32_t kCudaSiftAlgorithmVersion = 3;

class CudaSiftAlgorithm final : public IImageMatchingAlgorithm
{
public:
    explicit CudaSiftAlgorithm(ImageMatchingRuntimeConfig config);

    ImageMatchingAlgorithmDescriptor descriptor() const override;
    FeatureSet extract(const ImageFeatureInput &input) const override;
    MatchResult matchFeatures(const FeatureSet &features0,
                              const FeatureSet &features1) override;

private:
    ImageMatchingRuntimeConfig _config;
};

void registerCudaSiftAlgorithm();

} // namespace xjw::image_matching
