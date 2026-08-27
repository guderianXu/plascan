#pragma once

/**
 * @file OrbBinaryAlgorithm.h
 * @brief OpenCV ORB clean-room baseline for binary feature experiments.
 */

#include "../ImageMatchingAlgorithm.h"

namespace xjw::image_matching
{

inline constexpr const char *kOrbBinaryAlgorithmId = "orb_binary";
inline constexpr std::uint32_t kOrbBinaryAlgorithmVersion = 1;

/**
 * @brief 可选的 ORB 二进制基线，不代表任何商业软件的私有实现。
 *
 * 该实现用于评估二进制描述子在摄影测量粗匹配中的内存、速度和召回；默认
 * 生产算法仍是 Auto SIFT。匹配采用 Hamming 双向 top-2、ratio 和 mutual gate。
 */
class OrbBinaryAlgorithm final : public IImageMatchingAlgorithm
{
public:
    explicit OrbBinaryAlgorithm(ImageMatchingRuntimeConfig config);

    ImageMatchingAlgorithmDescriptor descriptor() const override;
    FeatureSet extract(const ImageFeatureInput &input) const override;
    MatchResult matchFeatures(const FeatureSet &features0,
                              const FeatureSet &features1) override;

private:
    ImageMatchingRuntimeConfig _config;
};

void registerOrbBinaryAlgorithm();

} // namespace xjw::image_matching
