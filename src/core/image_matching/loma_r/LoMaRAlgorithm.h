#pragma once

/**
 * @file LoMaRAlgorithm.h
 * @brief LoMa-R 的统一影像匹配算法适配器。
 */

#include "ImageMatchingAlgorithm.h"

#include <memory>

namespace xjw::image_matching
{

inline constexpr const char *kLoMaRAlgorithmId = "loma_r";
inline constexpr std::uint32_t kLoMaRAlgorithmVersion = 1;

class LoMaRAlgorithm final : public IImageMatchingAlgorithm
{
public:
    explicit LoMaRAlgorithm(ImageMatchingRuntimeConfig config);
    ~LoMaRAlgorithm() override;

    ImageMatchingAlgorithmDescriptor descriptor() const override;
    FeatureSet extract(const ImageFeatureInput &input) const override;
    MatchResult matchFeatures(const FeatureSet &features0,
                              const FeatureSet &features1) override;

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

void registerLoMaRAlgorithm();

} // namespace xjw::image_matching
