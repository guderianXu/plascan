#pragma once

/**
 * @file AutoSiftAlgorithm.h
 * @brief 跨平台 SIFT 算法入口，自动选择可用的提取和匹配后端。
 */

#include "../ImageMatchingAlgorithm.h"

namespace xjw::image_matching
{

    inline constexpr const char* kAutoSiftAlgorithmId = "auto_sift";
    inline constexpr std::uint32_t kAutoSiftAlgorithmVersion = 3;

    class AutoSiftAlgorithm final : public IImageMatchingAlgorithm
    {
    public:
        explicit AutoSiftAlgorithm(ImageMatchingRuntimeConfig config);

        ImageMatchingAlgorithmDescriptor descriptor() const override;
        FeatureSet extract(const ImageFeatureInput& input) const override;
        MatchResult matchFeatures(const FeatureSet& features0, const FeatureSet& features1) override;

    private:
        ImageMatchingRuntimeConfig _config;
    };

    void registerAutoSiftAlgorithm();

} // namespace xjw::image_matching
