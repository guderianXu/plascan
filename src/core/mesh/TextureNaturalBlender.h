#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>

#include <opencv2/core.hpp>

namespace xjw::mesh::texture_v4
{
    inline constexpr int kTextureBlendLevels = 5;

    struct TextureSourcePyramid
    {
        // Level zero comes directly from the validated source sample. Successive
        // levels are four times coarser, independent of the final UV placement.
        std::array<cv::Mat, kTextureBlendLevels> lowFrequency;
        std::array<cv::Mat, kTextureBlendLevels> weight;
    };

    struct TexturePyramidSample
    {
        const TextureSourcePyramid* pyramid = nullptr;
        cv::Point2d pixel;
        cv::Vec3f encodedColor{};
        float confidence = 1.0f;
        bool primary = false;
    };

    TextureSourcePyramid buildTextureSourcePyramid(const cv::Mat& image,
                                                   const cv::Mat& support,
                                                   const cv::Mat& winner,
                                                   float exposure_gain = 1.0f,
                                                   const std::function<bool()>& is_cancelled = {});

    cv::Vec3f textureSrgbToLinear(const cv::Vec3f& encoded);
    cv::Vec3b textureLinearToSrgb(const cv::Vec3f& linear);

    // Samples only supported taps of the 2x2 bilinear footprint.  A supported
    // black texel is valid; unsupported atlas background never darkens a seam.
    bool sampleSupportedBilinear(const cv::Mat& image,
                                 const cv::Mat& support,
                                 double x,
                                 double y,
                                 cv::Vec3f* color);

    // Returns linear RGB/BGR (same channel order as input) for supersampling.
    cv::Vec3f blendTexturePyramidSamples(std::span<const TexturePyramidSample> samples,
                                         bool filter_ghosts,
                                         float ghost_threshold,
                                         std::uint64_t* rejected_count = nullptr);
} // namespace xjw::mesh::texture_v4
