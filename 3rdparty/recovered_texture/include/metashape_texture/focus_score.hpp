#pragma once

#include <cstdint>
#include <vector>

namespace metashape_texture::recovered {

struct FocusBinStatistics {
    float minimum = 0.0F;
    float maximum = 0.0F;
    std::vector<float> sums;
    std::vector<std::int32_t> counts;
    std::vector<std::uint64_t> optional_counts;
};

// Recovered host worker semantics feeding sub_141905500.  Depth is the
// 160x120 metric-depth framebuffer.  Value and mask are the 320x240 DoG0 and
// 2700-shader outputs; each depth sample therefore covers scale*scale values.
FocusBinStatistics build_focus_bin_statistics(
    const std::vector<float> &depth,
    int depth_width,
    int depth_height,
    const std::vector<float> &value,
    const std::vector<std::uint8_t> &mask,
    int scale,
    const std::vector<std::uint8_t> *optional_mask = nullptr,
    int bin_count = 20);

// Exact float32 translation of Metashape 2.3.1 sub_141905500.
std::vector<float> build_focus_base_weight(
    const std::vector<float> &depth,
    const FocusBinStatistics &statistics);

// Exact host loop immediately before sub_141F40510.  It scans the base image
// in row-major order, selects base > 0.9f, and averages the corresponding
// top-left samples of the full-resolution Sobel grad2 image.  The sum is
// binary64, the mean narrows to binary32 before sqrtf, and the result is
// clamped to [0.01f, 1.0f].
float compute_focus_gradient_factor(
    const std::vector<float> &base_weight,
    int base_width,
    int base_height,
    const std::vector<float> &grad2,
    int grad2_width,
    int grad2_height);

// Exact sub_141F40510 consumer: base * max(raw_factor, 0.1f).
std::vector<float> apply_focus_gradient_factor(
    const std::vector<float> &base_weight,
    float raw_factor);

// Exact zero-preserving U8 conversion used by sub_141E8A7D0.
std::vector<std::uint8_t> quantize_focus_u8(
    const std::vector<float> &weight);

}  // namespace metashape_texture::recovered
