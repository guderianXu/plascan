#include "metashape_texture/focus_score.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace metashape_texture::recovered {
namespace {

constexpr float kFloatMax = std::numeric_limits<float>::max();

float clamp_native(float value, float lower, float upper) noexcept {
    if (value <= upper) return std::max(lower, value);
    return upper;
}

bool valid_depth(float depth) noexcept {
    return depth != 0.0F && depth != kFloatMax;
}

float depth_coordinate(float depth, float minimum, float maximum, int bin_count) noexcept {
    const float clipped = std::min(maximum, std::max(minimum, depth));
    float span = maximum - minimum;
    if (span == 0.0F) span = 1.0F;
    return ((clipped - minimum) / span) * static_cast<float>(bin_count);
}

int depth_bin(float depth, float minimum, float maximum, int bin_count) noexcept {
    return std::clamp(static_cast<int>(depth_coordinate(depth, minimum, maximum, bin_count)),
                      0, bin_count - 1);
}

}  // namespace

FocusBinStatistics build_focus_bin_statistics(
    const std::vector<float> &depth,
    int depth_width,
    int depth_height,
    const std::vector<float> &value,
    const std::vector<std::uint8_t> &mask,
    int scale,
    const std::vector<std::uint8_t> *optional_mask,
    int bin_count) {
    if (depth_width <= 0 || depth_height <= 0 || scale <= 0 || bin_count <= 0 ||
        depth.size() != static_cast<std::size_t>(depth_width * depth_height) ||
        value.size() != static_cast<std::size_t>(depth_width * scale * depth_height * scale) ||
        mask.size() != value.size() ||
        (optional_mask != nullptr && optional_mask->size() != depth.size())) {
        throw std::invalid_argument("invalid focus bin-statistic dimensions");
    }

    FocusBinStatistics result;
    result.minimum = kFloatMax;
    result.maximum = -kFloatMax;
    for (const float sample : depth) {
        if (!valid_depth(sample)) continue;
        result.minimum = std::min(result.minimum, sample);
        result.maximum = std::max(result.maximum, sample);
    }
    if (result.minimum == kFloatMax) {
        result.minimum = 0.0F;
        result.maximum = 0.0F;
    }
    result.sums.assign(static_cast<std::size_t>(bin_count), 0.0F);
    result.counts.assign(static_cast<std::size_t>(bin_count), 0);
    if (optional_mask != nullptr) {
        result.optional_counts.assign(static_cast<std::size_t>(bin_count), 0);
    }

    const int value_width = depth_width * scale;
    const int value_height = depth_height * scale;
    for (int y = 0; y < value_height; ++y) {
        for (int x = 0; x < value_width; ++x) {
            const std::size_t value_index = static_cast<std::size_t>(y * value_width + x);
            const std::size_t depth_index = static_cast<std::size_t>(
                (y / scale) * depth_width + x / scale);
            const float sample_depth = depth[depth_index];
            if (!valid_depth(sample_depth) || mask[value_index] != 0) continue;
            const int index = depth_bin(sample_depth, result.minimum, result.maximum, bin_count);
            result.sums[static_cast<std::size_t>(index)] += std::abs(value[value_index]);
            ++result.counts[static_cast<std::size_t>(index)];
        }
    }
    if (optional_mask != nullptr) {
        for (std::size_t pixel = 0; pixel < depth.size(); ++pixel) {
            if (!valid_depth(depth[pixel]) || (*optional_mask)[pixel] != 0) continue;
            const int index = depth_bin(depth[pixel], result.minimum, result.maximum, bin_count);
            ++result.optional_counts[static_cast<std::size_t>(index)];
        }
    }
    return result;
}

std::vector<float> build_focus_base_weight(
    const std::vector<float> &depth,
    const FocusBinStatistics &statistics) {
    const int bin_count = static_cast<int>(statistics.sums.size());
    if (bin_count <= 0 || statistics.counts.size() != statistics.sums.size() ||
        (!statistics.optional_counts.empty() &&
         statistics.optional_counts.size() != statistics.sums.size())) {
        throw std::invalid_argument("invalid focus bin statistics");
    }

    std::vector<std::uint64_t> population(static_cast<std::size_t>(bin_count), 0);
    for (const float sample : depth) {
        if (!valid_depth(sample)) continue;
        const int index = depth_bin(sample, statistics.minimum, statistics.maximum, bin_count);
        ++population[static_cast<std::size_t>(index)];
    }

    const bool optional_present = !statistics.optional_counts.empty();
    const float lower_optional = optional_present ? 0.0000024999999F : 0.0F;
    const float upper_optional = optional_present ? 0.003F : 1.0F;
    float optional_span = upper_optional - lower_optional;
    if (optional_span == 0.0F) optional_span = 1.0F;

    std::vector<float> weights(static_cast<std::size_t>(bin_count));
    int active_first = std::numeric_limits<int>::max();
    int active_last = -1;
    for (int index = 0; index < bin_count; ++index) {
        const float denominator = std::max(
            static_cast<float>(population[static_cast<std::size_t>(index)]), 10000.0F);
        const float optional_value = optional_present
            ? static_cast<float>(statistics.optional_counts[static_cast<std::size_t>(index)])
            : denominator;
        const float count_ratio =
            static_cast<float>(statistics.counts[static_cast<std::size_t>(index)]) /
            denominator;
        const float clamped_count_1 =
            clamp_native(count_ratio, 0.0000099999997F, 0.012F);
        const float clamped_count_2 =
            clamp_native(count_ratio, 0.0000099999997F, 0.012F);
        const float squared_count = clamped_count_1 * clamped_count_2;
        const float optional_ratio = optional_value / denominator;
        const float clamped_optional =
            clamp_native(optional_ratio, lower_optional, upper_optional);
        const float combined = clamped_optional * squared_count;
        const float numerator = combined - lower_optional * 9.9999994e-11F;
        const float score = (numerator / 0.00014376012F) / optional_span;
        weights[static_cast<std::size_t>(index)] = score <= 0.001F ? 0.001F : score;

        const float active_threshold = (upper_optional * 0.00014400001F) / 3.0F;
        if (combined >= active_threshold) {
            active_first = std::min(active_first, index);
            active_last = std::max(active_last, index);
        }
    }

    int filled = 0;
    int interior_count = 0;
    if (active_last != -1 && active_first + 1 < active_last) {
        interior_count = active_last - (active_first + 1);
        for (int index = active_first + 1; index < active_last; ++index) {
            float left_max = 0.0F;
            for (int cursor = index; cursor >= active_first; --cursor) {
                left_max = std::max(left_max, weights[static_cast<std::size_t>(cursor)]);
            }
            float right_max = 0.0F;
            for (int cursor = index; cursor <= active_last; ++cursor) {
                right_max = std::max(right_max, weights[static_cast<std::size_t>(cursor)]);
            }
            const float replacement = std::min(1.0F, std::min(left_max, right_max));
            if (replacement > weights[static_cast<std::size_t>(index)]) {
                weights[static_cast<std::size_t>(index)] = replacement;
                ++filled;
            }
        }
    }
    const bool force_one = filled >= 2 && filled >= (interior_count + 1) / 3;

    std::vector<float> output(depth.size(), 0.0F);
    for (std::size_t pixel = 0; pixel < depth.size(); ++pixel) {
        if (!valid_depth(depth[pixel])) continue;
        if (force_one) {
            output[pixel] = 1.0F;
            continue;
        }
        const float coordinate = depth_coordinate(
            depth[pixel], statistics.minimum, statistics.maximum, bin_count);
        int left = static_cast<int>(coordinate - 0.5F);
        left = std::clamp(left, 0, bin_count - 1);
        const int right = std::clamp(left + 1, 0, bin_count - 1);
        const float fraction = clamp_native(
            (coordinate - 0.5F) - static_cast<float>(left), 0.0F, 1.0F);
        const float first = (1.0F - fraction) * weights[static_cast<std::size_t>(left)];
        const float second = fraction * weights[static_cast<std::size_t>(right)];
        output[pixel] = clamp_native(first + second, 0.0F, 1.0F);
    }
    return output;
}

std::vector<float> apply_focus_gradient_factor(
    const std::vector<float> &base_weight,
    float raw_factor) {
    const float factor = std::max(raw_factor, 0.1F);
    std::vector<float> output(base_weight.size());
    for (std::size_t pixel = 0; pixel < base_weight.size(); ++pixel) {
        output[pixel] = base_weight[pixel] * factor;
    }
    return output;
}

float compute_focus_gradient_factor(
    const std::vector<float> &base_weight,
    int base_width,
    int base_height,
    const std::vector<float> &grad2,
    int grad2_width,
    int grad2_height) {
    if (base_width <= 0 || base_height <= 0 || grad2_width < base_width ||
        grad2_height < base_height ||
        base_weight.size() != static_cast<std::size_t>(base_width) * base_height ||
        grad2.size() != static_cast<std::size_t>(grad2_width) * grad2_height) {
        throw std::invalid_argument("focus gradient-factor image dimensions do not match");
    }
    double sum = 0.0;
    std::int64_t selected_count = 0;
    for (int y = 0; y < base_height; ++y) {
        for (int x = 0; x < base_width; ++x) {
            if (base_weight[static_cast<std::size_t>(y) * base_width + x] <= 0.9F) continue;
            sum += static_cast<double>(grad2[static_cast<std::size_t>(y) * grad2_width + x]);
            ++selected_count;
        }
    }
    const double denominator = static_cast<double>(std::max<std::int64_t>(selected_count, 1));
    const float mean = static_cast<float>(sum / denominator);
    const float root = std::sqrt(mean);
    const float factor = root / 0.2F;
    return std::clamp(factor, 0.01F, 1.0F);
}

std::vector<std::uint8_t> quantize_focus_u8(const std::vector<float> &weight) {
    std::vector<std::uint8_t> output(weight.size());
    for (std::size_t pixel = 0; pixel < weight.size(); ++pixel) {
        const float value = weight[pixel];
        if (value == 0.0F) continue;
        const float scaled = clamp_native(value, 0.0F, 1.0F) * 255.0F;
        const int rounded = static_cast<int>(std::round(scaled));
        output[pixel] = static_cast<std::uint8_t>(std::max(rounded, 1));
    }
    return output;
}

}  // namespace metashape_texture::recovered
