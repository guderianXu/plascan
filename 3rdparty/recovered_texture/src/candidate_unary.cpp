#include "metashape_texture/candidate_unary.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace metashape_texture::recovered {
namespace {

constexpr float kSentinel = -128.0F;
constexpr float kPartialResolution = -1.0F;
// Linux 2.3.2 embeds 0x3e4ccccc here, one binary32 ULP below the
// nearest-float spelling 0.2F (0x3e4ccccd).
constexpr float kResolutionRangeFraction =
    std::bit_cast<float>(std::uint32_t{0x3e4cccccU});

float clamp01(float value) noexcept {
    if (value < 0.0F) return 0.0F;
    if (value > 1.0F) return 1.0F;
    return value;
}

struct Bounds {
    float range = 1.0F;
    float lower = 0.0F;
};

template <class Getter>
Bounds quality_bounds(const std::vector<FaceCameraQuality> &options,
                      Getter get, float minimum_range) {
    float minimum = std::numeric_limits<float>::max();
    float maximum = -std::numeric_limits<float>::max();
    for (const auto &option : options) {
        minimum = std::min(minimum, get(option));
        maximum = std::max(maximum, get(option));
    }
    float range = maximum - minimum;
    if (range <= minimum_range) range = minimum_range;
    const float lower_candidate = maximum - range;
    return {range, lower_candidate <= minimum ? lower_candidate : minimum};
}

Bounds resolution_bounds(const std::vector<FaceCameraQuality> &options) {
    float minimum = std::numeric_limits<float>::max();
    float maximum = 0.0F;
    for (const auto &option : options) {
        if (option.resolution == kSentinel || option.resolution == kPartialResolution) continue;
        minimum = std::min(minimum, option.resolution);
        maximum = std::max(maximum, option.resolution);
    }
    float range = maximum - minimum;
    const float fractional_range = maximum * kResolutionRangeFraction;
    if (range <= fractional_range) range = fractional_range;
    const float lower_candidate = maximum - range;
    return {range, lower_candidate <= minimum ? lower_candidate : minimum};
}

float squared_quality_factor(float value, Bounds bounds) noexcept {
    const float normalized = clamp01((value - bounds.lower) / bounds.range);
    float factor = 1.0F - normalized * 0.9F;
    factor *= factor;
    return factor;
}

}  // namespace

std::vector<std::int32_t> build_face_camera_unary(
    const std::vector<FaceCameraQuality> &options,
    float face_weight,
    FaceCameraCostFlags flags) {
    if (options.empty()) return {};

    Bounds sharpness{};
    Bounds consistency{};
    Bounds resolution{};
    if (flags.sharpness) {
        sharpness = quality_bounds(options,
            [](const FaceCameraQuality &option) { return option.sharpness; }, 0.001F);
    }
    if (flags.photoconsistency) {
        consistency = quality_bounds(options,
            [](const FaceCameraQuality &option) { return option.photoconsistency; }, 0.15F);
    }
    if (flags.resolution) resolution = resolution_bounds(options);

    face_weight = std::clamp(face_weight, 0.1F, 1000.0F);
    std::vector<std::int32_t> result;
    result.reserve(options.size());
    for (const auto &option : options) {
        if ((flags.sharpness && option.sharpness == kSentinel) ||
            (flags.photoconsistency && option.photoconsistency == kSentinel) ||
            (flags.resolution && option.resolution == kSentinel) ||
            (flags.nadirness && option.nadirness == kSentinel)) {
            throw std::invalid_argument("enabled FaceCameraOption field contains -128 sentinel");
        }

        const float consistency_factor = flags.photoconsistency
            ? squared_quality_factor(option.photoconsistency, consistency) : 1.0F;
        const float sharpness_factor = flags.sharpness
            ? squared_quality_factor(option.sharpness, sharpness) : 1.0F;
        float resolution_factor = 1.0F;
        if (flags.resolution) {
            const float normalized = option.resolution == kPartialResolution
                ? 0.0F
                : clamp01((option.resolution - resolution.lower) / resolution.range);
            resolution_factor = 1.0F - normalized * 0.9F;
        }
        const float nadir_factor = flags.nadirness
            ? 1.0F - option.nadirness * 0.5F : 1.0F;

        float product = consistency_factor * face_weight;
        product *= sharpness_factor;
        product *= resolution_factor;
        product *= nadir_factor;
        const float scaled = product * 1000.0F;
        result.push_back(static_cast<std::int32_t>(std::round(scaled)));
    }
    return result;
}

}  // namespace metashape_texture::recovered
