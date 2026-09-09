#include "metashape_texture/focus_grayscale.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace metashape_texture::recovered {
namespace {

std::size_t pixel_index(int x, int y, int channel, int width) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x)) * 3U + static_cast<std::size_t>(channel);
}

std::vector<float> filter_axis_symmetric(
    const std::vector<float> &source,
    int width,
    int height,
    const std::vector<float> &kernel,
    bool horizontal) {
    std::vector<float> output(source.size());
    const int radius = static_cast<int>(kernel.size() / 2U);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float value = source[pixel_index(x, y, channel, width)] *
                              kernel[static_cast<std::size_t>(radius)];
                for (int distance = 1; distance <= radius; ++distance) {
                    const int x0 = horizontal ? std::max(0, x - distance) : x;
                    const int x1 = horizontal ? std::min(width - 1, x + distance) : x;
                    const int y0 = horizontal ? y : std::max(0, y - distance);
                    const int y1 = horizontal ? y : std::min(height - 1, y + distance);
                    const float pair =
                        source[pixel_index(x0, y0, channel, width)] +
                        source[pixel_index(x1, y1, channel, width)];
                    value = value + pair * kernel[static_cast<std::size_t>(radius + distance)];
                }
                output[pixel_index(x, y, channel, width)] = value;
            }
        }
    }
    return output;
}

std::uint8_t normalized_float_to_u8(float value) {
    const float rounded = value * 255.0F + 0.5F;
    return static_cast<std::uint8_t>(std::clamp(static_cast<int>(rounded), 0, 255));
}

}  // namespace

std::vector<float> build_focus_gaussian_kernel(float sigma, int kernel_size) {
    if (!(sigma > 0.0F) || kernel_size < 0) {
        throw std::invalid_argument("invalid Gaussian kernel parameters");
    }
    if (kernel_size == 0) {
        kernel_size = static_cast<int>(static_cast<double>(sigma * 8.0F) + 1.5) | 1;
    }
    if ((kernel_size & 1) == 0) {
        throw std::invalid_argument("Gaussian kernel size must be odd");
    }

    const int radius = kernel_size / 2;
    std::vector<float> half(static_cast<std::size_t>(radius + 1));
    const float sigma_squared = sigma * sigma;
    const double exponent_factor = -0.5 / static_cast<double>(sigma_squared);
    double sum = -1.0;
    for (int distance = 0; distance <= radius; ++distance) {
        const double index = static_cast<double>(distance);
        const float sample = static_cast<float>(std::exp(index * exponent_factor * index));
        half[static_cast<std::size_t>(distance)] = sample;
        sum += static_cast<double>(sample * 2.0F);
    }
    const double inverse_sum = 1.0 / sum;

    std::vector<float> kernel(static_cast<std::size_t>(kernel_size));
    for (int distance = 0; distance <= radius; ++distance) {
        const float normalized = static_cast<float>(
            static_cast<double>(half[static_cast<std::size_t>(distance)]) * inverse_sum);
        kernel[static_cast<std::size_t>(radius - distance)] = normalized;
        kernel[static_cast<std::size_t>(radius + distance)] = normalized;
    }
    return kernel;
}

FocusGrayCacheImage build_focus_gray_cache_d2(
    const std::vector<std::uint8_t> &rgb,
    int width,
    int height) {
    if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0 ||
        rgb.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U) {
        throw std::invalid_argument("invalid RGB image for focus gray d2 path");
    }

    std::vector<float> normalized(rgb.size());
    for (std::size_t index = 0; index < rgb.size(); ++index) {
        normalized[index] = static_cast<float>(rgb[index]) / 255.0F;
    }
    const auto kernel = build_focus_gaussian_kernel(0.55F);
    auto filtered = filter_axis_symmetric(normalized, width, height, kernel, true);
    filtered = filter_axis_symmetric(filtered, width, height, kernel, false);

    std::vector<std::uint8_t> filtered_u8(filtered.size());
    std::transform(filtered.begin(), filtered.end(), filtered_u8.begin(), normalized_float_to_u8);

    FocusGrayCacheImage result;
    result.width = width / 2;
    result.height = height / 2;
    result.pixels.resize(static_cast<std::size_t>(result.width) *
                         static_cast<std::size_t>(result.height));
    for (int y = 0; y < result.height; ++y) {
        for (int x = 0; x < result.width; ++x) {
            std::uint8_t downscaled[3]{};
            for (int channel = 0; channel < 3; ++channel) {
                unsigned int sum = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        sum += filtered_u8[pixel_index(x * 2 + dx, y * 2 + dy, channel, width)];
                    }
                }
                downscaled[channel] = static_cast<std::uint8_t>((sum + 2U) / 4U);
            }
            const double gray =
                static_cast<double>(downscaled[1]) * 0.587 +
                static_cast<double>(downscaled[0]) * 0.299 +
                static_cast<double>(downscaled[2]) * 0.114;
            result.pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(result.width) +
                          static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(gray);
        }
    }
    return result;
}

std::vector<float> build_focus_gray_upload(
    const std::vector<std::uint8_t> &decoded_gray) {
    std::vector<float> output(decoded_gray.size());
    for (std::size_t index = 0; index < decoded_gray.size(); ++index) {
        output[index] = static_cast<float>(std::max<std::uint8_t>(decoded_gray[index], 1U)) /
                        255.0F;
    }
    return output;
}

}  // namespace metashape_texture::recovered
