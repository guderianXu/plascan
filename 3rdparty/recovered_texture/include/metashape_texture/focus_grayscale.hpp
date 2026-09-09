#pragma once

#include <cstdint>
#include <vector>

namespace metashape_texture::recovered {

struct FocusGrayCacheImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

// Exact Metashape 2.3.1 Gaussian constructor used by sub_1410768A0.
std::vector<float> build_focus_gaussian_kernel(float sigma, int kernel_size = 0);

// Recovered out-of-focus input producer for the observed RGB/U8, downscale=2
// path: U8 -> normalized F32 -> separable Gaussian -> U8 -> 2x2 average -> L/U8.
FocusGrayCacheImage build_focus_gray_cache_d2(
    const std::vector<std::uint8_t> &rgb,
    int width,
    int height);

// Exact upload conversion after Metashape decodes the quality-90 gray JPEG.
std::vector<float> build_focus_gray_upload(
    const std::vector<std::uint8_t> &decoded_gray);

}  // namespace metashape_texture::recovered
