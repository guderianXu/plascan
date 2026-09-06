#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace metalign {

struct Image {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<float> gray;
    std::vector<std::uint8_t> rgb;
    std::optional<double> focal_length_35mm;

    float& at(std::size_t x, std::size_t y) { return gray[y * width + x]; }
    float at(std::size_t x, std::size_t y) const { return gray[y * width + x]; }
    bool empty() const { return width == 0 || height == 0 || gray.empty(); }
};

Image load_image(const std::filesystem::path& path, bool parallel_gray = false);
std::uint32_t jpeg_decoder_version_number();
// Metashape Accuracy=Highest lattice expansion: source samples occupy even
// output coordinates and the gaps are filled by 2-/4-neighbour float means.
// The resulting size is (2W-1)x(2H-1), not a generic 2W x 2H resize.
Image upsample_highest(const Image& source, bool parallel_rows = false);
Image gaussian_blur(const Image& source, double sigma, bool parallel_rows = false);
Image resize_bilinear(const Image& source, std::size_t width, std::size_t height);
Image downsample_half(const Image& source);
float sample_bilinear(const Image& image, double x, double y);
std::vector<std::filesystem::path> collect_images(const std::filesystem::path& directory);

}  // namespace metalign
