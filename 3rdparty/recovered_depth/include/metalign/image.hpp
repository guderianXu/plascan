#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace metalign
{
struct Image
{
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<float> gray;
    std::vector<std::uint8_t> rgb;
    std::optional<double> focal_length_35mm;

    float &at(std::size_t x, std::size_t y)
    {
        return gray[y * width + x];
    }

    float at(std::size_t x, std::size_t y) const
    {
        return gray[y * width + x];
    }

    bool empty() const
    {
        return width == 0 || height == 0 || gray.empty();
    }
};

Image load_gray_image(const std::filesystem::path &path, bool parallel_gray = false);
float sample_bilinear(const Image &image, double x, double y);
} // namespace metalign
