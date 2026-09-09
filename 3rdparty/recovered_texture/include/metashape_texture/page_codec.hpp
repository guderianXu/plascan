#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace metashape_texture::page_codec {

// Reproduce Metashape 2.3.2's temporary texture-page boundary: encode RGB/U8
// as 128x128 tiled YCbCr JPEG-in-TIFF at quality 100, then decode it back to
// RGB/U8 with the same libtiff/libjpeg-turbo ABI.
std::vector<std::uint8_t> jpeg_tiff_roundtrip_rgb(
    std::span<const std::uint8_t> rgb, int width, int height,
    const std::filesystem::path &temporary_tiff);

// Write the final page with Metashape 2.3.2's lossless TIFF layout: RGB/U8,
// LZW compression, 32 rows per strip, and no horizontal predictor.
void write_lzw_tiff_rgb(std::span<const std::uint8_t> rgb, int width, int height,
                        const std::filesystem::path &output_tiff);

}  // namespace metashape_texture::page_codec
