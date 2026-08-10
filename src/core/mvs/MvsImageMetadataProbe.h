#pragma once

#include <string>

namespace xjw::mvs
{

struct MvsImageMetadata
{
    int width = 0;
    int height = 0;
};

/// Reads raster dimensions without requesting any pixel block from the image.
bool probeMvsImageMetadata(const std::string &imagePath,
                           MvsImageMetadata *metadata,
                           std::string *errorMessage = nullptr);

} // namespace xjw::mvs
