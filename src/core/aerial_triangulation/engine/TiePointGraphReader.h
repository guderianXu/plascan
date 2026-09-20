#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "engine/TiePointGraph.h"

namespace xjw::aerial_triangulation::engine
{
    /// Optional resolver adapts archived project path tokens without coupling JSON parsing to project/UI types.
    using ImageTokenResolver = std::function<int(std::string_view)>;
    bool readTiePointGraph(const std::filesystem::path& path,
                           const std::vector<std::filesystem::path>& selectedImages,
                           TiePointGraph* graph,
                           std::string* error = nullptr,
                           const ImageTokenResolver& resolver = {});
} // namespace xjw::aerial_triangulation::engine
