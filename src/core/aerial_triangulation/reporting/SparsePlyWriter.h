#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace xjw::aerial_triangulation
{
    struct SparsePlyVertex
    {
        std::array<float, 3> xyz{};
        std::array<std::uint8_t, 3> color{};
    };
    /// Streams vertices in fixed little-endian format and atomically replaces the destination.
    bool writeSparsePly(const std::filesystem::path& path,
                        std::size_t count,
                        const std::function<SparsePlyVertex(std::size_t)>& vertex,
                        std::string_view comment,
                        std::string* error = nullptr);
} // namespace xjw::aerial_triangulation
