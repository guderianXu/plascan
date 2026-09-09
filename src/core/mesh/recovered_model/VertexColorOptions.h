#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>

#include "metmodel/mesh.hpp"

namespace metmodel
{
    struct RecoveredVertexColorVulkanOptions
    {
        std::optional<std::array<std::uint8_t, 16>> deviceUuid;
        std::function<bool()> isCancelled;
        std::function<void(std::size_t, std::size_t)> progress;
    };

    RecoveredVertexColorVulkanStats colorize_mesh_recovered_vulkan(Mesh& mesh,
                                                                   std::span<const Camera> cameras,
                                                                   const std::filesystem::path& shader_directory,
                                                                   const RecoveredVertexColorVulkanOptions& options);
} // namespace metmodel
