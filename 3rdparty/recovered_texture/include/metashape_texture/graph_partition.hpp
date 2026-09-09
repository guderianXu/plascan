#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace metashape_texture {

struct PartitionMesh {
    std::vector<std::array<float, 3>> vertices;
    std::vector<std::array<std::uint32_t, 3>> faces;
};

struct PartitionAcceptedUnion {
    std::size_t pass = 0;
    std::uint32_t left = 0;
    std::uint32_t right = 0;
    std::uint64_t recorded_size = 0;
};

struct TextureGraphPartition {
    std::vector<std::uint32_t> core_faces;
    std::vector<std::uint32_t> outskirts_faces;
    std::vector<std::uint32_t> global_faces;
    std::vector<std::uint8_t> core_mask;
};

struct GraphPartitionBuildResult {
    std::vector<std::vector<std::uint32_t>> connected_components;
    std::vector<std::vector<std::uint32_t>> preliminary_parts;
    std::vector<std::vector<std::vector<std::uint32_t>>> merge_passes;
    std::vector<PartitionAcceptedUnion> accepted_unions;
    std::vector<TextureGraphPartition> partitions;
};

// Recovered from Metashape 2.3.1's native MSVC x64 graph builder. The defaults
// are the constants used by Build Texture: 0.9*16384 recursive chunks, three
// directed-candidate merge passes, and a 20-edge optimizer halo.
GraphPartitionBuildResult build_graph_partitions(
    const PartitionMesh &mesh,
    std::size_t merge_limit = 16384,
    std::size_t merge_pass_count = 3,
    std::size_t outskirts_radius = 20);

}  // namespace metashape_texture
