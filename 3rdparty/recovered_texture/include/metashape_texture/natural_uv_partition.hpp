#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace metashape_texture {

struct NaturalUvPartitionStep {
    std::uint32_t cluster_count{};
    double objective{};
    std::vector<std::uint32_t> seeds;
};

struct NaturalUvNormalPartition {
    std::vector<std::int32_t> assignments;
    std::vector<std::array<double, 3>> prototypes;
    std::vector<std::uint32_t> seeds;
    std::vector<NaturalUvPartitionStep> history;
};

struct NaturalUvPartitionCandidate {
    std::uint32_t vertex{};
    std::int32_t target{};
    double gain{};
};

struct NaturalUvPartitionPostprocessStep {
    std::vector<NaturalUvPartitionCandidate> candidates;
    std::uint32_t applied{};
};

struct NaturalUvPartitionPostprocess {
    std::vector<std::int32_t> assignments;
    std::vector<NaturalUvPartitionPostprocessStep> history;
};

// Recovered no-camera chart split primitive from Linux Metashape 2.3.2.
// ordered_faces contains global triangle ordinals in the chart's current face
// order.  vertex_xyz and triangle_indices are tightly packed project float32
// positions and uint32 triangles.
NaturalUvNormalPartition partition_natural_uv_chart_by_normals(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const std::uint32_t> ordered_faces,
    std::uint32_t max_clusters = 10,
    std::uint32_t refinement_iterations = 6,
    double target_ratio = 0.75);

// Recovered 0x27a5e10 boundary cleanup.  It greedily moves eligible vertex
// fans to the neighboring label when that shortens the UV-space cut, while
// preserving two-manifold chart topology.
NaturalUvPartitionPostprocess postprocess_natural_uv_partition(
    std::span<const std::uint32_t> triangle_indices,
    std::span<const float> vertex_uv,
    std::span<const std::int32_t> initial_assignments,
    std::uint32_t max_iterations = 100U);

}  // namespace metashape_texture
