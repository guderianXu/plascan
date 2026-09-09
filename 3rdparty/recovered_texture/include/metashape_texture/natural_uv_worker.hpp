#pragma once

#include "metashape_texture/natural_uv_parameterization.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace metashape_texture {

struct NaturalUvChart {
    std::vector<float> vertex_xyz;
    std::vector<std::uint32_t> triangle_indices;
    std::vector<std::uint32_t> source_vertices;
    std::vector<std::uint32_t> source_faces;
    std::vector<float> uv;
};

struct NaturalUvWorkerOptions {
    NaturalUvValidationOptions validation;
    std::uint32_t partition_max_clusters{10U};
    std::uint32_t partition_refinement_iterations{6U};
    double partition_target_ratio{0.75};
    std::uint32_t partition_postprocess_iterations{100U};
    std::uint32_t max_processed_charts{1024U};
};

struct NaturalUvWorkerStep {
    std::vector<std::uint32_t> source_faces;
    std::uint32_t anchor_first{};
    std::uint32_t anchor_second{};
    std::vector<float> uv;
    NaturalUvValidationResult validation;
    std::vector<std::vector<std::uint32_t>> child_source_faces;
};

struct NaturalUvWorkerResult {
    std::vector<NaturalUvChart> charts;
    std::vector<NaturalUvWorkerStep> history;
};

// Recovered root-chart topology normalization.  Faces are ordered by the
// native binary32 area key, then vertices whose incident faces form multiple
// disconnected fans are duplicated.  Child charts are already normalized by
// construction and do not pass through this boundary again.
NaturalUvChart normalize_natural_uv_root_chart(NaturalUvChart chart);

// Native accepted-chart transforms.  The first normalizes a parameterized
// leaf to the queue's geometry scale and preferred bounding-box orientation;
// the second converts that chart to the face-density scale consumed by the UV
// packer.
void transform_accepted_natural_uv_chart(NaturalUvChart &chart,
                                         double scale_argument);

// The caller flattens the accepted FIFO leaves into one two-dimensional
// chart, restores faces in their original component order, normalizes its
// topology once more, and extracts edge-connected charts.  This step is what
// determines the chart order and vertex numbering passed to the packer.
std::vector<NaturalUvChart> rebuild_natural_uv_charts_for_packing(
    std::span<const std::uint32_t> source_face_order,
    std::span<const NaturalUvChart> accepted_charts);

void scale_natural_uv_chart_for_packing(NaturalUvChart &chart,
                                        std::uint32_t texture_size);

// Complete recovered no-camera Natural UV chart worker.  Charts are processed
// in the native FIFO order.  Closed charts are split directly by normal
// regions; rejected open parameterizations additionally receive the native UV
// boundary cleanup before their children are appended to the queue.
NaturalUvWorkerResult generate_natural_uv_charts(
    NaturalUvChart initial_chart,
    const NaturalUvWorkerOptions &options = {});

}  // namespace metashape_texture
