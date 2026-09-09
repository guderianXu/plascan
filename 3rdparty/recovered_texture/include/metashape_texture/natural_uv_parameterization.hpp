#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace metashape_texture {

struct NaturalUvAngleProjection {
    std::vector<double> angles;
    std::vector<bool> boundary_vertices;
    std::uint32_t constraint_count{};
};

struct NaturalUvParameterization {
    NaturalUvAngleProjection angle_projection;
    std::vector<float> uv;
    std::uint32_t anchor_first{};
    std::uint32_t anchor_second{};
};

struct NaturalUvSvd3x3 {
    std::array<double, 9> left_columns{};
    std::array<double, 9> right_columns{};
    std::array<double, 3> singular_values{};
    std::uint32_t dominant_index{};
};

// Stage-level diagnostic for the recovered 3x3 Golub-Reinsch path.  Matrices
// use row-major storage; each singular vector occupies one column.
NaturalUvSvd3x3 decompose_natural_uv_covariance(
    const std::array<double, 9> &matrix);

struct NaturalUvRasterMetrics {
    double area_ratio{};
    double conformal_error{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t occupied_pixels{};
    std::uint64_t overlap_pixels{};
};

struct NaturalUvValidationOptions {
    std::uint32_t small_chart_fast_accept_max_faces{10U};
    std::uint32_t raster_maximum_dimension{1024U};
    double max_stretch_ratio{20.0};
    double min_area_ratio{0.25};
    double max_conformal_error{0.005};
};

enum class NaturalUvValidationDecision {
    invalid_input,
    small_chart_fast_accept,
    reject_stretch,
    reject_area_ratio,
    reject_conformal_error,
    accept,
};

struct NaturalUvValidationResult {
    bool accepted{};
    NaturalUvValidationDecision decision{NaturalUvValidationDecision::invalid_input};
    bool stretch_evaluated{};
    bool raster_evaluated{};
    double stretch_ratio{};
    NaturalUvRasterMetrics raster;
};

// Select the two native mode-3 constraints from the longest oriented boundary
// loop using its dominant 3D principal axis.  The result is {min, max}
// projection in the native anchor order.
std::array<std::uint32_t, 2> select_natural_uv_anchors(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices);

// Recovered mode-3 angle projection used before Metashape's angle-based LSCM.
// vertex_xyz and triangle_indices describe one local chart with tightly packed
// XYZ triples and local uint32 triangle triples.  The output stores three
// angles per triangle in matching corner order.
NaturalUvAngleProjection project_natural_uv_angles(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices);

// Complete recovered mode-3 local parameterizer.  The native routine fixes
// anchor_first to (0, 0) and anchor_second to (0, 1), then solves the
// area-weighted angle-based LSCM system.  UV values are returned as packed
// float32 pairs in local vertex order.
NaturalUvParameterization parameterize_natural_uv_chart(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices,
    std::uint32_t anchor_first,
    std::uint32_t anchor_second);

NaturalUvParameterization parameterize_natural_uv_chart(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices);

// Recovered quality gates used by the native mode-3 worker after each local
// parameterization.  The stretch gate is evaluated first; only charts that
// pass it reach the 1024-pixel occupancy/overlap raster test.
double measure_natural_uv_stretch(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const float> vertex_uv);

NaturalUvRasterMetrics measure_natural_uv_raster_quality(
    std::span<const std::uint32_t> triangle_indices,
    std::span<const float> vertex_uv,
    std::uint32_t maximum_dimension = 1024U);

NaturalUvValidationResult validate_natural_uv_chart(
    std::span<const float> vertex_xyz,
    std::span<const std::uint32_t> triangle_indices,
    std::span<const float> vertex_uv,
    const NaturalUvValidationOptions &options = {});

}  // namespace metashape_texture
