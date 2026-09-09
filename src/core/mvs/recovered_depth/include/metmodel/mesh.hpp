#pragma once

#include "metmodel/options.hpp"
#include "metmodel/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace metmodel {

Mesh build_mesh(const Scene& scene, const std::vector<DepthMap>& depth_maps,
                const ProgramOptions& options);
void recompute_normals(Mesh& mesh);
void decimate_mesh(Mesh& mesh, std::size_t target_faces);

// Mode-3 area-weighted quadric edge collapse used by the recovered model
// path.  The result counters make rejected/stale heap entries visible to
// replay tools instead of hiding topology decisions behind a void API.
struct QemDecimationStats {
    std::size_t input_vertices{};
    std::size_t input_faces{};
    std::size_t output_vertices{};
    std::size_t output_faces{};
    std::size_t accepted_collapses{};
    std::size_t stale_candidates{};
    std::size_t topology_rejections{};
    std::size_t normal_rejections{};
    std::uint32_t first_source = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t first_target = std::numeric_limits<std::uint32_t>::max();
    float first_cost{};
    std::uint32_t first_popped_source = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t first_popped_target = std::numeric_limits<std::uint32_t>::max();
    float first_popped_cost{};
};

struct QemCollapseEvent {
    std::uint32_t source{};
    std::uint32_t target{};
    std::array<float, 3> position{};
    float cost{};
};

[[nodiscard]] QemDecimationStats decimate_mesh_qem_mode3(
    Mesh& mesh, std::size_t target_faces,
    std::span<const float> vertex_scale = {},
    std::vector<QemCollapseEvent>* trace = nullptr,
    std::vector<std::array<float, 4>>* vertex_attributes = nullptr,
    const std::function<bool()>& is_cancelled = {});

struct RecoveredFixBackTrianglesStats {
    std::size_t passes{};
    std::size_t corrected_vertices{};
};

// Exact post-QEM repair embedded in sub_17B9460.  In each of five passes it
// snapshots unit face normals and face-weighted one-ring vertex sums, finds a
// closed triangle whose three adjacent normals oppose it by at least 0.86,
// and moves one still-unmodified corner to its one-ring float32 mean.  The
// corner whose replacement normal has the smallest dot with the old face
// normal is selected.  Faces and the fourth vertex field are not changed.
[[nodiscard]] RecoveredFixBackTrianglesStats
fix_back_triangles_recovered(Mesh& mesh);

// Four additive values carried by the target marching/QEM path.  Before
// decimation they are respectively the source node's weight denominator,
// central histogram support, octree level, and one.  QEM adds them when a
// vertex is collapsed; trimming consumes channels 1/3 and 2/3.
using RecoveredMeshTrimAttribute = std::array<float, 4>;

// The target stores confidence in the fourth float of its 16-byte vertex
// record immediately before trimming.  For the four additive marching/QEM
// attributes this is channel 1 divided by channel 3 in scalar float
// precision (the numerator is retained when the denominator is not > 0).
void assign_recovered_vertex_confidence(
    Mesh& mesh,
    std::span<const RecoveredMeshTrimAttribute> attributes);

// RGB fusion produces three weighted sums followed by one weight per vertex.
// The target normalizes them as a float reciprocal followed by a float
// multiply, then converts toward zero to uint8.  Keeping this as a public
// boundary lets CUDA/CPU fusion implementations share the exact finalizer.
using RecoveredVertexColorAccumulator = std::array<float, 4>;

// One uint8 image level consumed by the target CPU colorizer.  Pixels use
// row_elements rather than an assumed tightly-packed row so target captures
// and production images share the same sampler contract.
struct RecoveredVertexColorImageLevel {
    std::size_t width{};
    std::size_t height{};
    std::size_t channels{};
    std::size_t row_elements{};
    std::vector<std::uint8_t> pixels;
};

// Per-camera mutable scratch at the boundary immediately after the target's
// face-weight pass.  The three arrays are respectively 3, 2 and 2 floats per
// mesh vertex.  The accumulation pass consumes and resets their active
// entries exactly like sub_214E470.
struct RecoveredVertexColorCameraScratch {
    std::vector<float> geometry;
    std::vector<float> projected_xy;
    std::vector<float> weighted_xy;
};

struct RecoveredVertexColorProjectionCamera {
    metalign::CameraModel model;
    std::array<double, 16> world_to_camera{};
    std::size_t width{};
    std::size_t height{};
};

// Evidence-backed source form for the 688-byte std430/std140 camera record
// consumed by the recovered Vulkan vertex-colour shaders.  The target camera
// object exposes these records independently: camera_to_world at +0, the
// auxiliary 16-double record at +584 and image_matrix at +736.  The target
// packer consumes auxiliary elements 0..2 and 12..14.  They are all zero in
// the captured ordinary South Building frame camera; image_matrix is identity.
// Keeping the auxiliary record explicit prevents that observed camera-class
// default from becoming an assumption for rolling-shutter or other cameras.
struct RecoveredVertexColorVulkanCameraSource {
    std::array<double, 16> camera_to_world{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0};
    metalign::CameraModel model;
    std::size_t width{};
    std::size_t height{};
    std::array<double, 16> auxiliary_transform{};
    std::array<double, 9> image_matrix{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
};

using RecoveredVertexColorVulkanCameraPayload =
    std::array<std::byte, 688>;

// Exact perspective-camera packer recovered from sub_2C3F720 and its callees.
// Unsupported target branches (RPC, Ebner and Brown s1..s4) are intentionally
// absent from the public source type.  The target copies uninitialised stack
// bytes into the inactive RPC coefficient array; this implementation
// deterministically zeroes that semantically dead region instead.
[[nodiscard]] RecoveredVertexColorVulkanCameraPayload
pack_recovered_vertex_color_vulkan_camera(
    const RecoveredVertexColorVulkanCameraSource& source);

// Project-boundary adapter for the ordinary perspective-frame domain.  The
// two target auxiliary matrices are explicit arguments so callers must not
// silently assume identity for rotated/cropped or otherwise transformed
// camera records.
[[nodiscard]] RecoveredVertexColorVulkanCameraSource
make_recovered_vertex_color_vulkan_camera_source(
    const Camera& camera,
    const std::array<double, 16>& auxiliary_transform,
    const std::array<double, 9>& image_matrix);

struct RecoveredVertexColorDepthImage {
    std::size_t width{};
    std::size_t height{};
    std::size_t row_elements{};
    std::vector<float> pixels;
};

using RecoveredVertexColorPosition = std::array<float, 4>;
using RecoveredVertexColorFace = std::array<std::uint32_t, 3>;

// Persistent geometry channel 2 prepared once before the target camera loop.
// Metashape reads the float4 mesh positions, promotes xyz to double, computes
// one half of the double-precision cross-product norm for every face, casts
// that area once to float, then adds it to the three incident vertices in
// source face order.  The ordering is observable because every later camera
// weight divides by this value.
[[nodiscard]] std::vector<float>
compute_recovered_vertex_color_geometry_denominator(
    std::span<const RecoveredVertexColorPosition> positions,
    std::span<const RecoveredVertexColorFace> faces);

// Production form of the recovered seven-stage Vulkan camera fusion chain.
// Pipelines and mesh buffers persist across the complete camera loop; only
// the camera payload and its RGB mip pyramid change between iterations.
// The current exact domain is ordinary perspective frames of one common,
// even resolution.  Unsupported camera/image layouts fail closed.
struct RecoveredVertexColorVulkanStats {
    std::string device_name;
    std::size_t input_vertices{};
    std::size_t input_faces{};
    std::size_t cameras{};
    std::size_t draw_calls{};
    std::size_t compute_dispatches{};
    std::size_t directly_colored_vertices{};
    std::size_t extrapolated_vertices{};
    std::size_t uncolored_vertices{};
    double gpu_seconds{};
    double finalize_seconds{};
};

[[nodiscard]] RecoveredVertexColorVulkanStats
colorize_mesh_recovered_vulkan(
    Mesh& mesh,
    std::span<const Camera> cameras,
    const std::filesystem::path& shader_directory);

// Exact perspective/depth visibility pass from sub_20E5AB0 for the float4
// mesh position and float32 depth-image domain used by South.  It resets the
// selected entries to the supplied invalid pair before projecting them.
void project_recovered_vertex_color_camera_cpu(
    std::span<const RecoveredVertexColorPosition> positions,
    std::span<const std::uint32_t> selected_vertices,
    const RecoveredVertexColorProjectionCamera& camera,
    const RecoveredVertexColorDepthImage& depth,
    RecoveredVertexColorCameraScratch& scratch,
    float invalid_x = -1.0F, float invalid_y = -1.0F);

// Exact scalar face-weight producer used by the target CPU colorizer
// (sub_20E9110).  geometry channel 2 is a persistent per-vertex denominator
// prepared before the camera loop; this stage only adds channels 0/1 and the
// weighted image coordinates for faces whose three vertices are visible.
void accumulate_recovered_vertex_color_geometry_cpu(
    std::span<const RecoveredVertexColorPosition> positions,
    std::span<const RecoveredVertexColorFace> faces,
    std::span<const std::uint32_t> selected_faces,
    const std::array<float, 3>& camera_center,
    RecoveredVertexColorCameraScratch& scratch,
    float invalid_x = -1.0F, float invalid_y = -1.0F,
    std::vector<std::uint8_t>* selected_face_acceptance = nullptr);

// Exact uint8/mipmap sampling and weighted RGB accumulation path shipped in
// the target's OpenMP colorizer (sub_214E470 -> sub_20E55A0 -> sub_20EE140).
// Projection, visibility and face-weight production are deliberately a
// separate boundary so each stage can be validated against target captures.
void accumulate_recovered_vertex_color_camera_cpu(
    std::span<const std::uint32_t> selected_vertices,
    std::span<const RecoveredVertexColorImageLevel> mipmaps,
    RecoveredVertexColorCameraScratch& scratch,
    std::vector<RecoveredVertexColorAccumulator>& accumulator,
    float invalid_x = -1.0F, float invalid_y = -1.0F);

[[nodiscard]] std::vector<std::int32_t>
extrapolate_recovered_vertex_color_accumulator(
    const Mesh& mesh,
    std::vector<RecoveredVertexColorAccumulator>& accumulator);
void assign_recovered_vertex_colors_from_accumulator(
    Mesh& mesh,
    std::span<const RecoveredVertexColorAccumulator> accumulator);

// Vulkan samples R8_UNORM in [0,1].  Its final boundary keeps the observable
// scalar order `(sum / weight) * 255`, converts toward zero, and writes black
// when weight is zero.  This is intentionally separate from the CPU uint8
// accumulator finalizer above because moving the 255 multiply across the
// division changes boundary bytes.
void assign_recovered_vertex_colors_from_normalized_accumulator(
    Mesh& mesh,
    std::span<const RecoveredVertexColorAccumulator> accumulator);

struct RecoveredMeshTrimParameters {
    // Target mode 1 is ordinary trimming.  Mode 2 is the separate noise-
    // cluster filter and deliberately skips the boundary morphology below.
    std::int32_t kind{1};
    float softness{0.1F};
};

struct RecoveredMeshTrimThresholds {
    float initial_support{};
    float frontier_level{};
    float lowest_level{-1.0F};
    float low_level{-1.0F};
    float median_level{-1.0F};
    float high_level{-1.0F};
    float maximum_level{-1.0F};
};

struct RecoveredMeshTrimTrace {
    RecoveredMeshTrimThresholds thresholds;
    std::vector<std::uint8_t> initial_keep;
    std::vector<std::vector<std::uint32_t>> frontier_inputs;
    std::vector<std::vector<std::uint32_t>> frontier_outputs;
    std::vector<std::uint8_t> after_frontier;
    std::vector<std::uint8_t> before_morphology_expanded;
    std::vector<Face> morphology_auxiliary_faces;
    std::vector<std::uint8_t> final_keep;
};

// Exact sub_17B9460 trimming state machine recovered from Metashape 2.3.2:
// dynamic support thresholds, source-gated frontier expansion, boundary
// padding and the fixed 3 dilate / 6 erode / 3 dilate face schedule.
[[nodiscard]] RecoveredMeshTrimTrace recover_mesh_trim_mask(
    std::span<const Face> faces,
    std::span<const RecoveredMeshTrimAttribute> attributes,
    const RecoveredMeshTrimParameters& parameters = {});

// Target compaction keeps every marked vertex, drops any face touching an
// unmarked vertex and remaps the additive attributes in the same order.
void compact_mesh_recovered_trim(
    Mesh& mesh,
    std::vector<RecoveredMeshTrimAttribute>& attributes,
    std::span<const std::uint8_t> keep);

// Raw geometry used by the target's post-trim region clipper.  The fourth
// float is deliberately retained: sub_2921AF0 copies confidence for old
// vertices and writes +0 for newly intersected vertices.
using RecoveredMeshClipVertex = std::array<float, 4>;
using RecoveredMeshClipFace = std::array<std::uint32_t, 3>;

struct RecoveredMeshClipPlane {
    std::array<double, 3> normal{};
    double offset{};
    double tolerance{};
};

struct RecoveredMeshClipStats {
    std::size_t input_vertices{};
    std::size_t input_faces{};
    std::size_t output_vertices{};
    std::size_t output_faces{};
    std::size_t inserted_vertices{};
    std::size_t removed_vertices{};
    std::size_t appended_faces{};
    std::size_t removed_faces{};
};

// Exact float topology path of target sub_2921AF0: classify by a double
// signed distance, split crossing edges once, triangulate clipped quads as a
// fan, then compact invalid faces and unreferenced vertices in stable order.
[[nodiscard]] RecoveredMeshClipStats clip_mesh_to_plane_recovered(
    std::vector<RecoveredMeshClipVertex>& vertices,
    std::vector<RecoveredMeshClipFace>& faces,
    const RecoveredMeshClipPlane& plane);

// Build the six planes consumed by sub_2922950 from the target's row-major
// oriented-region transform.  Its orthonormal columns become the inward plane
// normals; center[3] and size[3] provide the two offsets on each axis.
[[nodiscard]] std::array<RecoveredMeshClipPlane, 6>
make_recovered_region_clip_planes(
    const std::array<double, 15>& region, double tolerance);

// Production adapter. It is intentionally a geometry-finalization step and
// should run before target color/confidence/UV generation.
[[nodiscard]] std::array<RecoveredMeshClipStats, 6>
clip_mesh_to_region_recovered(
    Mesh& mesh, const std::array<double, 15>& region, double tolerance);

// Per-part request passed to the target mode-3 QEM worker.  The controller
// first distributes the global face goal proportionally to the part's raw
// faces, adds one before truncation, then applies its 1.01 (ordinary) or 1.10
// (expanded seam) allowance.
[[nodiscard]] std::size_t recovered_qem_part_face_budget(
    std::size_t part_faces, std::size_t total_faces,
    std::size_t global_target_faces, bool expanded_allowance = false);
void smooth_mesh(Mesh& mesh, double strength, bool fix_borders, bool preserve_edges);
std::size_t target_face_count(const ProgramOptions& options, std::size_t current_faces);

}  // namespace metmodel
