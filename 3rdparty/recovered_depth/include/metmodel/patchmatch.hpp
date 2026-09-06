#pragma once

#include "metmodel/options.hpp"
#include "metmodel/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace metmodel {

// Exact 272-byte by-value camera ABI used by the recovered PatchMatch
// CUDA/OpenCL kernels.  The record deliberately keeps role-dependent fields
// even when an ordinary perspective camera does not consume them.
struct alignas(16) PatchMatchCamera {
    float f = 0.0F;
    float cx = 0.0F;
    float cy = 0.0F;
    std::uint32_t width_original = 0;
    std::uint32_t height_original = 0;
    std::uint32_t pyramid_level0_downscale = 0;
    float b1 = 0.0F;
    float b2 = 0.0F;
    std::uint32_t type = 0;
    std::array<std::uint32_t, 3> padding12bytes{};
    std::array<float, 4> line_model_num{};
    std::array<float, 4> line_model_den{};
    std::array<float, 4> samp_model_num{};
    std::array<float, 4> samp_model_den{};
    std::array<float, 8> inv_line_model_num{};
    std::array<float, 4> inv_line_model_den{};
    std::array<float, 8> inv_samp_model_num{};
    std::array<float, 4> inv_samp_model_den{};
    std::array<float, 16> transform{};
};

PatchMatchCamera make_patchmatch_perspective_camera(const Camera& camera,
                                                    int downscale);

// Role-specific camera records built by host cost producer 0x25A1070.  The
// reference record carries the float camera->world transform.  Normal
// rotation records are derived from that float matrix by the target's full
// double cofactor inverse, row normalization, float rounding, and a second
// cofactor inverse; direct R/R-transpose substitution is not bit-equivalent.
PatchMatchCamera make_patchmatch_reference_camera(const Camera& camera,
                                                  int downscale);

struct PatchMatchNormalRotationCameras {
    PatchMatchCamera before;
    PatchMatchCamera after;
};

PatchMatchNormalRotationCameras make_patchmatch_normal_rotation_cameras(
    const PatchMatchCamera& reference_camera);

std::array<float, 12> make_patchmatch_propagation_rotation(
    const PatchMatchCamera& reference_camera);

std::array<float, 16> patchmatch_reference_to_neighbor_transform(
    const Camera& reference, const Camera& neighbor);

enum class DepthVoteRelation { Supports, Intersects, DoesNotReach, NoDepth, Occludes };
enum class DepthVotingClass { Empty, Bad, Normal, Good };

int depth_vote_weight(bool inlier, DepthVoteRelation relation);
DepthVotingClass classify_depth_voting(float depth, int vote_sum);

// Opaque, exact-size CUDA parameter records used by the recovered depth-map
// voting kernels.  Unknown calibration fields intentionally remain raw bytes:
// callers either use a captured target record or an independently verified
// packer; this interface does not invent semantic field names.
struct alignas(8) DepthVotingCalibrationCu {
    std::array<std::byte, 120> bytes{};
};

struct alignas(16) DepthVotingMatrix4x4f {
    std::array<float, 16> values{};
};

static_assert(sizeof(DepthVotingCalibrationCu) == 120);
static_assert(alignof(DepthVotingCalibrationCu) == 8);
static_assert(sizeof(DepthVotingMatrix4x4f) == 64);
static_assert(alignof(DepthVotingMatrix4x4f) == 16);

// Target wrapper 0x26835B0 packs the rectified frame-camera calibration into
// this exact by-value CUDA record.  The recovered input domain is deliberately
// narrow: ordinary perspective frame cameras, power-of-two downscales, and
// dimensions exactly divisible at the selected three-level voting pyramid.
// Unsupported domains fail instead of receiving guessed calibration fields.
DepthVotingCalibrationCu make_depth_voting_perspective_calibration(
    const Camera& camera,
    std::uint32_t downscale,
    std::uint32_t pyramid_level);

// Matrix records packed by the same target voting wrappers.  Radius uses the
// camera-to-world transform; direct and occlusion voting use the ordered
// camera-coordinate transform (from -> to).  Both functions preserve the
// target's double-operation order before the final float conversion.
DepthVotingMatrix4x4f make_depth_voting_camera_to_world_transform(
    const Camera& camera);
DepthVotingMatrix4x4f make_depth_voting_camera_transform(
    const Camera& from,
    const Camera& to);

// Exact sample-radius stage recovered from target wrapper 0x26835B0 and the
// target CUDA PTX.  depth/radius are complete persistent allocations;
// level_offset selects the current pyramid level inside them, while
// kernel_offset selects the first logical work item of this launch.
struct DepthRadiusEstimateInput {
    DepthVotingCalibrationCu calibration;
    DepthVotingMatrix4x4f transform;
    std::vector<float> depth_allocation;
    std::vector<float> radius_allocation;
    std::uint32_t level_offset = 0;
    std::uint32_t kernel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

bool run_recovered_depth_radius_estimate_cuda(
    const DepthRadiusEstimateInput& input,
    std::vector<float>& radius_output,
    std::string& error);

// Exact direct cross-camera voting contract recovered from wrapper 0x2681710
// and the target PTX.  Neighbor depth/radius contain a contiguous pyramid;
// all seven counters and the vote map are persistent accumulators.
struct DepthNeighborVotesInput {
    DepthVotingCalibrationCu reference_calibration;
    DepthVotingMatrix4x4f reference_to_neighbor;
    DepthVotingCalibrationCu neighbor_calibration;
    std::vector<std::int32_t> votes;
    std::vector<float> reference_depth;
    std::vector<float> reference_radius;
    std::vector<std::uint8_t> neighbor_inlier_mask;
    std::vector<float> neighbor_depth_levels;
    std::vector<float> neighbor_radius_levels;
    std::uint32_t reference_level = 0;
    std::uint32_t neighbor_levels = 0;
    std::array<std::int32_t, 7> counters{};
    std::uint32_t kernel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

struct DepthNeighborVotesOutput {
    std::vector<std::int32_t> votes;
    std::array<std::int32_t, 7> counters{};
};

bool run_recovered_depth_neighbor_votes_cuda(
    const DepthNeighborVotesInput& input,
    DepthNeighborVotesOutput& output,
    std::string& error);

// Exact reverse-projection occlusion voting contract recovered from wrapper
// 0x2682460, target PTX and the embedded OpenCL source.  The thread domain is
// one selected neighbor pyramid level; votes are atomically accumulated in
// reference-camera coordinates.
struct DepthNeighborOcclusionVotesInput {
    DepthVotingCalibrationCu reference_calibration;
    DepthVotingMatrix4x4f neighbor_to_reference;
    DepthVotingCalibrationCu neighbor_calibration;
    std::vector<std::int32_t> votes;
    std::vector<float> reference_depth;
    std::vector<float> reference_radius;
    std::vector<std::uint8_t> neighbor_inlier_mask;
    std::vector<float> neighbor_depth_levels;
    std::vector<float> neighbor_radius_levels;
    std::uint32_t neighbor_level = 0;
    std::uint32_t reference_level = 0;
    std::array<std::int32_t, 2> counters{};
    std::uint32_t kernel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

struct DepthNeighborOcclusionVotesOutput {
    std::vector<std::int32_t> votes;
    std::array<std::int32_t, 2> counters{};
};

bool run_recovered_depth_neighbor_occlusion_votes_cuda(
    const DepthNeighborOcclusionVotesInput& input,
    DepthNeighborOcclusionVotesOutput& output,
    std::string& error);

// Exact final cross-camera voting stage recovered from the target's CUDA PTX
// and wrapper at 0x2683F70.  Counters are accumulated, not reset, matching the
// device contract.  The launch range may be replayed in balanced batches.
struct DepthVotingFinalizeInput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> depth_allocation;
    std::vector<std::int32_t> votes_allocation;
    std::int32_t counter_empty = 0;
    std::int32_t counter_bad = 0;
    std::int32_t counter_normal = 0;
    std::int32_t counter_good = 0;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

struct DepthVotingFinalizeOutput {
    std::vector<float> depth_allocation;
    std::int32_t counter_empty = 0;
    std::int32_t counter_bad = 0;
    std::int32_t counter_normal = 0;
    std::int32_t counter_good = 0;
};

bool run_recovered_depth_voting_finalize_cuda(
    const DepthVotingFinalizeInput& input,
    DepthVotingFinalizeOutput& output,
    std::string& error);

// Host orchestration recovered for the target's fixed three-level voting
// pyramid.  These records contain launch metadata only; the chain owns and
// continuously carries every radius/vote/depth accumulator between launches.
struct DepthVotingRadiusLaunch {
    DepthVotingCalibrationCu calibration;
    DepthVotingMatrix4x4f transform;
    std::uint32_t level_offset = 0;
    std::uint32_t kernel_offset = 0;
    std::size_t global_work_items = 0;
};

struct DepthVotingDirectLaunch {
    DepthVotingCalibrationCu reference_calibration;
    DepthVotingMatrix4x4f reference_to_neighbor;
    DepthVotingCalibrationCu neighbor_calibration;
    std::uint32_t kernel_offset = 0;
    std::size_t global_work_items = 0;
};

struct DepthVotingOcclusionLaunch {
    DepthVotingCalibrationCu reference_calibration;
    DepthVotingMatrix4x4f neighbor_to_reference;
    DepthVotingCalibrationCu neighbor_calibration;
    std::uint32_t kernel_offset = 0;
    std::size_t global_work_items = 0;
};

struct DepthVotingChainReferenceLevelInput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> depth;
    std::span<const float> depth_view;
    std::vector<std::int32_t> initial_votes;
    DepthVotingRadiusLaunch radius;
};

struct DepthVotingChainNeighborInput {
    std::vector<float> depth_levels;
    std::array<std::span<const float>, 3> depth_level_views;
    std::array<DepthVotingRadiusLaunch, 3> radius;
    std::array<std::vector<std::uint8_t>, 3> inlier_masks;
    std::array<std::span<const std::uint8_t>, 3> inlier_mask_views;
    std::array<DepthVotingDirectLaunch, 3> direct;
    std::array<std::array<DepthVotingOcclusionLaunch, 3>, 3> occlusion;
};

struct DepthVotingChainInput {
    std::array<DepthVotingChainReferenceLevelInput, 3> reference;
    std::vector<DepthVotingChainNeighborInput> neighbors;
    std::array<std::int32_t, 7> direct_counters{};
    std::array<std::int32_t, 2> occlusion_counters{};
    std::array<std::int32_t, 4> final_counters{};
    std::uint32_t component_size_threshold = 0;
    std::size_t device_index = 0;
};

struct DepthVotingChainOutput {
    std::array<std::vector<float>, 3> depth_before_components;
    std::array<std::vector<float>, 3> depth_after_components;
    std::array<std::vector<std::int32_t>, 3> votes;
    std::array<std::int32_t, 7> direct_counters{};
    std::array<std::int32_t, 2> occlusion_counters{};
    std::array<std::int32_t, 4> final_counters{};
    std::array<std::size_t, 3> components_cleared{};
};

// Production-side resources emitted by the recovered PM/filter chain.  Depth
// remains split by level here so the adapter can validate every boundary
// before constructing the contiguous neighbor allocation expected by CUDA.
struct DepthVotingPreparedReference {
    std::size_t camera_index = 0;
    std::array<std::vector<float>, 3> depth_levels;
    std::array<std::span<const float>, 3> depth_level_views;
};

struct DepthVotingPreparedNeighbor {
    std::size_t camera_index = 0;
    std::array<std::vector<float>, 3> depth_levels;
    std::array<std::span<const float>, 3> depth_level_views;
    std::array<std::vector<std::uint8_t>, 3> inlier_masks;
    std::array<std::span<const std::uint8_t>, 3> inlier_mask_views;
};

DepthVotingChainInput make_recovered_depth_voting_chain_input(
    const Scene& scene,
    const DepthVotingPreparedReference& reference,
    const std::vector<DepthVotingPreparedNeighbor>& neighbors,
    std::uint32_t downscale,
    FilterMode filter_mode,
    std::size_t device_index = 0);

bool run_recovered_depth_voting_chain_cuda(
    const DepthVotingChainInput& input,
    DepthVotingChainOutput& output,
    std::string& error);

// Host post-pass recovered from 0x255EBC0 and 0x255FD60: form 8-connected
// components over non-zero depth pixels and clear components whose size is
// less than or equal to the supplied mode threshold.  Returns pixels cleared.
std::size_t filter_recovered_small_depth_components(
    std::vector<float>& depth,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t component_size_threshold);

// Dynamically observed target mapping (2.3.2 build 22956): Disabled and Mild
// both use internal mode 0; Moderate/Aggressive use modes 1/2.
std::uint32_t recovered_depth_component_threshold(FilterMode mode);

enum class PatchMatchScheduleOperation {
    Refinement,
    Wta,
    Propagation,
    CoarseToPrecise,
    CostPipeline,
    CostBatch,
    FinalRefinement,
    FilterCheckCost,
    FilterCheckNeighbours,
    FilterClearDepth,
    FilterNormals,
    FilterSpecklesEdges,
};

// Host-level call record recovered from the target's embedded OpenCL source,
// CUDA wrapper ABI, full runtime traces and the batching helper at 0x4168770.
// Fields unused by an operation remain zero.
struct PatchMatchScheduleEvent {
    PatchMatchScheduleOperation operation{};
    std::uint32_t downscale = 0;
    std::uint32_t iteration = 0;
    std::uint32_t hypotheses = 0;
    std::uint32_t is_checkboard = 0;
    std::uint32_t checkboard_step = 0;
    std::uint32_t only_each_fourth_pixel = 0;
    std::uint32_t pixel_offset = 0;
};

// Exact balanced batching helper used by the target:
// align_up(ceil(items / ceil(items / capacity)), alignment).
std::uint32_t patchmatch_balanced_batch_span(
    std::uint32_t items,
    std::uint32_t alignment = 128,
    std::uint32_t capacity = 128 * 1024);

std::vector<PatchMatchScheduleEvent> make_recovered_patchmatch_level_schedule(
    std::uint32_t width_original,
    std::uint32_t height_original,
    std::uint32_t target_downscale);

const char* patchmatch_schedule_operation_name(PatchMatchScheduleOperation operation);

// Exact neighbor-resource selector loop used by the uchar PatchMatch host
// batcher at 0x25A2917..0x25A297E.  The first three ranked neighbors are
// always used.  Remaining neighbors are split into four deterministic lanes
// selected by the low two iteration bits.  The raw all-neighbors state bit is
// exposed explicitly because its higher-level option meaning is not yet
// proven; when set, the target bypasses lane filtering.
std::vector<std::uint32_t> make_recovered_patchmatch_neighbor_subset(
    std::uint32_t neighbor_count,
    std::uint32_t iteration,
    bool all_neighbors_state = false);

// Exact sparse-point preparation and range reducer used by the perspective
// PatchMatch host path at 0x1CC0A10 -> 0x1CC20D0 -> 0x1CC1F80.  Point
// coordinates are first rounded to float, region tests are then performed in
// double precision, and accepted points are transformed to float camera depth.
struct PatchMatchDepthRangeInput {
    std::vector<std::uint32_t> track_ids;
    std::vector<float> camera_depths;
};

struct PatchMatchDepthRange {
    float minimum = 0.0F;
    float maximum = 0.0F;
};

PatchMatchDepthRangeInput make_recovered_patchmatch_depth_range_input(
    const Scene& scene,
    std::size_t camera_index);

PatchMatchDepthRange reduce_recovered_patchmatch_depth_range(
    const std::vector<float>& camera_depths);

PatchMatchDepthRange make_recovered_patchmatch_depth_range(
    const Scene& scene,
    std::size_t camera_index);

// Exact uint8 image-range reducer at 0x1CC1EC0 and the float composition at
// 0x1CF393C.  The image must be the already prepared/undistorted level passed
// by 0x1CEE400; image preparation itself is a separate recovered boundary.
float reduce_recovered_patchmatch_deviation_multiplier(
    std::span<const std::uint8_t> prepared_image);

float compose_recovered_patchmatch_deviation_multiplier(
    std::span<const std::uint8_t> prepared_image,
    float owner_multiplier);

// Exact uint8 source/preparation records used before PatchMatch.  The source
// image conversion is the recovered double-precision BT.601 truncation.  The
// CUDA preparation input carries the two 120-byte records verbatim because
// the target kernel consumes CalibrationCu by value.
struct RecoveredPatchMatchImageU8 {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> image;
    std::vector<std::uint8_t> rejection_mask;
};

std::vector<std::uint8_t> make_recovered_patchmatch_source_u8(
    const metalign::Image& image);

// Exact high-level calibration reducer called by Metashape when Brown radial
// coefficients change.  It solves the radial-map derivative polynomial
// [1, 3*k1, 5*k2, 7*k3, 9*k4] with the target's analytic real-root solver and
// returns the smallest strictly positive root, or 1e9 when none exists.
double recovered_calibration_maximum_radius_squared(
    const metalign::CameraModel& model);

DepthVotingCalibrationCu make_recovered_patchmatch_source_calibration(
    const Camera& camera);

DepthVotingCalibrationCu make_recovered_patchmatch_source_calibration(
    const Camera& camera,
    float maximum_radius_squared);

DepthVotingCalibrationCu make_recovered_patchmatch_undistorted_calibration(
    const Camera& camera);

RecoveredPatchMatchImageU8 reduce_recovered_patchmatch_image_half(
    const RecoveredPatchMatchImageU8& source);

struct PatchMatchUndistortU8Input {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> source_image;
    std::vector<std::uint8_t> source_mask;
    DepthVotingCalibrationCu source_calibration;
    DepthVotingCalibrationCu target_calibration;
    std::size_t device_index = 0;
};

bool run_recovered_patchmatch_undistort_u8_cuda(
    const PatchMatchUndistortU8Input& input,
    RecoveredPatchMatchImageU8& output,
    std::string& error);

// Evidence-backed composition of the recovered unmasked perspective-camera
// preparation path.  It performs exact RGB8->uint8 conversion, target CUDA
// undistortion/rejection-mask generation, and enum-2 ceil-half pyramid
// reduction through downscale 32.  A deviation ratio is retained for every
// prepared level so the still-being-recovered host state can select it without
// recomputation.  Explicit source masks and non-frame camera models are not
// silently approximated by this API.
struct RecoveredPatchMatchPreparedLevel {
    std::uint32_t downscale = 1;
    RecoveredPatchMatchImageU8 data;
    float deviation_ratio = 1.0F;
};

struct RecoveredPatchMatchPreparedCamera {
    std::size_t camera_index = 0;
    std::uint32_t target_downscale = 1;
    std::size_t device_index = 0;
    bool valid = false;
    PatchMatchDepthRange depth_range;
    DepthVotingCalibrationCu source_calibration;
    DepthVotingCalibrationCu undistorted_calibration;
    std::vector<RecoveredPatchMatchPreparedLevel> image_levels;
    // Fixed for the complete internal x32 -> target schedule.  Its pyramid
    // level-0 downscale is target_downscale/2, not the current depth level.
    PatchMatchCamera camera;
};

// Exact 72-byte image-area record produced by target worker 0x1CFDA70 for
// every reference/neighbor pair. Coordinates are half-open and expressed at
// target_downscale; full_width/full_height describe the neighbor at that same
// scale. The leading id occupies four bytes followed by four bytes ABI padding.
struct RecoveredPatchMatchCropDescriptor {
    std::int32_t camera_index = -1;
    std::uint32_t padding = 0;
    std::uint64_t left = 0;
    std::uint64_t right = 0;
    std::uint64_t top = 0;
    std::uint64_t bottom = 0;
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    std::uint64_t full_width = 0;
    std::uint64_t full_height = 0;
};

static_assert(sizeof(RecoveredPatchMatchCropDescriptor) == 72);

bool make_recovered_patchmatch_neighbor_crop_descriptor(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::size_t neighbor_camera_index,
    std::uint32_t target_downscale,
    RecoveredPatchMatchCropDescriptor& output,
    std::string& error);

bool apply_recovered_patchmatch_neighbor_crop(
    const Camera& source_camera,
    std::uint32_t target_downscale,
    const RecoveredPatchMatchCropDescriptor& crop,
    RecoveredPatchMatchPreparedCamera& prepared,
    std::string& error);

bool make_recovered_patchmatch_unmasked_camera_preparation(
    const Scene& scene,
    std::size_t camera_index,
    std::uint32_t target_downscale,
    std::size_t device_index,
    RecoveredPatchMatchPreparedCamera& output,
    std::string& error,
    std::uint32_t minimum_retained_downscale = 1U);

// Exact host-side transition from one filtered PatchMatch level to the next
// finer level, recovered from 0x1CE7440/0x1CE9A90 and
// 0x1CE74C0/0x1CE6B40/0x1CFA5F0.  The previous buffers are the bilateral
// depth/normal output at 2*depth_downscale.  The result is the complete state
// uploaded before the first refinement at depth_downscale: bilinear depth,
// max-weight source normal, {-1,0.5} cost, identical coarse depth, and the
// per-pixel 30th-percentile local 3D edge radius.  Only the instruction-backed
// ordinary perspective camera path is accepted.
struct RecoveredPatchMatchCrossLevelState {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    std::vector<float> coarse_depth;
    std::vector<float> coarse_radius;
};

bool make_recovered_patchmatch_cross_level_state(
    const Camera& camera,
    std::uint32_t depth_downscale,
    std::uint32_t previous_width,
    std::uint32_t previous_height,
    std::span<const float> previous_depth,
    std::span<const std::uint8_t> previous_normal,
    RecoveredPatchMatchCrossLevelState& output,
    std::string& error);

// Exact uchar specialization used after each level's six filtering kernels
// and before the host cross-level transition.  sigma_d/sigma_r and offset are
// explicit because they are target launch arguments; the South level boundary
// uses 1, 1 and 0 respectively.
struct PatchMatchBilateralU8Input {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<std::uint8_t> image;
    float sigma_d = 1.0F;
    float sigma_r = 1.0F;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

struct PatchMatchBilateralU8Output {
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
};

bool run_recovered_patchmatch_bilateral_u8_cuda(
    const PatchMatchBilateralU8Input& input,
    PatchMatchBilateralU8Output& output,
    std::string& error);

// Exact first-refinement device contract recovered from the target.  Candidate
// storage is slot-major with the target's fixed 131072-pixel stride.
struct PatchMatchRefinementInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    float depth_min = 0.0F;
    float depth_max = 0.0F;
    std::vector<std::uint8_t> reference_image;
    std::uint32_t image_one_step_more_detailed = 0;
    float deviation_threshold_multiplier = 1.0F;
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    std::vector<float> coarse_depth;
    std::vector<float> coarse_depth_radius;
    std::span<const std::uint8_t> reference_image_view;
    std::span<const float> depth_view;
    std::span<const std::uint8_t> normal_view;
    std::span<const float> cost_view;
    std::span<const float> coarse_depth_view;
    std::span<const float> coarse_depth_radius_view;
    std::vector<float> initial_candidate_depth;
    std::vector<float> initial_candidate_normal;
    std::uint32_t iteration = 0;
    std::uint32_t only_each_fourth_pixel = 0;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
    // Opaque token returned by the preceding WTA/inlier-copy pipeline.  A
    // nonzero value proves that the persistent cost scratch still matches the
    // host state supplied here.  The producer still materializes its public
    // host output, but carries that proof into the following cost batch.
    std::uint64_t cuda_workspace_handoff = 0;
    // Production coarsest iteration zero may assert the captured uniform
    // initial state. The CUDA wrapper verifies every asserted host byte before
    // replacing equivalent H2D copies with device memset operations.
    bool initial_cuda_uniform_state = false;
    // Production-only optimization: keep the candidate result resident for
    // the immediately following cost batch. This is accepted only inside an
    // active CUDA workspace session; standalone public calls remain fully
    // materialized by default.
    bool defer_cuda_host_output = false;
};

struct PatchMatchCandidateOutput {
    static constexpr std::size_t capacity = 128 * 1024;
    static constexpr std::size_t hypotheses = 8;
    std::vector<float> depth;
    std::vector<float> normal;
    // Opaque, session-local token proving that the candidate producer left
    // its inputs and outputs in the role-stable CUDA workspace. It is valid
    // only for the immediately following cost batch.
    std::uint64_t cuda_workspace_handoff = 0;
    // True only when cuda_workspace_handoff represents a producer reached
    // from the preceding inlier-copy state, so cost scratch may be retained.
    // The token is validated fail-closed; this flag alone grants no reuse.
    bool cuda_resident_cost_scratch = false;
    // False means depth/normal are intentionally stale host mirrors and may
    // only be consumed through the matching CUDA workspace handoff.
    bool cuda_host_output_materialized = true;
};

bool run_recovered_patchmatch_refinement_cuda(
    const PatchMatchRefinementInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error);
bool run_recovered_patchmatch_refinement_cuda_movable(
    PatchMatchRefinementInput&& input,
    PatchMatchCandidateOutput& output,
    std::string& error);

// Exact final-refinement contract.  Unlike the random refinement kernel this
// derives eight deterministic depth candidates around the current plane and
// has no coarse-depth or RNG iteration parameters.
struct PatchMatchFinalRefinementInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    std::vector<std::uint8_t> reference_image;
    std::uint32_t image_one_step_more_detailed = 0;
    float deviation_threshold_multiplier = 1.0F;
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    std::span<const std::uint8_t> reference_image_view;
    std::span<const float> depth_view;
    std::span<const std::uint8_t> normal_view;
    std::span<const float> cost_view;
    std::vector<float> initial_candidate_depth;
    std::vector<float> initial_candidate_normal;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
    std::uint64_t cuda_workspace_handoff = 0;
    bool defer_cuda_host_output = false;
};

bool run_recovered_patchmatch_final_refinement_cuda(
    const PatchMatchFinalRefinementInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error);

struct PatchMatchFinalRefinementOutput {
    PatchMatchCandidateOutput candidates;
    // The target final-refinement kernel also overwrites the active main-cost
    // map before the radius-2 neighbor-cost batch consumes it.
    std::vector<float> cost;
};

bool run_recovered_patchmatch_final_refinement_state_cuda(
    const PatchMatchFinalRefinementInput& input,
    PatchMatchFinalRefinementOutput& output,
    std::string& error);
bool run_recovered_patchmatch_final_refinement_state_cuda_movable(
    PatchMatchFinalRefinementInput&& input,
    PatchMatchFinalRefinementOutput& output,
    std::string& error);

struct PatchMatchWtaInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    std::uint32_t image_one_step_more_detailed = 0;
    std::uint32_t hypotheses_per_pixel = 8;
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    PatchMatchCandidateOutput candidates;
    std::vector<float> average_cost;
    // The target keeps this scratch map across WTA calls.  Checkerboard calls
    // update only their active parity and preserve the other entries.
    std::vector<std::uint8_t> winner;
    std::uint32_t is_checkboard = 0;
    std::uint32_t checkboard_step = 0;
    std::uint32_t only_each_fourth_pixel = 0;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
    // Opaque, session-local producer token returned by the immediately
    // preceding cost batch. Nonzero requests a fail-closed device-resident
    // handoff; callers must never synthesize this value.
    std::uint64_t cuda_workspace_handoff = 0;
    // False means candidates/average_cost are shape-only host mirrors and the
    // immediately preceding cost handoff is the sole authoritative storage.
    bool cuda_cost_output_materialized = true;
    // Production-only optimization: leave the post-WTA main state and winner
    // resident for the immediately following packed-mask copy and producer.
    // A nonzero, valid cost handoff is mandatory; standalone calls remain
    // fully materialized.
    bool defer_cuda_host_output = false;
};

struct PatchMatchWtaOutput {
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    std::vector<std::uint8_t> winner;
    std::uint64_t cuda_workspace_handoff = 0;
    // False means all four vectors are shape-only host mirrors and the stage-2
    // workspace handoff is their sole authoritative storage.
    bool cuda_host_output_materialized = true;
};

bool run_recovered_patchmatch_wta_cuda(
    const PatchMatchWtaInput& input,
    PatchMatchWtaOutput& output,
    std::string& error);
bool run_recovered_patchmatch_wta_cuda_movable(
    PatchMatchWtaInput&& input,
    PatchMatchWtaOutput& output,
    std::string& error);

// Exact post-WTA packed-inlier transfer.  The averaging kernel stores one
// byte for every (8-neighbor group, hypothesis, fixed-capacity pixel slot).
// This kernel selects the WTA-winning hypothesis and writes group-major bytes
// into the persistent full-resolution mask allocation.  Inactive pixels and
// allocation regions are preserved from initial_neighbor_inlier_masks.
struct PatchMatchCopyInlierMasksInput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t hypotheses_per_pixel = 8;
    std::uint32_t neighbor_count = 0;
    std::vector<std::uint8_t> temporary_inlier_masks;
    std::vector<std::uint8_t> winner;
    std::vector<std::uint8_t> initial_neighbor_inlier_masks;
    std::uint32_t is_checkboard = 0;
    std::uint32_t checkboard_step = 0;
    std::uint32_t only_each_fourth_pixel = 0;
    std::uint32_t pixel_offset = 0;
    // Explicit because the target splits the x4 grid into fixed-capacity
    // launches.  Zero is rejected rather than assigned a guessed launch size.
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
    // Opaque token returned by WTA for its immediately following packed-mask
    // copy. A stale/non-matching nonzero token is an error, not a fallback.
    std::uint64_t cuda_workspace_handoff = 0;
    // False means temporary_inlier_masks is a shape-only host mirror.  It is
    // valid only with the immediately preceding WTA handoff.
    bool cuda_temporary_inlier_masks_materialized = true;
    // False means winner is a shape-only host mirror produced by a deferred
    // WTA.  The stage-2 handoff then supplies the authoritative device winner.
    bool cuda_winner_materialized = true;
    // False means the packed-mask backing is a shape-only host mirror.  This
    // is accepted only when the same stage-2 generation proves that an earlier
    // copy left a newer backing resident on the device.
    bool cuda_initial_neighbor_inlier_masks_materialized = true;
    // Production-only optimization: keep the updated packed-mask backing on
    // the device until an explicit C2P/filter/level-product host boundary.
    bool defer_cuda_host_output = false;
};

struct PatchMatchCopyInlierMasksOutput {
    std::vector<std::uint8_t> neighbor_inlier_masks;
    // Opaque proof that main state, cost scratch, winner and packed masks
    // remain resident for the immediately following candidate producer.
    std::uint64_t cuda_workspace_handoff = 0;
    bool cuda_host_output_materialized = true;
};

bool run_recovered_patchmatch_copy_inlier_masks_cuda(
    const PatchMatchCopyInlierMasksInput& input,
    PatchMatchCopyInlierMasksOutput& output,
    std::string& error);
bool run_recovered_patchmatch_copy_inlier_masks_cuda_movable(
    PatchMatchCopyInlierMasksInput&& input,
    PatchMatchCopyInlierMasksOutput& output,
    std::string& error);

// Exact spatial-propagation device contract. Matrix3x3f is represented as
// three float4 rows because the target passes the 3x3 value in a 48-byte,
// 16-byte-aligned CUDA parameter slot.
struct PatchMatchPropagationInput {
    PatchMatchCamera camera;
    std::array<float, 12> reference_to_neighbor_rotation{};
    std::uint32_t depth_downscale = 1;
    std::vector<std::uint8_t> reference_image;
    std::uint32_t image_one_step_more_detailed = 0;
    float deviation_threshold_multiplier = 1.0F;
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    std::vector<float> coarse_depth;
    std::vector<float> coarse_depth_radius;
    std::span<const std::uint8_t> reference_image_view;
    std::span<const float> depth_view;
    std::span<const std::uint8_t> normal_view;
    std::span<const float> cost_view;
    std::span<const float> coarse_depth_view;
    std::span<const float> coarse_depth_radius_view;
    std::vector<float> initial_candidate_depth;
    std::vector<float> initial_candidate_normal;
    std::uint32_t checkboard_step = 0;
    std::uint32_t only_each_fourth_pixel = 0;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
    std::uint64_t cuda_workspace_handoff = 0;
    bool defer_cuda_host_output = false;
};

bool run_recovered_patchmatch_propagation_cuda(
    const PatchMatchPropagationInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error);
bool run_recovered_patchmatch_propagation_cuda_movable(
    PatchMatchPropagationInput&& input,
    PatchMatchCandidateOutput& output,
    std::string& error);

// Exact cross-level candidate construction contract.  The target preserves
// odd/odd pixels from the inherited lattice and writes two slot-major
// hypotheses for every other pixel: the first adjacent lattice value and the
// valid-neighbour mean (depth != 0 and cost < 0.15).
struct PatchMatchCoarseToPreciseInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    // The target reuses the same candidate allocation across kernels.  C2P
    // deliberately leaves odd/odd pixels untouched, so exact full-buffer
    // replay requires their incoming scratch state.  Empty means a diagnostic
    // fresh allocation and is not suitable for production state chaining.
    PatchMatchCandidateOutput initial_candidates;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
    // Production continuation from the preceding WTA/copy-inliers stage.
    // A valid token proves that main state and candidate backing are still in
    // the role-stable CUDA workspace.  Standalone calls keep the old fully
    // materialized contract.
    std::uint64_t cuda_workspace_handoff = 0;
    bool cuda_main_state_materialized = true;
    bool defer_cuda_host_output = false;
};

bool run_recovered_patchmatch_coarse_to_precise_cuda(
    const PatchMatchCoarseToPreciseInput& input,
    PatchMatchCandidateOutput& output,
    std::string& error);
bool run_recovered_patchmatch_coarse_to_precise_cuda_movable(
    PatchMatchCoarseToPreciseInput&& input,
    PatchMatchCandidateOutput& output,
    std::string& error);

struct PatchMatchCostResourceGroup {
    // Target `gpu_neighbImages`: compact mip chains for up to ten cameras are
    // concatenated into one linear source allocation. These offsets are kept
    // separate from the historical mask-offset table below.
    std::vector<std::uint64_t> image_offsets;
    std::vector<std::uint8_t> image;
    // Historical API name retained for capture-replay compatibility.  These
    // are camera-slot byte offsets into `mask`, not mip-level offsets.
    std::vector<std::uint64_t> level_offsets;
    std::vector<std::uint8_t> mask;
};

struct PatchMatchTextureCopyRegion {
    std::uint64_t source_offset = 0;
    std::uint32_t source_pitch = 0;
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct PatchMatchCostNeighbor {
    PatchMatchCamera camera;
    std::uint32_t resource_group = 0;
    std::uint32_t resource_index = 0;
    std::uint32_t output_index = 0;
    std::vector<std::uint8_t> texture;
    // Target `gpu_neighbImages` source: the mip images concatenated without
    // atlas padding.  texture_copy_regions maps these bytes into `texture`'s
    // destination layout with the captured tightly-packed source pitch.
    std::vector<std::uint8_t> texture_source;
    bool texture_source_grouped = false;
    // Production-only non-owning view into the immutable per-camera atlas in
    // RecoveredPatchMatchNeighborResources.  It avoids materializing a full
    // cumulative atlas for every synchronous cost launch.  Capture/replay
    // callers continue to use the owning `texture` vector above.
    std::span<const std::uint8_t> texture_view;
    // Host-side reconstruction aid: bytes copied by this camera into the
    // shared mip atlas. The target preserves all other bytes from the prior
    // selected neighbor instead of clearing the CUDA array.
    std::vector<std::uint8_t> texture_write_mask;
    // Exact mip rectangles copied by the target's prepare operation.  The
    // full host atlas remains available to capture-replay callers, while the
    // persistent CUDA path uploads only these regions after the first
    // neighbour of a reference-camera generation.
    std::vector<PatchMatchTextureCopyRegion> texture_copy_regions;
};

// Host resources recovered from 0x1CE8920, 0x2584C70 and 0x4160640.  Ranked
// neighbours are grouped ten at a time.  Each neighbour owns a mip atlas with
// common dimensions, while every group concatenates the per-camera packed
// rejection bitsets and carries a fixed ten-entry uint64 offset table.
struct RecoveredPatchMatchNeighborResources {
    std::uint32_t base_downscale = 1;
    std::uint32_t texture_width = 0;
    std::uint32_t texture_height = 0;
    std::vector<PatchMatchCostResourceGroup> resource_groups;
    std::vector<PatchMatchCostNeighbor> ranked_neighbors;
};

// Page-locks immutable per-neighbour atlas sources for one reference-camera
// lifetime.  The target's rectangle order and synchronous CUDA semantics are
// unchanged; this only removes pageable-memory staging from the recovered
// implementation.  Destruction unregisters every successfully pinned span
// before the owning neighbor resources can be released.
class RecoveredCudaPinnedNeighborTextures {
public:
    RecoveredCudaPinnedNeighborTextures() = default;
    ~RecoveredCudaPinnedNeighborTextures();
    RecoveredCudaPinnedNeighborTextures(
        const RecoveredCudaPinnedNeighborTextures&) = delete;
    RecoveredCudaPinnedNeighborTextures& operator=(
        const RecoveredCudaPinnedNeighborTextures&) = delete;
    RecoveredCudaPinnedNeighborTextures(
        RecoveredCudaPinnedNeighborTextures&& other) noexcept;
    RecoveredCudaPinnedNeighborTextures& operator=(
        RecoveredCudaPinnedNeighborTextures&& other) noexcept;

    void reset() noexcept;

private:
    friend bool pin_recovered_patchmatch_neighbor_textures_cuda(
        const RecoveredPatchMatchNeighborResources&, std::size_t,
        RecoveredCudaPinnedNeighborTextures&, std::string&);
    std::size_t device_index_ = 0;
    std::vector<void*> pointers_;
};

bool pin_recovered_patchmatch_neighbor_textures_cuda(
    const RecoveredPatchMatchNeighborResources& resources,
    std::size_t device_index,
    RecoveredCudaPinnedNeighborTextures& registration,
    std::string& error);

// Host mirror of the one CUDA neighbour mip atlas owned by a target camera.
// Dynamic capture proves that the target initializes this allocation once to
// zero, overwrites only the selected neighbour rectangles for every prepare,
// retains the untouched bytes throughout all x32..target-level cost bindings,
// and destroys it at the camera boundary.
struct RecoveredPatchMatchCostAtlasState {
    // Process-local identity for one reference-camera resource lifetime.  It
    // is independent of the producer handoff generation, which legitimately
    // changes at host cross-level boundaries while the target atlas and
    // grouped neighbour resources remain resident.
    std::uint64_t generation = 0;
    std::uint32_t texture_width = 0;
    std::uint32_t texture_height = 0;
    std::vector<std::uint8_t> texture;
    // Number of selected-neighbour prepare operations materialized into this
    // camera atlas. South camera 0 d4 has 603, matching the target's
    // 489+78+36 cost launches and independent atlas-lifecycle capture.
    std::size_t prepare_count = 0;
};

RecoveredPatchMatchCostAtlasState
make_recovered_patchmatch_cost_atlas_state(
    const RecoveredPatchMatchNeighborResources& resources);

// Metashape's x32/no-prior path uploads two unwritten host allocations and is
// therefore allocator-state dependent.  Production currently exposes only a
// deterministic, explicitly initialized policy.  A target-bug lifecycle mode
// must not be added by injecting capture payloads or pseudo-random values.
enum class RecoveredPatchMatchNoPriorPolicy : std::uint8_t {
    DeterministicZero = 0,
};

// Scene-derived host resources for the recovered full-frame, source-unmasked
// perspective path.  This object deliberately stops before allocating or
// mutating per-level PatchMatch state; it replaces capture-file camera/atlas
// inputs without pretending that the production orchestrator is wired.
struct RecoveredPatchMatchHostPreparation {
    std::size_t reference_camera_index = 0;
    std::uint32_t target_downscale = 1;
    // CUDA device selected when the preparation was built.  Every recovered
    // level launch must use the same device; silently falling back to device 0
    // breaks non-zero gpu-mask selections and target allocation identity.
    std::size_t device_index = 0;
    RecoveredPatchMatchPreparedCamera reference;
    // Stable scene-camera identities are retained independently of the large
    // prepared image pyramids. Production scene runs can therefore build
    // reference-specific resources directly from the immutable camera cache
    // without deep-copying every neighbour a second time.
    std::vector<std::size_t> ranked_neighbor_camera_indices;
    std::vector<RecoveredPatchMatchPreparedCamera> ranked_neighbors;
    RecoveredPatchMatchNeighborResources neighbor_resources;
    PatchMatchCamera reference_camera;
    PatchMatchNormalRotationCameras normal_rotations;
    std::array<float, 12> propagation_rotation{};
    float deviation_threshold_multiplier = 1.0F;
    RecoveredPatchMatchNoPriorPolicy no_prior_policy =
        RecoveredPatchMatchNoPriorPolicy::DeterministicZero;
};

bool make_recovered_patchmatch_unmasked_host_preparation(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::span<const std::size_t> ranked_neighbor_camera_indices,
    std::uint32_t target_downscale,
    std::size_t device_index,
    RecoveredPatchMatchHostPreparation& output,
    std::string& error,
    std::span<const RecoveredPatchMatchPreparedCamera> prepared_camera_cache = {},
    bool materialize_prepared_neighbors = true);

// Exact linear reference-image allocation used by the cost kernels.  The
// device buffer has the capacity of the final target's one-step-more-detailed
// level (target_downscale / 2), but each current depth level copies only its
// own one-step-more-detailed mip (depth_downscale / 2) at byte offset zero and
// leaves the entire tail zero.  This lifetime/layout rule is independently
// validated at South d32 and d16.
std::vector<std::uint8_t>
make_recovered_patchmatch_reference_image_allocation(
    const RecoveredPatchMatchPreparedCamera& reference,
    std::uint32_t target_downscale,
    std::uint32_t depth_downscale);

// Static part of a target cost batch.  Dynamic depth/normal/cost/candidate and
// checkerboard state remains explicit in PatchMatchCostInput and is not
// invented here.
struct RecoveredPatchMatchCostBinding {
    PatchMatchCamera reference_camera;
    PatchMatchNormalRotationCameras normal_rotations;
    std::vector<std::uint8_t> reference_image_allocation;
    std::span<const std::uint8_t> reference_image_view;
    std::uint32_t reference_image_level_downscale = 1;
    float deviation_threshold_multiplier = 1.0F;
    std::uint32_t neighbor_texture_width = 0;
    std::uint32_t neighbor_texture_height = 0;
    // Zero-backed camera-lifetime atlas used only for the first prepare of a
    // resource generation. Production resources otherwise carry compact mip
    // sources rather than one full atlas per ranked neighbour.
    std::span<const std::uint8_t> initial_neighbor_texture_view;
    std::vector<PatchMatchCostResourceGroup> resource_groups;
    std::span<const PatchMatchCostResourceGroup> resource_groups_view;
    std::vector<PatchMatchCostNeighbor> neighbor_batch;
    std::uint32_t neighbor_count = 0;
    std::uint32_t neighbor_cost_capacity = 0;
    std::uint64_t camera_resource_generation = 0;
};

RecoveredPatchMatchCostBinding make_recovered_patchmatch_cost_binding(
    const RecoveredPatchMatchHostPreparation& preparation,
    std::uint32_t depth_downscale,
    std::uint32_t iteration,
    bool all_neighbors_state = false);

// Stateful production path. Each selected neighbour overwrites its recovered
// rectangles in camera_atlas_state and every batch entry snapshots the full
// resulting atlas presented to that target cost launch.
RecoveredPatchMatchCostBinding make_recovered_patchmatch_cost_binding(
    const RecoveredPatchMatchHostPreparation& preparation,
    RecoveredPatchMatchCostAtlasState& camera_atlas_state,
    std::uint32_t depth_downscale,
    std::uint32_t iteration,
    bool all_neighbors_state = false,
    bool materialize_cumulative_atlas = true);

// Evidence-backed unmasked perspective path.  For a base level below the
// directly undistorted level, a non-zero rejection mask is rejected until the
// target's scale-transition producer is recovered; all-zero South masks are
// packed exactly rather than assigned a guessed 2x2 rule.
bool make_recovered_patchmatch_unmasked_neighbor_resources(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::span<const RecoveredPatchMatchPreparedCamera> ranked_neighbors,
    std::uint32_t target_downscale,
    RecoveredPatchMatchNeighborResources& output,
    std::string& error);

// Cache-backed equivalent used by the all-camera production path. It preserves
// the same crop and atlas construction while avoiding an otherwise redundant
// deep copy of every prepared neighbour pyramid.
bool make_recovered_patchmatch_unmasked_neighbor_resources(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::span<const RecoveredPatchMatchPreparedCamera> prepared_camera_cache,
    std::span<const std::size_t> ranked_neighbor_camera_indices,
    std::uint32_t target_downscale,
    RecoveredPatchMatchNeighborResources& output,
    std::string& error);

// Applies recovered ranked-neighbour selectors and rewrites output_index to
// the dense per-batch slot consumed by pm_avg_all_neighbs_costs.
std::vector<PatchMatchCostNeighbor> make_recovered_patchmatch_cost_batch(
    const RecoveredPatchMatchNeighborResources& resources,
    std::span<const std::uint32_t> ranked_selectors);

std::vector<PatchMatchCostNeighbor> make_recovered_patchmatch_cost_batch(
    const RecoveredPatchMatchNeighborResources& resources,
    std::span<const std::uint32_t> ranked_selectors,
    RecoveredPatchMatchCostAtlasState& camera_atlas_state,
    bool materialize_cumulative_atlas = true);

// Exact CUDA cost sub-pipeline used after propagation/refinement.  A target
// batch rotates normals once, writes multiple neighbor slots in one persistent
// allocation, rotates back once, and averages once.  The legacy scalar fields
// remain for the already validated one-neighbor captures; neighbor_batch and
// resource_groups select the recovered grouped path.
struct PatchMatchCostInput {
    PatchMatchCamera reference_camera;
    PatchMatchCamera neighbor_camera;
    PatchMatchCamera rotate_before_camera;
    PatchMatchCamera rotate_after_camera;
    std::uint32_t depth_downscale = 1;
    std::uint32_t image_one_step_more_detailed = 0;
    float deviation_threshold_multiplier = 1.0F;
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    std::vector<float> coarse_depth;
    std::vector<float> coarse_depth_radius;
    // Production orchestration keeps these host mirrors alive for the whole
    // level. Read-only views avoid copying all five arrays for every cost
    // batch; standalone replay callers may keep using the owning vectors.
    std::span<const float> depth_view;
    std::span<const std::uint8_t> normal_view;
    std::span<const float> cost_view;
    std::span<const float> coarse_depth_view;
    std::span<const float> coarse_depth_radius_view;
    std::vector<std::uint8_t> reference_image_allocation;
    std::span<const std::uint8_t> reference_image_view;
    std::uint32_t neighbor_texture_width = 0;
    std::uint32_t neighbor_texture_height = 0;
    std::vector<std::uint8_t> neighbor_texture;
    std::span<const std::uint8_t> neighbor_texture_view;
    std::vector<std::uint64_t> neighbor_level_offsets;
    std::vector<std::uint8_t> neighbor_mask;
    std::vector<PatchMatchCostResourceGroup> resource_groups;
    std::span<const PatchMatchCostResourceGroup> resource_groups_view;
    std::vector<PatchMatchCostNeighbor> neighbor_batch;
    PatchMatchCandidateOutput candidates;
    // Persistent target allocations. Checkerboard batches update one parity
    // and preserve the other, so exact chaining must supply their prior state.
    std::vector<float> initial_neighbor_cost;
    std::vector<float> initial_average_cost;
    std::vector<std::uint8_t> initial_auxiliary;
    std::uint32_t neighbor_camera_level = 0;
    std::uint32_t neighbor_output_index = 0;
    // Compile-time PATCH_RADIUS specialization of the recovered target
    // kernel. Ordinary batches use 3; the final-refinement cost batch uses 2.
    std::uint32_t reference_patch_radius = 3;
    std::uint32_t hypotheses_per_pixel = 8;
    std::uint32_t neighbor_count = 1;
    std::uint32_t neighbor_cost_capacity = 1;
    std::uint32_t is_checkboard = 0;
    std::uint32_t checkboard_step = 0;
    std::uint32_t only_each_fourth_pixel = 0;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
    std::uint64_t camera_resource_generation = 0;
    // Opaque token returned by refinement/propagation/final-refinement.
    // Nonzero requests a fail-closed producer-to-cost device handoff.
    std::uint64_t cuda_workspace_handoff = 0;
    // Scratch mirrors may be stale while the same session workspace carries
    // the authoritative values through a producer handoff.
    bool initial_cuda_cost_scratch_materialized = true;
    // Production coarsest initialization may assert that all three scratch
    // mirrors are byte-zero. The CUDA path verifies the assertion before
    // replacing their H2D copies with equal-range device memset operations.
    bool initial_cuda_cost_scratch_all_zero = false;
    // Production orchestration may keep the post-cost candidates and scratch
    // device-resident until a proven host/C2P/level boundary.
    bool defer_cuda_host_output = false;
};

struct PatchMatchCostOutput {
    PatchMatchCandidateOutput candidates;
    std::vector<float> per_neighbor_cost;
    std::vector<float> average_cost;
    std::vector<std::uint8_t> auxiliary;
    std::uint64_t cuda_workspace_handoff = 0;
    bool cuda_host_output_materialized = true;
};

bool run_recovered_patchmatch_cost_cuda(
    const PatchMatchCostInput& input,
    PatchMatchCostOutput& output,
    std::string& error);

// Production ownership-transfer form.  The input object is an rvalue so its
// large, device-stale host mirrors can be moved into the output without the
// repeated multi-megabyte copies retained by the replay-safe const API.
bool run_recovered_patchmatch_cost_cuda_movable(
    PatchMatchCostInput&& input,
    PatchMatchCostOutput& output,
    std::string& error);

// First target filtering kernel after final WTA.  The target keeps depth and
// cost in allocations sized for the finest requested level, so vectors may be
// larger than the active camera grid; inactive allocation tails are preserved.
struct PatchMatchFilterCheckCostInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    std::vector<float> depth_allocation;
    std::vector<float> cost_allocation;
    std::int32_t counter_no_cost_samples = 0;
    std::int32_t counter_big_cost_samples = 0;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

struct PatchMatchFilterCheckCostOutput {
    std::vector<float> depth_allocation;
    std::int32_t counter_no_cost_samples = 0;
    std::int32_t counter_big_cost_samples = 0;
};

bool run_recovered_patchmatch_filter_check_cost_cuda(
    const PatchMatchFilterCheckCostInput& input,
    PatchMatchFilterCheckCostOutput& output,
    std::string& error);

struct PatchMatchFilterCheckNeighboursInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    float depth_min = 0.0F;
    float depth_max = 0.0F;
    std::vector<float> depth_allocation;
    std::vector<std::uint8_t> filtered_mask_allocation;
    std::int32_t counter_no_neighbours = 0;
    std::int32_t counter_no_close_neighbours = 0;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

struct PatchMatchFilterCheckNeighboursOutput {
    std::vector<std::uint8_t> filtered_mask_allocation;
    std::int32_t counter_no_neighbours = 0;
    std::int32_t counter_no_close_neighbours = 0;
};

bool run_recovered_patchmatch_filter_check_neighbours_cuda(
    const PatchMatchFilterCheckNeighboursInput& input,
    PatchMatchFilterCheckNeighboursOutput& output,
    std::string& error);

struct PatchMatchFilterClearDepthInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    std::vector<float> depth_allocation;
    std::vector<std::uint8_t> filtered_mask_allocation;
    std::int32_t counter_not_empty = 0;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

struct PatchMatchFilterClearDepthOutput {
    std::vector<float> depth_allocation;
    std::int32_t counter_not_empty = 0;
};

bool run_recovered_patchmatch_filter_clear_depth_cuda(
    const PatchMatchFilterClearDepthInput& input,
    PatchMatchFilterClearDepthOutput& output,
    std::string& error);

struct PatchMatchFilterNormalsInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    std::vector<float> depth_allocation;
    std::vector<std::uint8_t> normal_allocation;
    std::vector<std::uint8_t> estimated_normal_allocation;
    bool estimate_normal_map = false;
    std::vector<std::uint8_t> filtered_mask_allocation;
    std::int32_t counter_inconsistent_normal = 0;
    std::int32_t counter_bad_view_angle_estimated_normal = 0;
    std::int32_t counter_bad_view_angle_found_normal = 0;
    float counter_cos_sum = 0.0F;
    std::int32_t counter_ncos_sum = 0;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

struct PatchMatchFilterNormalsOutput {
    std::vector<std::uint8_t> estimated_normal_allocation;
    std::vector<std::uint8_t> filtered_mask_allocation;
    std::int32_t counter_inconsistent_normal = 0;
    std::int32_t counter_bad_view_angle_estimated_normal = 0;
    std::int32_t counter_bad_view_angle_found_normal = 0;
    float counter_cos_sum = 0.0F;
    std::int32_t counter_ncos_sum = 0;
};

bool run_recovered_patchmatch_filter_normals_cuda(
    const PatchMatchFilterNormalsInput& input,
    PatchMatchFilterNormalsOutput& output,
    std::string& error);

// Final device stage of the first recovered filter pass.  The mask is both
// input scratch and output: every active pixel is replaced by the target's
// eight-neighbour connectivity bit mask; allocation tails are preserved.
struct PatchMatchFilterSpecklesEdgesInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    std::vector<float> depth_allocation;
    std::vector<std::uint8_t> filtered_mask_allocation;
    std::uint32_t pixel_offset = 0;
    std::size_t global_work_items = 0;
    std::size_t device_index = 0;
};

struct PatchMatchFilterSpecklesEdgesOutput {
    std::vector<std::uint8_t> filtered_mask_allocation;
};

bool run_recovered_patchmatch_filter_speckles_edges_cuda(
    const PatchMatchFilterSpecklesEdgesInput& input,
    PatchMatchFilterSpecklesEdgesOutput& output,
    std::string& error);

// Exact host composition of the six filtering stages observed after the final
// WTA at every PatchMatch level.  All allocations are persistent finest-level
// buffers; only the current level prefix is touched.  Counters are explicit
// accumulators because the target kernels do not clear them.  The two
// clear-depth launches reuse one target pointer, but the host clears it before
// each launch; they are represented as independent input values to preserve
// that observed reset boundary.
struct PatchMatchFilterChainInput {
    PatchMatchCamera camera;
    std::uint32_t depth_downscale = 1;
    float depth_min = 0.0F;
    float depth_max = 0.0F;
    std::vector<float> depth_allocation;
    std::vector<float> cost_allocation;
    std::vector<std::uint8_t> normal_allocation;
    std::vector<std::uint8_t> estimated_normal_allocation;
    std::vector<std::uint8_t> filtered_mask_allocation;
    bool estimate_normal_map = false;
    std::int32_t counter_no_cost_samples = 0;
    std::int32_t counter_big_cost_samples = 0;
    std::int32_t counter_no_neighbours = 0;
    std::int32_t counter_no_close_neighbours = 0;
    std::int32_t first_counter_not_empty = 0;
    std::int32_t counter_inconsistent_normal = 0;
    std::int32_t counter_bad_view_angle_estimated_normal = 0;
    std::int32_t counter_bad_view_angle_found_normal = 0;
    float counter_cos_sum = 0.0F;
    std::int32_t counter_ncos_sum = 0;
    std::int32_t second_counter_not_empty = 0;
    std::size_t device_index = 0;
    // Production-only handoff from the final copy-inliers launch.  A nonzero
    // token means main depth/normal/cost and the packed inlier backing are
    // authoritative in the camera workspace; diagnostic callers leave it 0.
    std::uint64_t cuda_workspace_handoff = 0;
    bool cuda_main_state_materialized = true;
    bool cuda_inlier_masks_materialized = true;
    std::vector<std::uint8_t> neighbor_inlier_masks_allocation;
};

struct PatchMatchFilterChainOutput {
    std::vector<float> depth_allocation;
    std::vector<std::uint8_t> estimated_normal_allocation;
    std::vector<std::uint8_t> filtered_mask_allocation;
    std::int32_t counter_no_cost_samples = 0;
    std::int32_t counter_big_cost_samples = 0;
    std::int32_t counter_no_neighbours = 0;
    std::int32_t counter_no_close_neighbours = 0;
    std::int32_t first_counter_not_empty = 0;
    std::int32_t counter_inconsistent_normal = 0;
    std::int32_t counter_bad_view_angle_estimated_normal = 0;
    std::int32_t counter_bad_view_angle_found_normal = 0;
    float counter_cos_sum = 0.0F;
    std::int32_t counter_ncos_sum = 0;
    std::int32_t second_counter_not_empty = 0;
    std::vector<std::uint8_t> neighbor_inlier_masks_allocation;
    // Stage-6 token: filter outputs needed by the host are materialized, while
    // candidate/cost scratch/winner/inlier backing remain resident for the
    // next level's first producer.
    std::uint64_t cuda_workspace_handoff = 0;
};

bool run_recovered_patchmatch_filter_chain_cuda(
    const PatchMatchFilterChainInput& input,
    PatchMatchFilterChainOutput& output,
    std::string& error);

// Host speckle-component pass at 0x1CEC530/0x1CE6270.  The target does not
// consume the GPU edge-mask bytes here: it recomputes a float3 point and local
// sample radius for every valid depth, joins directed 8-neighbour edges using
// the exact radius/distance tests, and clears components of size <= threshold.
bool filter_recovered_patchmatch_speckle_components(
    const Camera& camera,
    std::uint32_t depth_downscale,
    std::vector<float>& depth,
    std::uint32_t component_size_threshold,
    std::string& error);

// CUDA-path component pass at 0x1CED320 with workers 0x1CE6600 and
// 0x1CE68D0.  Unlike the CPU fallback above, it unions the compact edge bits
// emitted by pm_filtering_estimate_speckles_edges.  Border pixels pack only
// in-bounds neighbours, so their bit positions are not the fixed interior
// direction indices.
bool filter_recovered_patchmatch_cuda_speckle_components(
    std::uint32_t width,
    std::uint32_t height,
    std::span<const std::uint8_t> filtered_mask,
    std::vector<float>& depth,
    std::uint32_t component_size_threshold,
    std::string& error);

// Complete level-boundary composition used by the target before the next
// finer-level host transition: six filtering stages followed by the exact
// uchar bilateral kernel.  Same-run host-upload capture proves that bilateral
// consumes the normals-stage estimated-normal allocation, not the original
// main-normal allocation.
struct PatchMatchLevelBoundaryInput {
    PatchMatchFilterChainInput filter;
    std::uint32_t speckle_component_size_threshold = 6;
    std::vector<std::uint8_t> bilateral_image;
    float bilateral_sigma_d = 1.0F;
    float bilateral_sigma_r = 1.0F;
};

struct PatchMatchLevelBoundaryOutput {
    PatchMatchFilterChainOutput filter;
    PatchMatchBilateralU8Output bilateral;
};

bool run_recovered_patchmatch_level_boundary_cuda(
    const PatchMatchLevelBoundaryInput& input,
    PatchMatchLevelBoundaryOutput& output,
    std::string& error);

// Production-facing state for the recovered coarsest (x32) CUDA level.  The
// vectors use active-grid sizes except for the target's fixed candidate/cost
// scratch allocations.
struct RecoveredPatchMatchLevelState {
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    std::vector<std::uint8_t> winner;
    PatchMatchCandidateOutput candidates;
    std::vector<float> neighbor_cost;
    std::vector<float> average_cost;
    std::vector<std::uint8_t> auxiliary;
    // False is permitted only while the enclosing CUDA session workspace is
    // carrying these three arrays through an opaque producer handoff.
    bool cuda_cost_scratch_materialized = true;
    // False is permitted only between an explicitly deferred WTA and the next
    // proven host boundary while the pipeline handoff remains nonzero.
    bool cuda_main_state_materialized = true;
    // False is permitted only while the current handoff generation carries a
    // newer packed-mask backing to the next copy-inliers invocation.
    bool cuda_inlier_masks_materialized = true;
    // Nonzero only on the production path between a completed level filter
    // and the first producer of the next level.
    std::uint64_t cuda_workspace_handoff = 0;
    // Target-owned finest-level backing reused by copy-inliers at every WTA.
    // Each current level reinterprets its active prefix with a group stride of
    // current_width*current_height; the backing is never resized or cleared at
    // a level transition.
    std::vector<std::uint8_t> neighbor_inlier_masks;
};

struct RecoveredPatchMatchCoarsestLevelOutput {
    RecoveredPatchMatchLevelState state_before_filter;
    PatchMatchLevelBoundaryOutput boundary;
};

// Exact ordinary-frame CUDA coarsest-level composition recovered from the
// target: fresh state -> six refinement iterations, each with two spatial
// propagation checkerboards -> filter/component/bilateral boundary.
bool run_recovered_patchmatch_coarsest_level_cuda(
    const RecoveredPatchMatchHostPreparation& preparation,
    std::span<const std::uint8_t> bilateral_image,
    RecoveredPatchMatchCoarsestLevelOutput& output,
    std::string& error,
    RecoveredPatchMatchCostAtlasState* camera_atlas_state = nullptr,
    bool capture_diagnostic_checkpoints = true);

// Observable x16 checkpoints kept deliberately separate from the final state.
// They make the recovered continuous chain falsifiable against the target's
// before-C2P and after-WTA2 captures without feeding target intermediates back
// into the calculation.
struct RecoveredPatchMatchX16Checkpoint {
    std::vector<float> depth;
    std::vector<std::uint8_t> normal;
    std::vector<float> cost;
    std::vector<std::uint8_t> winner;
    PatchMatchCandidateOutput candidates;
    std::vector<float> neighbor_cost;
    std::vector<float> average_cost;
    std::vector<std::uint8_t> auxiliary;
    std::vector<std::uint8_t> neighbor_inlier_masks;
};

struct RecoveredPatchMatchX16LevelOutput {
    RecoveredPatchMatchCrossLevelState cross_level_initial;
    RecoveredPatchMatchX16Checkpoint before_c2p;
    RecoveredPatchMatchX16Checkpoint after_wta2;
    RecoveredPatchMatchLevelState state_before_filter;
    PatchMatchLevelBoundaryOutput boundary;
};

// Exact single-batch x16 continuation used by South d4/d8/d2 schedules:
// x32 bilateral host transition -> six inherited quarter-lattice iterations
// -> C2P/cost/WTA2 -> iterations 7 and 8 -> final refinement -> filter,
// component pass and bilateral boundary.  Persistent candidate/winner/cost
// scratch is inherited from x32 exactly; no diagnostic fresh allocation is
// accepted here.
bool run_recovered_patchmatch_x16_level_cuda(
    const Camera& camera,
    const RecoveredPatchMatchHostPreparation& preparation,
    const RecoveredPatchMatchCoarsestLevelOutput& coarsest,
    std::span<const std::uint8_t> bilateral_image,
    RecoveredPatchMatchX16LevelOutput& output,
    std::string& error,
    RecoveredPatchMatchCostAtlasState* camera_atlas_state = nullptr,
    bool capture_diagnostic_checkpoints = true,
    RecoveredPatchMatchLevelState* consumable_previous_state = nullptr);

// Evidence-backed continuation for every level finer than x16.  The target
// branch is selected by downscale == preparation.target_downscale:
// intermediate levels run six inherited quarter-lattice iterations, C2P at
// state 6, iterations 7/8, normal estimation and bilateral; the target level
// runs one inherited iteration, C2P at state 1, iteration 2, an all-neighbor
// final cost, no normal estimation and no bilateral.  Batch offsets use the
// recovered fixed-capacity balanced partition.
struct RecoveredPatchMatchFinerLevelOutput {
    std::uint32_t downscale = 0;
    bool target_level = false;
    bool bilateral_executed = false;
    RecoveredPatchMatchCrossLevelState cross_level_initial;
    RecoveredPatchMatchX16Checkpoint before_c2p;
    RecoveredPatchMatchX16Checkpoint after_wta2;
    RecoveredPatchMatchLevelState state_before_filter;
    PatchMatchLevelBoundaryOutput boundary;
    // Host snapshots appended before the optional bilateral.  These are the
    // products consumed by voting; bilateral output is only the next-level
    // prior and must never replace level_product_depth.
    std::vector<float> level_product_depth;
    std::vector<std::uint8_t> level_product_inlier_masks;
    std::vector<std::size_t> ranked_neighbor_camera_indices;
};

bool run_recovered_patchmatch_finer_level_cuda(
    const Camera& camera,
    const RecoveredPatchMatchHostPreparation& preparation,
    std::uint32_t downscale,
    const RecoveredPatchMatchLevelState& previous_state_before_filter,
    const PatchMatchLevelBoundaryOutput& previous_boundary,
    std::span<const std::uint8_t> bilateral_image,
    RecoveredPatchMatchFinerLevelOutput& output,
    std::string& error,
    RecoveredPatchMatchCostAtlasState* camera_atlas_state = nullptr,
    bool capture_diagnostic_checkpoints = true,
    RecoveredPatchMatchLevelState* consumable_previous_state = nullptr);

// Complete evidence-backed South d4 product boundary for one reference
// camera with the observed 6..16-neighbor closure.  Six neighbors are the
// smallest dynamically traced target case; the same recovered selector and
// packed-mask formulas cover every count through the 16-neighbor maximum.
// Levels are ordered
// d4, d8, d16 exactly as required by depth voting and already include the
// target PXR24 persistence round-trip plus final-depth mask gating.  Packed
// inlier masks retain one byte containing eight ranked-neighbor bits per pixel
// group; use unpack_recovered_patchmatch_inlier_mask before constructing a
// DepthVotingPreparedNeighbor.  Other target downscales and neighbor-count
// domains remain fail-closed until numerically validated.
struct RecoveredPatchMatchD4PyramidOutput {
    std::size_t camera_index = 0;
    std::array<std::vector<float>, 3> depth_levels;
    std::array<std::vector<std::uint8_t>, 3> packed_inlier_masks;
    std::vector<std::size_t> ranked_neighbor_camera_indices;
    std::size_t neighbor_atlas_prepare_count = 0;
    double host_preparation_seconds = 0.0;
    double host_registration_seconds = 0.0;
    double x32_seconds = 0.0;
    double x16_seconds = 0.0;
    double x8_seconds = 0.0;
    double x4_seconds = 0.0;
    double product_seconds = 0.0;
};

bool run_recovered_patchmatch_d4_pyramid_cuda(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::span<const std::size_t> ranked_neighbor_camera_indices,
    std::size_t device_index,
    RecoveredPatchMatchD4PyramidOutput& output,
    std::string& error,
    std::span<const RecoveredPatchMatchPreparedCamera> prepared_camera_cache = {},
    bool capture_diagnostic_checkpoints = true,
    RecoveredPatchMatchHostPreparation* prebuilt_host_preparation = nullptr,
    double prebuilt_host_preparation_seconds = 0.0);

// All-reference production boundary.  Every requested reference PM pyramid is
// completed first; only then are per-reference voting inputs assembled from
// the reference's own packed masks and the selected neighbors' depth
// pyramids.  This prevents partial per-camera results from being mistaken for
// a valid voting closure.
struct RecoveredPatchMatchD4CameraOutput {
    RecoveredPatchMatchD4PyramidOutput patchmatch;
    DepthVotingChainOutput voting;
    // Metashape DepthMap.image() default view.  This is not the persisted
    // level-0 image: valid coarse triangles are projected coarse-to-fine and
    // fill only zero-valued pixels.  BuildModel keeps the three persisted
    // levels separately, so callers must not substitute this composite for
    // voting.depth_after_components[0] without a proven consumer boundary.
    std::vector<float> public_depth;
    double patchmatch_seconds = 0.0;
    double voting_seconds = 0.0;
};

// Recovered native DepthMap.image() composition for the observed d4 pyramid.
// `persisted_depth_levels` are ordered d4/d8/d16.  The routine uses the exact
// target triangle diagonal choice, normalized-camera normal gate (0.2), and
// zero-only coarse-to-fine fill.  It is deliberately fail-closed for cameras
// outside the ordinary finite pinhole/divisible d4 domain.
bool compose_recovered_depthmap_default_image_d4(
    const Camera& camera,
    const std::array<std::span<const float>, 3>& persisted_depth_levels,
    std::vector<float>& output,
    std::string& error);

// Measured physical CUDA module/function activity for one explicit recovered
// execution session. A scene session may contain nested per-camera scopes;
// only the outermost scope owns and unloads the cached modules. These fields
// describe execution identity and cache behavior only: kernel ordering,
// synchronization and device-buffer lifetimes remain governed by the
// recovered algorithm wrappers.
struct RecoveredCudaModuleSessionStats {
    std::size_t device_index = 0;
    std::string device_name;
    std::uint64_t total_global_memory = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int driver_version = 0;
    int runtime_version = 0;
    std::uint64_t scope_entries = 0;
    std::uint64_t module_load_requests = 0;
    std::uint64_t module_cache_hits = 0;
    std::uint64_t physical_module_loads = 0;
    std::uint64_t module_release_requests = 0;
    std::uint64_t physical_module_unloads = 0;
    std::uint64_t function_requests = 0;
    std::uint64_t function_cache_hits = 0;
    std::uint64_t physical_function_lookups = 0;
    std::uint64_t set_device_calls = 0;
    std::uint64_t cuda_malloc_calls = 0;
    std::uint64_t cuda_malloc_bytes = 0;
    std::uint64_t cuda_free_calls = 0;
    std::uint64_t cuda_free_null_calls = 0;
    std::uint64_t host_to_device_copy_calls = 0;
    std::uint64_t host_to_device_copy_bytes = 0;
    std::uint64_t host_to_device_copy_nanoseconds = 0;
    std::uint64_t device_to_host_copy_calls = 0;
    std::uint64_t device_to_host_copy_bytes = 0;
    std::uint64_t device_to_host_copy_nanoseconds = 0;
    std::uint64_t device_to_device_copy_calls = 0;
    std::uint64_t device_to_device_copy_bytes = 0;
    std::uint64_t memset_calls = 0;
    std::uint64_t memset_bytes = 0;
    std::uint64_t array_create_calls = 0;
    std::uint64_t array_destroy_calls = 0;
    std::uint64_t array_copy_calls = 0;
    std::uint64_t array_copy_bytes = 0;
    std::uint64_t array_copy_nanoseconds = 0;
    std::uint64_t host_register_calls = 0;
    std::uint64_t host_register_bytes = 0;
    std::uint64_t host_register_nanoseconds = 0;
    std::uint64_t host_unregister_calls = 0;
    std::uint64_t host_unregister_nanoseconds = 0;
    std::uint64_t texture_create_calls = 0;
    std::uint64_t texture_destroy_calls = 0;
    std::uint64_t kernel_launches = 0;
    std::uint64_t context_synchronizations = 0;
    std::uint64_t stream_synchronizations = 0;
    std::uint64_t stream_synchronization_nanoseconds = 0;
    // First O2 optimization boundary: the role-stable linear buffers and
    // texture backing used by a complete PatchMatch cost batch are retained
    // for the enclosing CUDA session.  These counters distinguish logical
    // cost calls from physical workspace growth.
    std::uint64_t cost_workspace_acquires = 0;
    std::uint64_t cost_workspace_reused_buffers = 0;
    std::uint64_t cost_workspace_physical_allocations = 0;
    std::uint64_t cost_workspace_peak_bytes = 0;
    std::uint64_t producer_workspace_acquires = 0;
    std::uint64_t wta_workspace_acquires = 0;
    std::uint64_t inlier_workspace_acquires = 0;
    std::uint64_t coarse_to_precise_workspace_acquires = 0;
    std::uint64_t filter_workspace_acquires = 0;
    std::uint64_t workspace_handoff_skipped_h2d_calls = 0;
    std::uint64_t workspace_handoff_skipped_h2d_bytes = 0;
    std::uint64_t workspace_handoff_skipped_d2h_calls = 0;
    std::uint64_t workspace_handoff_skipped_d2h_bytes = 0;
    // A complete three-level voting chain keeps reference depth/radius/votes,
    // the current neighbor pyramid/masks and all counters in a role-stable
    // workspace.  Public single-kernel replay APIs remain fully materialized;
    // these counters describe only the production chain boundary.
    std::uint64_t voting_workspace_acquires = 0;
    std::uint64_t voting_workspace_reused_buffers = 0;
    std::uint64_t voting_workspace_physical_allocations = 0;
    std::uint64_t voting_workspace_peak_bytes = 0;
    std::uint64_t voting_depth_cache_entries = 0;
    std::uint64_t voting_depth_cache_bytes = 0;
    std::uint64_t voting_depth_cache_hits = 0;
    std::uint64_t voting_depth_cache_hit_bytes = 0;
    // Physical H2D attribution uses the same execution-phase scopes as D2H.
    // Keeping both directions symmetric makes host materialization and
    // device-resident handoff regressions visible in production manifests.
    std::uint64_t undistort_h2d_calls = 0;
    std::uint64_t undistort_h2d_bytes = 0;
    std::uint64_t producer_h2d_calls = 0;
    std::uint64_t producer_h2d_bytes = 0;
    std::uint64_t cost_h2d_calls = 0;
    std::uint64_t cost_h2d_bytes = 0;
    std::uint64_t cost_main_state_h2d_calls = 0;
    std::uint64_t cost_main_state_h2d_bytes = 0;
    std::uint64_t cost_group_resources_h2d_calls = 0;
    std::uint64_t cost_group_resources_h2d_bytes = 0;
    std::uint64_t cost_scratch_h2d_calls = 0;
    std::uint64_t cost_scratch_h2d_bytes = 0;
    std::uint64_t cost_texture_sources_h2d_calls = 0;
    std::uint64_t cost_texture_sources_h2d_bytes = 0;
    std::uint64_t wta_h2d_calls = 0;
    std::uint64_t wta_h2d_bytes = 0;
    std::uint64_t inlier_h2d_calls = 0;
    std::uint64_t inlier_h2d_bytes = 0;
    std::uint64_t coarse_to_precise_h2d_calls = 0;
    std::uint64_t coarse_to_precise_h2d_bytes = 0;
    std::uint64_t bilateral_h2d_calls = 0;
    std::uint64_t bilateral_h2d_bytes = 0;
    std::uint64_t filter_h2d_calls = 0;
    std::uint64_t filter_h2d_bytes = 0;
    std::uint64_t voting_h2d_calls = 0;
    std::uint64_t voting_h2d_bytes = 0;
    std::uint64_t uncategorized_h2d_calls = 0;
    std::uint64_t uncategorized_h2d_bytes = 0;
    // Diagnostic attribution for the physical D2H traffic that remains after
    // workspace handoff.  These counters do not alter scheduling; they make
    // the next materialization boundary an evidence-based choice.
    std::uint64_t undistort_d2h_calls = 0;
    std::uint64_t undistort_d2h_bytes = 0;
    std::uint64_t producer_d2h_calls = 0;
    std::uint64_t producer_d2h_bytes = 0;
    std::uint64_t cost_d2h_calls = 0;
    std::uint64_t cost_d2h_bytes = 0;
    std::uint64_t wta_d2h_calls = 0;
    std::uint64_t wta_d2h_bytes = 0;
    std::uint64_t inlier_d2h_calls = 0;
    std::uint64_t inlier_d2h_bytes = 0;
    std::uint64_t coarse_to_precise_d2h_calls = 0;
    std::uint64_t coarse_to_precise_d2h_bytes = 0;
    std::uint64_t bilateral_d2h_calls = 0;
    std::uint64_t bilateral_d2h_bytes = 0;
    std::uint64_t filter_d2h_calls = 0;
    std::uint64_t filter_d2h_bytes = 0;
    std::uint64_t voting_d2h_calls = 0;
    std::uint64_t voting_d2h_bytes = 0;
    std::uint64_t uncategorized_d2h_calls = 0;
    std::uint64_t uncategorized_d2h_bytes = 0;
};

bool begin_recovered_cuda_module_session(
    std::size_t device_index,
    std::string& error);

// Uploads immutable persisted PM depth levels once per scene session. Voting
// workers still copy into their target-shaped role buffers, but use exact D2D
// bytes instead of repeatedly staging the same camera pyramid through host
// memory. Calls outside an active session fail closed.
bool prime_recovered_cuda_voting_depth_cache(
    std::span<const std::span<const float>> depth_levels,
    std::size_t device_index,
    std::string& error);

// Ends one host-pointer identity epoch without closing the enclosing CUDA
// module session. Callers must clear the cache before freeing or relocating
// any primed host span. The operation fails while a voting workspace is in
// use, preventing a batch loader from publishing dangling cache identities.
bool clear_recovered_cuda_voting_depth_cache(
    std::size_t device_index,
    std::string& error);

bool end_recovered_cuda_module_session(
    RecoveredCudaModuleSessionStats& stats,
    std::string& error);

struct RecoveredPatchMatchD4SceneOutput {
    std::vector<RecoveredPatchMatchD4CameraOutput> cameras;
    std::size_t patchmatch_worker_count = 0;
    // Host-side dynamic allocation high-water accounting for the scene
    // scheduler. These counts use vector capacities (not logical sizes), so
    // they describe owned payload capacity without conflating it with RSS,
    // CUDA allocations, allocator metadata, or driver/JIT memory.
    std::uint64_t prepared_camera_cache_bytes = 0;
    std::uint64_t prepared_camera_cache_final_bytes = 0;
    std::uint64_t prepared_job_peak_bytes = 0;
    std::uint64_t resource_preparation_inflight_peak_bytes = 0;
    std::uint64_t prepared_queue_peak_bytes = 0;
    std::uint64_t active_prepared_jobs_peak_bytes = 0;
    std::uint64_t prepared_host_live_peak_bytes = 0;
    std::size_t prepared_queue_peak_jobs = 0;
    std::size_t resource_preparation_inflight_peak_count = 0;
    std::size_t active_prepared_jobs_peak_count = 0;
    std::uint64_t prepared_queue_byte_budget = 0;
    double camera_preparation_seconds = 0.0;
    double patchmatch_seconds = 0.0;
    double voting_wall_seconds = 0.0;
    double voting_seconds = 0.0;
    RecoveredCudaModuleSessionStats cuda_module_session;
    std::filesystem::path patchmatch_store_root;
    bool patchmatch_store_used = false;
    std::size_t voting_batch_size = 0U;
};

bool run_recovered_patchmatch_d4_scene_cuda(
    const Scene& scene,
    std::span<const std::size_t> reference_camera_indices,
    std::span<const std::vector<std::size_t>> ranked_neighbors_by_camera,
    FilterMode filter_mode,
    std::size_t device_index,
    RecoveredPatchMatchD4SceneOutput& output,
    std::string& error,
    bool retain_voting_diagnostics = true,
    const std::filesystem::path& patchmatch_store_root = {},
    std::size_t voting_batch_size = 16U);

bool unpack_recovered_patchmatch_inlier_mask(
    std::span<const std::uint8_t> packed_masks,
    std::size_t pixels,
    std::size_t neighbor_count,
    std::size_t ranked_neighbor_index,
    std::vector<std::uint8_t>& output,
    std::string& error);

// Reproduces the PM-to-voting persistence boundary.  Metashape stores each
// component-filtered depth level as a FLOAT/Z OpenEXR using PXR24 compression,
// then reloads it before voting.  PXR24 rounds away the low eight float bits.
// Packed inlier bits are independently retained only where that level's final
// depth is valid.  The function is deliberately fail-closed outside the
// observed finite, non-negative depth domain.
bool make_recovered_patchmatch_voting_level_product(
    std::span<const float> component_filtered_depth,
    std::span<const std::uint8_t> raw_packed_inlier_masks,
    std::size_t neighbor_count,
    std::vector<float>& persisted_depth,
    std::vector<std::uint8_t>& persisted_packed_inlier_masks,
    std::string& error);

}  // namespace metmodel
