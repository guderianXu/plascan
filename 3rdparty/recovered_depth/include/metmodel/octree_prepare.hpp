#pragma once

#include "metmodel/fusion.hpp"
#include "metmodel/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace metmodel {

// Exact CPU/CUDA HistVoxel ABI consumed by sub_1EB3B20 and
// cuda::ooc_histogram_init. This is a different 28-byte record from the
// Morton-keyed persistent OocOctreeRecord below: offsets +0..+11 are xyz
// float32 coordinates and the ten mutable bins are exactly +12..+21.
struct OocHistogramVoxel {
    std::array<float, 3> position{};
    std::array<std::uint8_t, 10> histogram{};
    std::uint16_t scalar_lut_index{};
    std::array<std::uint8_t, 4> metadata{};
};

static_assert(sizeof(OocHistogramVoxel) == 28);
static_assert(offsetof(OocHistogramVoxel, histogram) == 12);
static_assert(offsetof(OocHistogramVoxel, scalar_lut_index) == 22);
static_assert(offsetof(OocHistogramVoxel, metadata) == 24);

// State at 0x1EB448C after the four selected float2 pyramid texels have been
// accumulated. Field names describe their proven arithmetic role; the exact
// Metashape UI names for threshold_b and vote_base have not been recovered.
struct OocHistogramAccumulatedSample {
    float depth_sum{};
    float auxiliary_sum{};
    float valid_weight{};
    float level_threshold{};
    float geometry_distance{};
    std::uint32_t pyramid_scale{};
    float projected_depth{};
    float threshold_b{};
    std::uint32_t vote_base{};
};

struct OocHistogramPyramidSelection {
    float projected_x{};
    float projected_y{};
    std::int32_t level_width{};
    std::int32_t level_height{};
    std::uint32_t pyramid_scale{};
    std::uint32_t base_float_offset{};
};

// Numeric camera-model mode 0 fields shared by the recovered OOC point-scale
// and histogram-projection workers. Intrinsics use Metashape's centered
// representation: pixel principal point = (width/2 + cx, height/2 + cy).
struct OocSampleScaleCameraMode0 {
    std::int64_t width{};
    std::int64_t height{};
    double focal_length{};
    double principal_x{};
    double principal_y{};
    double additive_focal{};
    double shear{};
};

enum class OocHistogramProjectionDecision : std::uint8_t {
    Selected,
    BehindCamera,
    OutsideImage,
};

struct OocHistogramProjectionInput {
    OocSampleScaleCameraMode0 camera;
    std::array<double, 16> world_to_camera{};
    std::array<float, 16> camera_to_world{};
    std::span<const float> scalar_lut;
    std::uint32_t pyramid_levels{};
    float threshold_a{};
    std::uint32_t mode_b{};
};

// Exact by-value CUDA records used by cuda::ooc_histogram_init. Unknown or
// non-semantic ABI padding remains zero; the target wrapper leaves a few such
// bytes as stack residue, and they are not read by the recovered PTX.
struct alignas(8) OocHistogramCalibrationCu {
    std::array<std::byte, 120> bytes{};
};

struct alignas(4) OocHistogramRollingShutterCu {
    std::array<float, 6> values{};
};

struct alignas(16) OocHistogramMatrix3x3f {
    std::array<float, 12> values{};
};

struct alignas(16) OocHistogramCameraExteriorTransformCu {
    std::array<float, 16> values{};
};

static_assert(sizeof(OocHistogramCalibrationCu) == 120);
static_assert(sizeof(OocHistogramRollingShutterCu) == 24);
static_assert(sizeof(OocHistogramMatrix3x3f) == 48);
static_assert(sizeof(OocHistogramCameraExteriorTransformCu) == 64);

struct OocHistogramCudaCameraParameters {
    OocHistogramCalibrationCu calibration;
    OocHistogramRollingShutterCu rolling_shutter;
    OocHistogramMatrix3x3f before_rotation;
    OocHistogramMatrix3x3f after_rotation;
    OocHistogramCameraExteriorTransformCu exterior_transform;
};

// Exact ordinary, non-rolling-shutter mode-0 packer recovered from wrapper
// 0x25581B0. camera_to_world uses the row-major float4x4 representation of the
// CPU worker; the CUDA exterior record moves its translation column into the
// final row without any additional arithmetic.
[[nodiscard]] OocHistogramCudaCameraParameters
make_ooc_histogram_cuda_camera_parameters_mode0(
    const OocHistogramProjectionInput& input);

// One target CUDA camera pass. The six scalar fields and the five by-value
// records are the complete 14-parameter ooc_histogram_init ABI recovered from
// build 22956. The auxiliary pointer parameter is not exposed: the recovered
// sm_35 PTX declares it but never loads it.
struct OocHistogramCudaCameraInput {
    OocHistogramCudaCameraParameters parameters;
    std::vector<float> concatenated_float2_pyramid;
    std::uint32_t pyramid_levels{};
    std::uint32_t mode_b{};
    float threshold_a{};
    float threshold_b{};
};

// The target Shoe execution used four persistent partition buffers and ran
// one synchronized kernel per (camera, partition) pair. Partitions remain
// explicit here because the upstream general partitioner is not yet proven;
// this API does not infer a split from the record count.
struct OocHistogramCudaChainInput {
    std::array<std::vector<OocHistogramVoxel>, 4> initial_partitions;
    std::vector<OocHistogramCudaCameraInput> cameras;
    std::size_t device_index{};
};

struct OocHistogramCudaChainOutput {
    std::array<std::vector<OocHistogramVoxel>, 4> partitions;
    std::uint64_t kernel_launches{};
};

// Exact semantic inputs to sub_1778FD0/sub_17A00F0, the build-22956 bridge
// from balanced Morton nodes to the temporary HistVoxel table.  root_scale is
// the expanded cubic root used by Morton encoding; region_size remains the
// unexpanded oriented box, so the expanded cube is anchored at the box's
// negative corner just as in the target.
struct OocWeightedNodeRecord;

struct OocHistogramVoxelBuildInput {
    std::span<const OocWeightedNodeRecord> balanced_records;
    std::array<double, 9> region_rotation{};
    std::array<double, 3> region_center{};
    std::array<double, 3> region_size{};
    float root_scale{};
};

// Produces the sentinel followed by the target's level-major/Morton-major
// records, including the exact seven-nearest-neighbour radius adaptation.
// Target bytes +24/+25 are uninitialized stack residue and are never consumed;
// this deterministic API writes zero there while reproducing every semantic
// byte (+0..+23 and +26/+27).
[[nodiscard]] std::vector<OocHistogramVoxel>
build_ooc_histogram_voxels_mode0(
    const OocHistogramVoxelBuildInput& input);

// Exact four-way host partition at 0x1EA47E9..0x1EA4893. The target computes
// chunk=ceil(record_count/4), creates four independent buffers, and copies
// consecutive clamped slices; it does not round the record count for CUDA.
[[nodiscard]] std::array<std::vector<OocHistogramVoxel>, 4>
partition_ooc_histogram_voxels_four_way(
    std::span<const OocHistogramVoxel> voxels);

// Exact 33-bin tail-quantile selector at 0x1EC5461..0x1EC553B. The bins are
// uint64 counts for level codes 0..32. Build 22956 multiplies their total by
// float bits 0x3DCCCCD0, walks the tail from code 32 downward, and never
// returns a code below 1. The result is the registry field at +0x438.
[[nodiscard]] std::uint32_t select_ooc_pyramid_maximum_level(
    const std::array<std::uint64_t, 33>& level_counts) noexcept;

struct OocPyramidRegistryLevelMetadata {
    std::uint32_t maximum_level{};
    // Exact valid-pixel population of level codes 0..32.  The registry file
    // stores only maximum_level and the selected 32-bin histogram.  Retaining
    // the complete distribution here makes the preceding per-camera 10%
    // selector independently auditable.
    std::array<std::uint64_t, 33> level_counts{};
    std::array<float, 32> selected_level_histogram{};
};

struct OocAdaptiveRootMode0 {
    std::uint32_t maximum_level{};
    std::uint32_t winning_bin{};
    double scale_factor{};
    float root_scale{};
    std::uint64_t total_samples{};
    std::array<double, 33> level_counts{};
    std::array<double, 32> selected_level_histogram{};
};

// Exact no-point-cloud adaptive-root reduction in sub_178D9C0.  It combines
// all per-camera registry statistics, selects a 10% upper-tail level, then a
// 30% low-tail sub-bin at that level.  The resulting float is rw.data and the
// selected level is last_level.data.
[[nodiscard]] OocAdaptiveRootMode0 derive_ooc_adaptive_root_mode0(
    std::span<const OocPyramidRegistryLevelMetadata> metadata,
    const std::array<double, 3>& region_size);

// Build 22956 `pyramidsAll.registry` envelope. The per-camera 0x4d4-byte
// payload contains additional fields whose semantics are not all recovered,
// so this boundary preserves those bytes opaquely instead of naming or
// synthesizing them. On disk: little-endian uint64 record count followed by
// exactly count contiguous raw records and no trailer.
using OocPyramidRegistryRawRecord = std::array<std::byte, 0x4D4>;

struct OocPyramidRegistryPayloadLocation {
    std::uint64_t shard_index{};
    std::uint64_t byte_offset{};
    std::uint64_t byte_size{};
};

struct OocPyramidPayloadLevelLayout {
    std::uint64_t first_exr_offset{};
    std::uint64_t first_exr_size{};
    std::uint64_t second_exr_offset{};
    std::uint64_t second_exr_size{};
    std::uint64_t interlevel_plane_offset{};
    std::uint64_t interlevel_plane_size{};
    std::uint64_t next_width{};
    std::uint64_t next_height{};
};

struct OocPyramidPayloadLayout {
    std::uint64_t table_count{};
    std::uint64_t encoded_blob_offset{};
    std::uint64_t encoded_blob_size{};
    std::vector<OocPyramidPayloadLevelLayout> levels;
};

struct OocSampleScalePyramidOutput;

// Exact hidden-state reduction performed by OpenMP worker 0x1EBEC80 after
// the two camera-reprojection levels have been produced. The target retains
// the float temporary plane between levels but serializes only its clamped
// uint8 view; callers must therefore carry `temporary` without reconstructing
// it from the payload file. An empty gate means every output pixel is active.
struct OocPyramidReductionInput {
    std::uint64_t width{};
    std::uint64_t height{};
    std::span<const float> depth;
    std::span<const float> temporary;
    std::span<const float> sample_scale;
    std::span<const std::uint8_t> output_gate;
    bool special_invalid_depth{};
};

struct OocPyramidReductionOutput {
    std::uint64_t width{};
    std::uint64_t height{};
    std::vector<float> depth;
    std::vector<float> temporary;
    std::vector<float> sample_scale;
    std::vector<std::uint8_t> temporary_u8;
};

// Exact 0x1EBEC80 arithmetic and row-major instruction order: every output
// pixel consumes the corresponding 2x2 source block, discards odd tail rows
// and columns, ignores source depth 0/-2^40, averages depth and sample scale
// with the hidden temporary weights, doubles sample scale, and clamps the
// truncated new temporary to uint8. This is not the distinct 0x1EBFC60
// camera-reprojection worker used for the first two reductions.
[[nodiscard]] OocPyramidReductionOutput reduce_ooc_pyramid_level(
    const OocPyramidReductionInput& input);

// Exact record tail at +0x4bc/+0x4c4/+0x4cc. In the retained target runs the
// first word selects `pyramids{N}.data`; the next two delimit this record's
// byte range within that shard.
[[nodiscard]] OocPyramidRegistryPayloadLocation
ooc_pyramid_registry_payload_location(
    const OocPyramidRegistryRawRecord& record) noexcept;

// Exact per-camera `pyramidsN.data` segment envelope consumed by
// 0x1EA8F90..0x1EA9600. At every level the first PXR24 FLOAT-Z EXR stores depth
// and the second stores sample scale. Every non-final level additionally
// serializes the next level's clamped uint8 temporary plane, but not the float
// temporary weights consumed by the following reduction. The leading uint64
// table therefore has 4*level_count-2 entries, followed by one uint64 blob
// size and exactly that many bytes. The first two reductions are produced by
// the distinct camera-reprojection worker 0x1EBFC60 and remain outside the
// recovered 0x1EBEC80 arithmetic API above.
[[nodiscard]] OocPyramidPayloadLayout parse_ooc_pyramid_payload_layout(
    std::span<const std::byte> segment,
    std::uint32_t level_count);

// The target's PXR24 bytes are version-sensitive. Build 22956 is reproduced
// by OpenEXR 3.2.2 plus zlib 1.3.2; the serializer fails closed for any other
// runtime pair instead of emitting a readable but byte-different cache.
[[nodiscard]] bool ooc_pyramid_exact_encoder_available() noexcept;

// Exact per-camera `pyramidsN.data` segment writer. Each level stores a
// single-channel FLOAT-Z PXR24 tiled EXR for depth and sample scale; every
// non-final level then stores the following level's clamped temporary-u8
// plane. The target's fixed 18-byte minimal Exif blob is preserved.
[[nodiscard]] std::vector<std::byte> serialize_ooc_pyramid_payload(
    const OocSampleScalePyramidOutput& pyramid);

[[nodiscard]] std::vector<std::byte> serialize_ooc_pyramid_registry(
    std::span<const OocPyramidRegistryRawRecord> records);

[[nodiscard]] std::vector<OocPyramidRegistryRawRecord>
deserialize_ooc_pyramid_registry(std::span<const std::byte> bytes);

// Exact registry +0x438/+0x43c producer at
// 0x1EC5418..0x1EC5679. All three images are flattened in the target's common
// pixel order. Depth 0 and bits 0xD3800000 are excluded; NaNs follow the
// target unordered-comparison path and remain eligible. Level codes outside
// 0..32, or selected-level histogram codes outside 0..31, fail closed.
[[nodiscard]] OocPyramidRegistryLevelMetadata
make_ooc_pyramid_registry_level_metadata(
    std::span<const float> depth,
    std::span<const std::uint8_t> level_codes,
    std::span<const std::uint8_t> histogram_codes);

struct OocPyramidMetadataCodes {
    std::uint8_t level{};
    std::uint8_t histogram_code{};
};

// Exact finite-positive quantizer at 0x1EBFA3F..0x1EBFB19 plus helper
// 0x1EA3AE0. `base_scale` is the pre-level scale already multiplied by 0.5;
// `value` is the positive per-pixel scale ratio. The level is selected by the
// repeated 0.75/0.5 test and the residual is quantized through 255.89999 then
// shifted from eight bits to five. Other numeric domains fail closed.
[[nodiscard]] OocPyramidMetadataCodes quantize_ooc_pyramid_metadata_codes(
    float value, float base_scale);

// CUDA-only, exact-build mode-0 execution boundary. Device allocations, PTX
// module, stream, four histogram partitions and the pyramid buffer persist for
// the whole chain; every kernel is followed by a stream synchronization as in
// the target wrapper. Unsupported or incomplete inputs fail closed.
bool run_recovered_ooc_histogram_cuda_chain(
    const OocHistogramCudaChainInput& input,
    OocHistogramCudaChainOutput& output,
    std::string& error);

struct OocHistogramProjectionResult {
    OocHistogramProjectionDecision decision{};
    OocHistogramPyramidSelection selection;
    float support_scale{};
    float base_geometry_distance{};
    float geometry_distance{};
    float projected_depth{};
    float reference_depth{};
    float level_threshold{};
    std::uint32_t level_selection_vote{};

    [[nodiscard]] bool selected() const noexcept {
        return decision == OocHistogramProjectionDecision::Selected;
    }
};

// Exact observed ordinary-camera mode-0 path at 0x1EB3BD8..0x1EB420E:
// world-to-camera projection, float image bounds, two half-pixel
// unprojections, float4x4 world-space distance, optional diagonal adjustment,
// record/LUT threshold production, and concatenated-pyramid level selection.
// Other numeric calibration modes are intentionally outside this API.
[[nodiscard]] OocHistogramProjectionResult
prepare_ooc_histogram_pyramid_selection_mode0(
    const OocHistogramVoxel& voxel,
    const OocHistogramProjectionInput& input);

struct OocHistogramBilinearAccumulation {
    float depth_sum{};
    float auxiliary_sum{};
    float valid_weight{};

    [[nodiscard]] bool valid() const noexcept { return valid_weight > 0.0F; }
};

// Exact 0x1EB420E..0x1EB448C four-corner lookup. The pyramid is the target's
// concatenated float2 storage exposed as scalar floats; base_float_offset is
// therefore measured in floats, not pixels. A first component equal to zero
// or the -2^40 sentinel invalidates only that corner.
[[nodiscard]] OocHistogramBilinearAccumulation
accumulate_ooc_histogram_pyramid_sample(
    const OocHistogramPyramidSelection& selection,
    std::span<const float> concatenated_float2_pyramid) noexcept;

enum class OocHistogramSampleDecision : std::uint8_t {
    Accepted,
    NegativeDepthGate,
    GeometryGate,
};

struct OocHistogramAcceptedVote {
    OocHistogramSampleDecision decision{};
    float normalized_residual{};
    float raw_vote_weight{};

    [[nodiscard]] bool accepted() const noexcept {
        return decision == OocHistogramSampleDecision::Accepted;
    }
};

// Exact 0x1EB448C..0x1EB457C acceptance/weight path. This covers accumulator
// normalization, both rejection gates, the 1.3..1.7 geometry ramp, residual
// clamp and pyramid-scale vote multiplier. It deliberately starts after the
// projection, pyramid-level selection and four-texel sampling boundary.
[[nodiscard]] OocHistogramAcceptedVote evaluate_ooc_histogram_sample(
    const OocHistogramAccumulatedSample& sample) noexcept;

// Exact accepted-vote tail of sub_1EB3B20. normalized_residual is the value
// already clamped to [-1, 1] by the projection/depth gate. raw_vote_weight is
// the camera/level weight before its uint8 truncation. Weight <= 1 updates the
// nearest bin through float addition/truncation. Larger weights are divided
// between the adjacent bin centers with roundf and saturate at 254. A bin set
// to 255 is a locked sentinel and is never modified.
void accumulate_ooc_histogram_vote(
    OocHistogramVoxel& voxel,
    float normalized_residual,
    float raw_vote_weight) noexcept;

enum class OocHistogramCameraVoteDecision : std::uint8_t {
    Accepted,
    BehindCamera,
    OutsideImage,
    InvalidPyramidSample,
    NegativeDepthGate,
    GeometryGate,
};

struct OocHistogramCameraVoteResult {
    OocHistogramCameraVoteDecision decision{};
    OocHistogramProjectionResult projection;
    OocHistogramBilinearAccumulation accumulation;
    OocHistogramAcceptedVote vote;

    [[nodiscard]] bool accepted() const noexcept {
        return decision == OocHistogramCameraVoteDecision::Accepted;
    }
};

// Exact same-record mode-0 path from projection through the final mutable
// ten-bin histogram write. This is the continuously captured
// 0x1EB3BD8..0x1EB4994 path. The pyramid is fail-closed before lookup so a
// truncated production cache cannot be misinterpreted as an invalid sample.
[[nodiscard]] OocHistogramCameraVoteResult
accumulate_ooc_histogram_camera_vote_mode0(
    OocHistogramVoxel& voxel,
    const OocHistogramProjectionInput& input,
    float threshold_b,
    std::span<const float> concatenated_float2_pyramid);

// Exact 28-byte persistent node record consumed by sub_1EAC320 in Metashape
// 2.3.2 build 22956. The three words form one MSB-first 96-bit Morton path.
struct OocOctreeRecord {
    std::array<std::uint32_t, 3> morton_words{};
    std::uint8_t level{};
    std::uint8_t weight{};
    std::array<std::uint8_t, 10> histogram{};
    std::uint16_t scalar_lut_index{};
    std::uint16_t trailing_field{};
};

static_assert(sizeof(OocOctreeRecord) == 28);

// Exact 16-byte weighted Morton node used before the persistent 28-byte form.
// The unaligned little-endian uint16 at +13 and uint8 denominator at +15 are
// represented as bytes so the layout remains portable and explicit.
struct OocWeightedNodeRecord {
    std::array<std::uint32_t, 3> morton_words{};
    std::uint8_t level{};
    std::array<std::uint8_t, 2> weight_sum_le{};
    std::uint8_t weight_denominator{};
};

static_assert(sizeof(OocWeightedNodeRecord) == 16);

struct OocInitialWeightSelection {
    std::uint8_t level{};
    float cell_scale{};
    float sample_scale{};
    bool direct_zero{};
};

[[nodiscard]] std::uint16_t ooc_weight_sum(
    const OocWeightedNodeRecord& record) noexcept;

void set_ooc_weight_sum(
    OocWeightedNodeRecord& record, std::uint16_t value) noexcept;

// Exact sub_1EA3AE0 initial-sample weight quantizer. The target evaluates this
// in float32 and truncates the clamped [0, 1] result after multiplying by the
// literal float32 value 255.9f (bits 0x437fe666).
[[nodiscard]] std::uint8_t quantize_ooc_initial_weight(
    float cell_scale, float sample_scale) noexcept;

// Exact adaptive level choice at 0x17A5150..0x17A57B9. The two root scales
// drive independent 0.75*cell threshold searches; the selected level is then
// constrained by maximum_level. Clamp-path samples are restricted to
// [0.75*cell, 1.5*cell], while the selected-level-zero fallback bypasses the
// quantizer and writes zero directly.
[[nodiscard]] OocInitialWeightSelection select_ooc_initial_weight(
    float root_scale,
    float alternate_scale,
    float raw_sample_scale,
    std::uint32_t maximum_level) noexcept;

// Exact record writes at 0x17A553F..0x17A5565. raw_denominator is multiplied
// as uint32; the target then stores only the product's low 16 bits and the
// denominator's low 8 bits. The selected_level==0 branch passes weight zero.
void set_initial_ooc_weight(
    OocWeightedNodeRecord& record,
    std::uint32_t raw_denominator,
    std::uint8_t quantized_weight) noexcept;

// Exact coordinate quantization and XYZ bit spreading at
// 0x17A51F6..0x17A551E. Coordinates are divided in float32, promoted to
// double, scaled by 2^32, truncated/saturated, and written as an MSB-first
// 96-bit path with X/Y/Z occupying bits 2/1/0 of each level.
[[nodiscard]] std::array<std::uint32_t, 3> encode_ooc_morton_words(
    const std::array<float, 3>& position,
    float root_scale,
    std::uint8_t level) noexcept;

// Exact float4 sample generated by sub_1EC26A0 before Morton encoding.
// position is expressed in the shifted adaptive-root coordinate system and
// scale is the filtered depth-map sample radius.
struct OocDepthCandidate {
    std::array<float, 3> position{};
    float scale{};
};

static_assert(sizeof(OocDepthCandidate) == 16);

// Exact per-candidate record production chain at 0x17A5110..0x17A5564.
// This joins adaptive level selection, Morton encoding, weight quantization,
// and the final unaligned 16-byte record writes.
[[nodiscard]] OocWeightedNodeRecord make_initial_ooc_weighted_node(
    const OocDepthCandidate& candidate,
    float root_scale,
    float alternate_scale,
    std::uint32_t maximum_level,
    std::uint32_t raw_denominator) noexcept;

// Exact cache-boundary quantizer observed between sub_1EBF020's full-resolution
// sample-scale output and sub_1EC26A0's loaded input. Metashape adds 0x80 to
// the unsigned float32 bit pattern and then discards its lowest eight bits.
// Exact midpoint values therefore always carry into the retained part; this
// is deliberately not IEEE round-to-nearest-even.
// This retains sign, exponent and the upper 15 fraction bits (24 stored bits
// including the cleared byte in the in-memory float32 representation).
[[nodiscard]] float quantize_ooc_sample_scale_cache(float value) noexcept;

void quantize_ooc_sample_scale_cache(std::span<float> values) noexcept;

// Exact 24-byte per-pixel source record consumed by sub_1EBF020. The first
// triple is the reconstructed point in the worker's transform domain; the
// second triple is the associated direction used by the incidence-angle
// correction.
struct OocSampleScalePointRecord {
    std::array<float, 3> point{};
    std::array<float, 3> direction{};
};

static_assert(sizeof(OocSampleScalePointRecord) == 24);

struct OocSampleScaleWorkerInput {
    std::size_t width{};
    std::size_t height{};
    std::span<const float> depth;
    std::span<const OocSampleScalePointRecord> point_records;
    OocSampleScaleCameraMode0 camera;
    std::array<double, 16> camera_to_record{};
    float maximum_sample_scale{};
    bool diagonal_pixel_scale{};
};

struct OocSampleScaleWorkerOutput {
    std::vector<float> depth;
    std::vector<float> temporary;
    std::vector<float> sample_scale;
    std::vector<std::uint8_t> level;
    std::vector<std::uint8_t> level_weight;
};

// Exact normal-execution input of the distinct first-two-pyramid-level worker
// sub_1EBFC60. Unlike sub_1EBF020, depth is both the prepopulated input and an
// output, the byte plane is only a one-bit accepted marker, and there is no
// maximum-sample-scale rejection. Numeric camera modes other than mode 0 and
// cancellation during the worker remain deliberately outside this API.
struct OocPyramidFinerLevelInput {
    std::size_t width{};
    std::size_t height{};
    std::span<const float> depth;
    std::span<const OocSampleScalePointRecord> point_records;
    OocSampleScaleCameraMode0 camera;
    std::array<double, 16> camera_to_record{};
    std::span<const std::uint8_t> gate;
    bool diagonal_pixel_scale{};
    bool special_invalid_depth{};
};

struct OocPyramidFinerLevelOutput {
    std::vector<float> depth;
    std::vector<float> temporary;
    std::vector<float> sample_scale;
    std::vector<std::uint8_t> temporary_u8;
    std::uint64_t accepted{};
    float maximum_depth{};
};

[[nodiscard]] OocPyramidFinerLevelOutput build_ooc_pyramid_finer_level(
    const OocPyramidFinerLevelInput& input);

// Exact normal-path float-depth ROI expansion performed by sub_1EC1950 before
// the three OOC sample-scale seeds are dispatched. The cache reader supplies
// only [x_begin,x_end) x [y_begin,y_end); the target allocates the full image,
// fills every other float with positive zero, and copies each ROI row without
// numeric conversion. The coordinates are expressed in the full image.
struct OocDepthRoiExpansionInput {
    std::size_t full_width{};
    std::size_t full_height{};
    std::size_t x_begin{};
    std::size_t y_begin{};
    std::size_t x_end{};
    std::size_t y_end{};
    std::span<const float> roi_depth;
};

[[nodiscard]] std::vector<float> expand_ooc_depth_roi(
    const OocDepthRoiExpansionInput& input);

struct OocDepthRoiBounds;

// Adapter-equivalent inverse of the recovered ROI expansion for an in-memory
// voted depth product. It extracts the same tightly packed row-major rectangle
// that the target cache later reloads before sub_1EC1950 expands it. This is a
// verified voting-to-OOC boundary, not a claim about the cache file envelope.
[[nodiscard]] std::vector<float> extract_ooc_depth_roi(
    std::size_t full_width,
    std::size_t full_height,
    const OocDepthRoiBounds& bounds,
    std::span<const float> full_depth);

// Exact normal-path perspective ROI producer at 0x1EC88A3..0x1EC9492.  The
// target samples a 4x4x4 grid in the oriented reconstruction box, projects
// every point through the rectified pinhole camera, truncates projected
// coordinates toward zero, adds a five-percent integer margin, and clamps the
// result to the depth-product dimensions.  region_rotation is the row-major
// matrix stored in the project; the target uses its columns as the three box
// axes.  The currently proven domain is the target's numeric camera mode 1
// with disabled distortion/correction/rolling shutter and identity final
// projective transform--that is, the rectified perspective depth camera.
struct OocDepthRoiBoundsMode0Input {
    OocSampleScaleCameraMode0 camera;
    std::array<double, 16> world_to_camera{};
    std::array<double, 9> region_rotation{};
    std::array<double, 3> region_center{};
    std::array<double, 3> region_size{};
};

// Public project-boundary form. camera_to_world is the raw 4x4 matrix parsed
// from the project; unlike OocDepthRoiBoundsMode0Input, callers do not need to
// reproduce Camera.transform's hidden SVD conditioning or its full inverse.
struct OocDepthRoiProjectMode0Input {
    OocSampleScaleCameraMode0 camera;
    std::array<double, 16> camera_to_world{};
    std::array<double, 9> region_rotation{};
    std::array<double, 3> region_center{};
    std::array<double, 3> region_size{};
};

struct OocDepthRoiBounds {
    std::int64_t x_begin{-1};
    std::int64_t y_begin{-1};
    std::int64_t x_end{};
    std::int64_t y_end{};

    [[nodiscard]] bool valid() const noexcept {
        return x_begin >= 0 && y_begin >= 0 &&
               x_begin < x_end && y_begin < y_end;
    }
};

struct OocDepthRoiProjectionSample {
    std::array<double, 3> world{};
    std::array<double, 2> projected{};
    bool valid{};
};

struct OocDepthRoiBoundsOutput {
    OocDepthRoiBounds bounds;
    std::array<OocDepthRoiProjectionSample, 64> samples;
    std::size_t valid_samples{};
};

// Exact Matrix -> Camera.transform conditioning primitive at sub_2BA66F0.
// The target first divides all 16 elements by the homogeneous bottom-right
// value when it is nonzero, projects the upper-left 3x3 block onto its
// orthogonal polar factor using sub_4109450's recovered Golub-Reinsch SVD,
// preserves the normalized translation column, and clears the first three
// elements of the bottom row. No rotation-log/Rodrigues round trip occurs.
[[nodiscard]] std::array<double, 16> condition_ooc_camera_transform_mode0(
    const std::array<double, 16>& matrix);

// Exact full-projective cofactor inverse used by the target before the ROI
// producer and again by the OOC depth-candidate bridge. It deliberately does
// not replace the camera transform with a rigid transpose: the project XML
// contains rounded doubles, so that shortcut changes low bits.
[[nodiscard]] std::array<double, 16> invert_ooc_projective_matrix4(
    const std::array<double, 16>& matrix);

[[nodiscard]] OocDepthRoiBoundsOutput compute_ooc_depth_roi_bounds_mode0(
    const OocDepthRoiBoundsMode0Input& input);

[[nodiscard]] OocDepthRoiBoundsOutput
compute_ooc_depth_roi_bounds_from_project_mode0(
    const OocDepthRoiProjectMode0Input& input);

// Exact scalar passed to sub_1EC4970. The caller compares the three region
// dimensions in X/Y/Z order with ordered greater-than comparisons, then
// converts the selected double to float32.
[[nodiscard]] float derive_ooc_maximum_sample_scale(
    const std::array<double, 3>& region_size) noexcept;

// Exact ROI transition used for the two saved finer depth products: lower
// coordinates use floor(x/2), upper coordinates use ceil(x/2).
[[nodiscard]] OocDepthRoiBounds downsample_ooc_depth_roi_bounds(
    const OocDepthRoiBounds& bounds);

// Three independently supplied depth products seed the OOC sample-scale
// pyramid observed in build 22956: the full depth product is processed by
// 0x1EBF020, the next two products by 0x1EBFC60, and only then does
// 0x1EBEC80 recursively reduce the third result. The seed cameras remain
// explicit because their calibration belongs to each incoming depth product;
// this API does not infer an unproven odd-dimension scaling convention.
struct OocSampleScalePyramidSeedInput {
    std::size_t width{};
    std::size_t height{};
    std::span<const float> depth;
    OocSampleScaleCameraMode0 camera;
    std::span<const std::uint8_t> gate;
};

struct OocSampleScalePyramidInput {
    std::array<OocSampleScalePyramidSeedInput, 3> seeds;
    std::array<double, 16> camera_to_record{};
    float maximum_sample_scale{};
    bool diagonal_pixel_scale{};
    bool special_invalid_depth{};

    // Empty means that every recursive output pixel is active. Otherwise the
    // vector must contain exactly one gate for every 0x1EBEC80 output level.
    std::vector<std::span<const std::uint8_t>> reduction_gates;
};

struct OocSampleScalePyramidRoiSeedInput {
    OocDepthRoiExpansionInput depth;
    OocSampleScaleCameraMode0 camera;
    std::span<const std::uint8_t> gate;
};

// Production-facing form of the recovered bridge: three independently
// decoded cache ROIs are expanded exactly as sub_1EC1950 and then passed to
// the already recovered full/finer/reduction state machine. ROI derivation
// from an arbitrary camera model and reconstruction region is intentionally a
// separate boundary; callers must provide the target coordinates.
struct OocSampleScalePyramidRoiInput {
    std::array<OocSampleScalePyramidRoiSeedInput, 3> seeds;
    std::array<double, 16> camera_to_record{};
    float maximum_sample_scale{};
    bool diagonal_pixel_scale{};
    bool special_invalid_depth{};
    std::vector<std::span<const std::uint8_t>> reduction_gates;
};

// Autonomous rectified-camera bridge.  Unlike OocSampleScalePyramidRoiInput,
// this form does not accept target ROI coordinates: it derives the full ROI
// from camera + reconstruction region and derives the next two by the exact
// lower-floor/upper-ceil transitions before expanding the three decoded ROI
// payloads.
struct OocSampleScalePyramidRegionInput {
    OocDepthRoiProjectMode0Input bounds;
    std::array<std::span<const float>, 3> roi_depths;
    std::array<OocSampleScaleCameraMode0, 3> cameras;
    std::array<std::span<const std::uint8_t>, 3> gates;
    std::array<double, 16> camera_to_record{};
    float maximum_sample_scale{};
    bool diagonal_pixel_scale{};
    bool special_invalid_depth{};
    std::vector<std::span<const std::uint8_t>> reduction_gates;
};

struct OocSampleScalePyramidLevel {
    std::size_t width{};
    std::size_t height{};
    std::vector<float> depth;
    std::vector<float> temporary;
    std::vector<float> sample_scale;
    // The full-resolution 0x1EBF020 worker has no corresponding byte output.
    // Every subsequent level carries the exact byte plane produced with it.
    std::vector<std::uint8_t> temporary_u8;
};

struct OocSampleScalePyramidOutput {
    std::vector<OocSampleScalePyramidLevel> levels;
    std::vector<std::uint8_t> initial_level;
    std::vector<std::uint8_t> initial_level_weight;
    std::array<std::uint64_t, 2> finer_accepted{};
    std::array<float, 2> finer_maximum_depth{};
};

// Exact normal-path, numeric-camera-mode-0 `pyramidsAll.registry` record.
// Build 22956 constructs the 0x4d4-byte POD from the SVD-conditioned camera
// transform, the rectified calibration, the autonomous ROI, the recovered
// pyramid metadata, and the payload location assigned by the outer shard
// writer. The three input depth products are a fixed part of this proven
// path; the output level count is taken from `pyramid.levels`.
struct OocPyramidRegistryMode0Input {
    std::uint32_t camera_id{};
    OocDepthRoiProjectMode0Input project;
    OocPyramidRegistryPayloadLocation payload;
};

[[nodiscard]] OocPyramidRegistryRawRecord
make_ooc_pyramid_registry_record_mode0(
    const OocPyramidRegistryMode0Input& input,
    const OocSampleScalePyramidOutput& pyramid);

// Proven single-shard outer writer for numeric camera mode 0. Items are
// serialized in caller order because retained target runs show that registry
// order follows parallel worker completion rather than camera ID. Payload
// offsets and sizes are derived from the exact encoded segments; callers
// cannot inject them independently.
struct OocPyramidBundleMode0Item {
    std::uint32_t camera_id{};
    OocDepthRoiProjectMode0Input project;
};

struct OocPyramidBundleSingleShardMode0Output {
    std::vector<std::byte> registry;
    std::vector<std::byte> payload;
};

[[nodiscard]] OocPyramidBundleSingleShardMode0Output
serialize_ooc_pyramid_bundle_single_shard_mode0(
    std::span<const OocPyramidBundleMode0Item> items,
    std::span<const OocSampleScalePyramidOutput> pyramids);

// Exact balanced camera-group sizing performed by sub_17F4D90. A zero public
// work-item or group limit is normalized to one by the target. For N>0,
// G=min(max_groups, ceil(N/workitem_size)); the first N%G groups contain one
// extra item and every range is consecutive in the caller's upstream order.
// Empty input is rejected because the retained target path never schedules a
// zero-camera BuildModel group.
struct OocPyramidCameraGroupRange {
    std::size_t begin_index{};
    std::size_t item_count{};
};

[[nodiscard]] std::vector<OocPyramidCameraGroupRange>
partition_ooc_pyramid_camera_groups(
    std::size_t item_count,
    std::size_t workitem_size_cameras,
    std::size_t max_workgroup_size);

// Per-group CPU scheduling recovered from sub_17DAD50. The target first asks
// OpenMP for its available team size, budgets one concurrent camera as
// 40*max_depthmap_pixels bytes, and requests the memory-capped team size for
// the outer dynamic camera loop. The inner thread count passed to each camera
// worker is ceil(available_team/min(outer_threads,camera_count)).
struct OocPyramidWorkerPlan {
    std::uint64_t target_memory_bytes{};
    std::uint32_t available_openmp_threads{};
    std::size_t max_depthmap_pixels{};
    std::size_t camera_task_count{};
    std::uint32_t requested_outer_threads{};
    std::uint32_t active_camera_slots{};
    std::uint32_t per_camera_inner_threads{};
    bool memory_limited{};
};

[[nodiscard]] OocPyramidWorkerPlan plan_ooc_pyramid_workers(
    std::uint64_t available_memory_bytes,
    std::uint32_t available_openmp_threads,
    std::size_t max_depthmap_pixels,
    std::size_t camera_task_count);

// Multi-group outer writer recovered from sub_17DAD50 and sub_17DC830. One
// upstream camera group maps to one pyramids{group}.data shard. Items within
// each group must be supplied in observed worker-completion order; the total
// registry concatenates local registries in increasing group index. Payload
// offsets restart at zero in every shard and +0x4bc stores the group index.
struct OocPyramidBundleMode0Group {
    std::span<const OocPyramidBundleMode0Item> items;
    std::span<const OocSampleScalePyramidOutput> pyramids;
};

struct OocPyramidBundleMode0Output {
    std::vector<std::byte> registry;
    std::vector<std::vector<std::byte>> payload_shards;
};

[[nodiscard]] OocPyramidBundleMode0Output
serialize_ooc_pyramid_bundle_mode0(
    std::span<const OocPyramidBundleMode0Group> groups);

// Production-facing recovered bridge from three full-image cross-camera
// voting products. It derives project ROIs autonomously, extracts the cache
// rectangles without changing float bits, expands them through sub_1EC1950's
// semantics, and runs the recovered mode-0 pyramid state machine.
struct OocSampleScalePyramidVotedDepthInput {
    OocDepthRoiProjectMode0Input bounds;
    std::array<std::span<const float>, 3> voted_depths;
    std::array<OocSampleScaleCameraMode0, 3> cameras;
    std::array<std::span<const std::uint8_t>, 3> gates;
    bool diagonal_pixel_scale{};
    bool special_invalid_depth{};
    std::vector<std::span<const std::uint8_t>> reduction_gates;
};

// Public Scene adapter for the recovered rectified perspective mode-0 path
// with BuildModel.volumetric_masks=false. Scene calibration stores an absolute
// principal point and a world-to-camera pose; the target OOC worker consumes
// centered intrinsics at each voted-depth level plus the raw camera-to-world
// project matrix. The reconstruction region is mandatory because it defines
// the autonomous ROI.
//
// captured_diagonal_pixel_scale is deliberately explicit. In target build
// 22956 the type-6 constructor leaves context+0xd8 uninitialized; independent
// runs have reached the worker with both 1 and 0x0000cc01. Exact replay must
// therefore carry the observed branch value instead of inventing a setting.
[[nodiscard]] OocDepthRoiProjectMode0Input
make_ooc_depth_roi_project_mode0_input(
    const Scene& scene,
    std::size_t camera_index,
    std::uint32_t depth_downscale);

[[nodiscard]] OocSampleScalePyramidOutput
build_ooc_sample_scale_pyramid_from_scene_voted_depths_mode0(
    const Scene& scene,
    std::size_t camera_index,
    const std::array<std::span<const float>, 3>& voted_depths,
    std::uint32_t depth_downscale,
    bool captured_diagonal_pixel_scale);

[[nodiscard]] OocSampleScalePyramidOutput
build_ooc_sample_scale_pyramid_from_scene_voted_depths_mode0(
    const Scene& scene,
    std::size_t camera_index,
    const std::array<std::vector<float>, 3>& voted_depths,
    std::uint32_t depth_downscale,
    bool captured_diagonal_pixel_scale);

// Deterministic all-camera bridge from completed voting products to the
// exact mode-0 pyramid payload shards and registry. Input order is preserved
// within the recovered balanced camera groups. Target registry order follows
// nondeterministic worker completion, so callers seeking same-run file bytes
// must provide views in the observed completion order; camera IDs and payload
// contents remain semantic regardless of that ordering.
struct OocSceneVotedDepthMode0View {
    std::size_t camera_index{};
    std::array<std::span<const float>, 3> voted_depths;
    bool captured_diagonal_pixel_scale{};
};

struct OocSceneVotedDepthBundleMode0Output {
    std::vector<OocPyramidCameraGroupRange> camera_groups;
    std::vector<OocPyramidBundleMode0Item> items;
    std::vector<OocSampleScalePyramidOutput> pyramids;
    OocPyramidBundleMode0Output bundle;
};

[[nodiscard]] OocSceneVotedDepthBundleMode0Output
build_ooc_pyramid_bundle_from_scene_voted_depths_mode0(
    const Scene& scene,
    std::span<const OocSceneVotedDepthMode0View> cameras,
    std::uint32_t depth_downscale,
    std::size_t workitem_size_cameras,
    std::size_t max_workgroup_size);

// Exact normal-execution orchestration for numeric camera mode 0, no
// mid-worker cancellation, and caller-supplied gates. It never derives the
// second or third depth seed from the first and never reconstructs hidden
// float temporary weights from the serialized uint8 planes.
[[nodiscard]] OocSampleScalePyramidOutput
build_ooc_sample_scale_pyramid_mode0(
    const OocSampleScalePyramidInput& input);

[[nodiscard]] OocSampleScalePyramidOutput
build_ooc_sample_scale_pyramid_from_rois_mode0(
    const OocSampleScalePyramidRoiInput& input);

[[nodiscard]] OocSampleScalePyramidOutput
build_ooc_sample_scale_pyramid_from_region_mode0(
    const OocSampleScalePyramidRegionInput& input);

[[nodiscard]] OocSampleScalePyramidOutput
build_ooc_sample_scale_pyramid_from_voted_depths_mode0(
    const OocSampleScalePyramidVotedDepthInput& input);

// Continuous production bridge from the exact mode-0 sample-scale worker to
// registry +0x438/+0x43c. This overload deliberately consumes the worker's
// computed images rather than a captured/externally supplied intermediate.
[[nodiscard]] OocPyramidRegistryLevelMetadata
make_ooc_pyramid_registry_level_metadata(
    const OocSampleScaleWorkerOutput& output);

// Exact full-rectangle, numeric-camera-mode-0 point-record producer feeding
// sub_1EBF020. This is the observed sub_209F160/sub_209F190/sub_209F3B0 path:
// zero depth is invalid, the four 2x2 topology bits select depth-continuous
// triangles, and area-weight-independent unit face normals are accumulated and
// normalized per vertex. The caller-visible result is the same 24-byte record
// layout consumed by OocSampleScaleWorkerInput.
[[nodiscard]] std::vector<OocSampleScalePointRecord>
build_ooc_sample_scale_point_records_mode0(
    std::size_t width,
    std::size_t height,
    std::span<const float> depth,
    const OocSampleScaleCameraMode0& camera,
    const std::array<double, 16>& camera_to_record);

// Exact empty-validity-mask, numeric-camera-mode-0 path of sub_1EBF020. It
// reproduces the three sub_27CE000 calls, homogeneous matrix transforms,
// float32 distance/incidence arithmetic, threshold gate, adaptive 0.75 level
// search, and the quantized five-bit weight output. Other numeric camera modes
// and the optional validity-mask mutation path remain deliberately outside
// this API rather than being approximated.
[[nodiscard]] OocSampleScaleWorkerOutput build_ooc_sample_scale_workspace(
    const OocSampleScaleWorkerInput& input);

struct OocDepthCandidateInput {
    std::size_t width{};
    std::size_t height{};
    std::span<const float> depth;
    std::span<const float> sample_scale;

    // Effective zero-index depth-camera calibration. For a rectified Metashape
    // depth map these are original f/downscale and original absolute principal
    // point/downscale - 0.5.
    double focal_length{};
    double principal_x{};
    double principal_y{};

    // Row-major homogeneous transforms. camera_to_world is the camera pose
    // stored in the 288-byte work item. The target obtains world_to_root by a
    // full cofactor inverse of root_to_world; it does not assume a rigid matrix.
    // root_extent belongs to the adaptive root frame in the shared context.
    std::array<double, 16> camera_to_world{};
    std::array<double, 16> root_to_world{};
    std::array<double, 3> root_extent{};
};

// Exact projection, two homogeneous transforms/divides, root shift and
// inclusive bounds gate at 0x1EC3180..0x1EC347A. Invalid pixels remain
// all-zero records.
[[nodiscard]] std::vector<OocDepthCandidate> build_ooc_depth_candidate_workspace(
    const OocDepthCandidateInput& input);

// Exact positive-threshold branch at 0x1EC3549..0x1EC3D6A. The target forms a
// cross-neighbor normal, centers the plane on the center plus four axial
// neighbors, and rejects one outlier by using the second-largest 3x3 residual.
// Passing records have only their scale doubled.
void apply_ooc_depth_candidate_planarity(
    std::span<OocDepthCandidate> workspace,
    std::size_t width,
    std::size_t height,
    float threshold);

// Exact row-major nonzero-scale compaction performed by the common tail.
[[nodiscard]] std::vector<OocDepthCandidate> compact_ooc_depth_candidates(
    std::span<const OocDepthCandidate> workspace);

// Exact finite-sample aggregation performed by the latter half of
// sub_1ED34B0. At most eight scalar/weight pairs are accepted because that is
// the target sampler's fixed neighborhood capacity.
[[nodiscard]] float aggregate_ooc_marching_scalar(
    std::span<const float> scalars,
    std::span<const float> weights);

// Exact sub_1EA3C00 quotient, including the denominator-zero representation.
[[nodiscard]] std::uint8_t normalized_ooc_weight(
    const OocWeightedNodeRecord& record);

// Exact sub_1EA3B60 merge and its >254 denominator renormalization.
void merge_ooc_weighted_node(
    OocWeightedNodeRecord& target,
    const OocWeightedNodeRecord& source);

// Exact unsigned lexicographic key used by the target sort helpers at
// 0x1804280/0x1804410: morton_words[0..2], then level. Weight fields do not
// participate in ordering.
[[nodiscard]] bool ooc_weighted_node_less(
    const OocWeightedNodeRecord& left,
    const OocWeightedNodeRecord& right) noexcept;

// Exact per-worker reduction performed by sub_17A4DD0 after sorting. For one
// key it retains the record with the larger denominator; equal denominators
// retain the smaller integer quotient weight_sum/denominator. The final
// occurrence supplies the payload when these comparisons tie.
[[nodiscard]] std::vector<OocWeightedNodeRecord>
reduce_ooc_weighted_nodes_local(std::span<const OocWeightedNodeRecord> records);

// Exact all-camera sort and key reduction at 0x17A7C22..0x17A7D65. Duplicate
// payloads are combined in sorted encounter order through sub_1EA3B60.
[[nodiscard]] std::vector<OocWeightedNodeRecord>
merge_ooc_weighted_nodes_all_cameras(
    std::span<const OocWeightedNodeRecord> records);

// Production ownership-transfer form. The caller no longer needs the
// published per-camera concatenation after this boundary, so its allocation
// becomes the merge sort input instead of being copied once more.
[[nodiscard]] std::vector<OocWeightedNodeRecord>
merge_ooc_weighted_nodes_all_cameras(
    std::vector<OocWeightedNodeRecord>&& records);

// Exact single-camera rejection at 0x178D4F2..0x178D586 for the observed
// arbitrary-surface depth-map path. Records whose camera-vote denominator is
// zero or one are omitted before octree balancing; encounter order is kept.
[[nodiscard]] std::vector<OocWeightedNodeRecord>
filter_ooc_multi_camera_nodes(
    std::span<const OocWeightedNodeRecord> records);

// Exact topology and payload result of sub_17CB620 / OpenMP worker
// sub_17A86F0 for one in-memory part. Descending from the maximum level, the
// trigger set at level L is the union of original level-L records and the
// closed 3x3x3 neighbourhood of every level-(L+1) trigger's parent. Each
// trigger's sibling octet is then completed. Original triggers retain their
// payload, neighbourhood triggers are zero, and non-trigger siblings inherit
// the normalized payload of the last trigger in their octet. This recurrence
// is equivalent to full-child 26-neighbour 2:1 balancing, while also retaining
// the target's otherwise observable synthetic-node weights. Output uses the
// target's unsigned Morton/level ordering.
[[nodiscard]] std::vector<OocWeightedNodeRecord>
balance_ooc_weighted_nodes(
    std::span<const OocWeightedNodeRecord> records);

// Exact multi-worker wrapper observed for the South single-group balance.
// Input is first put in target Morton/level order and split into consecutive
// 0x40000-record parts. Each part runs the exact worker above. At overlapping
// halo keys an original record (non-zero denominator) owns the payload;
// otherwise the later spatial part owns the generated payload.
[[nodiscard]] std::vector<OocWeightedNodeRecord>
balance_ooc_weighted_nodes_partitioned(
    std::span<const OocWeightedNodeRecord> records,
    std::size_t partition_records = 0x40000U);

// Exact non-histogram initialization of the persistent 28-byte record seen at
// the first OOC solve boundary. Keys and order are preserved, weight is
// sub_1EA3C00's normalized byte, the ten bins and scalar index start at zero,
// and the trailing half field starts with bits 0xC000. Camera histogram voting
// is a later mutation of the ten bins.
[[nodiscard]] std::vector<OocOctreeRecord>
initialize_ooc_octree_records(
    std::span<const OocWeightedNodeRecord> balanced_records);

// Exact sub_1EA3C00 bridge after camera histogram accumulation.  HistVoxel
// records are sentinel + level-major/Morton-major balanced nodes.  The
// maximum-level nodes are sampling support only and are omitted from the
// persistent Morton table consumed by the variational solve.
[[nodiscard]] std::vector<OocOctreeRecord>
build_ooc_octree_records_from_histogram_mode0(
    std::span<const OocWeightedNodeRecord> balanced_records,
    std::span<const OocHistogramVoxel> histogram_voxels);

// Exact target tree order for one OOC part: root first, then each parent's
// complete child octet in parent encounter order.  This produces the
// root-plus-octets layout required by the marching-tree bridge.
[[nodiscard]] std::vector<std::uint32_t>
select_ooc_octree_nodes_breadth_first(
    std::span<const OocOctreeRecord> records);

// Target active byte at the fusion-preparation boundary: one for a selected
// leaf and zero for an internal node.  selected_indices is kept explicit so
// the byte vector shares the solver's breadth-first order.
[[nodiscard]] std::vector<std::uint8_t> make_ooc_leaf_active_mask(
    std::span<const OocOctreeRecord> records,
    std::span<const std::uint32_t> selected_indices);

// Final node layout consumed by sub_1E9EC90 before adaptive marching.
struct OocMarchingNode {
    std::array<std::uint32_t, 3> morton_words{};
    std::uint8_t level{};
    std::array<std::uint8_t, 3> reserved{};
    std::uint32_t weight_denominator{};
    std::uint32_t central_histogram_sum{};
    float scalar{};
    float weighted_cell_scale{};
};

static_assert(sizeof(OocMarchingNode) == 32);

struct OocMarchingExtractNode {
    float weighted_cell_scale{};
    float scalar{};
    std::uint32_t child_group{std::numeric_limits<std::uint32_t>::max()};
};

static_assert(sizeof(OocMarchingExtractNode) == 12);

// Exact pre-sort 16-byte entry emitted by sub_1ED49D0/sub_1ECEB50.  The
// second field is the encounter-order cell-state ordinal; sub_1ED56D0 later
// sorts a copy by selected_node_index before initializing the 152-byte state
// vector.
struct OocMarchingActiveCellEntry {
    std::uint64_t selected_node_index{};
    std::uint64_t cell_state_index{};

    friend bool operator==(const OocMarchingActiveCellEntry&,
                           const OocMarchingActiveCellEntry&) = default;
};

static_assert(sizeof(OocMarchingActiveCellEntry) == 16);

struct OocMarchingActiveCells {
    // Target-compatible little-endian bit words, rounded up to 64 nodes.
    std::vector<std::uint64_t> bits;
    // Deliberately retained in target recursive encounter order.
    std::vector<OocMarchingActiveCellEntry> entries;
    std::uint32_t maximum_level{};
};

// Integer-grid spatial domain consumed by the target's bounded marching
// workers. Coordinates are expressed at `level`; both ends are retained
// because the target separately applies closed containment and strict overlap
// tests at adaptive cell resolutions.
struct OocMarchingGridBounds {
    std::array<std::uint32_t, 3> minimum{};
    std::array<std::uint32_t, 3> maximum{};
    std::uint32_t level{};
};

// Spatial work frontier selected by the target before invoking sub_1ED56D0.
// South's target trace proves a fixed 0x80000-node subtree capacity: an
// over-capacity node is replaced by its child octet, then the final frontier
// is consumed in ascending reordered-tree index order.
struct OocMarchingWorkCell {
    std::uint32_t node_index{};
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t z{};
    std::uint32_t level{};
    std::uint64_t subtree_node_count{};
};

[[nodiscard]] std::vector<OocMarchingWorkCell>
plan_ooc_marching_capacity_frontier(
    std::span<const OocMarchingExtractNode> nodes,
    std::uint64_t maximum_subtree_nodes = 0x80000ULL);

// Exact sub_1ED5240 active-cell discovery.  It performs the target's
// sign-classification pass, sign-changing-edge neighbor marking and second
// recursive closure over the reordered 12-byte marching tree.  No captured
// active mask or entry is consumed.
[[nodiscard]] OocMarchingActiveCells build_ooc_marching_active_cells(
    std::span<const OocMarchingExtractNode> nodes,
    std::optional<OocMarchingGridBounds> bounds = std::nullopt);

// Raw sub_1ED56D0 outputs before per-part decimation/finalization. The fourth
// vertex word is initialized to zero by the target edge builder and is retained
// because it is part of the exact 16-byte ABI.
struct OocMarchingVertex {
    std::array<float, 3> position{};
    std::uint32_t trailing_word{};
};

static_assert(sizeof(OocMarchingVertex) == 16);

struct OocMarchingTriangle {
    std::array<std::uint32_t, 3> vertices{};

    friend bool operator==(const OocMarchingTriangle&,
                           const OocMarchingTriangle&) = default;
};

static_assert(sizeof(OocMarchingTriangle) == 12);

struct OocMarchingRawOutput {
    std::vector<OocMarchingVertex> vertices;
    std::vector<OocMarchingTriangle> triangles;
    std::vector<float> vertex_scale;
    std::vector<std::uint32_t> vertex_source;
    std::vector<std::uint32_t> face_source;
    // Target's additive four-float trimming payload before QEM:
    // weight denominator, central histogram support, source level, one.
    std::vector<std::array<float, 4>> vertex_trim_attribute;
};

void validate_ooc_marching_raw_output(const OocMarchingRawOutput& output);

// Exact input slice consumed when sub_1ECF0F0 creates one previously unseen
// sign-changing adaptive-cell edge. Corner numbering uses xyz bits 0,1,2;
// edges 0..3 vary x, 4..7 vary y, and 8..11 vary z.
struct OocMarchingEdgeInput {
    std::uint32_t edge{};
    std::uint32_t child_slot{};
    std::array<std::int32_t, 3> cell{};
    std::uint32_t level{};
    float weighted_cell_scale{};
    std::uint32_t source_node_index{};
    std::array<float, 8> corner_scalar{};
    double root_scale{};
    std::array<double, 16> transform{};
};

struct OocMarchingEdgeOutput {
    OocMarchingVertex vertex{};
    float vertex_scale{};
    std::uint32_t source_node_index{};
    // Pre-transform physical root-grid coordinate. This is retained by the
    // bounded worker so spatial ownership does not depend on the world matrix.
    std::array<double, 3> root_grid_position{};
};

// Exact zero-crossing interpolation and row-major homogeneous transform at
// 0x1ECF479..0x1ECF760. Arithmetic order intentionally follows the scalar SSE
// instruction stream; callers must only pass an edge whose endpoint signs
// differ, as the target does before entering this block.
[[nodiscard]] OocMarchingEdgeOutput make_ooc_marching_edge_vertex(
    const OocMarchingEdgeInput& input);

struct OocMarchingPolygonDpResult {
    std::vector<double> cost;
    std::vector<std::int32_t> split;
    std::vector<OocMarchingTriangle> triangles;
};

// Exact 152-byte adaptive-cell working record built by sub_1ED3830 and
// consumed by sub_1ED07C0. A child index or edge vertex of -1 is the target's
// null sentinel.
struct OocMarchingCellState {
    float weighted_cell_scale{};
    std::uint32_t source_node_index{};
    std::array<float, 8> corner_scalar{};
    std::array<std::int32_t, 12> edge_vertex{};
    std::array<std::int64_t, 8> child_state{};
};

static_assert(sizeof(OocMarchingCellState) == 152);

// Exact edge-materialization output of sub_1ED0520/sub_1ECF0F0.  The returned
// cell vector retains the complete adaptive tree and has every sign-changing
// leaf edge filled with the shared raw-vertex index used by the target.
struct OocMarchingEdgeMesh {
    std::vector<OocMarchingCellState> cell_states;
    std::vector<OocMarchingVertex> vertices;
    std::vector<float> vertex_scale;
    std::vector<std::uint32_t> vertex_source;
    std::vector<std::array<double, 3>> vertex_root_grid_position;
};

// Exact depth-first edge coordinator.  Same-resolution cube edges share one
// vertex; across a balanced coarse/fine boundary the sign-changing fine half
// owns the vertex and propagates its index back to the containing coarse edge.
// Vertex numbering follows the target's child-slot/edge encounter order.
[[nodiscard]] OocMarchingEdgeMesh build_ooc_marching_edge_mesh(
    std::vector<OocMarchingCellState> cell_states,
    std::uint64_t root_state_index,
    std::uint32_t maximum_level,
    double root_scale,
    const std::array<double, 16>& transform);

// Exact sub_1ED4410/sub_1ED3830 initial adaptive-cell state construction.
// Entries are consumed with their target encounter ordinals; captured states
// are not accepted as input.
[[nodiscard]] std::vector<OocMarchingCellState>
build_ooc_marching_initial_cell_states(
    std::span<const OocMarchingExtractNode> nodes,
    const OocMarchingActiveCells& active_cells);

// Exact sub_1ECEAF0/sub_1ECD1E0 adaptive crack-closure schedule: one mode-0
// pass followed by mode-1 passes until the 152-byte vector stops growing.
// New child octets inherit the parent source/scale and obtain shared corner
// values through the target's 27-neighborhood descent, with its edge/face/
// center fallbacks.
[[nodiscard]] std::vector<OocMarchingCellState>
complete_ooc_marching_cell_states(
    std::vector<OocMarchingCellState> initial_states,
    const OocMarchingActiveCells& active_cells,
    std::vector<std::size_t>* pass_state_counts = nullptr);

struct OocMarchingContourTrace {
    std::vector<std::uint32_t> edges;
    std::vector<std::uint32_t> vertex_indices;
};

struct OocMarchingAdaptiveGridStep {
    std::uint32_t next_edge{};
    std::vector<std::uint32_t> vertex_indices;
};

// Exact local-cell branch of the sub_1ED07C0 contour walker. This covers
// contours for which the neighbor descent does not expose finer edge vertices;
// adaptive 5x5 crack stitching is intentionally a separate API/subproblem.
[[nodiscard]] OocMarchingContourTrace trace_ooc_marching_local_contour(
    const OocMarchingCellState& cell,
    std::uint32_t start_edge);

// Exact 5x5-grid traversal/exit half of the adaptive branch in sub_1ED07C0.
// Grid construction from the two-level neighbor descent is kept separate so
// both state machines can be validated at their real runtime boundary.
[[nodiscard]] OocMarchingAdaptiveGridStep
trace_ooc_marching_adaptive_grid_step(
    const OocMarchingCellState& cell,
    std::uint32_t edge,
    const std::array<std::int32_t, 25>& grid);

// Exact sub_1ED1FE0/sub_1ED07C0 contour coordinator for a balanced adaptive
// tree after edge materialization.  The target's normal in-domain path passes
// penalize_axis_aligned=true; the flag remains explicit for clipped part
// boundaries where sub_1EF1D30 can disable that penalty.
[[nodiscard]] OocMarchingRawOutput build_ooc_marching_raw_mesh(
    const OocMarchingEdgeMesh& edges,
    std::uint64_t root_state_index,
    std::uint32_t maximum_level,
    bool penalize_axis_aligned = true);

// Exact interval dynamic program and sub_1ED0640 backtracking used after
// sub_1ED07C0 has traced one closed adaptive-cell contour. The target adds a
// 2*max_initial_area penalty to consecutive triples sharing any exactly equal
// xyz coordinate when penalize_axis_aligned is set.
[[nodiscard]] OocMarchingPolygonDpResult triangulate_ooc_marching_polygon(
    std::span<const OocMarchingVertex> polygon_vertices,
    std::span<const std::uint32_t> vertex_indices,
    bool penalize_axis_aligned);

// Rebuild the CPU SoA input made by sub_1EAC320. selected_indices defines the
// output order. active[i] == 0 sets flags bit 2 (numeric value 4). The optional
// partition_excluded vector represents the separately recovered partition
// predicate and sets flags bit 1 (numeric value 2).
[[nodiscard]] OocFusionState prepare_ooc_fusion_state(
    const std::vector<OocOctreeRecord>& records,
    const std::vector<std::uint32_t>& selected_indices,
    const std::vector<std::uint8_t>& active,
    const std::vector<float>& scalar_lut,
    const std::vector<std::uint8_t>& partition_excluded = {});

// sub_1EAB480 clamps the fused primal scalar to [-1, 1] and converts it to
// binary16. sub_1EAE0B0 then writes those bits to record offset +26 for nodes
// included by the current partition.
[[nodiscard]] std::vector<std::uint16_t> pack_ooc_fused_scalars(
    const std::vector<float>& fused_u);

void write_ooc_fused_scalars(
    std::vector<OocOctreeRecord>& records,
    const std::vector<std::uint32_t>& selected_indices,
    const std::vector<std::uint16_t>& packed_scalars,
    const std::vector<std::uint8_t>& partition_included = {});

// Reconstruct the verified record28 -> record32 bridge. weight_denominator is the
// sole byte at offset +15 of the upstream 16-byte node that is not retained in
// OocOctreeRecord. Field +20 is derived from histogram bins 3..6; scalar and
// scale use the target's exact float operation order.
[[nodiscard]] std::vector<OocMarchingNode> make_ooc_marching_nodes(
    const std::vector<OocOctreeRecord>& records,
    const std::vector<std::uint32_t>& selected_indices,
    const std::vector<std::uint8_t>& weight_denominator,
    std::uint8_t denominator_cap = 127U);

// Recover the marching denominator from its proven producer instead of a
// captured record32.  Metashape reads byte +15 of the upstream 16-byte
// weighted node and clamps it before constructing the 32-byte marching node.
// The table and weighted records are joined by Morton key and level rather
// than by vector position because their traversal orders differ.
[[nodiscard]] std::vector<std::uint8_t>
select_ooc_marching_weight_denominators(
    const std::vector<OocOctreeRecord>& records,
    const std::vector<std::uint32_t>& selected_indices,
    std::span<const OocWeightedNodeRecord> balanced_records);

[[nodiscard]] std::vector<OocMarchingNode> make_ooc_marching_nodes(
    const std::vector<OocOctreeRecord>& records,
    const std::vector<std::uint32_t>& selected_indices,
    std::span<const OocWeightedNodeRecord> balanced_records,
    std::uint8_t denominator_cap = 127U);

// Exact sub_1E9EC90 projection. A parent either references a contiguous group
// of eight children through (first_child_index - 1) / 8, or UINT32_MAX.
[[nodiscard]] std::vector<OocMarchingExtractNode> build_ooc_marching_extract(
    const std::vector<OocMarchingNode>& nodes);

// sub_1E9E190 changes each consecutive child octet from binary Morton order to
// the corner order required by the marching sampler.
void reorder_ooc_marching_child_groups(
    std::vector<OocMarchingExtractNode>& nodes);

// Continuous, target-ordered CPU boundary from one persisted OOC part through
// the 200-round functional-type-2 solve, reordered adaptive-marching input and
// exact active-cell discovery.  It deliberately stops before sub_1ED3830's
// 152-byte state initialization/fixed-point grid closure.
struct RecoveredOocContinuousPartOutput {
    std::vector<OocOctreeRecord> records;
    OocFusionState fusion;
    std::vector<std::uint16_t> packed_scalars;
    std::vector<OocMarchingNode> marching_nodes;
    std::vector<OocMarchingExtractNode> marching_extract_before_reorder;
    std::vector<OocMarchingExtractNode> marching_extract;
    OocMarchingActiveCells marching_active_cells;
    std::vector<OocMarchingCellState> marching_initial_cell_states;
    std::vector<OocMarchingCellState> marching_cell_states;
};

[[nodiscard]] RecoveredOocContinuousPartOutput
run_recovered_ooc_continuous_part_cpu(
    std::vector<OocOctreeRecord> records,
    const std::vector<std::uint32_t>& selected_indices,
    const std::vector<std::uint8_t>& active,
    const std::vector<float>& scalar_lut,
    std::span<const OocWeightedNodeRecord> balanced_records,
    const std::vector<std::uint8_t>& partition_excluded = {},
    const std::vector<std::uint8_t>& partition_included = {},
    const OocFusionParameters& fusion_parameters = {});

// One stage of the target mode-0 coarse-to-fine fusion schedule.  The target
// loads all balanced nodes through `support_level`, omits that last level from
// the persistent solve table, and uses the half scalar written by the previous
// stage as the next stage's input scalar for nodes already visited.
struct RecoveredOocFusionStageStats {
    std::uint32_t support_level{};
    std::uint64_t balanced_node_count{};
    std::uint64_t persistent_node_count{};
    std::uint64_t active_node_count{};
    double filter_seconds{};
    double topology_seconds{};
    double preparation_seconds{};
    double fusion_seconds{};
    double commit_seconds{};
};

struct RecoveredOocMultilevelModelOutput {
    std::vector<RecoveredOocFusionStageStats> stages;
    std::vector<OocOctreeRecord> records;
    std::vector<OocMarchingWorkCell> marching_work_cells;
    OocMarchingRawOutput raw_mesh;
    std::uint32_t marching_maximum_level{};
    std::uint64_t marching_initial_cell_count{};
    std::uint64_t marching_closed_cell_count{};
    double marching_nodes_seconds{};
    double marching_extract_seconds{};
    double marching_active_seconds{};
    double marching_closure_seconds{};
    double marching_edges_seconds{};
    double marching_raw_seconds{};
};

// Recovered one-part mode-0 scheduler used by the South-building target path.
// Stages are cumulative and normally advance by two levels (6,8,10,12).  The
// explicit stage list is intentionally part of the API: target part sizing can
// raise the first stage for smaller scenes, so an unverified generic heuristic
// must not silently change the numerical result.  An empty list is accepted
// only when maximum_level <= 6 and becomes a single final stage.
[[nodiscard]] RecoveredOocMultilevelModelOutput
run_recovered_ooc_multilevel_model_cpu(
    std::span<const OocOctreeRecord> histogram_records,
    std::span<const OocWeightedNodeRecord> balanced_records,
    std::span<const float> scalar_lut,
    std::span<const std::uint32_t> support_levels,
    double root_scale,
    const std::array<double, 16>& root_grid_to_world,
    const OocFusionParameters& fusion_parameters = {});

// CUDA variant of the same recovered scheduler.  Each cumulative support
// stage keeps all twelve variational arrays resident for its complete 200
// update rounds and uses the exact target PTX/launch partitioning.
[[nodiscard]] RecoveredOocMultilevelModelOutput
run_recovered_ooc_multilevel_model_cuda(
    std::span<const OocOctreeRecord> histogram_records,
    std::span<const OocWeightedNodeRecord> balanced_records,
    std::span<const float> scalar_lut,
    std::span<const std::uint32_t> support_levels,
    double root_scale,
    const std::array<double, 16>& root_grid_to_world,
    std::size_t device_index,
    OocFusionCudaStats& cuda_stats,
    const OocFusionParameters& fusion_parameters = {});

}  // namespace metmodel
