#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace metmodel {

struct PatchMatchCamera;
struct DepthVotingCalibrationCu;
struct DepthVotingMatrix4x4f;
struct OocHistogramVoxel;
struct OocHistogramCalibrationCu;
struct OocHistogramCameraExteriorTransformCu;

cudaError_t launch_recovered_patchmatch_undistort_u8_source(
    const std::uint8_t* source, const std::uint8_t* source_mask,
    std::uint8_t* result, std::uint8_t* result_mask,
    const DepthVotingCalibrationCu& source_calibration,
    const DepthVotingCalibrationCu& target_calibration,
    std::uint32_t width, std::uint32_t height, bool with_mask,
    std::uint32_t pixel_offset, std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_undistort_source(
    const void* source, const std::uint8_t* source_mask,
    void* result, std::uint8_t* result_mask,
    std::uint32_t sample, std::uint32_t channels,
    const DepthVotingCalibrationCu& source_calibration,
    const DepthVotingCalibrationCu& target_calibration,
    std::uint32_t width, std::uint32_t height, bool with_mask,
    std::uint32_t pixel_offset, std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_ooc_histogram_mode0_source(
    OocHistogramVoxel* voxels, const float* pyramid,
    std::uint32_t pyramid_levels, std::uint32_t mode_b,
    const OocHistogramCalibrationCu& calibration,
    const OocHistogramCameraExteriorTransformCu& exterior,
    float threshold_a, float threshold_b,
    std::uint32_t begin, std::uint32_t end,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_depth_radius_estimate_source(
    const float* depth,
    float* radius,
    std::uint32_t level_offset,
    const DepthVotingCalibrationCu& calibration,
    const DepthVotingMatrix4x4f& transform,
    std::uint32_t kernel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_depth_neighbor_votes_source(
    std::int32_t* votes,
    const float* reference_depth,
    const float* reference_radius,
    const std::uint8_t* neighbor_inlier_mask,
    const float* neighbor_depth_levels,
    const float* neighbor_radius_levels,
    std::uint32_t reference_level,
    const DepthVotingCalibrationCu& reference_calibration,
    const DepthVotingMatrix4x4f& reference_to_neighbor,
    std::uint32_t neighbor_levels,
    const DepthVotingCalibrationCu& neighbor_calibration,
    std::uint32_t* counter_inlier_supports,
    std::uint32_t* counter_inlier_intersects,
    std::uint32_t* counter_inlier_does_not_reach,
    std::uint32_t* counter_inlier_no_depth,
    std::uint32_t* counter_outlier_supports,
    std::uint32_t* counter_outlier_intersects,
    std::uint32_t* counter_outlier_does_not_reach,
    std::uint32_t kernel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_depth_neighbor_occlusion_votes_source(
    std::int32_t* votes,
    const float* reference_depth,
    const float* reference_radius,
    const std::uint8_t* neighbor_inlier_mask,
    std::uint32_t neighbor_level,
    const float* neighbor_depth_levels,
    const float* neighbor_radius_levels,
    std::uint32_t reference_level,
    const DepthVotingCalibrationCu& reference_calibration,
    const DepthVotingMatrix4x4f& neighbor_to_reference,
    const DepthVotingCalibrationCu& neighbor_calibration,
    std::uint32_t* counter_inlier_occludes,
    std::uint32_t* counter_outlier_occludes,
    std::uint32_t kernel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_depth_voting_finalize_source(
    float* depth,
    const std::int32_t* votes,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t* counter_empty,
    std::uint32_t* counter_bad,
    std::uint32_t* counter_normal,
    std::uint32_t* counter_good,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_copy_inlier_masks_source(
    std::uint8_t* neighbor_inlier_masks,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t hypotheses_per_pixel,
    std::uint32_t neighbor_count,
    const std::uint8_t* temporary_inlier_masks,
    const std::uint8_t* winner,
    std::uint32_t is_checkboard,
    std::uint32_t checkboard_step,
    std::uint32_t only_each_fourth_pixel,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_coarse_to_precise_source(
    const float* depth,
    const std::uint8_t* normal,
    const float* cost,
    std::uint32_t width_original,
    std::uint32_t height_original,
    std::uint32_t depth_downscale,
    float* candidate_depth,
    float* candidate_normal,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_rotate_normals_source(
    float* normals,
    const PatchMatchCamera& camera_transform,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_average_costs_source(
    const float* per_neighbor_cost,
    float* average_cost,
    std::uint8_t* auxiliary,
    std::uint32_t width_original,
    std::uint32_t height_original,
    std::uint32_t depth_downscale,
    std::uint32_t hypotheses_per_pixel,
    std::uint32_t neighbor_count,
    std::uint32_t is_checkboard,
    std::uint32_t checkboard_step,
    std::uint32_t only_each_fourth_pixel,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_wta_source(
    float* depth,
    std::uint8_t* normal,
    float* cost,
    std::uint32_t width_original,
    std::uint32_t height_original,
    std::uint32_t depth_downscale,
    std::uint32_t hypotheses_per_pixel,
    const float* candidate_depth,
    const float* candidate_normal,
    const float* average_cost,
    std::uint8_t* winner,
    std::uint32_t is_checkboard,
    std::uint32_t checkboard_step,
    std::uint32_t only_each_fourth_pixel,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

// Source reconstruction of all three pm_refinement_final sample templates.
cudaError_t launch_recovered_patchmatch_final_refinement_source(
    float* depth,
    std::uint8_t* normal,
    float* cost,
    const PatchMatchCamera& camera,
    std::uint32_t depth_downscale,
    const void* reference_image,
    std::uint32_t reference_image_sample_bytes,
    std::uint32_t image_one_step_more_detailed,
    float deviation_threshold_multiplier,
    float* candidate_depth,
    float* candidate_normal,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_refinement_source(
    const float* depth, const std::uint8_t* normal, const float* cost,
    const float* coarse_depth, const float* coarse_radius,
    const PatchMatchCamera& camera, std::uint32_t depth_downscale,
    float depth_min, float depth_max, const void* reference_image,
    std::uint32_t reference_image_sample_bytes,
    std::uint32_t image_one_step_more_detailed,
    float deviation_threshold_multiplier, float* candidate_depth,
    float* candidate_normal, std::uint32_t iteration,
    std::uint32_t only_each_fourth_pixel, std::uint32_t pixel_offset,
    std::size_t work_items, cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_bilateral_u8_source(
    const float* depth,
    const std::uint8_t* normal,
    const std::uint8_t* image,
    float* filtered_depth,
    std::uint8_t* filtered_normal,
    std::uint32_t width,
    std::uint32_t height,
    float sigma_d,
    float sigma_r,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_cost_u8_source(
    const float* depth, const std::uint8_t* normal, const float* cost,
    const float* coarse_depth, const float* coarse_radius,
    const PatchMatchCamera& reference_camera, std::uint32_t depth_downscale,
    const std::uint8_t* reference_image,
    std::uint32_t image_one_step_more_detailed,
    float deviation_threshold_multiplier,
    const PatchMatchCamera& neighbor_camera, std::uint32_t neighbor_level,
    std::uint32_t result_index, cudaTextureObject_t neighbor_texture,
    const std::uint64_t* neighbor_mask_offsets,
    const std::uint8_t* neighbor_masks, const float* candidate_depth,
    const float* candidate_normal, float* neighbor_cost,
    std::uint32_t reference_patch_radius,
    std::uint32_t hypotheses_per_pixel, std::uint32_t is_checkerboard,
    std::uint32_t checkerboard_step,
    std::uint32_t only_each_fourth_pixel, std::uint32_t pixel_offset,
    std::size_t work_items, cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_cost_source(
    const float* depth, const std::uint8_t* normal, const float* cost,
    const float* coarse_depth, const float* coarse_radius,
    const PatchMatchCamera& reference_camera, std::uint32_t depth_downscale,
    const void* reference_image, std::uint32_t image_sample_bytes,
    std::uint32_t image_one_step_more_detailed,
    float deviation_threshold_multiplier,
    const PatchMatchCamera& neighbor_camera, std::uint32_t neighbor_level,
    std::uint32_t result_index, cudaTextureObject_t neighbor_texture,
    const std::uint64_t* neighbor_mask_offsets,
    const std::uint8_t* neighbor_masks, const float* candidate_depth,
    const float* candidate_normal, float* neighbor_cost,
    std::uint32_t reference_patch_radius,
    std::uint32_t hypotheses_per_pixel, std::uint32_t is_checkerboard,
    std::uint32_t checkerboard_step,
    std::uint32_t only_each_fourth_pixel, std::uint32_t pixel_offset,
    std::size_t work_items, cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_bilateral_u16_source(
    const float* depth, const std::uint8_t* normal,
    const std::uint16_t* image, float* filtered_depth,
    std::uint8_t* filtered_normal, std::uint32_t width,
    std::uint32_t height, float sigma_d, float sigma_r,
    std::uint32_t pixel_offset, std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_bilateral_f32_source(
    const float* depth, const std::uint8_t* normal,
    const float* image, float* filtered_depth,
    std::uint8_t* filtered_normal, std::uint32_t width,
    std::uint32_t height, float sigma_d, float sigma_r,
    std::uint32_t pixel_offset, std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_filter_clear_depth_source(
    float* depth,
    const std::uint8_t* filtered_mask,
    std::uint32_t* counter_not_empty,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_filter_check_cost_source(
    float* depth,
    const float* cost,
    std::uint32_t* counter_no_cost,
    std::uint32_t* counter_big_cost,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_filter_check_neighbours_source(
    const float* depth,
    std::uint8_t* filtered_mask,
    const PatchMatchCamera& camera,
    std::uint32_t depth_downscale,
    float depth_min,
    float depth_max,
    std::uint32_t* counter_no_neighbours,
    std::uint32_t* counter_no_close_neighbours,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_filter_normals_source(
    const float* depth,
    const std::uint8_t* normal,
    std::uint8_t* estimated_normal,
    bool estimate_normal_map,
    std::uint8_t* filtered_mask,
    const PatchMatchCamera& camera,
    std::uint32_t depth_downscale,
    std::uint32_t* counter_inconsistent_normal,
    std::uint32_t* counter_bad_view_angle_estimated_normal,
    std::uint32_t* counter_bad_view_angle_found_normal,
    float* counter_cos_sum,
    std::uint32_t* counter_ncos_sum,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_filter_speckles_edges_source(
    const float* depth,
    std::uint8_t* filtered_mask,
    const PatchMatchCamera& camera,
    std::uint32_t depth_downscale,
    std::uint32_t pixel_offset,
    std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_ooc_update_u_source(
    const std::uint8_t* weights, const std::uint8_t* histogram,
    const std::uint32_t* neighbors, const std::uint8_t* connectivity,
    const std::uint8_t* refinement, const std::uint8_t* flags,
    float* u, float* u_old, const std::uint16_t* p,
    std::uint16_t* v, std::uint16_t* v_old,
    const std::uint16_t* q, std::uint32_t count,
    float spatial_scale, float threshold, float alpha, float beta,
    float data_weight, std::uint32_t functional_mode,
    std::uint32_t offset, std::size_t work_items,
    cudaStream_t stream = nullptr);

cudaError_t launch_recovered_ooc_update_p_source(
    const std::uint32_t* neighbors, const std::uint8_t* connectivity,
    const std::uint8_t* refinement, const std::uint8_t* flags,
    const float* u, const float* u_old, std::uint16_t* p,
    const std::uint16_t* v, const std::uint16_t* v_old,
    std::uint16_t* q, std::uint32_t count, float spatial_scale,
    float alpha, float beta, float data_weight,
    std::uint32_t functional_mode, std::uint32_t offset,
    std::size_t work_items, cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_propagation_u8_perspective_source(
    float* depth, std::uint8_t* normal, float* cost,
    const float* coarse_depth, const float* coarse_radius,
    const PatchMatchCamera& camera, const float* rotation_to_local,
    std::uint32_t depth_downscale, const std::uint8_t* reference_image,
    std::uint32_t image_one_step_more_detailed,
    float deviation_threshold_multiplier, float* candidate_depth,
    float* candidate_normal, std::uint32_t checkerboard_step,
    std::uint32_t only_each_fourth_pixel, std::uint32_t pixel_offset,
    std::size_t work_items, cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_propagation_u16_perspective_source(
    float* depth, std::uint8_t* normal, float* cost,
    const float* coarse_depth, const float* coarse_radius,
    const PatchMatchCamera& camera, const float* rotation_to_local,
    std::uint32_t depth_downscale, const std::uint16_t* reference_image,
    std::uint32_t image_one_step_more_detailed,
    float deviation_threshold_multiplier, float* candidate_depth,
    float* candidate_normal, std::uint32_t checkerboard_step,
    std::uint32_t only_each_fourth_pixel, std::uint32_t pixel_offset,
    std::size_t work_items, cudaStream_t stream = nullptr);

cudaError_t launch_recovered_patchmatch_propagation_f32_perspective_source(
    float* depth, std::uint8_t* normal, float* cost,
    const float* coarse_depth, const float* coarse_radius,
    const PatchMatchCamera& camera, const float* rotation_to_local,
    std::uint32_t depth_downscale, const float* reference_image,
    std::uint32_t image_one_step_more_detailed,
    float deviation_threshold_multiplier, float* candidate_depth,
    float* candidate_normal, std::uint32_t checkerboard_step,
    std::uint32_t only_each_fourth_pixel, std::uint32_t pixel_offset,
    std::size_t work_items, cudaStream_t stream = nullptr);

}  // namespace metmodel
