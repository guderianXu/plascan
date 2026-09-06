#include "metmodel/patchmatch.hpp"
#include "metmodel/patchmatch_store.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>


namespace metmodel {
namespace {

class RecoveredCudaModuleSessionScope {
public:
    RecoveredCudaModuleSessionScope() = default;
    RecoveredCudaModuleSessionScope(
        const RecoveredCudaModuleSessionScope&) = delete;
    RecoveredCudaModuleSessionScope& operator=(
        const RecoveredCudaModuleSessionScope&) = delete;

    bool open(std::size_t device_index, std::string& error) {
        if (!begin_recovered_cuda_module_session(device_index, error))
            return false;
        active_ = true;
        return true;
    }

    bool close(RecoveredCudaModuleSessionStats& stats,
               std::string& error) {
        if (!active_) {
            error = "recovered CUDA module session scope is not active";
            return false;
        }
        if (!end_recovered_cuda_module_session(stats, error))
            return false;
        active_ = false;
        return true;
    }

    ~RecoveredCudaModuleSessionScope() {
        if (!active_) return;
        RecoveredCudaModuleSessionStats ignored_stats;
        std::string ignored_error;
        (void)end_recovered_cuda_module_session(
            ignored_stats, ignored_error);
    }

private:
    bool active_ = false;
};

std::uint64_t saturating_add_bytes(std::uint64_t left,
                                   std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        return std::numeric_limits<std::uint64_t>::max();
    return left + right;
}

template <class T>
std::uint64_t vector_capacity_bytes(const std::vector<T>& values) {
    constexpr std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();
    if (values.capacity() > maximum / sizeof(T)) return maximum;
    return static_cast<std::uint64_t>(values.capacity()) * sizeof(T);
}

std::uint64_t prepared_camera_dynamic_bytes(
    const RecoveredPatchMatchPreparedCamera& camera) {
    std::uint64_t bytes = vector_capacity_bytes(camera.image_levels);
    for (const auto& level : camera.image_levels) {
        bytes = saturating_add_bytes(
            bytes, vector_capacity_bytes(level.data.image));
        bytes = saturating_add_bytes(
            bytes, vector_capacity_bytes(level.data.rejection_mask));
    }
    return bytes;
}

std::uint64_t neighbor_resources_dynamic_bytes(
    const RecoveredPatchMatchNeighborResources& resources) {
    std::uint64_t bytes = vector_capacity_bytes(resources.resource_groups);
    for (const auto& group : resources.resource_groups) {
        bytes = saturating_add_bytes(bytes,
                                     vector_capacity_bytes(group.image_offsets));
        bytes = saturating_add_bytes(bytes,
                                     vector_capacity_bytes(group.image));
        bytes = saturating_add_bytes(bytes,
                                     vector_capacity_bytes(group.level_offsets));
        bytes = saturating_add_bytes(bytes,
                                     vector_capacity_bytes(group.mask));
    }
    bytes = saturating_add_bytes(
        bytes, vector_capacity_bytes(resources.ranked_neighbors));
    for (const auto& neighbor : resources.ranked_neighbors) {
        bytes = saturating_add_bytes(bytes,
                                     vector_capacity_bytes(neighbor.texture));
        bytes = saturating_add_bytes(
            bytes, vector_capacity_bytes(neighbor.texture_source));
        bytes = saturating_add_bytes(
            bytes, vector_capacity_bytes(neighbor.texture_write_mask));
        bytes = saturating_add_bytes(
            bytes, vector_capacity_bytes(neighbor.texture_copy_regions));
    }
    return bytes;
}

std::uint64_t host_preparation_dynamic_bytes(
    const RecoveredPatchMatchHostPreparation& preparation) {
    std::uint64_t bytes =
        prepared_camera_dynamic_bytes(preparation.reference);
    bytes = saturating_add_bytes(
        bytes,
        vector_capacity_bytes(preparation.ranked_neighbor_camera_indices));
    bytes = saturating_add_bytes(
        bytes, vector_capacity_bytes(preparation.ranked_neighbors));
    for (const auto& neighbor : preparation.ranked_neighbors) {
        bytes = saturating_add_bytes(
            bytes, prepared_camera_dynamic_bytes(neighbor));
    }
    return saturating_add_bytes(
        bytes, neighbor_resources_dynamic_bytes(preparation.neighbor_resources));
}

std::uint64_t prepared_camera_cache_dynamic_bytes(
    const std::vector<RecoveredPatchMatchPreparedCamera>& cameras) {
    std::uint64_t bytes = vector_capacity_bytes(cameras);
    for (const auto& camera : cameras) {
        bytes = saturating_add_bytes(
            bytes, prepared_camera_dynamic_bytes(camera));
    }
    return bytes;
}

}  // namespace

bool run_recovered_patchmatch_coarsest_level_cuda(
    const RecoveredPatchMatchHostPreparation& preparation,
    std::span<const std::uint8_t> bilateral_image,
    RecoveredPatchMatchCoarsestLevelOutput& output,
    std::string& error,
    RecoveredPatchMatchCostAtlasState* camera_atlas_state,
    bool capture_diagnostic_checkpoints) {
    try {
        constexpr std::uint32_t downscale = 32U;
        constexpr std::size_t hypotheses =
            PatchMatchCandidateOutput::hypotheses;
        constexpr std::size_t capacity =
            PatchMatchCandidateOutput::capacity;
        const PatchMatchCamera& camera = preparation.reference_camera;
        const bool valid_target_downscale =
            preparation.target_downscale >= 2U &&
            preparation.target_downscale <= 8U &&
            (preparation.target_downscale &
             (preparation.target_downscale - 1U)) == 0U;
        if (!valid_target_downscale) {
            error = "PatchMatch coarsest-level orchestration requires power-of-two target downscale 2..8";
            return false;
        }
        if (preparation.no_prior_policy !=
            RecoveredPatchMatchNoPriorPolicy::DeterministicZero) {
            error = "PatchMatch coarsest-level no-prior policy is unsupported";
            return false;
        }
        if (camera.width_original == 0U || camera.height_original == 0U ||
            preparation.neighbor_resources.ranked_neighbors.empty()) {
            error = "PatchMatch coarsest-level host preparation is empty";
            return false;
        }
        RecoveredPatchMatchCostAtlasState local_atlas_state;
        if (camera_atlas_state == nullptr) {
            local_atlas_state = make_recovered_patchmatch_cost_atlas_state(
                preparation.neighbor_resources);
            camera_atlas_state = &local_atlas_state;
        }
        const std::size_t width =
            (camera.width_original + downscale - 1U) / downscale;
        const std::size_t height =
            (camera.height_original + downscale - 1U) / downscale;
        const std::size_t pixels = width * height;
        const std::size_t finest_width =
            (camera.width_original + preparation.target_downscale - 1U) /
            preparation.target_downscale;
        const std::size_t finest_height =
            (camera.height_original + preparation.target_downscale - 1U) /
            preparation.target_downscale;
        const std::size_t finest_pixels = finest_width * finest_height;
        const std::uint32_t estimated_downscale =
            preparation.target_downscale * 2U;
        const std::size_t estimated_width =
            (camera.width_original + estimated_downscale - 1U) /
            estimated_downscale;
        const std::size_t estimated_height =
            (camera.height_original + estimated_downscale - 1U) /
            estimated_downscale;
        const std::size_t estimated_pixels =
            estimated_width * estimated_height;
        if (pixels == 0U || pixels > capacity ||
            bilateral_image.size() != pixels) {
            error = "PatchMatch coarsest-level active grid is invalid";
            return false;
        }

        RecoveredPatchMatchLevelState state;
        state.depth.assign(pixels, 0.0F);
        state.normal.assign(pixels * 3U, std::uint8_t{0x80});
        state.cost.assign(pixels, -1.0F);
        state.winner.assign(capacity, std::uint8_t{0});
        state.candidates.depth.assign(hypotheses * capacity, 0.0F);
        state.candidates.normal.assign(hypotheses * capacity * 3U, 0.0F);
        const std::size_t neighbor_capacity =
            preparation.neighbor_resources.ranked_neighbors.size();
        const std::size_t inlier_groups = (neighbor_capacity + 7U) / 8U;
        state.neighbor_cost.assign(
            neighbor_capacity * hypotheses * capacity, 0.0F);
        state.average_cost.assign(hypotheses * capacity, 0.0F);
        state.auxiliary.assign(
            inlier_groups * hypotheses * capacity, std::uint8_t{0});
        state.neighbor_inlier_masks.assign(
            inlier_groups * finest_pixels, std::uint8_t{0});
        // This is an explicit production compatibility policy, not a claim
        // about the target's allocator residue.  Reuse this one pair at every
        // x32 producer/consumer boundary so the behavior cannot silently
        // depend on temporary-vector allocation state.
        const std::vector<float> no_prior_coarse_depth(pixels, 0.0F);
        const std::vector<float> no_prior_coarse_radius(pixels, 0.0F);
        const auto reference_image =
            make_recovered_patchmatch_reference_image_allocation(
                preparation.reference, preparation.target_downscale,
                downscale);
        std::uint64_t next_producer_handoff = 0U;
        bool initial_cost_scratch_all_zero = true;
        const bool run_final_refinement =
            downscale <= preparation.target_downscale * 4U;

        const auto cost_and_wta = [&](
            PatchMatchCandidateOutput candidates,
            std::uint32_t iteration,
            bool all_neighbors_state,
            std::uint32_t checkerboard, std::uint32_t step,
            std::size_t work_items,
            std::uint32_t reference_patch_radius,
            bool materialize_cost_output) {
            auto binding = make_recovered_patchmatch_cost_binding(
                preparation, *camera_atlas_state, downscale, iteration,
                all_neighbors_state, false);
            PatchMatchCostInput cost_input;
            cost_input.reference_camera = binding.reference_camera;
            cost_input.rotate_before_camera = binding.normal_rotations.before;
            cost_input.rotate_after_camera = binding.normal_rotations.after;
            cost_input.depth_downscale = downscale;
            cost_input.image_one_step_more_detailed = 1U;
            cost_input.deviation_threshold_multiplier =
                binding.deviation_threshold_multiplier;
            cost_input.depth_view = state.depth;
            cost_input.normal_view = state.normal;
            cost_input.cost_view = state.cost;
            cost_input.coarse_depth_view = no_prior_coarse_depth;
            cost_input.coarse_depth_radius_view = no_prior_coarse_radius;
            cost_input.reference_image_view = reference_image;
            cost_input.neighbor_texture_width = binding.neighbor_texture_width;
            cost_input.neighbor_texture_height =
                binding.neighbor_texture_height;
            cost_input.neighbor_texture_view =
                binding.initial_neighbor_texture_view;
            cost_input.resource_groups_view = binding.resource_groups_view;
            cost_input.neighbor_batch = std::move(binding.neighbor_batch);
            cost_input.candidates = std::move(candidates);
            cost_input.initial_neighbor_cost = std::move(state.neighbor_cost);
            cost_input.initial_average_cost = std::move(state.average_cost);
            cost_input.initial_auxiliary = std::move(state.auxiliary);
            cost_input.reference_patch_radius = reference_patch_radius;
            cost_input.hypotheses_per_pixel = 8U;
            cost_input.neighbor_count = binding.neighbor_count;
            cost_input.neighbor_cost_capacity =
                binding.neighbor_cost_capacity;
            cost_input.is_checkboard = checkerboard;
            cost_input.checkboard_step = step;
            cost_input.global_work_items = work_items;
            cost_input.device_index = preparation.device_index;
            cost_input.camera_resource_generation =
                binding.camera_resource_generation;
            cost_input.cuda_workspace_handoff =
                cost_input.candidates.cuda_workspace_handoff;
            cost_input.initial_cuda_cost_scratch_materialized =
                state.cuda_cost_scratch_materialized;
            cost_input.initial_cuda_cost_scratch_all_zero =
                initial_cost_scratch_all_zero;
            cost_input.defer_cuda_host_output = !materialize_cost_output;
            PatchMatchCostOutput cost_output;
            if (!run_recovered_patchmatch_cost_cuda_movable(
                    std::move(cost_input), cost_output, error))
                return false;
            initial_cost_scratch_all_zero = false;

            PatchMatchWtaInput wta;
            wta.camera = binding.reference_camera;
            wta.depth_downscale = downscale;
            wta.image_one_step_more_detailed = 1U;
            wta.hypotheses_per_pixel = 8U;
            wta.depth = std::move(state.depth);
            wta.normal = std::move(state.normal);
            wta.cost = std::move(state.cost);
            wta.candidates = std::move(cost_output.candidates);
            wta.average_cost = std::move(cost_output.average_cost);
            wta.winner = std::move(state.winner);
            wta.is_checkboard = checkerboard;
            wta.checkboard_step = step;
            wta.global_work_items = work_items;
            wta.device_index = preparation.device_index;
            wta.cuda_workspace_handoff =
                cost_output.cuda_workspace_handoff;
            wta.cuda_cost_output_materialized =
                cost_output.cuda_host_output_materialized;
            wta.defer_cuda_host_output =
                !materialize_cost_output &&
                cost_output.cuda_workspace_handoff != 0U;
            PatchMatchWtaOutput wta_output;
            if (!run_recovered_patchmatch_wta_cuda_movable(
                    std::move(wta), wta_output, error))
                return false;
            cost_output.candidates = std::move(wta.candidates);
            cost_output.average_cost = std::move(wta.average_cost);

            PatchMatchCopyInlierMasksInput copy_inliers;
            copy_inliers.width = static_cast<std::uint32_t>(width);
            copy_inliers.height = static_cast<std::uint32_t>(height);
            copy_inliers.hypotheses_per_pixel = 8U;
            copy_inliers.neighbor_count =
                static_cast<std::uint32_t>(neighbor_capacity);
            copy_inliers.temporary_inlier_masks =
                std::move(cost_output.auxiliary);
            copy_inliers.winner = std::move(wta_output.winner);
            copy_inliers.initial_neighbor_inlier_masks =
                std::move(state.neighbor_inlier_masks);
            copy_inliers.is_checkboard = checkerboard;
            copy_inliers.checkboard_step = step;
            copy_inliers.global_work_items = work_items;
            copy_inliers.device_index = preparation.device_index;
            copy_inliers.cuda_workspace_handoff =
                wta_output.cuda_workspace_handoff;
            copy_inliers.cuda_temporary_inlier_masks_materialized =
                cost_output.cuda_host_output_materialized;
            copy_inliers.cuda_winner_materialized =
                wta_output.cuda_host_output_materialized;
            copy_inliers.cuda_initial_neighbor_inlier_masks_materialized =
                state.cuda_inlier_masks_materialized;
            copy_inliers.defer_cuda_host_output =
                !materialize_cost_output &&
                wta_output.cuda_workspace_handoff != 0U;
            PatchMatchCopyInlierMasksOutput copied_inliers;
            if (!run_recovered_patchmatch_copy_inlier_masks_cuda_movable(
                    std::move(copy_inliers), copied_inliers, error))
                return false;
            cost_output.auxiliary =
                std::move(copy_inliers.temporary_inlier_masks);
            wta_output.winner = std::move(copy_inliers.winner);
            next_producer_handoff =
                copied_inliers.cuda_workspace_handoff;
            state.depth = std::move(wta_output.depth);
            state.normal = std::move(wta_output.normal);
            state.cost = std::move(wta_output.cost);
            state.winner = std::move(wta_output.winner);
            state.candidates = std::move(cost_output.candidates);
            state.neighbor_cost = std::move(cost_output.per_neighbor_cost);
            state.average_cost = std::move(cost_output.average_cost);
            state.auxiliary = std::move(cost_output.auxiliary);
            state.cuda_cost_scratch_materialized =
                cost_output.cuda_host_output_materialized;
            state.cuda_main_state_materialized =
                wta_output.cuda_host_output_materialized;
            state.neighbor_inlier_masks =
                std::move(copied_inliers.neighbor_inlier_masks);
            state.cuda_inlier_masks_materialized =
                copied_inliers.cuda_host_output_materialized;
            return true;
        };

        for (std::uint32_t iteration = 0U; iteration < 6U; ++iteration) {
            PatchMatchRefinementInput refinement;
            refinement.camera = camera;
            refinement.depth_downscale = downscale;
            refinement.depth_min = preparation.reference.depth_range.minimum;
            refinement.depth_max = preparation.reference.depth_range.maximum;
            refinement.reference_image_view = reference_image;
            refinement.image_one_step_more_detailed = 1U;
            refinement.deviation_threshold_multiplier =
                preparation.deviation_threshold_multiplier;
            refinement.depth_view = state.depth;
            refinement.normal_view = state.normal;
            refinement.cost_view = state.cost;
            refinement.coarse_depth_view = no_prior_coarse_depth;
            refinement.coarse_depth_radius_view = no_prior_coarse_radius;
            refinement.initial_candidate_depth =
                std::move(state.candidates.depth);
            refinement.initial_candidate_normal =
                std::move(state.candidates.normal);
            refinement.iteration = iteration;
            refinement.global_work_items = pixels;
            refinement.device_index = preparation.device_index;
            refinement.cuda_workspace_handoff =
                std::exchange(next_producer_handoff, 0U);
            refinement.initial_cuda_uniform_state = iteration == 0U;
            refinement.defer_cuda_host_output = true;
            PatchMatchCandidateOutput refined;
            if (!run_recovered_patchmatch_refinement_cuda_movable(
                    std::move(refinement), refined, error) ||
                !cost_and_wta(std::move(refined), iteration, false,
                              0U, 0U, pixels, 3U, false))
                return false;

            for (std::uint32_t step = 0U; step < 2U; ++step) {
                PatchMatchPropagationInput propagation;
                propagation.camera = camera;
                propagation.reference_to_neighbor_rotation =
                    preparation.propagation_rotation;
                propagation.depth_downscale = downscale;
                propagation.reference_image_view = reference_image;
                propagation.image_one_step_more_detailed = 1U;
                propagation.deviation_threshold_multiplier =
                    preparation.deviation_threshold_multiplier;
                propagation.depth_view = state.depth;
                propagation.normal_view = state.normal;
                propagation.cost_view = state.cost;
                propagation.coarse_depth_view = no_prior_coarse_depth;
                propagation.coarse_depth_radius_view = no_prior_coarse_radius;
                propagation.initial_candidate_depth =
                    std::move(state.candidates.depth);
                propagation.initial_candidate_normal =
                    std::move(state.candidates.normal);
                propagation.checkboard_step = step;
                propagation.global_work_items = (pixels + 1U) / 2U;
                propagation.device_index = preparation.device_index;
                propagation.cuda_workspace_handoff =
                    std::exchange(next_producer_handoff, 0U);
                propagation.defer_cuda_host_output =
                    !capture_diagnostic_checkpoints ||
                    run_final_refinement || iteration != 5U || step != 1U;
                PatchMatchCandidateOutput propagated;
                if (!run_recovered_patchmatch_propagation_cuda_movable(
                        std::move(propagation), propagated, error) ||
                    !cost_and_wta(std::move(propagated), iteration, false,
                                  1U, step, (pixels + 1U) / 2U, 3U,
                                  capture_diagnostic_checkpoints &&
                                      !run_final_refinement &&
                                      iteration == 5U && step == 1U))
                    return false;
            }
        }

        // The target only emits coarsest final refinement when this level is
        // at most four times coarser than the requested target.  Thus d8 has
        // an x32 final state (reusing iteration 5), while d4/d2 do not.
        if (run_final_refinement) {
            PatchMatchFinalRefinementInput final_refinement;
            final_refinement.camera = camera;
            final_refinement.depth_downscale = downscale;
            final_refinement.reference_image_view = reference_image;
            final_refinement.image_one_step_more_detailed = 1U;
            final_refinement.deviation_threshold_multiplier =
                preparation.deviation_threshold_multiplier;
            final_refinement.depth_view = state.depth;
            final_refinement.normal_view = state.normal;
            final_refinement.cost_view = state.cost;
            final_refinement.initial_candidate_depth =
                std::move(state.candidates.depth);
            final_refinement.initial_candidate_normal =
                std::move(state.candidates.normal);
            final_refinement.global_work_items = pixels;
            final_refinement.device_index = preparation.device_index;
            final_refinement.cuda_workspace_handoff =
                std::exchange(next_producer_handoff, 0U);
            final_refinement.defer_cuda_host_output =
                !capture_diagnostic_checkpoints;
            PatchMatchFinalRefinementOutput final_output;
            if (!run_recovered_patchmatch_final_refinement_state_cuda_movable(
                    std::move(final_refinement), final_output, error))
                return false;
            state.cost = std::move(final_output.cost);
            if (!cost_and_wta(std::move(final_output.candidates),
                              5U, false, 0U, 0U, pixels, 2U,
                              capture_diagnostic_checkpoints))
                return false;
        }

        if (capture_diagnostic_checkpoints) {
            if (!state.candidates.cuda_host_output_materialized ||
                !state.cuda_cost_scratch_materialized ||
                !state.cuda_main_state_materialized ||
                !state.cuda_inlier_masks_materialized) {
                error = "PatchMatch coarsest diagnostic boundary has deferred state";
                return false;
            }
        } else if (next_producer_handoff == 0U) {
            error = "PatchMatch coarsest production filter lacks a resident handoff";
            return false;
        }
        output.state_before_filter = state;
        PatchMatchLevelBoundaryInput boundary;
        boundary.filter.camera = camera;
        boundary.filter.depth_downscale = downscale;
        boundary.filter.depth_min = preparation.reference.depth_range.minimum;
        boundary.filter.depth_max = preparation.reference.depth_range.maximum;
        boundary.filter.depth_allocation = state.depth;
        boundary.filter.cost_allocation = state.cost;
        boundary.filter.normal_allocation = state.normal;
        boundary.filter.estimated_normal_allocation.assign(
            estimated_pixels * 3U, std::uint8_t{0});
        boundary.filter.filtered_mask_allocation.assign(
            finest_pixels, std::uint8_t{0});
        boundary.filter.estimate_normal_map = true;
        boundary.filter.device_index = preparation.device_index;
        if (!capture_diagnostic_checkpoints) {
            boundary.filter.cuda_workspace_handoff =
                std::exchange(next_producer_handoff, 0U);
            boundary.filter.cuda_main_state_materialized =
                state.cuda_main_state_materialized;
            boundary.filter.cuda_inlier_masks_materialized =
                state.cuda_inlier_masks_materialized;
            boundary.filter.neighbor_inlier_masks_allocation =
                state.neighbor_inlier_masks;
        }
        boundary.speckle_component_size_threshold = 6U;
        boundary.bilateral_image.assign(bilateral_image.begin(),
                                        bilateral_image.end());
        if (!run_recovered_patchmatch_level_boundary_cuda(
                boundary, output.boundary, error))
            return false;
        if (!capture_diagnostic_checkpoints) {
            output.state_before_filter.neighbor_inlier_masks =
                output.boundary.filter.neighbor_inlier_masks_allocation;
            output.state_before_filter.cuda_inlier_masks_materialized = true;
            output.state_before_filter.cuda_workspace_handoff =
                output.boundary.filter.cuda_workspace_handoff;
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool run_recovered_patchmatch_x16_level_cuda(
    const Camera& camera,
    const RecoveredPatchMatchHostPreparation& preparation,
    const RecoveredPatchMatchCoarsestLevelOutput& coarsest,
    std::span<const std::uint8_t> bilateral_image,
    RecoveredPatchMatchX16LevelOutput& output,
    std::string& error,
    RecoveredPatchMatchCostAtlasState* camera_atlas_state,
    bool capture_diagnostic_checkpoints,
    RecoveredPatchMatchLevelState* consumable_previous_state) {
    try {
        constexpr std::uint32_t downscale = 16U;
        constexpr std::size_t hypotheses =
            PatchMatchCandidateOutput::hypotheses;
        constexpr std::size_t capacity =
            PatchMatchCandidateOutput::capacity;
        const PatchMatchCamera& patch_camera = preparation.reference_camera;
        if (preparation.target_downscale > 8U ||
            preparation.target_downscale < 2U ||
            preparation.reference.camera_index != camera.index ||
            preparation.reference_camera_index != camera.index ||
            patch_camera.width_original == 0U ||
            patch_camera.height_original == 0U ||
            preparation.neighbor_resources.ranked_neighbors.empty()) {
            error = "PatchMatch x16 continuation requires a matching d2/d4/d8 host preparation";
            return false;
        }
        RecoveredPatchMatchCostAtlasState local_atlas_state;
        if (camera_atlas_state == nullptr) {
            local_atlas_state = make_recovered_patchmatch_cost_atlas_state(
                preparation.neighbor_resources);
            camera_atlas_state = &local_atlas_state;
        }

        const std::size_t width =
            (patch_camera.width_original + downscale - 1U) / downscale;
        const std::size_t height =
            (patch_camera.height_original + downscale - 1U) / downscale;
        const std::size_t pixels = width * height;
        const std::size_t previous_width =
            (patch_camera.width_original + 31U) / 32U;
        const std::size_t previous_height =
            (patch_camera.height_original + 31U) / 32U;
        const std::size_t previous_pixels = previous_width * previous_height;
        const std::size_t neighbor_capacity =
            preparation.neighbor_resources.ranked_neighbors.size();
        const std::size_t inlier_groups = (neighbor_capacity + 7U) / 8U;
        const std::size_t finest_width =
            (patch_camera.width_original + preparation.target_downscale - 1U) /
            preparation.target_downscale;
        const std::size_t finest_height =
            (patch_camera.height_original + preparation.target_downscale - 1U) /
            preparation.target_downscale;
        const std::size_t finest_pixels = finest_width * finest_height;
        const std::uint32_t estimated_downscale =
            preparation.target_downscale * 2U;
        const std::size_t estimated_width =
            (patch_camera.width_original + estimated_downscale - 1U) /
            estimated_downscale;
        const std::size_t estimated_height =
            (patch_camera.height_original + estimated_downscale - 1U) /
            estimated_downscale;
        const std::size_t estimated_pixels =
            estimated_width * estimated_height;
        if (pixels == 0U || pixels > capacity || previous_pixels == 0U ||
            bilateral_image.size() != pixels ||
            coarsest.boundary.bilateral.depth.size() != previous_pixels ||
            coarsest.boundary.bilateral.normal.size() != previous_pixels * 3U ||
            coarsest.state_before_filter.candidates.depth.size() !=
                hypotheses * capacity ||
            coarsest.state_before_filter.candidates.normal.size() !=
                hypotheses * capacity * 3U ||
            coarsest.state_before_filter.winner.size() != capacity ||
            coarsest.state_before_filter.neighbor_cost.size() !=
                neighbor_capacity * hypotheses * capacity ||
            coarsest.state_before_filter.average_cost.size() !=
                hypotheses * capacity ||
            coarsest.state_before_filter.auxiliary.size() !=
                inlier_groups * hypotheses * capacity ||
            (!coarsest.state_before_filter.cuda_cost_scratch_materialized &&
             coarsest.state_before_filter.cuda_workspace_handoff == 0U) ||
            coarsest.state_before_filter.neighbor_inlier_masks.size() !=
                inlier_groups * finest_pixels) {
            error = "PatchMatch x16 continuation persistent state is incomplete";
            return false;
        }

        RecoveredPatchMatchX16LevelOutput result;
        if (!make_recovered_patchmatch_cross_level_state(
                camera, downscale,
                static_cast<std::uint32_t>(previous_width),
                static_cast<std::uint32_t>(previous_height),
                coarsest.boundary.bilateral.depth,
                coarsest.boundary.bilateral.normal,
                result.cross_level_initial, error))
            return false;
        if (result.cross_level_initial.width != width ||
            result.cross_level_initial.height != height) {
            error = "PatchMatch x16 cross-level grid does not match preparation";
            return false;
        }

        RecoveredPatchMatchLevelState state;
        state.depth = result.cross_level_initial.depth;
        state.normal = result.cross_level_initial.normal;
        state.cost = result.cross_level_initial.cost;
        if (consumable_previous_state != nullptr &&
            consumable_previous_state != &coarsest.state_before_filter) {
            error = "PatchMatch x16 consumable state does not match its input";
            return false;
        }
        const bool consume_previous = !capture_diagnostic_checkpoints &&
            consumable_previous_state != nullptr;
        if (consume_previous) {
            state.winner = std::move(consumable_previous_state->winner);
            state.candidates = std::move(consumable_previous_state->candidates);
            state.neighbor_cost =
                std::move(consumable_previous_state->neighbor_cost);
            state.average_cost =
                std::move(consumable_previous_state->average_cost);
            state.auxiliary = std::move(consumable_previous_state->auxiliary);
        } else {
            state.winner = coarsest.state_before_filter.winner;
            state.candidates = coarsest.state_before_filter.candidates;
            state.neighbor_cost = coarsest.state_before_filter.neighbor_cost;
            state.average_cost = coarsest.state_before_filter.average_cost;
            state.auxiliary = coarsest.state_before_filter.auxiliary;
        }
        state.cuda_cost_scratch_materialized =
            coarsest.state_before_filter.cuda_cost_scratch_materialized;
        state.cuda_main_state_materialized =
            coarsest.state_before_filter.cuda_main_state_materialized;
        state.cuda_inlier_masks_materialized =
            coarsest.state_before_filter.cuda_inlier_masks_materialized;
        state.cuda_workspace_handoff =
            coarsest.state_before_filter.cuda_workspace_handoff;
        state.neighbor_inlier_masks =
            coarsest.state_before_filter.neighbor_inlier_masks;
        const std::vector<float>& coarse_depth =
            result.cross_level_initial.coarse_depth;
        const std::vector<float>& coarse_radius =
            result.cross_level_initial.coarse_radius;
        const auto reference_image =
            make_recovered_patchmatch_reference_image_allocation(
                preparation.reference, preparation.target_downscale,
                downscale);
        std::uint64_t next_producer_handoff =
            std::exchange(state.cuda_workspace_handoff, 0U);
        const bool run_final_refinement =
            downscale <= preparation.target_downscale * 4U;

        const auto checkpoint = [](const RecoveredPatchMatchLevelState& source) {
            RecoveredPatchMatchX16Checkpoint value;
            value.depth = source.depth;
            value.normal = source.normal;
            value.cost = source.cost;
            value.winner = source.winner;
            value.candidates = source.candidates;
            value.neighbor_cost = source.neighbor_cost;
            value.average_cost = source.average_cost;
            value.auxiliary = source.auxiliary;
            value.neighbor_inlier_masks = source.neighbor_inlier_masks;
            return value;
        };

        const auto cost_and_wta = [&state, &coarse_depth, &coarse_radius,
                                   &reference_image,
                                   &preparation, downscale, width, height,
                                   neighbor_capacity, camera_atlas_state,
                                   &next_producer_handoff, &error](
            PatchMatchCandidateOutput candidates,
            std::uint32_t iteration,
            bool all_neighbors_state,
            std::uint32_t hypothesis_count,
            std::uint32_t patch_radius,
            std::uint32_t checkerboard,
            std::uint32_t step,
            std::uint32_t only_each_fourth,
            std::size_t work_items,
            bool materialize_cost_output) {
            auto binding = make_recovered_patchmatch_cost_binding(
                preparation, *camera_atlas_state, downscale, iteration,
                all_neighbors_state, false);
            PatchMatchCostInput cost_input;
            cost_input.reference_camera = binding.reference_camera;
            cost_input.rotate_before_camera = binding.normal_rotations.before;
            cost_input.rotate_after_camera = binding.normal_rotations.after;
            cost_input.depth_downscale = downscale;
            cost_input.image_one_step_more_detailed = 1U;
            cost_input.deviation_threshold_multiplier =
                binding.deviation_threshold_multiplier;
            cost_input.depth_view = state.depth;
            cost_input.normal_view = state.normal;
            cost_input.cost_view = state.cost;
            cost_input.coarse_depth_view = coarse_depth;
            cost_input.coarse_depth_radius_view = coarse_radius;
            cost_input.reference_image_view = reference_image;
            cost_input.neighbor_texture_width = binding.neighbor_texture_width;
            cost_input.neighbor_texture_height =
                binding.neighbor_texture_height;
            cost_input.neighbor_texture_view =
                binding.initial_neighbor_texture_view;
            cost_input.resource_groups_view = binding.resource_groups_view;
            cost_input.neighbor_batch = std::move(binding.neighbor_batch);
            cost_input.candidates = std::move(candidates);
            cost_input.initial_neighbor_cost = std::move(state.neighbor_cost);
            cost_input.initial_average_cost = std::move(state.average_cost);
            cost_input.initial_auxiliary = std::move(state.auxiliary);
            cost_input.reference_patch_radius = patch_radius;
            cost_input.hypotheses_per_pixel = hypothesis_count;
            cost_input.neighbor_count = binding.neighbor_count;
            cost_input.neighbor_cost_capacity =
                binding.neighbor_cost_capacity;
            cost_input.is_checkboard = checkerboard;
            cost_input.checkboard_step = step;
            cost_input.only_each_fourth_pixel = only_each_fourth;
            cost_input.global_work_items = work_items;
            cost_input.device_index = preparation.device_index;
            cost_input.camera_resource_generation =
                binding.camera_resource_generation;
            cost_input.cuda_workspace_handoff =
                cost_input.candidates.cuda_workspace_handoff;
            cost_input.initial_cuda_cost_scratch_materialized =
                state.cuda_cost_scratch_materialized;
            cost_input.defer_cuda_host_output = !materialize_cost_output;
            PatchMatchCostOutput cost_output;
            if (!run_recovered_patchmatch_cost_cuda_movable(
                    std::move(cost_input), cost_output, error))
                return false;

            PatchMatchWtaInput wta;
            wta.camera = binding.reference_camera;
            wta.depth_downscale = downscale;
            wta.image_one_step_more_detailed = 1U;
            wta.hypotheses_per_pixel = hypothesis_count;
            wta.depth = std::move(state.depth);
            wta.normal = std::move(state.normal);
            wta.cost = std::move(state.cost);
            wta.candidates = std::move(cost_output.candidates);
            wta.average_cost = std::move(cost_output.average_cost);
            wta.winner = std::move(state.winner);
            wta.is_checkboard = checkerboard;
            wta.checkboard_step = step;
            wta.only_each_fourth_pixel = only_each_fourth;
            wta.global_work_items = work_items;
            wta.device_index = preparation.device_index;
            wta.cuda_workspace_handoff =
                cost_output.cuda_workspace_handoff;
            wta.cuda_cost_output_materialized =
                cost_output.cuda_host_output_materialized;
            wta.defer_cuda_host_output =
                !materialize_cost_output &&
                cost_output.cuda_workspace_handoff != 0U;
            PatchMatchWtaOutput wta_output;
            if (!run_recovered_patchmatch_wta_cuda_movable(
                    std::move(wta), wta_output, error))
                return false;
            cost_output.candidates = std::move(wta.candidates);
            cost_output.average_cost = std::move(wta.average_cost);

            PatchMatchCopyInlierMasksInput copy_inliers;
            copy_inliers.width = static_cast<std::uint32_t>(width);
            copy_inliers.height = static_cast<std::uint32_t>(height);
            copy_inliers.hypotheses_per_pixel = hypothesis_count;
            copy_inliers.neighbor_count =
                static_cast<std::uint32_t>(neighbor_capacity);
            copy_inliers.temporary_inlier_masks =
                std::move(cost_output.auxiliary);
            copy_inliers.winner = std::move(wta_output.winner);
            copy_inliers.initial_neighbor_inlier_masks =
                std::move(state.neighbor_inlier_masks);
            copy_inliers.is_checkboard = checkerboard;
            copy_inliers.checkboard_step = step;
            copy_inliers.only_each_fourth_pixel = only_each_fourth;
            copy_inliers.global_work_items = work_items;
            copy_inliers.device_index = preparation.device_index;
            copy_inliers.cuda_workspace_handoff =
                wta_output.cuda_workspace_handoff;
            copy_inliers.cuda_temporary_inlier_masks_materialized =
                cost_output.cuda_host_output_materialized;
            copy_inliers.cuda_winner_materialized =
                wta_output.cuda_host_output_materialized;
            copy_inliers.cuda_initial_neighbor_inlier_masks_materialized =
                state.cuda_inlier_masks_materialized;
            copy_inliers.defer_cuda_host_output =
                !materialize_cost_output &&
                wta_output.cuda_workspace_handoff != 0U;
            PatchMatchCopyInlierMasksOutput copied_inliers;
            if (!run_recovered_patchmatch_copy_inlier_masks_cuda_movable(
                    std::move(copy_inliers), copied_inliers, error))
                return false;
            cost_output.auxiliary =
                std::move(copy_inliers.temporary_inlier_masks);
            wta_output.winner = std::move(copy_inliers.winner);
            next_producer_handoff =
                copied_inliers.cuda_workspace_handoff;
            state.depth = std::move(wta_output.depth);
            state.normal = std::move(wta_output.normal);
            state.cost = std::move(wta_output.cost);
            state.winner = std::move(wta_output.winner);
            state.candidates = std::move(cost_output.candidates);
            state.neighbor_cost = std::move(cost_output.per_neighbor_cost);
            state.average_cost = std::move(cost_output.average_cost);
            state.auxiliary = std::move(cost_output.auxiliary);
            state.cuda_cost_scratch_materialized =
                cost_output.cuda_host_output_materialized;
            state.cuda_main_state_materialized =
                wta_output.cuda_host_output_materialized;
            state.neighbor_inlier_masks =
                std::move(copied_inliers.neighbor_inlier_masks);
            state.cuda_inlier_masks_materialized =
                copied_inliers.cuda_host_output_materialized;
            return true;
        };

        const std::size_t fourth_items =
            ((width + 1U) / 2U) * ((height + 1U) / 2U);
        const std::size_t fourth_checker_items =
            ((width + 3U) / 4U) * ((height + 1U) / 2U);
        const std::size_t checker_items = ((width + 1U) / 2U) * height;

        for (std::uint32_t iteration = 0U; iteration < 6U; ++iteration) {
            PatchMatchRefinementInput refinement;
            refinement.camera = patch_camera;
            refinement.depth_downscale = downscale;
            refinement.depth_min = preparation.reference.depth_range.minimum;
            refinement.depth_max = preparation.reference.depth_range.maximum;
            refinement.reference_image_view = reference_image;
            refinement.image_one_step_more_detailed = 1U;
            refinement.deviation_threshold_multiplier =
                preparation.deviation_threshold_multiplier;
            refinement.depth_view = state.depth;
            refinement.normal_view = state.normal;
            refinement.cost_view = state.cost;
            refinement.coarse_depth_view = coarse_depth;
            refinement.coarse_depth_radius_view = coarse_radius;
            refinement.initial_candidate_depth =
                std::move(state.candidates.depth);
            refinement.initial_candidate_normal =
                std::move(state.candidates.normal);
            refinement.iteration = iteration;
            refinement.only_each_fourth_pixel = 1U;
            refinement.global_work_items = fourth_items;
            refinement.device_index = preparation.device_index;
            refinement.cuda_workspace_handoff =
                std::exchange(next_producer_handoff, 0U);
            refinement.defer_cuda_host_output = true;
            PatchMatchCandidateOutput refined;
            if (!run_recovered_patchmatch_refinement_cuda_movable(
                    std::move(refinement), refined, error) ||
                !cost_and_wta(std::move(refined), iteration, false,
                              8U, 3U, 0U, 0U, 1U, fourth_items, false))
                return false;

            for (std::uint32_t step = 0U; step < 2U; ++step) {
                PatchMatchPropagationInput propagation;
                propagation.camera = patch_camera;
                propagation.reference_to_neighbor_rotation =
                    preparation.propagation_rotation;
                propagation.depth_downscale = downscale;
                propagation.reference_image_view = reference_image;
                propagation.image_one_step_more_detailed = 1U;
                propagation.deviation_threshold_multiplier =
                    preparation.deviation_threshold_multiplier;
                propagation.depth_view = state.depth;
                propagation.normal_view = state.normal;
                propagation.cost_view = state.cost;
                propagation.coarse_depth_view = coarse_depth;
                propagation.coarse_depth_radius_view = coarse_radius;
                propagation.initial_candidate_depth =
                    std::move(state.candidates.depth);
                propagation.initial_candidate_normal =
                    std::move(state.candidates.normal);
                propagation.checkboard_step = step;
                propagation.only_each_fourth_pixel = 1U;
                propagation.global_work_items = fourth_checker_items;
                propagation.device_index = preparation.device_index;
                propagation.cuda_workspace_handoff =
                    std::exchange(next_producer_handoff, 0U);
                propagation.defer_cuda_host_output =
                    !capture_diagnostic_checkpoints ||
                    iteration != 5U || step != 1U;
                PatchMatchCandidateOutput propagated;
                if (!run_recovered_patchmatch_propagation_cuda_movable(
                        std::move(propagation), propagated, error) ||
                    !cost_and_wta(std::move(propagated), iteration, false,
                                  8U, 3U, 1U, step, 1U,
                                  fourth_checker_items,
                                  capture_diagnostic_checkpoints &&
                                      iteration == 5U && step == 1U))
                    return false;
            }
        }
        if (capture_diagnostic_checkpoints) {
            if (!state.candidates.cuda_host_output_materialized ||
                !state.cuda_cost_scratch_materialized ||
                !state.cuda_main_state_materialized ||
                !state.cuda_inlier_masks_materialized) {
                error = "PatchMatch x16 C2P diagnostic boundary has deferred state";
                return false;
            }
            result.before_c2p = checkpoint(state);
        } else if (next_producer_handoff == 0U) {
            error = "PatchMatch x16 production C2P boundary lacks a resident handoff";
            return false;
        }

        PatchMatchCoarseToPreciseInput c2p;
        c2p.camera = patch_camera;
        c2p.depth_downscale = downscale;
        c2p.depth = state.depth;
        c2p.normal = state.normal;
        c2p.cost = state.cost;
        c2p.initial_candidates = std::move(state.candidates);
        c2p.global_work_items = pixels;
        c2p.device_index = preparation.device_index;
        c2p.cuda_workspace_handoff =
            std::exchange(next_producer_handoff, 0U);
        c2p.cuda_main_state_materialized =
            state.cuda_main_state_materialized;
        c2p.defer_cuda_host_output =
            c2p.cuda_workspace_handoff != 0U;
        PatchMatchCandidateOutput c2p_candidates;
        if (!run_recovered_patchmatch_coarse_to_precise_cuda_movable(
                std::move(c2p), c2p_candidates, error))
            return false;
        if (!cost_and_wta(std::move(c2p_candidates), 6U, false,
                          2U, 3U, 0U, 0U, 0U, pixels,
                          capture_diagnostic_checkpoints))
            return false;
        if (capture_diagnostic_checkpoints)
            result.after_wta2 = checkpoint(state);

        for (const std::uint32_t iteration : {7U, 8U}) {
            PatchMatchRefinementInput refinement;
            refinement.camera = patch_camera;
            refinement.depth_downscale = downscale;
            refinement.depth_min = preparation.reference.depth_range.minimum;
            refinement.depth_max = preparation.reference.depth_range.maximum;
            refinement.reference_image_view = reference_image;
            refinement.image_one_step_more_detailed = 1U;
            refinement.deviation_threshold_multiplier =
                preparation.deviation_threshold_multiplier;
            refinement.depth_view = state.depth;
            refinement.normal_view = state.normal;
            refinement.cost_view = state.cost;
            refinement.coarse_depth_view = coarse_depth;
            refinement.coarse_depth_radius_view = coarse_radius;
            refinement.initial_candidate_depth =
                std::move(state.candidates.depth);
            refinement.initial_candidate_normal =
                std::move(state.candidates.normal);
            refinement.iteration = iteration;
            refinement.global_work_items = pixels;
            refinement.device_index = preparation.device_index;
            refinement.cuda_workspace_handoff =
                std::exchange(next_producer_handoff, 0U);
            refinement.defer_cuda_host_output = true;
            PatchMatchCandidateOutput refined;
            if (!run_recovered_patchmatch_refinement_cuda_movable(
                    std::move(refinement), refined, error) ||
                !cost_and_wta(std::move(refined), iteration, false,
                              8U, 3U, 0U, 0U, 0U, pixels, false))
                return false;

            for (std::uint32_t step = 0U; step < 2U; ++step) {
                PatchMatchPropagationInput propagation;
                propagation.camera = patch_camera;
                propagation.reference_to_neighbor_rotation =
                    preparation.propagation_rotation;
                propagation.depth_downscale = downscale;
                propagation.reference_image_view = reference_image;
                propagation.image_one_step_more_detailed = 1U;
                propagation.deviation_threshold_multiplier =
                    preparation.deviation_threshold_multiplier;
                propagation.depth_view = state.depth;
                propagation.normal_view = state.normal;
                propagation.cost_view = state.cost;
                propagation.coarse_depth_view = coarse_depth;
                propagation.coarse_depth_radius_view = coarse_radius;
                propagation.initial_candidate_depth =
                    std::move(state.candidates.depth);
                propagation.initial_candidate_normal =
                    std::move(state.candidates.normal);
                propagation.checkboard_step = step;
                propagation.global_work_items = checker_items;
                propagation.device_index = preparation.device_index;
                propagation.cuda_workspace_handoff =
                    std::exchange(next_producer_handoff, 0U);
                propagation.defer_cuda_host_output =
                    !capture_diagnostic_checkpoints || run_final_refinement ||
                    iteration != 8U || step != 1U;
                PatchMatchCandidateOutput propagated;
                if (!run_recovered_patchmatch_propagation_cuda_movable(
                        std::move(propagation), propagated, error) ||
                    !cost_and_wta(std::move(propagated), iteration, false,
                                  8U, 3U, 1U, step, 0U, checker_items,
                                  capture_diagnostic_checkpoints &&
                                      !run_final_refinement &&
                                      iteration == 8U && step == 1U))
                    return false;
            }
        }

        // x16 final is present for d4/d8, but absent from the observed d2
        // schedule.  When present it reuses the iteration-8 resource binding.
        if (run_final_refinement) {
            PatchMatchFinalRefinementInput final_refinement;
            final_refinement.camera = patch_camera;
            final_refinement.depth_downscale = downscale;
            final_refinement.reference_image_view = reference_image;
            final_refinement.image_one_step_more_detailed = 1U;
            final_refinement.deviation_threshold_multiplier =
                preparation.deviation_threshold_multiplier;
            final_refinement.depth_view = state.depth;
            final_refinement.normal_view = state.normal;
            final_refinement.cost_view = state.cost;
            final_refinement.initial_candidate_depth =
                std::move(state.candidates.depth);
            final_refinement.initial_candidate_normal =
                std::move(state.candidates.normal);
            final_refinement.global_work_items = pixels;
            final_refinement.device_index = preparation.device_index;
            final_refinement.cuda_workspace_handoff =
                std::exchange(next_producer_handoff, 0U);
            final_refinement.defer_cuda_host_output =
                !capture_diagnostic_checkpoints;
            PatchMatchFinalRefinementOutput final_output;
            if (!run_recovered_patchmatch_final_refinement_state_cuda_movable(
                    std::move(final_refinement), final_output, error))
                return false;
            state.cost = std::move(final_output.cost);
            if (!cost_and_wta(std::move(final_output.candidates),
                              8U, false, 8U, 2U, 0U, 0U, 0U, pixels,
                              capture_diagnostic_checkpoints))
                return false;
        }

        PatchMatchLevelBoundaryInput boundary;
        boundary.filter.camera = patch_camera;
        boundary.filter.depth_downscale = downscale;
        boundary.filter.depth_min = preparation.reference.depth_range.minimum;
        boundary.filter.depth_max = preparation.reference.depth_range.maximum;
        boundary.filter.depth_allocation = state.depth;
        boundary.filter.cost_allocation = state.cost;
        boundary.filter.normal_allocation = state.normal;
        boundary.filter.estimated_normal_allocation.assign(
            estimated_pixels * 3U, std::uint8_t{0});
        boundary.filter.filtered_mask_allocation.assign(
            finest_pixels, std::uint8_t{0});
        const auto& previous_mask =
            coarsest.boundary.filter.filtered_mask_allocation;
        if (previous_mask.size() != finest_pixels) {
            error = "PatchMatch x16 inherited filter mask is incomplete";
            return false;
        }
        boundary.filter.filtered_mask_allocation = previous_mask;
        boundary.filter.estimate_normal_map = true;
        boundary.filter.device_index = preparation.device_index;
        if (!capture_diagnostic_checkpoints) {
            boundary.filter.cuda_workspace_handoff =
                std::exchange(next_producer_handoff, 0U);
            boundary.filter.cuda_main_state_materialized =
                state.cuda_main_state_materialized;
            boundary.filter.cuda_inlier_masks_materialized =
                state.cuda_inlier_masks_materialized;
            boundary.filter.neighbor_inlier_masks_allocation =
                state.neighbor_inlier_masks;
        }
        boundary.speckle_component_size_threshold = 6U;
        boundary.bilateral_image.assign(bilateral_image.begin(),
                                        bilateral_image.end());

        if (capture_diagnostic_checkpoints) {
            if (!state.candidates.cuda_host_output_materialized ||
                !state.cuda_cost_scratch_materialized ||
                !state.cuda_main_state_materialized ||
                !state.cuda_inlier_masks_materialized) {
                error = "PatchMatch x16 diagnostic filter boundary has deferred state";
                return false;
            }
        } else if (boundary.filter.cuda_workspace_handoff == 0U) {
            error = "PatchMatch x16 production filter lacks a resident handoff";
            return false;
        }
        result.state_before_filter = std::move(state);
        if (!run_recovered_patchmatch_level_boundary_cuda(
                boundary, result.boundary, error))
            return false;
        if (!capture_diagnostic_checkpoints) {
            result.state_before_filter.neighbor_inlier_masks =
                result.boundary.filter.neighbor_inlier_masks_allocation;
            result.state_before_filter.cuda_inlier_masks_materialized = true;
            result.state_before_filter.cuda_workspace_handoff =
                result.boundary.filter.cuda_workspace_handoff;
        }
        output = std::move(result);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool run_recovered_patchmatch_finer_level_cuda(
    const Camera& camera,
    const RecoveredPatchMatchHostPreparation& preparation,
    std::uint32_t downscale,
    const RecoveredPatchMatchLevelState& previous_state_before_filter,
    const PatchMatchLevelBoundaryOutput& previous_boundary,
    std::span<const std::uint8_t> bilateral_image,
    RecoveredPatchMatchFinerLevelOutput& output,
    std::string& error,
    RecoveredPatchMatchCostAtlasState* camera_atlas_state,
    bool capture_diagnostic_checkpoints,
    RecoveredPatchMatchLevelState* consumable_previous_state) {
    try {
        constexpr std::size_t hypotheses =
            PatchMatchCandidateOutput::hypotheses;
        constexpr std::size_t capacity =
            PatchMatchCandidateOutput::capacity;
        const PatchMatchCamera& patch_camera = preparation.reference_camera;
        const bool valid_power_of_two =
            downscale != 0U && (downscale & (downscale - 1U)) == 0U;
        const bool target_level =
            downscale == preparation.target_downscale;
        if (!valid_power_of_two || downscale >= 16U ||
            downscale < preparation.target_downscale ||
            preparation.target_downscale < 2U ||
            preparation.reference.camera_index != camera.index ||
            preparation.reference_camera_index != camera.index ||
            patch_camera.width_original == 0U ||
            patch_camera.height_original == 0U ||
            preparation.neighbor_resources.ranked_neighbors.empty()) {
            error = "PatchMatch finer-level continuation parameters are unsupported";
            return false;
        }

        const std::size_t width =
            (patch_camera.width_original + downscale - 1U) / downscale;
        const std::size_t height =
            (patch_camera.height_original + downscale - 1U) / downscale;
        const std::size_t pixels = width * height;
        const std::uint32_t previous_downscale = downscale * 2U;
        const std::size_t previous_width =
            (patch_camera.width_original + previous_downscale - 1U) /
            previous_downscale;
        const std::size_t previous_height =
            (patch_camera.height_original + previous_downscale - 1U) /
            previous_downscale;
        const std::size_t previous_pixels =
            previous_width * previous_height;
        const std::size_t finest_width =
            (patch_camera.width_original + preparation.target_downscale - 1U) /
            preparation.target_downscale;
        const std::size_t finest_height =
            (patch_camera.height_original + preparation.target_downscale - 1U) /
            preparation.target_downscale;
        const std::size_t finest_pixels = finest_width * finest_height;
        const std::uint32_t estimated_downscale =
            preparation.target_downscale * 2U;
        const std::size_t estimated_width =
            (patch_camera.width_original + estimated_downscale - 1U) /
            estimated_downscale;
        const std::size_t estimated_height =
            (patch_camera.height_original + estimated_downscale - 1U) /
            estimated_downscale;
        const std::size_t estimated_pixels =
            estimated_width * estimated_height;
        const std::size_t neighbor_capacity =
            preparation.neighbor_resources.ranked_neighbors.size();
        const std::size_t inlier_groups = (neighbor_capacity + 7U) / 8U;
        if (pixels > std::numeric_limits<std::uint32_t>::max()) {
            error = "PatchMatch finer-level active grid exceeds the recovered uint32 batch domain";
            return false;
        }
        if (pixels == 0U || previous_pixels == 0U ||
            previous_boundary.bilateral.depth.size() != previous_pixels ||
            previous_boundary.bilateral.normal.size() != previous_pixels * 3U ||
            (!target_level && bilateral_image.size() != pixels) ||
            (target_level && !bilateral_image.empty()) ||
            previous_state_before_filter.candidates.depth.size() !=
                hypotheses * capacity ||
            previous_state_before_filter.candidates.normal.size() !=
                hypotheses * capacity * 3U ||
            previous_state_before_filter.winner.size() != capacity ||
            previous_state_before_filter.neighbor_cost.size() !=
                neighbor_capacity * hypotheses * capacity ||
            previous_state_before_filter.average_cost.size() !=
                hypotheses * capacity ||
            previous_state_before_filter.auxiliary.size() !=
                inlier_groups * hypotheses * capacity ||
            (!previous_state_before_filter.cuda_cost_scratch_materialized &&
             previous_state_before_filter.cuda_workspace_handoff == 0U) ||
            previous_state_before_filter.neighbor_inlier_masks.size() !=
                inlier_groups * finest_pixels ||
            previous_boundary.filter.filtered_mask_allocation.size() !=
                finest_pixels ||
            previous_boundary.filter.estimated_normal_allocation.size() !=
                estimated_pixels * 3U) {
            error = "PatchMatch finer-level persistent state is incomplete";
            return false;
        }
        RecoveredPatchMatchCostAtlasState local_atlas_state;
        if (camera_atlas_state == nullptr) {
            local_atlas_state = make_recovered_patchmatch_cost_atlas_state(
                preparation.neighbor_resources);
            camera_atlas_state = &local_atlas_state;
        }

        RecoveredPatchMatchFinerLevelOutput result;
        result.downscale = downscale;
        result.target_level = target_level;
        if (!make_recovered_patchmatch_cross_level_state(
                camera, downscale,
                static_cast<std::uint32_t>(previous_width),
                static_cast<std::uint32_t>(previous_height),
                previous_boundary.bilateral.depth,
                previous_boundary.bilateral.normal,
                result.cross_level_initial, error))
            return false;
        if (result.cross_level_initial.width != width ||
            result.cross_level_initial.height != height) {
            error = "PatchMatch finer-level cross-level grid is inconsistent";
            return false;
        }

        RecoveredPatchMatchLevelState state;
        state.depth = result.cross_level_initial.depth;
        state.normal = result.cross_level_initial.normal;
        state.cost = result.cross_level_initial.cost;
        if (consumable_previous_state != nullptr &&
            consumable_previous_state != &previous_state_before_filter) {
            error = "PatchMatch finer-level consumable state does not match its input";
            return false;
        }
        const bool consume_previous = !capture_diagnostic_checkpoints &&
            consumable_previous_state != nullptr;
        if (consume_previous) {
            state.winner = std::move(consumable_previous_state->winner);
            state.candidates = std::move(consumable_previous_state->candidates);
            state.neighbor_cost =
                std::move(consumable_previous_state->neighbor_cost);
            state.average_cost =
                std::move(consumable_previous_state->average_cost);
            state.auxiliary = std::move(consumable_previous_state->auxiliary);
        } else {
            state.winner = previous_state_before_filter.winner;
            state.candidates = previous_state_before_filter.candidates;
            state.neighbor_cost = previous_state_before_filter.neighbor_cost;
            state.average_cost = previous_state_before_filter.average_cost;
            state.auxiliary = previous_state_before_filter.auxiliary;
        }
        state.cuda_cost_scratch_materialized =
            previous_state_before_filter.cuda_cost_scratch_materialized;
        state.cuda_main_state_materialized =
            previous_state_before_filter.cuda_main_state_materialized;
        state.cuda_inlier_masks_materialized =
            previous_state_before_filter.cuda_inlier_masks_materialized;
        state.cuda_workspace_handoff =
            previous_state_before_filter.cuda_workspace_handoff;
        state.neighbor_inlier_masks =
            previous_state_before_filter.neighbor_inlier_masks;
        const std::vector<float>& coarse_depth =
            result.cross_level_initial.coarse_depth;
        const std::vector<float>& coarse_radius =
            result.cross_level_initial.coarse_radius;
        const auto reference_image =
            make_recovered_patchmatch_reference_image_allocation(
                preparation.reference, preparation.target_downscale,
                downscale);
        std::uint64_t next_producer_handoff =
            std::exchange(state.cuda_workspace_handoff, 0U);

        const auto checkpoint = [](const RecoveredPatchMatchLevelState& source) {
            RecoveredPatchMatchX16Checkpoint value;
            value.depth = source.depth;
            value.normal = source.normal;
            value.cost = source.cost;
            value.winner = source.winner;
            value.candidates = source.candidates;
            value.neighbor_cost = source.neighbor_cost;
            value.average_cost = source.average_cost;
            value.auxiliary = source.auxiliary;
            value.neighbor_inlier_masks = source.neighbor_inlier_masks;
            return value;
        };
        const auto for_each_batch = [&error](std::size_t items,
                                             auto&& callback) {
            if (items > std::numeric_limits<std::uint32_t>::max()) {
                error = "PatchMatch finer-level work-item count exceeds the recovered uint32 batch domain";
                return false;
            }
            const auto span = patchmatch_balanced_batch_span(
                static_cast<std::uint32_t>(items));
            for (std::size_t offset = 0; offset < items; offset += span) {
                const std::size_t work_items =
                    std::min<std::size_t>(span, items - offset);
                if (offset > std::numeric_limits<std::uint32_t>::max() ||
                    work_items > std::numeric_limits<std::uint32_t>::max()) {
                    error = "PatchMatch finer-level batch offset exceeds the recovered uint32 launch domain";
                    return false;
                }
                if (!callback(static_cast<std::uint32_t>(offset), work_items))
                    return false;
            }
            return true;
        };

        const auto cost_and_wta =
            [&state, &coarse_depth, &coarse_radius, &reference_image,
             &preparation,
             downscale, width, height, neighbor_capacity,
             camera_atlas_state, &next_producer_handoff, &error](
                PatchMatchCandidateOutput candidates,
                std::uint32_t iteration,
                bool all_neighbors_state,
                std::uint32_t hypothesis_count,
                std::uint32_t patch_radius,
                std::uint32_t checkerboard,
                std::uint32_t step,
                std::uint32_t only_each_fourth,
                std::uint32_t offset,
                std::size_t work_items,
                bool materialize_cost_output) {
                auto binding = make_recovered_patchmatch_cost_binding(
                    preparation, *camera_atlas_state, downscale, iteration,
                    all_neighbors_state, false);
                PatchMatchCostInput cost_input;
                cost_input.reference_camera = binding.reference_camera;
                cost_input.rotate_before_camera =
                    binding.normal_rotations.before;
                cost_input.rotate_after_camera =
                    binding.normal_rotations.after;
                cost_input.depth_downscale = downscale;
                cost_input.image_one_step_more_detailed = 1U;
                cost_input.deviation_threshold_multiplier =
                    binding.deviation_threshold_multiplier;
                cost_input.depth_view = state.depth;
                cost_input.normal_view = state.normal;
                cost_input.cost_view = state.cost;
                cost_input.coarse_depth_view = coarse_depth;
                cost_input.coarse_depth_radius_view = coarse_radius;
                cost_input.reference_image_view = reference_image;
                cost_input.neighbor_texture_width =
                    binding.neighbor_texture_width;
                cost_input.neighbor_texture_height =
                    binding.neighbor_texture_height;
                cost_input.neighbor_texture_view =
                    binding.initial_neighbor_texture_view;
                cost_input.resource_groups_view = binding.resource_groups_view;
                cost_input.neighbor_batch = std::move(binding.neighbor_batch);
                cost_input.candidates = std::move(candidates);
                cost_input.initial_neighbor_cost =
                    std::move(state.neighbor_cost);
                cost_input.initial_average_cost =
                    std::move(state.average_cost);
                cost_input.initial_auxiliary = std::move(state.auxiliary);
                cost_input.reference_patch_radius = patch_radius;
                cost_input.hypotheses_per_pixel = hypothesis_count;
                cost_input.neighbor_count = binding.neighbor_count;
                cost_input.neighbor_cost_capacity =
                    binding.neighbor_cost_capacity;
                cost_input.is_checkboard = checkerboard;
                cost_input.checkboard_step = step;
                cost_input.only_each_fourth_pixel = only_each_fourth;
                cost_input.pixel_offset = offset;
                cost_input.global_work_items = work_items;
                cost_input.device_index = preparation.device_index;
                cost_input.camera_resource_generation =
                    binding.camera_resource_generation;
                cost_input.cuda_workspace_handoff =
                    cost_input.candidates.cuda_workspace_handoff;
                cost_input.initial_cuda_cost_scratch_materialized =
                    state.cuda_cost_scratch_materialized;
                cost_input.defer_cuda_host_output = !materialize_cost_output;
                PatchMatchCostOutput cost_output;
                if (!run_recovered_patchmatch_cost_cuda_movable(
                        std::move(cost_input), cost_output, error))
                    return false;

                PatchMatchWtaInput wta;
                wta.camera = binding.reference_camera;
                wta.depth_downscale = downscale;
                wta.image_one_step_more_detailed = 1U;
                wta.hypotheses_per_pixel = hypothesis_count;
                wta.depth = std::move(state.depth);
                wta.normal = std::move(state.normal);
                wta.cost = std::move(state.cost);
                wta.candidates = std::move(cost_output.candidates);
                wta.average_cost = std::move(cost_output.average_cost);
                wta.winner = std::move(state.winner);
                wta.is_checkboard = checkerboard;
                wta.checkboard_step = step;
                wta.only_each_fourth_pixel = only_each_fourth;
                wta.pixel_offset = offset;
                wta.global_work_items = work_items;
                wta.device_index = preparation.device_index;
                wta.cuda_workspace_handoff =
                    cost_output.cuda_workspace_handoff;
                wta.cuda_cost_output_materialized =
                    cost_output.cuda_host_output_materialized;
                wta.defer_cuda_host_output =
                    !materialize_cost_output &&
                    cost_output.cuda_workspace_handoff != 0U;
                PatchMatchWtaOutput wta_output;
                if (!run_recovered_patchmatch_wta_cuda_movable(
                        std::move(wta), wta_output, error))
                    return false;
                cost_output.candidates = std::move(wta.candidates);
                cost_output.average_cost = std::move(wta.average_cost);

                PatchMatchCopyInlierMasksInput copy_inliers;
                copy_inliers.width = static_cast<std::uint32_t>(width);
                copy_inliers.height = static_cast<std::uint32_t>(height);
                copy_inliers.hypotheses_per_pixel = hypothesis_count;
                copy_inliers.neighbor_count =
                    static_cast<std::uint32_t>(neighbor_capacity);
                copy_inliers.temporary_inlier_masks =
                    std::move(cost_output.auxiliary);
                copy_inliers.winner = std::move(wta_output.winner);
                copy_inliers.initial_neighbor_inlier_masks =
                    std::move(state.neighbor_inlier_masks);
                copy_inliers.is_checkboard = checkerboard;
                copy_inliers.checkboard_step = step;
                copy_inliers.only_each_fourth_pixel = only_each_fourth;
                copy_inliers.pixel_offset = offset;
                copy_inliers.global_work_items = work_items;
                copy_inliers.device_index = preparation.device_index;
                copy_inliers.cuda_workspace_handoff =
                    wta_output.cuda_workspace_handoff;
                copy_inliers.cuda_temporary_inlier_masks_materialized =
                    cost_output.cuda_host_output_materialized;
                copy_inliers.cuda_winner_materialized =
                    wta_output.cuda_host_output_materialized;
                copy_inliers.cuda_initial_neighbor_inlier_masks_materialized =
                    state.cuda_inlier_masks_materialized;
                copy_inliers.defer_cuda_host_output =
                    !materialize_cost_output &&
                    wta_output.cuda_workspace_handoff != 0U;
                PatchMatchCopyInlierMasksOutput copied_inliers;
                if (!run_recovered_patchmatch_copy_inlier_masks_cuda_movable(
                        std::move(copy_inliers), copied_inliers, error))
                    return false;
                cost_output.auxiliary =
                    std::move(copy_inliers.temporary_inlier_masks);
                wta_output.winner = std::move(copy_inliers.winner);
                next_producer_handoff =
                    copied_inliers.cuda_workspace_handoff;

                state.depth = std::move(wta_output.depth);
                state.normal = std::move(wta_output.normal);
                state.cost = std::move(wta_output.cost);
                state.winner = std::move(wta_output.winner);
                state.candidates = std::move(cost_output.candidates);
                state.neighbor_cost =
                    std::move(cost_output.per_neighbor_cost);
                state.average_cost = std::move(cost_output.average_cost);
                state.auxiliary = std::move(cost_output.auxiliary);
                state.cuda_cost_scratch_materialized =
                    cost_output.cuda_host_output_materialized;
                state.cuda_main_state_materialized =
                    wta_output.cuda_host_output_materialized;
                state.neighbor_inlier_masks =
                    std::move(copied_inliers.neighbor_inlier_masks);
                state.cuda_inlier_masks_materialized =
                    copied_inliers.cuda_host_output_materialized;
                return true;
            };

        const std::size_t full_items = pixels;
        const std::size_t checker_items =
            ((width + 1U) / 2U) * height;
        const std::size_t fourth_items =
            ((width + 1U) / 2U) * ((height + 1U) / 2U);
        const std::size_t fourth_checker_items =
            ((width + 3U) / 4U) * ((height + 1U) / 2U);

        const auto run_iteration =
            [&](std::uint32_t iteration, bool only_fourth,
                bool materialize_last_propagation) {
                const std::size_t refinement_items =
                    only_fourth ? fourth_items : full_items;
                if (!for_each_batch(
                        refinement_items,
                        [&](std::uint32_t offset, std::size_t work_items) {
                            PatchMatchRefinementInput refinement;
                            refinement.camera = patch_camera;
                            refinement.depth_downscale = downscale;
                            refinement.depth_min =
                                preparation.reference.depth_range.minimum;
                            refinement.depth_max =
                                preparation.reference.depth_range.maximum;
                            refinement.reference_image_view = reference_image;
                            refinement.image_one_step_more_detailed = 1U;
                            refinement.deviation_threshold_multiplier =
                                preparation.deviation_threshold_multiplier;
                            refinement.depth_view = state.depth;
                            refinement.normal_view = state.normal;
                            refinement.cost_view = state.cost;
                            refinement.coarse_depth_view = coarse_depth;
                            refinement.coarse_depth_radius_view = coarse_radius;
                            refinement.initial_candidate_depth =
                                std::move(state.candidates.depth);
                            refinement.initial_candidate_normal =
                                std::move(state.candidates.normal);
                            refinement.iteration = iteration;
                            refinement.only_each_fourth_pixel =
                                only_fourth ? 1U : 0U;
                            refinement.pixel_offset = offset;
                            refinement.global_work_items = work_items;
                            refinement.device_index = preparation.device_index;
                            refinement.cuda_workspace_handoff =
                                std::exchange(next_producer_handoff, 0U);
                            refinement.defer_cuda_host_output = true;
                            PatchMatchCandidateOutput refined;
                            return run_recovered_patchmatch_refinement_cuda_movable(
                                       std::move(refinement), refined, error) &&
                                   cost_and_wta(
                                       std::move(refined), iteration, false,
                                       8U, 3U, 0U, 0U,
                                       only_fourth ? 1U : 0U,
                                       offset, work_items, false);
                        }))
                    return false;

                const std::size_t propagation_items =
                    only_fourth ? fourth_checker_items : checker_items;
                for (std::uint32_t step = 0U; step < 2U; ++step) {
                    if (!for_each_batch(
                            propagation_items,
                            [&](std::uint32_t offset,
                                std::size_t work_items) {
                                PatchMatchPropagationInput propagation;
                                propagation.camera = patch_camera;
                                propagation.reference_to_neighbor_rotation =
                                    preparation.propagation_rotation;
                                propagation.depth_downscale = downscale;
                                propagation.reference_image_view = reference_image;
                                propagation.image_one_step_more_detailed = 1U;
                                propagation.deviation_threshold_multiplier =
                                    preparation.deviation_threshold_multiplier;
                                propagation.depth_view = state.depth;
                                propagation.normal_view = state.normal;
                                propagation.cost_view = state.cost;
                                propagation.coarse_depth_view = coarse_depth;
                                propagation.coarse_depth_radius_view = coarse_radius;
                                propagation.initial_candidate_depth =
                                    std::move(state.candidates.depth);
                                propagation.initial_candidate_normal =
                                    std::move(state.candidates.normal);
                                propagation.checkboard_step = step;
                                propagation.only_each_fourth_pixel =
                                    only_fourth ? 1U : 0U;
                                propagation.pixel_offset = offset;
                                propagation.global_work_items = work_items;
                                propagation.device_index =
                                    preparation.device_index;
                                propagation.cuda_workspace_handoff =
                                    std::exchange(next_producer_handoff, 0U);
                                propagation.defer_cuda_host_output =
                                    !(materialize_last_propagation &&
                                      step == 1U &&
                                      static_cast<std::size_t>(offset) +
                                              work_items ==
                                          propagation_items);
                                PatchMatchCandidateOutput propagated;
                                return run_recovered_patchmatch_propagation_cuda_movable(
                                           std::move(propagation), propagated,
                                           error) &&
                                       cost_and_wta(
                                           std::move(propagated), iteration,
                                           false, 8U, 3U, 1U, step,
                                           only_fourth ? 1U : 0U,
                                           offset, work_items,
                                           materialize_last_propagation &&
                                               step == 1U &&
                                               static_cast<std::size_t>(offset) +
                                                       work_items ==
                                                   propagation_items);
                            }))
                        return false;
                }
                return true;
            };

        const std::uint32_t inherited_iterations = target_level ? 1U : 6U;
        for (std::uint32_t iteration = 0U;
             iteration < inherited_iterations; ++iteration) {
            if (!run_iteration(
                    iteration, true,
                    capture_diagnostic_checkpoints &&
                        iteration + 1U == inherited_iterations))
                return false;
        }
        if (capture_diagnostic_checkpoints) {
            if (!state.candidates.cuda_host_output_materialized ||
                !state.cuda_cost_scratch_materialized ||
                !state.cuda_main_state_materialized ||
                !state.cuda_inlier_masks_materialized) {
                error = "PatchMatch finer C2P diagnostic boundary has deferred state";
                return false;
            }
            result.before_c2p = checkpoint(state);
        } else if (next_producer_handoff == 0U) {
            error = "PatchMatch finer production C2P boundary lacks a resident handoff";
            return false;
        }

        const std::uint32_t c2p_iteration = target_level ? 1U : 6U;
        if (!for_each_batch(
                full_items,
                [&](std::uint32_t offset, std::size_t work_items) {
                    PatchMatchCoarseToPreciseInput c2p;
                    c2p.camera = patch_camera;
                    c2p.depth_downscale = downscale;
                    c2p.depth = state.depth;
                    c2p.normal = state.normal;
                    c2p.cost = state.cost;
                    c2p.initial_candidates = std::move(state.candidates);
                    c2p.pixel_offset = offset;
                    c2p.global_work_items = work_items;
                    c2p.device_index = preparation.device_index;
                    c2p.cuda_workspace_handoff =
                        std::exchange(next_producer_handoff, 0U);
                    c2p.cuda_main_state_materialized =
                        state.cuda_main_state_materialized;
                    c2p.defer_cuda_host_output =
                        c2p.cuda_workspace_handoff != 0U;
                    PatchMatchCandidateOutput c2p_candidates;
                    return run_recovered_patchmatch_coarse_to_precise_cuda_movable(
                               std::move(c2p), c2p_candidates, error) &&
                           cost_and_wta(
                               std::move(c2p_candidates), c2p_iteration,
                               false, 2U, 3U, 0U, 0U, 0U, offset,
                               work_items, capture_diagnostic_checkpoints);
                }))
            return false;
        if (capture_diagnostic_checkpoints)
            result.after_wta2 = checkpoint(state);

        if (target_level) {
            if (!run_iteration(2U, false, false)) return false;
        } else {
            if (!run_iteration(7U, false, false) ||
                !run_iteration(8U, false, false))
                return false;
        }

        const std::uint32_t final_iteration = target_level ? 2U : 8U;
        if (!for_each_batch(
                full_items,
                [&](std::uint32_t offset, std::size_t work_items) {
                    PatchMatchFinalRefinementInput final_refinement;
                    final_refinement.camera = patch_camera;
                    final_refinement.depth_downscale = downscale;
                    final_refinement.reference_image_view = reference_image;
                    final_refinement.image_one_step_more_detailed = 1U;
                    final_refinement.deviation_threshold_multiplier =
                        preparation.deviation_threshold_multiplier;
                    final_refinement.depth_view = state.depth;
                    final_refinement.normal_view = state.normal;
                    final_refinement.cost_view = state.cost;
                    final_refinement.initial_candidate_depth =
                        std::move(state.candidates.depth);
                    final_refinement.initial_candidate_normal =
                        std::move(state.candidates.normal);
                    final_refinement.pixel_offset = offset;
                    final_refinement.global_work_items = work_items;
                    final_refinement.device_index = preparation.device_index;
                    final_refinement.cuda_workspace_handoff =
                        std::exchange(next_producer_handoff, 0U);
                    final_refinement.defer_cuda_host_output =
                        !capture_diagnostic_checkpoints ||
                        static_cast<std::size_t>(offset) + work_items !=
                        full_items;
                    PatchMatchFinalRefinementOutput final_output;
                    if (!run_recovered_patchmatch_final_refinement_state_cuda_movable(
                            std::move(final_refinement), final_output, error))
                        return false;
                    state.cost = std::move(final_output.cost);
                    return cost_and_wta(
                        std::move(final_output.candidates), final_iteration,
                        target_level, 8U, 2U, 0U, 0U, 0U, offset,
                        work_items,
                        capture_diagnostic_checkpoints &&
                            static_cast<std::size_t>(offset) + work_items ==
                                full_items);
                }))
            return false;

        PatchMatchLevelBoundaryInput boundary;
        boundary.filter.camera = patch_camera;
        boundary.filter.depth_downscale = downscale;
        boundary.filter.depth_min =
            preparation.reference.depth_range.minimum;
        boundary.filter.depth_max =
            preparation.reference.depth_range.maximum;
        boundary.filter.depth_allocation = state.depth;
        boundary.filter.cost_allocation = state.cost;
        boundary.filter.normal_allocation = state.normal;
        if (target_level) {
            boundary.filter.estimated_normal_allocation =
                previous_boundary.filter.estimated_normal_allocation;
        } else {
            boundary.filter.estimated_normal_allocation.assign(
                estimated_pixels * 3U, std::uint8_t{0});
        }
        boundary.filter.filtered_mask_allocation =
            previous_boundary.filter.filtered_mask_allocation;
        boundary.filter.estimate_normal_map = !target_level;
        boundary.filter.device_index = preparation.device_index;
        if (!capture_diagnostic_checkpoints) {
            boundary.filter.cuda_workspace_handoff =
                std::exchange(next_producer_handoff, 0U);
            boundary.filter.cuda_main_state_materialized =
                state.cuda_main_state_materialized;
            boundary.filter.cuda_inlier_masks_materialized =
                state.cuda_inlier_masks_materialized;
            boundary.filter.neighbor_inlier_masks_allocation =
                state.neighbor_inlier_masks;
        }
        // sub_1D0A360 selects 30 for the target level (v818 == 0) and 6
        // for inherited intermediate levels before applying the scale shift.
        boundary.speckle_component_size_threshold = target_level ? 30U : 6U;
        if (!target_level)
            boundary.bilateral_image.assign(
                bilateral_image.begin(), bilateral_image.end());

        if (capture_diagnostic_checkpoints) {
            if (!state.candidates.cuda_host_output_materialized ||
                !state.cuda_cost_scratch_materialized ||
                !state.cuda_main_state_materialized ||
                !state.cuda_inlier_masks_materialized) {
                error = "PatchMatch finer diagnostic filter boundary has deferred state";
                return false;
            }
        } else if (boundary.filter.cuda_workspace_handoff == 0U) {
            error = "PatchMatch finer production filter lacks a resident handoff";
            return false;
        }
        result.state_before_filter = std::move(state);
        if (target_level) {
            PatchMatchFilterChainOutput filtered;
            if (!run_recovered_patchmatch_filter_chain_cuda(
                    boundary.filter, filtered, error) ||
                !filter_recovered_patchmatch_cuda_speckle_components(
                    static_cast<std::uint32_t>(width),
                    static_cast<std::uint32_t>(height),
                    filtered.filtered_mask_allocation,
                    filtered.depth_allocation,
                    boundary.speckle_component_size_threshold,
                    error))
                return false;
            result.boundary.filter = std::move(filtered);
            result.bilateral_executed = false;
        } else {
            if (!run_recovered_patchmatch_level_boundary_cuda(
                    boundary, result.boundary, error))
                return false;
            result.bilateral_executed = true;
        }
        if (!capture_diagnostic_checkpoints) {
            result.state_before_filter.neighbor_inlier_masks =
                result.boundary.filter.neighbor_inlier_masks_allocation;
            result.state_before_filter.cuda_inlier_masks_materialized = true;
            result.state_before_filter.cuda_workspace_handoff =
                result.boundary.filter.cuda_workspace_handoff;
        }

        result.level_product_depth.assign(
            result.boundary.filter.depth_allocation.begin(),
            result.boundary.filter.depth_allocation.begin() + pixels);
        const std::size_t current_inlier_bytes = inlier_groups * pixels;
        result.level_product_inlier_masks.assign(
            result.state_before_filter.neighbor_inlier_masks.begin(),
            result.state_before_filter.neighbor_inlier_masks.begin() +
                current_inlier_bytes);
        result.ranked_neighbor_camera_indices.reserve(
            preparation.ranked_neighbor_camera_indices.size());
        result.ranked_neighbor_camera_indices =
            preparation.ranked_neighbor_camera_indices;

        output = std::move(result);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool unpack_recovered_patchmatch_inlier_mask(
    std::span<const std::uint8_t> packed_masks,
    std::size_t pixels,
    std::size_t neighbor_count,
    std::size_t ranked_neighbor_index,
    std::vector<std::uint8_t>& output,
    std::string& error) {
    try {
        if (pixels == 0U || neighbor_count == 0U ||
            ranked_neighbor_index >= neighbor_count) {
            error = "PatchMatch packed-mask extraction indices are invalid";
            return false;
        }
        const std::size_t groups = (neighbor_count + 7U) / 8U;
        if (groups > std::numeric_limits<std::size_t>::max() / pixels ||
            packed_masks.size() != groups * pixels) {
            error = "PatchMatch packed-mask extraction backing is inconsistent";
            return false;
        }
        const std::size_t group = ranked_neighbor_index / 8U;
        const std::uint8_t bit = static_cast<std::uint8_t>(
            std::uint8_t{1} << (ranked_neighbor_index % 8U));
        const std::size_t group_offset = group * pixels;
        std::vector<std::uint8_t> result(pixels);
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            result[pixel] = static_cast<std::uint8_t>(
                (packed_masks[group_offset + pixel] & bit) != 0U);
        }
        output = std::move(result);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool compose_recovered_depthmap_default_image_d4(
    const Camera& camera,
    const std::array<std::span<const float>, 3>& persisted_depth_levels,
    std::vector<float>& output,
    std::string& error) {
    try {
        if (!camera.aligned || camera.image.width == 0U ||
            camera.image.height == 0U || camera.image.width % 16U != 0U ||
            camera.image.height % 16U != 0U ||
            !std::isfinite(camera.model.f) || camera.model.f <= 0.0 ||
            !std::isfinite(camera.model.cx) ||
            !std::isfinite(camera.model.cy)) {
            error = "DepthMap default-image composition requires an aligned finite pinhole camera with dimensions divisible by 16";
            return false;
        }

        constexpr std::array<std::uint32_t, 3> downscales{4U, 8U, 16U};
        std::array<std::size_t, 3> widths{};
        std::array<std::size_t, 3> heights{};
        for (std::size_t level = 0; level < 3U; ++level) {
            widths[level] = camera.image.width / downscales[level];
            heights[level] = camera.image.height / downscales[level];
            if (widths[level] >
                    std::numeric_limits<std::size_t>::max() /
                        heights[level] ||
                persisted_depth_levels[level].size() !=
                    widths[level] * heights[level]) {
                error = "DepthMap default-image composition pyramid dimensions are inconsistent";
                return false;
            }
            for (const float depth : persisted_depth_levels[level]) {
                if (!std::isfinite(depth) || depth < 0.0F) {
                    error = "DepthMap default-image composition depth is outside the recovered domain";
                    return false;
                }
            }
        }
        if (widths[0] != 2U * widths[1] ||
            heights[0] != 2U * heights[1] ||
            widths[1] != 2U * widths[2] ||
            heights[1] != 2U * heights[2]) {
            error = "DepthMap default-image composition requires exact 2x pyramid levels";
            return false;
        }

        struct Point3 {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
        };
        struct Sample {
            double u = 0.0;
            double v = 0.0;
            float depth = 0.0F;
        };
        constexpr float normal_threshold = 0.2F;
        constexpr float normal_threshold_squared =
            normal_threshold * normal_threshold;
        const double principal_x_over_focal =
            camera.model.cx / camera.model.f;
        const double principal_y_over_focal =
            camera.model.cy / camera.model.f;

        const auto upsample = [&](std::span<const float> source,
                                  std::size_t source_width,
                                  std::size_t source_height,
                                  std::uint32_t source_downscale,
                                  std::size_t target_width,
                                  std::size_t target_height,
                                  std::vector<float>& target) -> bool {
            if (source_width < 2U || source_height < 2U ||
                target_width != 2U * source_width ||
                target_height != 2U * source_height) {
                error = "DepthMap default-image upsample dimensions are outside the recovered domain";
                return false;
            }
            target.assign(target_width * target_height, 0.0F);
            const double source_scale =
                static_cast<double>(source_downscale) / camera.model.f;
            const auto make_point = [&](const Sample& sample) {
                const double depth = static_cast<double>(sample.depth);
                return Point3{
                    (sample.u * source_scale - principal_x_over_focal) * depth,
                    (sample.v * source_scale - principal_y_over_focal) * depth,
                    depth};
            };

            for (std::size_t y = 0; y < target_height; ++y) {
                const double source_y =
                    (static_cast<double>(y) + 0.5) *
                        static_cast<double>(source_height) /
                        static_cast<double>(target_height) -
                    0.5;
                const auto iy = static_cast<std::ptrdiff_t>(source_y);
                if (iy < 0 ||
                    iy >= static_cast<std::ptrdiff_t>(source_height - 1U))
                    continue;
                const double fy = source_y - static_cast<double>(iy);
                for (std::size_t x = 0; x < target_width; ++x) {
                    const double source_x =
                        (static_cast<double>(x) + 0.5) *
                            static_cast<double>(source_width) /
                            static_cast<double>(target_width) -
                        0.5;
                    const auto ix = static_cast<std::ptrdiff_t>(source_x);
                    if (ix < 0 ||
                        ix >= static_cast<std::ptrdiff_t>(source_width - 1U))
                        continue;
                    const double fx = source_x - static_cast<double>(ix);
                    const std::size_t top_left =
                        static_cast<std::size_t>(iy) * source_width +
                        static_cast<std::size_t>(ix);
                    const float d00 = source[top_left];
                    const float d10 = source[top_left + 1U];
                    const float d01 = source[top_left + source_width];
                    const float d11 = source[top_left + source_width + 1U];
                    const double diagonal_main = d00 != 0.0F && d11 != 0.0F
                        ? std::abs(static_cast<double>(d00) - d11)
                        : std::numeric_limits<double>::infinity();
                    const double diagonal_other = d10 != 0.0F && d01 != 0.0F
                        ? std::abs(static_cast<double>(d10) - d01)
                        : std::numeric_limits<double>::infinity();

                    std::array<Sample, 3> triangle{};
                    bool has_triangle = false;
                    if (diagonal_main < diagonal_other) {
                        if (fx <= fy && d01 != 0.0F) {
                            triangle = {{{static_cast<double>(ix) + 0.5,
                                          static_cast<double>(iy) + 0.5, d00},
                                         {static_cast<double>(ix) + 0.5,
                                          static_cast<double>(iy) + 1.5, d01},
                                         {static_cast<double>(ix) + 1.5,
                                          static_cast<double>(iy) + 1.5, d11}}};
                            has_triangle = true;
                        } else if (fx > fy && d10 != 0.0F) {
                            triangle = {{{static_cast<double>(ix) + 1.5,
                                          static_cast<double>(iy) + 0.5, d10},
                                         {static_cast<double>(ix) + 0.5,
                                          static_cast<double>(iy) + 0.5, d00},
                                         {static_cast<double>(ix) + 1.5,
                                          static_cast<double>(iy) + 1.5, d11}}};
                            has_triangle = true;
                        }
                    } else if (diagonal_other < diagonal_main) {
                        if (fx + fy <= 1.0 && d00 != 0.0F) {
                            triangle = {{{static_cast<double>(ix) + 0.5,
                                          static_cast<double>(iy) + 0.5, d00},
                                         {static_cast<double>(ix) + 0.5,
                                          static_cast<double>(iy) + 1.5, d01},
                                         {static_cast<double>(ix) + 1.5,
                                          static_cast<double>(iy) + 0.5, d10}}};
                            has_triangle = true;
                        } else if (fx + fy > 1.0 && d11 != 0.0F) {
                            triangle = {{{static_cast<double>(ix) + 1.5,
                                          static_cast<double>(iy) + 0.5, d10},
                                         {static_cast<double>(ix) + 0.5,
                                          static_cast<double>(iy) + 1.5, d01},
                                         {static_cast<double>(ix) + 1.5,
                                          static_cast<double>(iy) + 1.5, d11}}};
                            has_triangle = true;
                        }
                    }
                    if (!has_triangle ||
                        triangle[0].depth == 0.0F ||
                        triangle[1].depth == 0.0F ||
                        triangle[2].depth == 0.0F)
                        continue;

                    const Point3 p0 = make_point(triangle[0]);
                    const Point3 p1 = make_point(triangle[1]);
                    const Point3 p2 = make_point(triangle[2]);
                    const Point3 a{p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
                    const Point3 b{p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};
                    const Point3 normal{
                        a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x};
                    const Point3 center{
                        (p0.x + p1.x + p2.x) / 3.0,
                        (p0.y + p1.y + p2.y) / 3.0,
                        (p0.z + p1.z + p2.z) / 3.0};
                    const double facing = -(center.x * normal.x +
                                             center.y * normal.y +
                                             center.z * normal.z);
                    const double normal2 = normal.x * normal.x +
                                           normal.y * normal.y +
                                           normal.z * normal.z;
                    const double center2 = center.x * center.x +
                                           center.y * center.y +
                                           center.z * center.z;
                    if (facing < 0.0 ||
                        normal2 * (center2 * normal_threshold_squared) >
                            facing * facing)
                        continue;

                    double depth = 0.0;
                    if (diagonal_main < diagonal_other) {
                        depth = fx <= fy
                            ? static_cast<double>(d00) * (1.0 - fy) +
                                  static_cast<double>(d01) * (fy - fx) +
                                  static_cast<double>(d11) * fx
                            : static_cast<double>(d00) * (1.0 - fx) +
                                  static_cast<double>(d10) * (fx - fy) +
                                  static_cast<double>(d11) * fy;
                    } else if (fx + fy <= 1.0) {
                        depth = static_cast<double>(d00) * (1.0 - fx - fy) +
                                static_cast<double>(d10) * fx +
                                static_cast<double>(d01) * fy;
                    } else {
                        depth = static_cast<double>(d10) * (1.0 - fy) +
                                static_cast<double>(d01) * (1.0 - fx) +
                                static_cast<double>(d11) * (fx + fy - 1.0);
                    }
                    if (std::isfinite(depth) && depth > 0.0)
                        target[y * target_width + x] =
                            static_cast<float>(depth);
                }
            }
            return true;
        };

        std::vector<float> composed_middle(
            persisted_depth_levels[1].begin(),
            persisted_depth_levels[1].end());
        std::vector<float> upsampled_middle;
        if (!upsample(persisted_depth_levels[2], widths[2], heights[2],
                      downscales[2], widths[1], heights[1],
                      upsampled_middle))
            return false;
        for (std::size_t pixel = 0; pixel < composed_middle.size(); ++pixel) {
            if (composed_middle[pixel] == 0.0F)
                composed_middle[pixel] = upsampled_middle[pixel];
        }

        std::vector<float> result(persisted_depth_levels[0].begin(),
                                  persisted_depth_levels[0].end());
        std::vector<float> upsampled_fine;
        if (!upsample(composed_middle, widths[1], heights[1], downscales[1],
                      widths[0], heights[0], upsampled_fine))
            return false;
        for (std::size_t pixel = 0; pixel < result.size(); ++pixel) {
            if (result[pixel] == 0.0F)
                result[pixel] = upsampled_fine[pixel];
        }
        output = std::move(result);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool make_recovered_patchmatch_voting_level_product(
    std::span<const float> component_filtered_depth,
    std::span<const std::uint8_t> raw_packed_inlier_masks,
    std::size_t neighbor_count,
    std::vector<float>& persisted_depth,
    std::vector<std::uint8_t>& persisted_packed_inlier_masks,
    std::string& error) {
    try {
        const std::size_t pixels = component_filtered_depth.size();
        const std::size_t groups = (neighbor_count + 7U) / 8U;
        if (pixels == 0U || neighbor_count == 0U ||
            groups > std::numeric_limits<std::size_t>::max() / pixels ||
            raw_packed_inlier_masks.size() != groups * pixels) {
            error = "PatchMatch voting level-product dimensions are inconsistent";
            return false;
        }

        std::vector<float> depth(component_filtered_depth.begin(),
                                 component_filtered_depth.end());
        std::vector<std::uint8_t> masks(raw_packed_inlier_masks.begin(),
                                        raw_packed_inlier_masks.end());
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const float value = depth[pixel];
            if (!std::isfinite(value) || value < 0.0F) {
                error = "PatchMatch voting level-product depth is outside the recovered domain";
                return false;
            }
            if (value == 0.0F) {
                depth[pixel] = 0.0F;
                for (std::size_t group = 0; group < groups; ++group)
                    masks[group * pixels + pixel] = 0U;
                continue;
            }

            // OpenEXR PXR24 stores FLOAT samples as sign+exponent+15 mantissa
            // bits.  For positive finite target depths its decode result is
            // exactly round-to-nearest followed by clearing the low 8 bits.
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
            const std::uint32_t rounded = (bits + 0x80U) & 0xFFFFFF00U;
            depth[pixel] = std::bit_cast<float>(rounded);
            if (!std::isfinite(depth[pixel]) || depth[pixel] <= 0.0F) {
                error = "PatchMatch PXR24 depth round-trip left the recovered domain";
                return false;
            }
        }

        persisted_depth = std::move(depth);
        persisted_packed_inlier_masks = std::move(masks);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool run_recovered_patchmatch_d4_pyramid_cuda(
    const Scene& scene,
    std::size_t reference_camera_index,
    std::span<const std::size_t> ranked_neighbor_camera_indices,
    std::size_t device_index,
    RecoveredPatchMatchD4PyramidOutput& output,
    std::string& error,
    std::span<const RecoveredPatchMatchPreparedCamera> prepared_camera_cache,
    bool capture_diagnostic_checkpoints,
    RecoveredPatchMatchHostPreparation* prebuilt_host_preparation,
    double prebuilt_host_preparation_seconds) {
    try {
        using Clock = std::chrono::steady_clock;
        constexpr std::uint32_t target_downscale = 4U;
        const std::size_t neighbor_count =
            ranked_neighbor_camera_indices.size();
        if (reference_camera_index >= scene.cameras.size() ||
            neighbor_count < 6U || neighbor_count > 16U) {
            error = "recovered d4 PatchMatch pyramid requires one reference and 6..16 ranked neighbors";
            return false;
        }
        RecoveredCudaModuleSessionScope cuda_session;
        if (!cuda_session.open(device_index, error))
            return false;
        const Camera& camera = scene.cameras[reference_camera_index];
        if (camera.index != reference_camera_index ||
            camera.image.width == 0U || camera.image.height == 0U ||
            camera.image.width % 16U != 0U ||
            camera.image.height % 16U != 0U) {
            error = "recovered d4 PatchMatch pyramid requires index-stable dimensions divisible by 16";
            return false;
        }

        const auto preparation_started = Clock::now();
        RecoveredPatchMatchHostPreparation preparation;
        double host_preparation_seconds = 0.0;
        if (prebuilt_host_preparation == nullptr) {
            if (!make_recovered_patchmatch_unmasked_host_preparation(
                    scene, reference_camera_index,
                    ranked_neighbor_camera_indices, target_downscale,
                    device_index, preparation, error, prepared_camera_cache,
                    capture_diagnostic_checkpoints))
                return false;
            host_preparation_seconds = std::chrono::duration<double>(
                Clock::now() - preparation_started).count();
        } else {
            preparation = std::move(*prebuilt_host_preparation);
            host_preparation_seconds = prebuilt_host_preparation_seconds;
            if (preparation.reference_camera_index != reference_camera_index ||
                preparation.target_downscale != target_downscale ||
                preparation.device_index != device_index ||
                preparation.ranked_neighbor_camera_indices.size() !=
                    ranked_neighbor_camera_indices.size() ||
                !std::equal(
                    preparation.ranked_neighbor_camera_indices.begin(),
                    preparation.ranked_neighbor_camera_indices.end(),
                    ranked_neighbor_camera_indices.begin())) {
                error = "prebuilt PatchMatch host preparation identity is invalid";
                return false;
            }
        }
        const auto registration_started = Clock::now();
        RecoveredCudaPinnedNeighborTextures pinned_neighbor_textures;
        if (!pin_recovered_patchmatch_neighbor_textures_cuda(
                preparation.neighbor_resources, device_index,
                pinned_neighbor_textures, error))
            return false;
        const double host_registration_seconds =
            std::chrono::duration<double>(Clock::now() - registration_started)
                .count();

        const auto find_level = [&](std::uint32_t downscale)
            -> const RecoveredPatchMatchPreparedLevel* {
            const auto level = std::find_if(
                preparation.reference.image_levels.begin(),
                preparation.reference.image_levels.end(),
                [downscale](const RecoveredPatchMatchPreparedLevel& value) {
                    return value.downscale == downscale;
                });
            return level == preparation.reference.image_levels.end()
                ? nullptr : &*level;
        };
        const auto* level32 = find_level(32U);
        const auto* level16 = find_level(16U);
        const auto* level8 = find_level(8U);
        if (level32 == nullptr || level16 == nullptr || level8 == nullptr) {
            error = "recovered d4 PatchMatch preparation lacks d32/d16/d8 images";
            return false;
        }

        // Dynamic target capture proves one full-zero CUDA neighbour atlas is
        // constructed for this reference camera and retained across every
        // x32, x16, x8 and x4 prepare.  Never reset it at a level/binding
        // boundary.
        auto camera_atlas_state = make_recovered_patchmatch_cost_atlas_state(
            preparation.neighbor_resources);

        const auto x32_started = Clock::now();
        RecoveredPatchMatchCoarsestLevelOutput x32;
        if (!run_recovered_patchmatch_coarsest_level_cuda(
                preparation, level32->data.image, x32, error,
                &camera_atlas_state, capture_diagnostic_checkpoints))
            return false;
        const double x32_seconds =
            std::chrono::duration<double>(Clock::now() - x32_started).count();
        const auto x16_started = Clock::now();
        RecoveredPatchMatchX16LevelOutput x16;
        if (!run_recovered_patchmatch_x16_level_cuda(
                camera, preparation, x32, level16->data.image, x16, error,
                &camera_atlas_state, capture_diagnostic_checkpoints,
                capture_diagnostic_checkpoints
                    ? nullptr
                    : &x32.state_before_filter))
            return false;
        const double x16_seconds =
            std::chrono::duration<double>(Clock::now() - x16_started).count();
        // x16 has consumed the coarsest boundary and moved the persistent
        // production state forward. No later stage reads x32.
        x32 = {};
        const auto x8_started = Clock::now();
        RecoveredPatchMatchFinerLevelOutput x8;
        if (!run_recovered_patchmatch_finer_level_cuda(
                camera, preparation, 8U, x16.state_before_filter,
                x16.boundary, level8->data.image, x8, error,
                &camera_atlas_state, capture_diagnostic_checkpoints,
                capture_diagnostic_checkpoints
                    ? nullptr
                    : &x16.state_before_filter))
            return false;
        const double x8_seconds =
            std::chrono::duration<double>(Clock::now() - x8_started).count();

        const std::size_t x16_pixels =
            (camera.image.width / 16U) * (camera.image.height / 16U);
        const std::size_t inlier_groups = (neighbor_count + 7U) / 8U;
        const std::size_t x16_packed_bytes = inlier_groups * x16_pixels;
        if (x16.boundary.filter.depth_allocation.size() < x16_pixels ||
            x16.state_before_filter.neighbor_inlier_masks.size() <
                x16_packed_bytes) {
            error = "recovered d4 PatchMatch x16 level product is incomplete";
            return false;
        }
        std::vector<float> x16_level_product_depth(
            x16.boundary.filter.depth_allocation.begin(),
            x16.boundary.filter.depth_allocation.begin() + x16_pixels);
        std::vector<std::uint8_t> x16_level_product_inlier_masks(
            x16.state_before_filter.neighbor_inlier_masks.begin(),
            x16.state_before_filter.neighbor_inlier_masks.begin() +
                x16_packed_bytes);
        // x8 has consumed the x16 state/boundary. Preserve only the compact
        // persisted products that are needed after x4.
        x16 = {};
        std::vector<float> x8_level_product_depth =
            std::move(x8.level_product_depth);
        std::vector<std::uint8_t> x8_level_product_inlier_masks =
            std::move(x8.level_product_inlier_masks);
        std::vector<std::size_t> x8_ranked_neighbor_camera_indices =
            std::move(x8.ranked_neighbor_camera_indices);
        const auto x4_started = Clock::now();
        RecoveredPatchMatchFinerLevelOutput x4;
        if (!run_recovered_patchmatch_finer_level_cuda(
                camera, preparation, 4U, x8.state_before_filter,
                x8.boundary, {}, x4, error, &camera_atlas_state,
                capture_diagnostic_checkpoints,
                capture_diagnostic_checkpoints
                    ? nullptr
                    : &x8.state_before_filter))
            return false;
        const double x4_seconds =
            std::chrono::duration<double>(Clock::now() - x4_started).count();
        // x4 has consumed the inherited x8 state and bilateral boundary.
        x8 = {};
        const auto product_started = Clock::now();

        if (x8_ranked_neighbor_camera_indices.size() != neighbor_count ||
            x4.ranked_neighbor_camera_indices !=
                x8_ranked_neighbor_camera_indices) {
            error = "recovered d4 PatchMatch level products are incomplete";
            return false;
        }

        RecoveredPatchMatchD4PyramidOutput result;
        result.camera_index = reference_camera_index;
        result.neighbor_atlas_prepare_count =
            camera_atlas_state.prepare_count;
        result.host_preparation_seconds = host_preparation_seconds;
        result.host_registration_seconds = host_registration_seconds;
        result.x32_seconds = x32_seconds;
        result.x16_seconds = x16_seconds;
        result.x8_seconds = x8_seconds;
        result.x4_seconds = x4_seconds;
        if (camera.image.width == 3072U &&
            camera.image.height == 2304U &&
            neighbor_count == 16U &&
            result.neighbor_atlas_prepare_count != 603U) {
            error = "South d4 camera-wide atlas prepare count differs from the 603-launch target trace";
            return false;
        }
        result.depth_levels[0] = std::move(x4.level_product_depth);
        result.depth_levels[1] = std::move(x8_level_product_depth);
        result.depth_levels[2] = std::move(x16_level_product_depth);
        result.packed_inlier_masks[0] =
            std::move(x4.level_product_inlier_masks);
        result.packed_inlier_masks[1] =
            std::move(x8_level_product_inlier_masks);
        result.packed_inlier_masks[2] =
            std::move(x16_level_product_inlier_masks);
        result.ranked_neighbor_camera_indices =
            std::move(x4.ranked_neighbor_camera_indices);
        if (!std::equal(result.ranked_neighbor_camera_indices.begin(),
                        result.ranked_neighbor_camera_indices.end(),
                        ranked_neighbor_camera_indices.begin())) {
            error = "recovered d4 PatchMatch ranked-neighbor mapping changed during execution";
            return false;
        }

        for (std::size_t level = 0; level < result.depth_levels.size();
             ++level) {
            std::vector<float> persisted_depth;
            std::vector<std::uint8_t> persisted_masks;
            if (!make_recovered_patchmatch_voting_level_product(
                    result.depth_levels[level],
                    result.packed_inlier_masks[level],
                    neighbor_count, persisted_depth,
                    persisted_masks, error)) {
                error = "recovered d4 PatchMatch level " +
                        std::to_string(level) + " voting product: " + error;
                return false;
            }
            result.depth_levels[level] = std::move(persisted_depth);
            result.packed_inlier_masks[level] = std::move(persisted_masks);
        }
        result.product_seconds =
            std::chrono::duration<double>(Clock::now() - product_started)
                .count();

        RecoveredCudaModuleSessionStats ignored_session_stats;
        if (!cuda_session.close(ignored_session_stats, error))
            return false;
        output = std::move(result);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool run_recovered_patchmatch_d4_scene_cuda(
    const Scene& scene,
    std::span<const std::size_t> reference_camera_indices,
    std::span<const std::vector<std::size_t>> ranked_neighbors_by_camera,
    FilterMode filter_mode,
    std::size_t device_index,
    RecoveredPatchMatchD4SceneOutput& output,
    std::string& error,
    bool retain_voting_diagnostics,
    const std::filesystem::path& patchmatch_store_root,
    std::size_t voting_batch_size) {
    try {
        using Clock = std::chrono::steady_clock;
        if (reference_camera_indices.empty() ||
            ranked_neighbors_by_camera.size() != scene.cameras.size()) {
            error = "recovered d4 scene requires explicit references and one neighbor row per scene camera";
            return false;
        }
        RecoveredCudaModuleSessionScope cuda_session;
        if (!cuda_session.open(device_index, error))
            return false;

        std::vector<std::size_t> product_for_camera(
            scene.cameras.size(), std::numeric_limits<std::size_t>::max());
        for (std::size_t ordinal = 0;
             ordinal < reference_camera_indices.size(); ++ordinal) {
            const std::size_t camera_index = reference_camera_indices[ordinal];
            if (camera_index >= scene.cameras.size() ||
                !scene.cameras[camera_index].aligned ||
                product_for_camera[camera_index] !=
                    std::numeric_limits<std::size_t>::max()) {
                error = "recovered d4 scene reference set is invalid or duplicated";
                return false;
            }
            const auto& neighbors = ranked_neighbors_by_camera[camera_index];
            if (neighbors.size() < 6U || neighbors.size() > 16U) {
                error = "recovered d4 scene reference lies outside the observed 6..16-neighbor domain";
                return false;
            }
            for (const std::size_t neighbor : neighbors) {
                if (neighbor >= scene.cameras.size() ||
                    !scene.cameras[neighbor].aligned) {
                    error = "recovered d4 scene neighbor mapping is invalid";
                    return false;
                }
            }
            product_for_camera[camera_index] = ordinal;
        }
        for (const std::size_t camera_index : reference_camera_indices) {
            for (const std::size_t neighbor :
                 ranked_neighbors_by_camera[camera_index]) {
                if (product_for_camera[neighbor] ==
                    std::numeric_limits<std::size_t>::max()) {
                    error = "recovered d4 scene reference closure omits a voting neighbor";
                    return false;
                }
            }
        }

        // Image undistortion and the complete ceil-half pyramid depend only
        // on (scene camera, target downscale, device), not on which reference
        // later consumes that camera as a neighbor.  Build each camera once
        // for this closed scene request and reuse immutable copies while the
        // reference-specific crop/transform resources are still rebuilt.
        const auto camera_preparation_started = Clock::now();
        std::vector<RecoveredPatchMatchPreparedCamera> prepared_camera_cache(
            scene.cameras.size());
        const std::size_t preparation_worker_count = std::min<std::size_t>(
            2U, reference_camera_indices.size());
        std::atomic<std::size_t> next_preparation_ordinal{0U};
        std::atomic<bool> preparation_failed{false};
        std::mutex preparation_error_mutex;
        std::string preparation_error;
        const auto run_preparation_worker = [&]() {
            while (!preparation_failed.load(std::memory_order_relaxed)) {
                const std::size_t ordinal =
                    next_preparation_ordinal.fetch_add(
                        1U, std::memory_order_relaxed);
                if (ordinal >= reference_camera_indices.size()) return;
                const std::size_t camera_index =
                    reference_camera_indices[ordinal];
                std::string local_error;
                if (!make_recovered_patchmatch_unmasked_camera_preparation(
                        scene, camera_index, 4U, device_index,
                        prepared_camera_cache[camera_index], local_error, 2U)) {
                    local_error = "recovered d4 scene camera preparation " +
                                  std::to_string(camera_index) + ": " +
                                  local_error;
                    {
                        std::lock_guard lock(preparation_error_mutex);
                        if (preparation_error.empty())
                            preparation_error = std::move(local_error);
                    }
                    preparation_failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        };
        std::vector<std::thread> preparation_workers;
        preparation_workers.reserve(preparation_worker_count);
        for (std::size_t worker = 0; worker < preparation_worker_count; ++worker)
            preparation_workers.emplace_back(run_preparation_worker);
        for (std::thread& worker : preparation_workers) worker.join();
        if (preparation_failed.load(std::memory_order_relaxed)) {
            error = preparation_error.empty()
                ? "recovered d4 scene camera preparation worker failed"
                : preparation_error;
            return false;
        }

        RecoveredPatchMatchD4SceneOutput result;
        result.prepared_camera_cache_bytes =
            prepared_camera_cache_dynamic_bytes(prepared_camera_cache);
        result.patchmatch_store_root = patchmatch_store_root;
        result.patchmatch_store_used = !patchmatch_store_root.empty();
        result.voting_batch_size = patchmatch_store_root.empty()
            ? reference_camera_indices.size() : voting_batch_size;
        result.camera_preparation_seconds = std::chrono::duration<double>(
            Clock::now() - camera_preparation_started).count();
        result.cameras.resize(reference_camera_indices.size());
        result.patchmatch_worker_count = std::min<std::size_t>(
            2U, reference_camera_indices.size());
        std::vector<std::size_t> prepared_camera_consumers(
            scene.cameras.size(), 0U);
        for (const std::size_t camera_index : reference_camera_indices) {
            ++prepared_camera_consumers[camera_index];
            for (const std::size_t neighbor :
                 ranked_neighbors_by_camera[camera_index])
                ++prepared_camera_consumers[neighbor];
        }
        std::uint64_t prepared_camera_cache_live_bytes =
            result.prepared_camera_cache_bytes;
        const auto patchmatch_started = Clock::now();
        std::atomic<bool> worker_failed{false};
        std::mutex worker_error_mutex;
        std::string worker_error;
        struct PreparedJob {
            std::size_t ordinal = 0U;
            RecoveredPatchMatchHostPreparation preparation;
            double preparation_seconds = 0.0;
            std::uint64_t dynamic_bytes = 0U;
        };
        constexpr std::size_t prepared_queue_capacity = 4U;
        constexpr std::uint64_t prepared_queue_byte_budget =
            64U * 1024U * 1024U;
        result.prepared_queue_byte_budget = prepared_queue_byte_budget;
        std::deque<PreparedJob> prepared_jobs;
        std::mutex prepared_jobs_mutex;
        std::condition_variable prepared_jobs_changed;
        std::uint64_t resource_preparation_inflight_bytes = 0U;
        std::uint64_t prepared_queue_bytes = 0U;
        std::uint64_t active_prepared_jobs_bytes = 0U;
        std::size_t resource_preparation_inflight = 0U;
        std::size_t active_prepared_jobs = 0U;
        const auto update_prepared_host_peaks_locked = [&]() {
            result.resource_preparation_inflight_peak_bytes = std::max(
                result.resource_preparation_inflight_peak_bytes,
                resource_preparation_inflight_bytes);
            result.prepared_queue_peak_bytes = std::max(
                result.prepared_queue_peak_bytes, prepared_queue_bytes);
            result.active_prepared_jobs_peak_bytes = std::max(
                result.active_prepared_jobs_peak_bytes,
                active_prepared_jobs_bytes);
            result.prepared_queue_peak_jobs = std::max(
                result.prepared_queue_peak_jobs, prepared_jobs.size());
            result.resource_preparation_inflight_peak_count = std::max(
                result.resource_preparation_inflight_peak_count,
                resource_preparation_inflight);
            result.active_prepared_jobs_peak_count = std::max(
                result.active_prepared_jobs_peak_count,
                active_prepared_jobs);
            // The initial allocation total remains a useful manifest
            // identity; live ownership is decremented independently as the
            // last resource consumer releases each entry.
            result.prepared_host_live_peak_bytes = std::max(
                result.prepared_host_live_peak_bytes,
                result.prepared_camera_cache_bytes);
            result.prepared_host_live_peak_bytes = std::max(
                result.prepared_host_live_peak_bytes,
                saturating_add_bytes(
                    prepared_camera_cache_live_bytes,
                    saturating_add_bytes(
                        resource_preparation_inflight_bytes,
                        saturating_add_bytes(prepared_queue_bytes,
                                             active_prepared_jobs_bytes))));
        };
        update_prepared_host_peaks_locked();
        std::atomic<std::size_t> next_resource_ordinal{0U};
        const std::size_t resource_preparation_worker_count =
            std::min<std::size_t>(2U, reference_camera_indices.size());
        std::size_t resource_preparation_workers_finished = 0U;
        const auto fail_worker = [&](std::string message) {
            {
                std::lock_guard lock(worker_error_mutex);
                if (worker_error.empty()) worker_error = std::move(message);
            }
            worker_failed.store(true, std::memory_order_relaxed);
            prepared_jobs_changed.notify_all();
        };
        const auto run_resource_preparation_worker = [&]() {
            while (!worker_failed.load(std::memory_order_relaxed)) {
                const std::size_t ordinal =
                    next_resource_ordinal.fetch_add(
                        1U, std::memory_order_relaxed);
                if (ordinal >= reference_camera_indices.size()) break;
                const std::size_t camera_index =
                    reference_camera_indices[ordinal];
                PreparedJob job;
                job.ordinal = ordinal;
                std::string local_error;
                const auto started = Clock::now();
                if (!make_recovered_patchmatch_unmasked_host_preparation(
                        scene, camera_index,
                        ranked_neighbors_by_camera[camera_index], 4U,
                        device_index, job.preparation, local_error,
                        prepared_camera_cache, false)) {
                    fail_worker(
                        "recovered d4 scene resource preparation camera " +
                        std::to_string(camera_index) + ": " + local_error);
                    break;
                }
                job.preparation_seconds = std::chrono::duration<double>(
                    Clock::now() - started).count();
                job.dynamic_bytes =
                    host_preparation_dynamic_bytes(job.preparation);
                std::unique_lock lock(prepared_jobs_mutex);
                bool consumer_accounting_valid = true;
                const auto release_prepared_camera_consumer =
                    [&](std::size_t consumed_camera_index) {
                        if (!consumer_accounting_valid) return;
                        std::size_t& remaining =
                            prepared_camera_consumers[consumed_camera_index];
                        if (remaining == 0U) {
                            consumer_accounting_valid = false;
                            return;
                        }
                        --remaining;
                        if (remaining != 0U) return;
                        const std::uint64_t released_bytes =
                            prepared_camera_dynamic_bytes(
                                prepared_camera_cache[consumed_camera_index]);
                        if (released_bytes >
                            prepared_camera_cache_live_bytes) {
                            consumer_accounting_valid = false;
                            return;
                        }
                        prepared_camera_cache_live_bytes -= released_bytes;
                        prepared_camera_cache[consumed_camera_index] = {};
                    };
                release_prepared_camera_consumer(camera_index);
                for (const std::size_t neighbor :
                     ranked_neighbors_by_camera[camera_index])
                    release_prepared_camera_consumer(neighbor);
                if (!consumer_accounting_valid) {
                    lock.unlock();
                    fail_worker(
                        "recovered d4 scene prepared-camera consumer accounting underflow");
                    break;
                }
                resource_preparation_inflight_bytes = saturating_add_bytes(
                    resource_preparation_inflight_bytes, job.dynamic_bytes);
                ++resource_preparation_inflight;
                result.prepared_job_peak_bytes = std::max(
                    result.prepared_job_peak_bytes, job.dynamic_bytes);
                update_prepared_host_peaks_locked();
                prepared_jobs_changed.wait(lock, [&]() {
                    if (worker_failed.load(std::memory_order_relaxed))
                        return true;
                    if (prepared_jobs.size() >= prepared_queue_capacity)
                        return false;
                    return prepared_jobs.empty() ||
                           (job.dynamic_bytes <= prepared_queue_byte_budget &&
                            prepared_queue_bytes <=
                                prepared_queue_byte_budget - job.dynamic_bytes);
                });
                if (job.dynamic_bytes > resource_preparation_inflight_bytes ||
                    resource_preparation_inflight == 0U) {
                    lock.unlock();
                    fail_worker(
                        "recovered d4 scene resource-preparation byte accounting underflow");
                    break;
                }
                resource_preparation_inflight_bytes -= job.dynamic_bytes;
                --resource_preparation_inflight;
                if (worker_failed.load(std::memory_order_relaxed)) {
                    update_prepared_host_peaks_locked();
                    break;
                }
                prepared_queue_bytes = saturating_add_bytes(
                    prepared_queue_bytes, job.dynamic_bytes);
                prepared_jobs.push_back(std::move(job));
                update_prepared_host_peaks_locked();
                lock.unlock();
                prepared_jobs_changed.notify_all();
            }
            {
                std::lock_guard lock(prepared_jobs_mutex);
                ++resource_preparation_workers_finished;
            }
            prepared_jobs_changed.notify_all();
        };
        const auto run_worker = [&]() {
            while (!worker_failed.load(std::memory_order_relaxed)) {
                PreparedJob job;
                {
                    std::unique_lock lock(prepared_jobs_mutex);
                    prepared_jobs_changed.wait(lock, [&]() {
                        return worker_failed.load(std::memory_order_relaxed) ||
                               !prepared_jobs.empty() ||
                               resource_preparation_workers_finished ==
                                   resource_preparation_worker_count;
                    });
                    if (worker_failed.load(std::memory_order_relaxed)) return;
                    if (prepared_jobs.empty()) return;
                    job = std::move(prepared_jobs.front());
                    prepared_jobs.pop_front();
                    if (job.dynamic_bytes > prepared_queue_bytes) {
                        fail_worker(
                            "recovered d4 scene prepared queue byte accounting underflow");
                        return;
                    }
                    prepared_queue_bytes -= job.dynamic_bytes;
                    active_prepared_jobs_bytes = saturating_add_bytes(
                        active_prepared_jobs_bytes, job.dynamic_bytes);
                    ++active_prepared_jobs;
                    update_prepared_host_peaks_locked();
                }
                prepared_jobs_changed.notify_all();
                const std::size_t ordinal = job.ordinal;
                const std::size_t camera_index =
                    reference_camera_indices[ordinal];
                const auto started = Clock::now();
                std::string local_error;
                try {
                    if (!run_recovered_patchmatch_d4_pyramid_cuda(
                            scene, camera_index,
                            ranked_neighbors_by_camera[camera_index],
                            device_index,
                            result.cameras[ordinal].patchmatch, local_error,
                            prepared_camera_cache, false,
                            &job.preparation, job.preparation_seconds)) {
                        local_error = "recovered d4 scene PatchMatch camera " +
                                      std::to_string(camera_index) + ": " +
                                      local_error;
                    }
                    if (local_error.empty() &&
                        !patchmatch_store_root.empty()) {
                        if (!write_recovered_patchmatch_store_camera(
                                patchmatch_store_root,
                                result.cameras[ordinal].patchmatch,
                                local_error)) {
                            local_error =
                                "recovered d4 scene PatchMatch store camera " +
                                std::to_string(camera_index) + ": " +
                                local_error;
                        } else {
                            for (auto& level : result.cameras[ordinal]
                                                   .patchmatch.depth_levels)
                                std::vector<float>().swap(level);
                            for (auto& level : result.cameras[ordinal]
                                                   .patchmatch
                                                   .packed_inlier_masks)
                                std::vector<std::uint8_t>().swap(level);
                        }
                    }
                } catch (const std::exception& exception) {
                    local_error = "recovered d4 scene PatchMatch camera " +
                                  std::to_string(camera_index) + ": " +
                                  exception.what();
                }
                // The complete camera product and optional store publication
                // no longer read the reference/neighbor preparation. Drop its
                // owning vectors before accounting the active job as done.
                job.preparation = {};
                {
                    std::lock_guard lock(prepared_jobs_mutex);
                    if (job.dynamic_bytes > active_prepared_jobs_bytes ||
                        active_prepared_jobs == 0U) {
                        if (local_error.empty())
                            local_error = "recovered d4 scene active prepared byte accounting underflow";
                    } else {
                        active_prepared_jobs_bytes -= job.dynamic_bytes;
                        --active_prepared_jobs;
                    }
                }
                prepared_jobs_changed.notify_all();
                if (!local_error.empty()) {
                    fail_worker(std::move(local_error));
                    return;
                }
                result.cameras[ordinal].patchmatch_seconds =
                    std::chrono::duration<double>(
                        Clock::now() - started).count();
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(result.patchmatch_worker_count);
        for (std::size_t worker = 0;
             worker < result.patchmatch_worker_count; ++worker)
            workers.emplace_back(run_worker);
        std::vector<std::thread> resource_preparation_workers;
        resource_preparation_workers.reserve(
            resource_preparation_worker_count);
        for (std::size_t worker = 0;
             worker < resource_preparation_worker_count; ++worker)
            resource_preparation_workers.emplace_back(
                run_resource_preparation_worker);
        for (std::thread& worker : resource_preparation_workers)
            worker.join();
        prepared_jobs_changed.notify_all();
        for (std::thread& worker : workers) worker.join();
        if (worker_failed.load(std::memory_order_relaxed)) {
            error = worker_error.empty()
                ? "recovered d4 scene PatchMatch worker failed"
                : worker_error;
            return false;
        }
        result.prepared_camera_cache_final_bytes =
            prepared_camera_cache_live_bytes;
        result.patchmatch_seconds = std::chrono::duration<double>(
            Clock::now() - patchmatch_started).count();

        // All reference-specific PatchMatch products have crossed their host
        // level-product boundary. Voting reads only the persisted depth
        // pyramids and packed masks in result.cameras, never these source
        // image pyramids. Release the immutable preparation cache before the
        // voting cache and voting outputs are allocated.
        std::vector<RecoveredPatchMatchPreparedCamera>().swap(
            prepared_camera_cache);

        const auto voting_wall_started = Clock::now();
        std::vector<RecoveredPatchMatchStoreBatch> voting_batches;
        if (patchmatch_store_root.empty()) {
            RecoveredPatchMatchStoreBatch batch;
            batch.references.assign(reference_camera_indices.begin(),
                                    reference_camera_indices.end());
            batch.closure = batch.references;
            voting_batches.push_back(std::move(batch));
        } else if (!plan_recovered_patchmatch_store_batches(
                       reference_camera_indices, ranked_neighbors_by_camera,
                       voting_batch_size, voting_batches, error)) {
            error = "recovered d4 scene voting batch plan: " + error;
            return false;
        }

        for (const auto& voting_batch : voting_batches) {
            std::vector<RecoveredPatchMatchD4PyramidOutput> loaded_products(
                scene.cameras.size());
            std::vector<const RecoveredPatchMatchD4PyramidOutput*>
                voting_products(scene.cameras.size(), nullptr);
            if (patchmatch_store_root.empty()) {
                for (const std::size_t camera_index : voting_batch.closure)
                    voting_products[camera_index] =
                        &result.cameras[product_for_camera[camera_index]].patchmatch;
            } else {
                for (const std::size_t camera_index : voting_batch.closure) {
                    if (!read_recovered_patchmatch_store_camera(
                            patchmatch_store_root, camera_index,
                            loaded_products[camera_index], error)) {
                        error = "recovered d4 scene voting store load camera " +
                                std::to_string(camera_index) + ": " + error;
                        return false;
                    }
                    voting_products[camera_index] =
                        &loaded_products[camera_index];
                }
            }
            std::vector<std::span<const float>> voting_depth_levels;
            voting_depth_levels.reserve(voting_batch.closure.size() * 3U);
            for (const std::size_t camera_index : voting_batch.closure) {
                for (const auto& level :
                     voting_products[camera_index]->depth_levels)
                    voting_depth_levels.emplace_back(level);
            }
            if (!prime_recovered_cuda_voting_depth_cache(
                    voting_depth_levels, device_index, error)) {
                error = "recovered d4 scene voting depth cache: " + error;
                return false;
            }

            std::atomic<std::size_t> next_voting_position{0U};
            worker_failed.store(false, std::memory_order_relaxed);
            worker_error.clear();
            const auto run_voting_worker = [&]() {
            while (!worker_failed.load(std::memory_order_relaxed)) {
                const std::size_t position = next_voting_position.fetch_add(
                    1U, std::memory_order_relaxed);
                if (position >= voting_batch.references.size()) return;
                const std::size_t camera_index =
                    voting_batch.references[position];
                const std::size_t ordinal = product_for_camera[camera_index];
                auto& camera_output = result.cameras[ordinal];
                const auto& patchmatch = *voting_products[camera_index];
                std::string local_error;
                try {
                    DepthVotingPreparedReference reference;
                    reference.camera_index = patchmatch.camera_index;
                    for (std::size_t level = 0; level < 3U; ++level)
                        reference.depth_level_views[level] =
                            patchmatch.depth_levels[level];

                    std::vector<DepthVotingPreparedNeighbor> voting_neighbors;
                    voting_neighbors.reserve(
                        patchmatch.ranked_neighbor_camera_indices.size());
                    for (std::size_t rank = 0;
                         rank < patchmatch.ranked_neighbor_camera_indices.size();
                         ++rank) {
                        const std::size_t neighbor_camera =
                            patchmatch.ranked_neighbor_camera_indices[rank];
                        if (neighbor_camera >= voting_products.size() ||
                            voting_products[neighbor_camera] == nullptr) {
                            local_error = "recovered d4 scene voting neighbor product is missing";
                            break;
                        }
                        DepthVotingPreparedNeighbor neighbor;
                        neighbor.camera_index = neighbor_camera;
                        for (std::size_t level = 0; level < 3U; ++level)
                            neighbor.depth_level_views[level] =
                                voting_products[neighbor_camera]
                                    ->depth_levels[level];
                        for (std::size_t level = 0;
                             local_error.empty() && level < 3U; ++level) {
                            const std::size_t pixels =
                                patchmatch.depth_levels[level].size();
                            if (!unpack_recovered_patchmatch_inlier_mask(
                                    patchmatch.packed_inlier_masks[level], pixels,
                                    patchmatch.ranked_neighbor_camera_indices.size(),
                                    rank, neighbor.inlier_masks[level],
                                    local_error)) {
                                local_error =
                                    "recovered d4 scene voting mask camera " +
                                    std::to_string(patchmatch.camera_index) +
                                    " rank " + std::to_string(rank) + ": " +
                                    local_error;
                                break;
                            }
                            neighbor.inlier_mask_views[level] =
                                neighbor.inlier_masks[level];
                        }
                        if (!local_error.empty()) break;
                        voting_neighbors.push_back(std::move(neighbor));
                    }

                    DepthVotingChainInput voting_input;
                    if (local_error.empty()) {
                        voting_input = make_recovered_depth_voting_chain_input(
                            scene, reference, voting_neighbors, 4U, filter_mode,
                            device_index);
                        const auto started = Clock::now();
                        if (!run_recovered_depth_voting_chain_cuda(
                                voting_input, camera_output.voting,
                                local_error)) {
                            local_error =
                                "recovered d4 scene voting camera " +
                                std::to_string(patchmatch.camera_index) + ": " +
                                local_error;
                        } else {
                            std::array<std::span<const float>, 3>
                                persisted_levels{};
                            for (std::size_t level = 0; level < 3U; ++level)
                                persisted_levels[level] =
                                    camera_output.voting
                                        .depth_after_components[level];
                            if (!compose_recovered_depthmap_default_image_d4(
                                    scene.cameras[patchmatch.camera_index],
                                    persisted_levels,
                                    camera_output.public_depth,
                                    local_error)) {
                                local_error =
                                    "recovered d4 scene public depth camera " +
                                    std::to_string(patchmatch.camera_index) +
                                    ": " + local_error;
                            }
                        }
                        if (local_error.empty() &&
                            !retain_voting_diagnostics) {
                            // Production OOC consumes only the final
                            // component-filtered depths. Preserve counters and
                            // final levels, but release replay-only snapshots
                            // as soon as this camera finishes voting.
                            for (auto& values :
                                 camera_output.voting.depth_before_components)
                                std::vector<float>().swap(values);
                            for (auto& values : camera_output.voting.votes)
                                std::vector<std::int32_t>().swap(values);
                        }
                        if (local_error.empty()) {
                            camera_output.voting_seconds =
                                std::chrono::duration<double>(
                                    Clock::now() - started).count();
                        }
                    }
                } catch (const std::exception& exception) {
                    local_error = "recovered d4 scene voting input camera " +
                                  std::to_string(patchmatch.camera_index) +
                                  ": " + exception.what();
                }
                if (!local_error.empty()) {
                    {
                        std::lock_guard lock(worker_error_mutex);
                        if (worker_error.empty()) worker_error = local_error;
                    }
                    worker_failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        };
        workers.clear();
        for (std::size_t worker = 0;
             worker < result.patchmatch_worker_count; ++worker)
            workers.emplace_back(run_voting_worker);
        for (std::thread& worker : workers) worker.join();
        if (worker_failed.load(std::memory_order_relaxed)) {
            error = worker_error.empty()
                ? "recovered d4 scene voting worker failed"
                : worker_error;
            return false;
        }
            if (!patchmatch_store_root.empty() &&
                !clear_recovered_cuda_voting_depth_cache(
                    device_index, error)) {
                error = "recovered d4 scene voting cache epoch clear: " + error;
                return false;
            }
        }
        for (const auto& camera_output : result.cameras)
            result.voting_seconds += camera_output.voting_seconds;
        result.voting_wall_seconds = std::chrono::duration<double>(
            Clock::now() - voting_wall_started).count();

        if (!cuda_session.close(result.cuda_module_session, error))
            return false;
        output = std::move(result);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

}  // namespace metmodel
