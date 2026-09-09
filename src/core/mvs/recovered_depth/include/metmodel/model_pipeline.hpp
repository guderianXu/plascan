#pragma once

#include "metmodel/octree_prepare.hpp"
#include "metmodel/patchmatch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace metmodel
{

    // The target type-6 OOC constructor leaves its diagonal-scale selector
    // uninitialized.  Keep same-run forensic replay separate from autonomous,
    // deterministic production instead of hiding either behavior behind a bool.
    enum class RecoveredOocDiagonalPolicy : std::uint8_t
    {
        StrictCaptured = 0,
        DeterministicDisabled = 1,
        DeterministicEnabled = 2,
    };

    // Numerical work and ordinal serialization are identical in both modes. The
    // serial mode exists for same-build causal performance measurements; parallel
    // remains the production default.
    enum class RecoveredOocCameraPyramidExecution : std::uint8_t
    {
        Serial = 0,
        Parallel = 1,
    };

    struct RecoveredD4VotingToOocMode0Input
    {
        std::span<const std::size_t> reference_camera_indices;
        RecoveredOocDiagonalPolicy diagonal_policy = RecoveredOocDiagonalPolicy::DeterministicDisabled;
        // Required, one byte (0/1) per reference, only for StrictCaptured.
        // Autonomous policies reject a non-empty span so captured state cannot be
        // accepted and silently ignored.
        std::span<const std::uint8_t> captured_diagonal_pixel_scale;
        std::uint32_t depth_downscale = 4U;
        std::size_t workitem_size_cameras = 20U;
        std::size_t max_workgroup_size = 100U;
        bool volumetric_masks = false;
        RecoveredOocCameraPyramidExecution camera_pyramid_execution = RecoveredOocCameraPyramidExecution::Parallel;
    };

    // Boundary manifest for the first continuous recovered production bridge.
    // Hashes are FNV-1a over explicitly little-endian scalar bytes so repeated
    // autonomous runs can be compared without serializing legacy DepthMap data.
    struct RecoveredD4VotingToOocMode0Manifest
    {
        std::uint32_t abi_version = 1U;
        std::uint32_t depth_downscale = 4U;
        RecoveredPatchMatchNoPriorPolicy no_prior_policy = RecoveredPatchMatchNoPriorPolicy::DeterministicZero;
        RecoveredOocDiagonalPolicy diagonal_policy = RecoveredOocDiagonalPolicy::DeterministicDisabled;
        bool volumetric_masks = false;
        std::size_t workitem_size_cameras = 20U;
        std::size_t max_workgroup_size = 100U;
        std::vector<std::size_t> reference_camera_indices;
        std::vector<std::uint32_t> stable_camera_ids;
        std::vector<std::uint8_t> diagonal_pixel_scale_used;
        std::vector<std::array<std::uint64_t, 3>> voted_depth_hashes;
        std::uint64_t registry_hash = 0U;
        std::vector<std::uint64_t> payload_shard_hashes;
    };

    struct RecoveredD4VotingToOocMode0Output
    {
        RecoveredD4VotingToOocMode0Manifest manifest;
        OocSceneVotedDepthBundleMode0Output ooc;
        // Wall spans are populated only by the consuming production bridge. The
        // preserving overload remains the serial oracle and leaves both at zero.
        double camera_pyramid_wall_seconds{};
        double bundle_serialization_wall_seconds{};
        RecoveredOocCameraPyramidExecution camera_pyramid_execution = RecoveredOocCameraPyramidExecution::Parallel;
    };

    // Continuous, capture-free execution boundary:
    // RecoveredPatchMatchD4SceneOutput::voting.depth_after_components ->
    // Scene ROI -> size-derived mode-0 pyramid -> PXR24 shards + registry.
    // Target captures are never accepted as depth/pyramid/payload inputs.
    [[nodiscard]] RecoveredD4VotingToOocMode0Output
    build_recovered_d4_voting_to_ooc_bundle_mode0(const Scene& scene,
                                                  const RecoveredPatchMatchD4SceneOutput& recovered_depth,
                                                  const RecoveredD4VotingToOocMode0Input& input);

    // Production memory variant. The caller must persist any PatchMatch/voting
    // diagnostics first. Once a camera's OOC pyramid owns all required values,
    // that camera's recovered depth/mask/voting payload vectors are released.
    [[nodiscard]] RecoveredD4VotingToOocMode0Output
    build_recovered_d4_voting_to_ooc_bundle_mode0_consuming(const Scene& scene,
                                                            RecoveredPatchMatchD4SceneOutput& recovered_depth,
                                                            const RecoveredD4VotingToOocMode0Input& input);

    // Per-camera accounting for the next recovered production boundary.  The
    // target consumes only the first saved depth/sample-scale pyramid level in
    // the observed depth-map mode-0 BuildModel path, converts every compact sample
    // to a denominator-one Morton record, and then performs its worker-local
    // reduction before publishing the records.
    struct RecoveredOocWeightedNodeCameraStats
    {
        std::size_t camera_index{};
        std::uint32_t stable_camera_id{};
        std::uint32_t maximum_level{};
        std::uint64_t candidate_count{};
        std::uint64_t local_node_count{};
    };

    // Capture-free scene boundary from the recovered voting/OOC pyramids through
    // the exact two-stage Morton reduction and 26-neighbour balancing.  Histogram
    // voting and part/halo scheduling are deliberately not folded into this type:
    // they are later boundaries and must not be approximated with fabricated
    // histogram bytes or partition metadata.
    struct RecoveredOocWeightedNodeOutput
    {
        float root_scale{};
        float alternate_scale{};
        std::uint32_t adaptive_maximum_level{};
        std::uint32_t adaptive_winning_bin{};
        double adaptive_scale_factor{};
        std::uint64_t adaptive_total_samples{};
        std::vector<RecoveredOocWeightedNodeCameraStats> cameras;
        std::vector<OocWeightedNodeRecord> merged_nodes;
        std::vector<OocWeightedNodeRecord> multi_camera_nodes;
        std::vector<OocWeightedNodeRecord> balanced_nodes;
        std::vector<OocOctreeRecord> records_before_histogram;
        // Wall-clock diagnostics only; these never participate in recovered state.
        double camera_nodes_seconds{};
        double publish_seconds{};
        double merge_seconds{};
        double filter_seconds{};
        double balance_seconds{};
        double initialize_records_seconds{};
        // Diagnostic process high-water marks sampled after each ownership stage.
        // Linux reports KiB through getrusage; zero means unavailable.
        std::uint64_t hwm_after_camera_nodes_kib{};
        std::uint64_t hwm_after_publish_kib{};
        std::uint64_t hwm_after_merge_kib{};
        std::uint64_t hwm_after_filter_kib{};
        std::uint64_t hwm_after_balance_kib{};
        std::uint64_t hwm_after_initialize_kib{};
    };

    [[nodiscard]] RecoveredOocWeightedNodeOutput
    build_recovered_ooc_weighted_nodes_mode0(const Scene& scene, const RecoveredD4VotingToOocMode0Output& input);

    // Capture-free boundary from balanced Morton nodes through all-camera CUDA
    // histogram voting and the exact one-part persistent-tree preparation.
    // `selected_indices` and `active` are already in the solver's target breadth-
    // first order; scalar_lut is the canonical binary16-to-float table.
    struct RecoveredOocHistogramOutput
    {
        std::uint64_t kernel_launches{};
        std::vector<OocHistogramVoxel> histogram_voxels;
        std::vector<OocOctreeRecord> records;
        std::vector<std::uint32_t> selected_indices;
        std::vector<std::uint8_t> active;
        std::vector<float> scalar_lut;
    };

    [[nodiscard]] RecoveredOocHistogramOutput
    build_recovered_ooc_histogram_mode0_cuda(const Scene& scene,
                                             const RecoveredD4VotingToOocMode0Output& pyramid_input,
                                             const RecoveredOocWeightedNodeOutput& weighted_input,
                                             std::size_t device_index);

    // Production memory variant. Each camera pyramid is released immediately
    // after its depth/sample-scale levels have been packed into the CUDA chain.
    // Numerical construction and launch order are identical to the preserving
    // overload; callers must stage any required pyramid diagnostics first.
    [[nodiscard]] RecoveredOocHistogramOutput
    build_recovered_ooc_histogram_mode0_cuda_consuming(const Scene& scene,
                                                       RecoveredD4VotingToOocMode0Output& pyramid_input,
                                                       const RecoveredOocWeightedNodeOutput& weighted_input,
                                                       std::size_t device_index);

} // namespace metmodel
