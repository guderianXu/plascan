#include "metmodel/model_pipeline.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace metmodel
{
    namespace
    {

        std::uint64_t process_high_water_kib() noexcept
        {
#if defined(__linux__)
            rusage usage{};
            if (::getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0)
                return static_cast<std::uint64_t>(usage.ru_maxrss);
#endif
            return 0U;
        }

        constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
        constexpr std::uint64_t fnv_prime = 1099511628211ULL;

        void fnv_byte(std::uint64_t& hash, std::uint8_t value) noexcept
        {
            hash ^= value;
            hash *= fnv_prime;
        }

        std::uint64_t hash_bytes(std::span<const std::byte> values) noexcept
        {
            std::uint64_t hash = fnv_offset;
            for (const std::byte value : values)
                fnv_byte(hash, std::to_integer<std::uint8_t>(value));
            return hash;
        }

        std::uint64_t hash_floats(std::span<const float> values) noexcept
        {
            std::uint64_t hash = fnv_offset;
            for (const float value : values)
            {
                const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
                for (std::uint32_t shift = 0U; shift != 32U; shift += 8U)
                    fnv_byte(hash, static_cast<std::uint8_t>(bits >> shift));
            }
            return hash;
        }

        std::uint8_t diagonal_for_camera(const RecoveredD4VotingToOocMode0Input& input, std::size_t ordinal)
        {
            switch (input.diagonal_policy)
            {
            case RecoveredOocDiagonalPolicy::StrictCaptured:
            {
                const std::uint8_t value = input.captured_diagonal_pixel_scale[ordinal];
                if (value > 1U)
                {
                    throw std::invalid_argument("captured OOC diagonal selector must be zero or one");
                }
                return value;
            }
            case RecoveredOocDiagonalPolicy::DeterministicDisabled:
                return 0U;
            case RecoveredOocDiagonalPolicy::DeterministicEnabled:
                return 1U;
            }
            throw std::invalid_argument("unknown recovered OOC diagonal policy");
        }

    } // namespace

    RecoveredD4VotingToOocMode0Output
    build_recovered_d4_voting_to_ooc_bundle_mode0_impl(const Scene& scene,
                                                       const RecoveredPatchMatchD4SceneOutput& recovered_depth,
                                                       const RecoveredD4VotingToOocMode0Input& input,
                                                       RecoveredPatchMatchD4SceneOutput* consumable_recovered_depth)
    {
        if (input.reference_camera_indices.empty() ||
            input.reference_camera_indices.size() != recovered_depth.cameras.size())
        {
            throw std::invalid_argument("recovered voting-to-OOC bridge requires one output per reference");
        }
        if (input.depth_downscale != 4U)
        {
            throw std::invalid_argument("recovered voting-to-OOC bridge is validated only for d4");
        }
        if (input.volumetric_masks)
        {
            throw std::invalid_argument("recovered OOC mode-0 bridge does not support volumetric masks");
        }
        if (input.camera_pyramid_execution != RecoveredOocCameraPyramidExecution::Serial &&
            input.camera_pyramid_execution != RecoveredOocCameraPyramidExecution::Parallel)
        {
            throw std::invalid_argument("unknown recovered OOC camera-pyramid execution mode");
        }
        const bool strict = input.diagonal_policy == RecoveredOocDiagonalPolicy::StrictCaptured;
        if ((strict && input.captured_diagonal_pixel_scale.size() != input.reference_camera_indices.size()) ||
            (!strict && !input.captured_diagonal_pixel_scale.empty()))
        {
            throw std::invalid_argument("OOC diagonal capture values do not match the selected policy");
        }

        RecoveredD4VotingToOocMode0Output output;
        output.camera_pyramid_execution = consumable_recovered_depth == nullptr
                                              ? RecoveredOocCameraPyramidExecution::Serial
                                              : input.camera_pyramid_execution;
        auto& manifest = output.manifest;
        manifest.depth_downscale = input.depth_downscale;
        manifest.diagonal_policy = input.diagonal_policy;
        manifest.volumetric_masks = input.volumetric_masks;
        manifest.workitem_size_cameras = input.workitem_size_cameras;
        manifest.max_workgroup_size = input.max_workgroup_size;
        manifest.reference_camera_indices.assign(input.reference_camera_indices.begin(),
                                                 input.reference_camera_indices.end());
        manifest.stable_camera_ids.reserve(recovered_depth.cameras.size());
        manifest.diagonal_pixel_scale_used.reserve(recovered_depth.cameras.size());
        manifest.voted_depth_hashes.reserve(recovered_depth.cameras.size());

        std::vector<OocSceneVotedDepthMode0View> views;
        views.reserve(recovered_depth.cameras.size());
        std::unordered_set<std::size_t> scene_indices;
        std::unordered_set<std::uint32_t> stable_ids;
        scene_indices.reserve(recovered_depth.cameras.size());
        stable_ids.reserve(recovered_depth.cameras.size());
        for (std::size_t ordinal = 0U; ordinal != recovered_depth.cameras.size(); ++ordinal)
        {
            const std::size_t camera_index = input.reference_camera_indices[ordinal];
            if (camera_index >= scene.cameras.size() || !scene_indices.insert(camera_index).second)
            {
                throw std::invalid_argument("recovered voting-to-OOC camera index is invalid or duplicated");
            }
            const Camera& camera = scene.cameras[camera_index];
            if (!camera.aligned || camera.index > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
            {
                throw std::invalid_argument("recovered voting-to-OOC camera is unaligned or has an invalid stable ID");
            }
            const std::uint32_t stable_id = static_cast<std::uint32_t>(camera.index);
            if (!stable_ids.insert(stable_id).second)
            {
                throw std::invalid_argument("recovered voting-to-OOC stable camera ID is duplicated");
            }

            const auto& camera_output = recovered_depth.cameras[ordinal];
            if (camera_output.patchmatch.camera_index != camera_index)
            {
                throw std::invalid_argument("recovered voting output order does not match the reference order");
            }
            std::array<std::span<const float>, 3> voted_depths;
            std::array<std::uint64_t, 3> hashes{};
            for (std::size_t level = 0U; level != voted_depths.size(); ++level)
            {
                const std::size_t level_downscale = static_cast<std::size_t>(input.depth_downscale) << level;
                if (camera.image.width == 0U || camera.image.height == 0U ||
                    camera.image.width % level_downscale != 0U || camera.image.height % level_downscale != 0U)
                {
                    throw std::invalid_argument("recovered voting-to-OOC camera dimensions are outside the d4 domain");
                }
                const std::size_t width = camera.image.width / level_downscale;
                const std::size_t height = camera.image.height / level_downscale;
                if (width > std::numeric_limits<std::size_t>::max() / height)
                {
                    throw std::overflow_error("recovered voting-to-OOC level pixel count overflows");
                }
                const auto& depth = camera_output.voting.depth_after_components[level];
                if (depth.size() != width * height)
                {
                    throw std::invalid_argument("recovered voting-to-OOC depth level has the wrong size");
                }
                voted_depths[level] = depth;
                hashes[level] = hash_floats(depth);
            }
            const std::uint8_t diagonal = diagonal_for_camera(input, ordinal);
            views.push_back({camera_index, voted_depths, diagonal != 0U});
            manifest.stable_camera_ids.push_back(stable_id);
            manifest.diagonal_pixel_scale_used.push_back(diagonal);
            manifest.voted_depth_hashes.push_back(hashes);
        }

        if (consumable_recovered_depth == nullptr)
        {
            output.ooc = build_ooc_pyramid_bundle_from_scene_voted_depths_mode0(
                scene, views, input.depth_downscale, input.workitem_size_cameras, input.max_workgroup_size);
        }
        else
        {
            auto& ooc = output.ooc;
            ooc.camera_groups = partition_ooc_pyramid_camera_groups(
                views.size(), input.workitem_size_cameras, input.max_workgroup_size);
            // Each ordinal reads immutable scene/view data and writes its own
            // pre-sized slots. Grouping and serialization below retain the
            // original ordinal order.
            ooc.items.resize(views.size());
            ooc.pyramids.resize(views.size());
            std::exception_ptr first_camera_error;
            std::mutex camera_error_mutex;
            const auto camera_pyramid_started = std::chrono::steady_clock::now();
            const auto build_camera_pyramid = [&](std::size_t ordinal)
            {
                try
                {
                    const auto& view = views[ordinal];
                    OocDepthRoiProjectMode0Input project =
                        make_ooc_depth_roi_project_mode0_input(scene, view.camera_index, input.depth_downscale);
                    ooc.items[ordinal] = {output.manifest.stable_camera_ids[ordinal], std::move(project)};
                    ooc.pyramids[ordinal] = build_ooc_sample_scale_pyramid_from_scene_voted_depths_mode0(
                        scene,
                        view.camera_index,
                        view.voted_depths,
                        input.depth_downscale,
                        view.captured_diagonal_pixel_scale);
                    auto& consumed = consumable_recovered_depth->cameras[ordinal];
                    for (auto& level : consumed.patchmatch.depth_levels)
                        std::vector<float>().swap(level);
                    for (auto& level : consumed.patchmatch.packed_inlier_masks)
                        std::vector<std::uint8_t>().swap(level);
                    std::vector<std::size_t>().swap(consumed.patchmatch.ranked_neighbor_camera_indices);
                    consumed.voting = {};
                    std::vector<float>().swap(consumed.public_depth);
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(camera_error_mutex);
                    if (!first_camera_error)
                    {
                        first_camera_error = std::current_exception();
                    }
                }
            };
            if (input.camera_pyramid_execution == RecoveredOocCameraPyramidExecution::Serial)
            {
                for (std::size_t ordinal = 0U; ordinal != views.size(); ++ordinal)
                {
                    build_camera_pyramid(ordinal);
                }
            }
            else if (input.camera_pyramid_execution == RecoveredOocCameraPyramidExecution::Parallel)
            {
#pragma omp parallel for schedule(static)
                for (std::ptrdiff_t signed_ordinal = 0; signed_ordinal < static_cast<std::ptrdiff_t>(views.size());
                     ++signed_ordinal)
                {
                    build_camera_pyramid(static_cast<std::size_t>(signed_ordinal));
                }
            }
            else
            {
                throw std::invalid_argument("unknown recovered OOC camera-pyramid execution mode");
            }
            output.camera_pyramid_wall_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - camera_pyramid_started).count();
            if (first_camera_error)
            {
                std::rethrow_exception(first_camera_error);
            }
            std::vector<OocPyramidBundleMode0Group> groups;
            const auto bundle_serialization_started = std::chrono::steady_clock::now();
            groups.reserve(ooc.camera_groups.size());
            for (const auto& range : ooc.camera_groups)
            {
                groups.push_back(
                    {std::span<const OocPyramidBundleMode0Item>(ooc.items).subspan(range.begin_index, range.item_count),
                     std::span<const OocSampleScalePyramidOutput>(ooc.pyramids)
                         .subspan(range.begin_index, range.item_count)});
            }
            ooc.bundle = serialize_ooc_pyramid_bundle_mode0(groups);
            output.bundle_serialization_wall_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - bundle_serialization_started).count();
        }
        manifest.registry_hash = hash_bytes(output.ooc.bundle.registry);
        manifest.payload_shard_hashes.reserve(output.ooc.bundle.payload_shards.size());
        for (const auto& shard : output.ooc.bundle.payload_shards)
            manifest.payload_shard_hashes.push_back(hash_bytes(shard));
        return output;
    }

    RecoveredD4VotingToOocMode0Output
    build_recovered_d4_voting_to_ooc_bundle_mode0(const Scene& scene,
                                                  const RecoveredPatchMatchD4SceneOutput& recovered_depth,
                                                  const RecoveredD4VotingToOocMode0Input& input)
    {
        return build_recovered_d4_voting_to_ooc_bundle_mode0_impl(scene, recovered_depth, input, nullptr);
    }

    RecoveredD4VotingToOocMode0Output
    build_recovered_d4_voting_to_ooc_bundle_mode0_consuming(const Scene& scene,
                                                            RecoveredPatchMatchD4SceneOutput& recovered_depth,
                                                            const RecoveredD4VotingToOocMode0Input& input)
    {
        return build_recovered_d4_voting_to_ooc_bundle_mode0_impl(scene, recovered_depth, input, &recovered_depth);
    }

    RecoveredOocWeightedNodeOutput
    build_recovered_ooc_weighted_nodes_mode0(const Scene& scene, const RecoveredD4VotingToOocMode0Output& input)
    {
        if (!scene.region.specified || !(scene.region.size.x > 0.0) || !(scene.region.size.y > 0.0) ||
            !(scene.region.size.z > 0.0))
        {
            throw std::invalid_argument("recovered OOC weighted-node builder requires a positive region");
        }
        if (input.manifest.depth_downscale != 4U)
        {
            throw std::invalid_argument("recovered OOC weighted-node builder is validated only for d4");
        }
        const std::size_t count = input.manifest.reference_camera_indices.size();
        if (count == 0U || input.ooc.items.size() != count || input.ooc.pyramids.size() != count ||
            input.manifest.stable_camera_ids.size() != count)
        {
            throw std::invalid_argument("recovered OOC weighted-node camera products are inconsistent");
        }

        RecoveredOocWeightedNodeOutput output;
        const std::array<double, 3> region_size{scene.region.size.x, scene.region.size.y, scene.region.size.z};
        output.alternate_scale = derive_ooc_maximum_sample_scale(region_size);

        std::vector<OocPyramidRegistryLevelMetadata> registry_metadata;
        registry_metadata.reserve(count);
        for (const auto& pyramid : input.ooc.pyramids)
        {
            if (pyramid.levels.empty() || pyramid.initial_level.size() != pyramid.levels.front().depth.size() ||
                pyramid.initial_level_weight.size() != pyramid.levels.front().depth.size())
            {
                throw std::invalid_argument("recovered OOC adaptive-root pyramid is incomplete");
            }
            registry_metadata.push_back(make_ooc_pyramid_registry_level_metadata(
                pyramid.levels.front().depth, pyramid.initial_level, pyramid.initial_level_weight));
        }
        const auto adaptive = derive_ooc_adaptive_root_mode0(registry_metadata, region_size);
        output.root_scale = adaptive.root_scale;
        output.adaptive_maximum_level = adaptive.maximum_level;
        output.adaptive_winning_bin = adaptive.winning_bin;
        output.adaptive_scale_factor = adaptive.scale_factor;
        output.adaptive_total_samples = adaptive.total_samples;

        std::array<double, 16> root_to_world{scene.region.rotation[0],
                                             scene.region.rotation[1],
                                             scene.region.rotation[2],
                                             scene.region.center.x,
                                             scene.region.rotation[3],
                                             scene.region.rotation[4],
                                             scene.region.rotation[5],
                                             scene.region.center.y,
                                             scene.region.rotation[6],
                                             scene.region.rotation[7],
                                             scene.region.rotation[8],
                                             scene.region.center.z,
                                             0.0,
                                             0.0,
                                             0.0,
                                             1.0};
        std::vector<std::vector<OocWeightedNodeRecord>> camera_local_nodes(count);
        std::vector<std::uint64_t> camera_candidate_counts(count, 0U);
        output.cameras.reserve(count);

        // sub_17A4F50 computes min(work+d0, work+d4, settings+0x370).  The
        // observed depth-map mode-0 BuildModel path supplies (9, 3, 1), so the
        // candidate producer is called exactly once, for level zero.  Consuming
        // all three independently saved seed levels over-counts both image cubes
        // and the balanced octree.
        constexpr std::size_t candidate_levels = 1U;
        constexpr float planarity_threshold = 0.5F;
        constexpr std::uint32_t raw_denominator = 1U;
        // Validate all public identities before entering the parallel region.  Each
        // camera then owns its candidate workspace and local reduction completely;
        // publishing remains ordinal-ordered below, preserving the target merge
        // order and every per-key floating-point operation.
        for (std::size_t ordinal = 0U; ordinal != count; ++ordinal)
        {
            const std::size_t camera_index = input.manifest.reference_camera_indices[ordinal];
            if (camera_index >= scene.cameras.size() ||
                input.ooc.items[ordinal].camera_id != input.manifest.stable_camera_ids[ordinal])
            {
                throw std::invalid_argument("recovered OOC weighted-node camera identity is inconsistent");
            }
            const auto& pyramid = input.ooc.pyramids[ordinal];
            if (pyramid.levels.size() < candidate_levels ||
                pyramid.initial_level.size() != pyramid.levels.front().depth.size() ||
                pyramid.initial_level_weight.size() != pyramid.levels.front().depth.size())
            {
                throw std::invalid_argument("recovered OOC weighted-node pyramid is incomplete");
            }
        }

        const auto camera_nodes_started = std::chrono::steady_clock::now();
        std::exception_ptr camera_failure;
        std::mutex camera_failure_mutex;
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t signed_ordinal = 0; signed_ordinal < static_cast<std::ptrdiff_t>(count); ++signed_ordinal)
        {
            const std::size_t ordinal = static_cast<std::size_t>(signed_ordinal);
            try
            {
                const std::size_t camera_index = input.manifest.reference_camera_indices[ordinal];
                const auto& pyramid = input.ooc.pyramids[ordinal];
                const auto& metadata = registry_metadata[ordinal];

                std::vector<OocWeightedNodeRecord> camera_nodes;
                std::uint64_t candidate_count = 0U;
                for (std::size_t level = 0U; level != candidate_levels; ++level)
                {
                    const auto& product = pyramid.levels[level];
                    if (product.width == 0U || product.height == 0U ||
                        product.width > std::numeric_limits<std::size_t>::max() / product.height ||
                        product.depth.size() != product.width * product.height ||
                        product.sample_scale.size() != product.depth.size())
                    {
                        throw std::invalid_argument("recovered OOC weighted-node level dimensions are invalid");
                    }
                    const std::uint32_t downscale = input.manifest.depth_downscale << static_cast<std::uint32_t>(level);
                    const auto project = make_ooc_depth_roi_project_mode0_input(scene, camera_index, downscale);
                    if (project.camera.width != static_cast<std::int64_t>(product.width) ||
                        project.camera.height != static_cast<std::int64_t>(product.height))
                    {
                        throw std::invalid_argument("recovered OOC weighted-node calibration/level mismatch");
                    }

                    std::vector<float> cached_depth = product.depth;
                    std::vector<float> cached_scale = product.sample_scale;
                    quantize_ooc_sample_scale_cache(cached_depth);
                    quantize_ooc_sample_scale_cache(cached_scale);
                    OocDepthCandidateInput candidate_input;
                    candidate_input.width = product.width;
                    candidate_input.height = product.height;
                    candidate_input.depth = cached_depth;
                    candidate_input.sample_scale = cached_scale;
                    candidate_input.focal_length = project.camera.focal_length;
                    candidate_input.principal_x =
                        static_cast<double>(project.camera.width) * 0.5 + project.camera.principal_x - 0.5;
                    candidate_input.principal_y =
                        static_cast<double>(project.camera.height) * 0.5 + project.camera.principal_y - 0.5;
                    candidate_input.camera_to_world = project.camera_to_world;
                    candidate_input.root_to_world = root_to_world;
                    candidate_input.root_extent = region_size;

                    auto workspace = build_ooc_depth_candidate_workspace(candidate_input);
                    apply_ooc_depth_candidate_planarity(workspace, product.width, product.height, planarity_threshold);
                    const auto candidates = compact_ooc_depth_candidates(workspace);
                    candidate_count += candidates.size();
                    if (camera_nodes.size() > std::numeric_limits<std::size_t>::max() - candidates.size())
                    {
                        throw std::overflow_error("recovered OOC weighted-node camera vector overflows");
                    }
                    camera_nodes.reserve(camera_nodes.size() + candidates.size());
                    for (const auto& candidate : candidates)
                    {
                        camera_nodes.push_back(make_initial_ooc_weighted_node(candidate,
                                                                              output.root_scale,
                                                                              output.alternate_scale,
                                                                              metadata.maximum_level,
                                                                              raw_denominator));
                    }
                }
                camera_local_nodes[ordinal] = reduce_ooc_weighted_nodes_local(camera_nodes);
                camera_candidate_counts[ordinal] = candidate_count;
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(camera_failure_mutex);
                if (!camera_failure)
                    camera_failure = std::current_exception();
            }
        }
        if (camera_failure)
            std::rethrow_exception(camera_failure);
        output.camera_nodes_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - camera_nodes_started).count();
        output.hwm_after_camera_nodes_kib = process_high_water_kib();

        const auto publish_started = std::chrono::steady_clock::now();
        std::size_t published_size = 0U;
        for (const auto& local : camera_local_nodes)
        {
            if (published_size > std::numeric_limits<std::size_t>::max() - local.size())
            {
                throw std::overflow_error("recovered OOC weighted-node scene vector overflows");
            }
            published_size += local.size();
        }
        std::vector<OocWeightedNodeRecord> published_nodes;
        published_nodes.reserve(published_size);
        for (std::size_t ordinal = 0U; ordinal != count; ++ordinal)
        {
            auto& local = camera_local_nodes[ordinal];
            const std::uint64_t local_size = static_cast<std::uint64_t>(local.size());
            published_nodes.insert(
                published_nodes.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
            output.cameras.push_back({
                input.manifest.reference_camera_indices[ordinal],
                input.manifest.stable_camera_ids[ordinal],
                registry_metadata[ordinal].maximum_level,
                camera_candidate_counts[ordinal],
                local_size,
            });
            std::vector<OocWeightedNodeRecord>().swap(local);
        }
        output.publish_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - publish_started).count();
        output.hwm_after_publish_kib = process_high_water_kib();

        const auto merge_started = std::chrono::steady_clock::now();
        output.merged_nodes = merge_ooc_weighted_nodes_all_cameras(std::move(published_nodes));
        output.merge_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - merge_started).count();
        output.hwm_after_merge_kib = process_high_water_kib();
        const auto filter_started = std::chrono::steady_clock::now();
        output.multi_camera_nodes = filter_ooc_multi_camera_nodes(output.merged_nodes);
        output.filter_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - filter_started).count();
        output.hwm_after_filter_kib = process_high_water_kib();
        const auto balance_started = std::chrono::steady_clock::now();
        output.balanced_nodes = balance_ooc_weighted_nodes_partitioned(output.multi_camera_nodes);
        output.balance_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - balance_started).count();
        output.hwm_after_balance_kib = process_high_water_kib();
        const auto initialize_started = std::chrono::steady_clock::now();
        output.records_before_histogram = initialize_ooc_octree_records(output.balanced_nodes);
        output.initialize_records_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - initialize_started).count();
        output.hwm_after_initialize_kib = process_high_water_kib();
        return output;
    }

    namespace
    {

        RecoveredOocHistogramOutput
        build_recovered_ooc_histogram_mode0_cuda_impl(const Scene& scene,
                                                      const RecoveredD4VotingToOocMode0Output& pyramid_input,
                                                      const RecoveredOocWeightedNodeOutput& weighted_input,
                                                      std::size_t device_index,
                                                      RecoveredD4VotingToOocMode0Output* consumable_pyramid_input)
        {
            if (!scene.region.specified || weighted_input.balanced_nodes.empty() ||
                pyramid_input.ooc.pyramids.empty() ||
                pyramid_input.ooc.pyramids.size() != pyramid_input.manifest.reference_camera_indices.size() ||
                pyramid_input.ooc.items.size() != pyramid_input.ooc.pyramids.size())
            {
                throw std::invalid_argument("recovered OOC histogram inputs are incomplete");
            }
            const std::array<double, 9> rotation = scene.region.rotation;
            const std::array<double, 3> center{scene.region.center.x, scene.region.center.y, scene.region.center.z};
            const std::array<double, 3> size{scene.region.size.x, scene.region.size.y, scene.region.size.z};
            auto initial = build_ooc_histogram_voxels_mode0(
                {weighted_input.balanced_nodes, rotation, center, size, weighted_input.root_scale});

            RecoveredOocHistogramOutput output;
            output.scalar_lut.resize(65536U);
            for (std::uint32_t bits = 0U; bits != 65536U; ++bits)
            {
                output.scalar_lut[bits] = half_to_float(static_cast<std::uint16_t>(bits));
            }

            OocHistogramCudaChainInput chain;
            chain.device_index = device_index;
            chain.initial_partitions = partition_ooc_histogram_voxels_four_way(initial);
            chain.cameras.reserve(pyramid_input.ooc.pyramids.size());
            for (std::size_t ordinal = 0U; ordinal != pyramid_input.ooc.pyramids.size(); ++ordinal)
            {
                const auto& pyramid = pyramid_input.ooc.pyramids[ordinal];
                if (pyramid.levels.empty() ||
                    pyramid.levels.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
                {
                    throw std::invalid_argument("recovered OOC histogram pyramid level count is invalid");
                }
                OocHistogramCudaCameraInput camera;
                std::size_t pixel_count = 0U;
                for (const auto& level : pyramid.levels)
                {
                    if (level.depth.size() != level.sample_scale.size() ||
                        level.width >
                            std::numeric_limits<std::size_t>::max() / std::max<std::size_t>(level.height, 1U) ||
                        level.depth.size() != level.width * level.height ||
                        pixel_count > std::numeric_limits<std::size_t>::max() - level.depth.size())
                    {
                        throw std::invalid_argument("recovered OOC histogram pyramid payload is invalid");
                    }
                    pixel_count += level.depth.size();
                }
                if (pixel_count > std::numeric_limits<std::size_t>::max() / 2U)
                {
                    throw std::overflow_error("recovered OOC histogram float2 pyramid overflows");
                }
                camera.concatenated_float2_pyramid.reserve(pixel_count * 2U);
                for (const auto& level : pyramid.levels)
                {
                    std::vector<float> depth = level.depth;
                    std::vector<float> scale = level.sample_scale;
                    // Both images cross the target PXR24 cache boundary before the
                    // histogram pass; reproducing only scale leaves low mantissa bits
                    // and downstream vote bins observably different.
                    quantize_ooc_sample_scale_cache(depth);
                    quantize_ooc_sample_scale_cache(scale);
                    for (std::size_t pixel = 0U; pixel != depth.size(); ++pixel)
                    {
                        camera.concatenated_float2_pyramid.push_back(depth[pixel]);
                        camera.concatenated_float2_pyramid.push_back(scale[pixel]);
                    }
                }

                const auto& project = pyramid_input.ooc.items[ordinal].project;
                OocHistogramProjectionInput projection;
                projection.camera = project.camera;
                projection.world_to_camera = invert_ooc_projective_matrix4(project.camera_to_world);
                for (std::size_t index = 0U; index != 16U; ++index)
                {
                    projection.camera_to_world[index] = static_cast<float>(project.camera_to_world[index]);
                }
                projection.scalar_lut = output.scalar_lut;
                projection.pyramid_levels = static_cast<std::uint32_t>(pyramid.levels.size());
                projection.threshold_a = weighted_input.root_scale;
                projection.mode_b = 1U;
                camera.parameters = make_ooc_histogram_cuda_camera_parameters_mode0(projection);
                camera.pyramid_levels = projection.pyramid_levels;
                camera.mode_b = projection.mode_b;
                camera.threshold_a = projection.threshold_a;
                camera.threshold_b = 6.0F;
                chain.cameras.push_back(std::move(camera));
                if (consumable_pyramid_input != nullptr)
                {
                    // All values read by this camera's histogram input now live in
                    // concatenated_float2_pyramid and parameters. No later camera can
                    // reference this per-camera pyramid.
                    consumable_pyramid_input->ooc.pyramids[ordinal] = {};
                }
            }

            OocHistogramCudaChainOutput histogram;
            std::string error;
            if (!run_recovered_ooc_histogram_cuda_chain(chain, histogram, error))
            {
                throw std::runtime_error(error);
            }
            output.kernel_launches = histogram.kernel_launches;
            output.histogram_voxels.reserve(initial.size());
            for (auto& partition : histogram.partitions)
            {
                output.histogram_voxels.insert(output.histogram_voxels.end(),
                                               std::make_move_iterator(partition.begin()),
                                               std::make_move_iterator(partition.end()));
            }
            if (output.histogram_voxels.size() != initial.size())
            {
                throw std::runtime_error("recovered OOC histogram partition join changed the record count");
            }
            output.records =
                build_ooc_octree_records_from_histogram_mode0(weighted_input.balanced_nodes, output.histogram_voxels);
            output.selected_indices = select_ooc_octree_nodes_breadth_first(output.records);
            output.active = make_ooc_leaf_active_mask(output.records, output.selected_indices);
            return output;
        }

    } // namespace

    RecoveredOocHistogramOutput
    build_recovered_ooc_histogram_mode0_cuda(const Scene& scene,
                                             const RecoveredD4VotingToOocMode0Output& pyramid_input,
                                             const RecoveredOocWeightedNodeOutput& weighted_input,
                                             std::size_t device_index)
    {
        return build_recovered_ooc_histogram_mode0_cuda_impl(
            scene, pyramid_input, weighted_input, device_index, nullptr);
    }

    RecoveredOocHistogramOutput
    build_recovered_ooc_histogram_mode0_cuda_consuming(const Scene& scene,
                                                       RecoveredD4VotingToOocMode0Output& pyramid_input,
                                                       const RecoveredOocWeightedNodeOutput& weighted_input,
                                                       std::size_t device_index)
    {
        return build_recovered_ooc_histogram_mode0_cuda_impl(
            scene, pyramid_input, weighted_input, device_index, &pyramid_input);
    }

} // namespace metmodel
