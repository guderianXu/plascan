#include <gtest/gtest.h>
#include "metmodel/mesh.hpp"
#include "metmodel/model_pipeline.hpp"
#include "metmodel/octree_prepare.hpp"
#include "metmodel/patchmatch.hpp"
#include "metmodel/neighbor_selection.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace
{
    TEST(RecoveredModelReference, NeighborGraphUsesInclusiveFloatRegionBounds)
    {
        metmodel::Scene scene;
        scene.neighbor_common_threshold = 1;
        scene.region.specified = true;
        scene.region.size = {2, 2, 2};
        scene.cameras.resize(3);
        for (auto& camera : scene.cameras)
            camera.aligned = true;
        scene.sparse_points = {{{3, 0, 0}, {255, 255, 255}, 0},
                               {{4, 0, 0}, {255, 255, 255}, 1},
                               {{1.0 + 1.0e-10, 0, 0}, {255, 255, 255}, 2}};
        scene.cameras[0].track_ids = {0, 1, 2};
        scene.cameras[1].track_ids = {0, 1};
        scene.cameras[2].track_ids = {2};
        const auto bounded = metmodel::select_recovered_neighbors(scene, 16);
        EXPECT_EQ(bounded[0], (std::vector<std::size_t>{2}));
        EXPECT_TRUE(bounded[1].empty());
        EXPECT_EQ(bounded[2], (std::vector<std::size_t>{0}));
        scene.region.size = {};
        EXPECT_EQ(metmodel::select_recovered_neighbors(scene, 16)[0], (std::vector<std::size_t>{1, 2}));
        scene.cameras[1].aligned = false;
        EXPECT_EQ(metmodel::select_recovered_neighbors(scene, 16)[0], (std::vector<std::size_t>{2}));
        for (auto& camera : scene.cameras)
            camera.track_ids.clear();
        EXPECT_THROW(metmodel::select_recovered_neighbors(scene, 16), std::invalid_argument);
    }

    TEST(RecoveredModelReference, SourceCudaFiltersGuardPartialTailBlocks)
    {
        constexpr std::size_t pixel_count = 130U;
        metmodel::PatchMatchCamera camera;
        camera.f = 100.0F;
        camera.width_original = static_cast<std::uint32_t>(pixel_count);
        camera.height_original = 1U;
        camera.type = 0U;

        metmodel::PatchMatchFilterCheckCostInput check_cost;
        check_cost.camera = camera;
        check_cost.depth_allocation.assign(pixel_count, 1.0F);
        check_cost.cost_allocation.assign(pixel_count, 0.0F);
        check_cost.global_work_items = pixel_count;
        metmodel::PatchMatchFilterCheckCostOutput check_cost_output;
        std::string error;
        if (!metmodel::run_recovered_patchmatch_filter_check_cost_cuda_source(check_cost, check_cost_output, error))
        {
            if (error.find("not compiled") != std::string::npos)
                GTEST_SKIP() << error;
            FAIL() << error;
        }
        EXPECT_EQ(check_cost_output.depth_allocation, check_cost.depth_allocation);

        metmodel::PatchMatchFilterCheckNeighboursInput check_neighbours;
        check_neighbours.camera = camera;
        check_neighbours.depth_allocation.assign(pixel_count, 0.0F);
        check_neighbours.filtered_mask_allocation.assign(pixel_count, 255U);
        check_neighbours.global_work_items = pixel_count;
        metmodel::PatchMatchFilterCheckNeighboursOutput check_neighbours_output;
        ASSERT_TRUE(metmodel::run_recovered_patchmatch_filter_check_neighbours_cuda_source(
            check_neighbours, check_neighbours_output, error))
            << error;
        EXPECT_EQ(check_neighbours_output.filtered_mask_allocation, std::vector<std::uint8_t>(pixel_count, 0U));

        metmodel::PatchMatchFilterClearDepthInput clear_depth;
        clear_depth.camera = camera;
        clear_depth.depth_allocation.assign(pixel_count, 1.0F);
        clear_depth.filtered_mask_allocation.assign(pixel_count, 0U);
        clear_depth.global_work_items = pixel_count;
        metmodel::PatchMatchFilterClearDepthOutput clear_depth_output;
        ASSERT_TRUE(
            metmodel::run_recovered_patchmatch_filter_clear_depth_cuda_source(clear_depth, clear_depth_output, error))
            << error;
        EXPECT_EQ(clear_depth_output.depth_allocation, clear_depth.depth_allocation);
        EXPECT_EQ(clear_depth_output.counter_not_empty, static_cast<std::int32_t>(pixel_count));

        metmodel::PatchMatchFilterNormalsInput normals;
        normals.camera = camera;
        normals.depth_allocation.assign(pixel_count, 0.0F);
        normals.normal_allocation.assign(pixel_count * 3U, 127U);
        normals.estimated_normal_allocation.assign(pixel_count * 3U, 0U);
        normals.estimate_normal_map = true;
        normals.filtered_mask_allocation.assign(pixel_count, 255U);
        normals.global_work_items = pixel_count;
        metmodel::PatchMatchFilterNormalsOutput normals_output;
        ASSERT_TRUE(metmodel::run_recovered_patchmatch_filter_normals_cuda_source(normals, normals_output, error))
            << error;
        EXPECT_EQ(normals_output.filtered_mask_allocation, std::vector<std::uint8_t>(pixel_count, 0U));

        metmodel::PatchMatchFilterSpecklesEdgesInput speckles;
        speckles.camera = camera;
        speckles.depth_allocation.assign(pixel_count, 0.0F);
        speckles.filtered_mask_allocation.assign(pixel_count, 255U);
        speckles.global_work_items = pixel_count;
        metmodel::PatchMatchFilterSpecklesEdgesOutput speckles_output;
        ASSERT_TRUE(
            metmodel::run_recovered_patchmatch_filter_speckles_edges_cuda_source(speckles, speckles_output, error))
            << error;
        EXPECT_EQ(speckles_output.filtered_mask_allocation, std::vector<std::uint8_t>(pixel_count, 0U));

        // The optimized source kernel dispatches exact affine perspective
        // transforms once on the host.  A mathematically equivalent generic
        // projective transform must retain identical filtering semantics.
        metmodel::PatchMatchCamera affine_camera;
        affine_camera.f = 100.0F;
        affine_camera.width_original = 5U;
        affine_camera.height_original = 3U;
        affine_camera.type = 0U;
        affine_camera.transform[0] = 1.0F;
        affine_camera.transform[5] = 1.0F;
        affine_camera.transform[10] = 1.0F;
        affine_camera.transform[15] = 1.0F;
        metmodel::PatchMatchFilterSpecklesEdgesInput affine_speckles;
        affine_speckles.camera = affine_camera;
        affine_speckles.depth_allocation.assign(15U, 2.0F);
        affine_speckles.filtered_mask_allocation.assign(15U, 0U);
        affine_speckles.global_work_items = 15U;
        metmodel::PatchMatchFilterSpecklesEdgesOutput affine_output;
        ASSERT_TRUE(
            metmodel::run_recovered_patchmatch_filter_speckles_edges_cuda_source(affine_speckles, affine_output, error))
            << error;

        auto projective_speckles = affine_speckles;
        for (std::size_t index = 0U; index != 16U; ++index)
            projective_speckles.camera.transform[index] *= 2.0F;
        metmodel::PatchMatchFilterSpecklesEdgesOutput projective_output;
        ASSERT_TRUE(metmodel::run_recovered_patchmatch_filter_speckles_edges_cuda_source(
            projective_speckles, projective_output, error))
            << error;
        EXPECT_EQ(affine_output.filtered_mask_allocation, projective_output.filtered_mask_allocation);
    }

    TEST(RecoveredModelReference, ParallelCudaSpeckleBandsRemainConnected)
    {
        constexpr std::uint32_t width = 512U;
        constexpr std::uint32_t height = 300U;
        constexpr std::uint32_t line_x = 100U;
        std::vector<float> depth(static_cast<std::size_t>(width) * height, 0.0F);
        std::vector<std::uint8_t> mask(depth.size(), 0U);
        for (std::uint32_t y = 0U; y < height; ++y)
        {
            const std::size_t index = static_cast<std::size_t>(y) * width + line_x;
            depth[index] = 1.0F;
            if (y + 1U < height)
            {
                mask[index] = static_cast<std::uint8_t>(1U << (y == 0U ? 3U : 6U));
            }
        }
        const std::size_t isolated = static_cast<std::size_t>(height / 2U) * width + 200U;
        depth[isolated] = 2.0F;
        std::string error;
        ASSERT_TRUE(
            metmodel::filter_recovered_patchmatch_cuda_speckle_components(width, height, mask, depth, 30U, error))
            << error;
        for (std::uint32_t y = 0U; y < height; ++y)
        {
            EXPECT_EQ(depth[static_cast<std::size_t>(y) * width + line_x], 1.0F);
        }
        EXPECT_EQ(depth[isolated], 0.0F);
    }

    TEST(RecoveredModelReference, CudaOocNeighborsMatchCpuReference)
    {
        std::vector<metmodel::OocOctreeRecord> records(9U);
        records[0].level = 0U;
        for (std::uint32_t child = 0U; child < 8U; ++child)
        {
            records[child + 1U].morton_words[0] = child << 29U;
            records[child + 1U].level = 1U;
        }
        std::vector<std::uint32_t> selected(records.size());
        std::iota(selected.begin(), selected.end(), 0U);
        const std::vector<std::uint8_t> active(records.size(), 1U);
        const std::vector<float> scalar_lut(65536U, 0.0F);
        const auto cpu = metmodel::prepare_ooc_fusion_state(records, selected, active, scalar_lut);
        metmodel::OocFusionState cuda;
        try
        {
            cuda = metmodel::prepare_ooc_fusion_state(records, selected, active, scalar_lut, {}, 0U);
        }
        catch (const std::exception& exception)
        {
            const std::string message = exception.what();
            if (message.find("CUDA-disabled") != std::string::npos ||
                message.find("no CUDA-capable device") != std::string::npos)
            {
                GTEST_SKIP() << message;
            }
            throw;
        }
        EXPECT_EQ(cuda.neighbors, cpu.neighbors);
        EXPECT_EQ(cuda.connectivity, cpu.connectivity);
        EXPECT_EQ(cuda.refinement, cpu.refinement);
        EXPECT_EQ(cuda.weights, cpu.weights);
        EXPECT_EQ(cuda.flags, cpu.flags);
    }

    void require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }
    void test_recovered_qem_part_budget()
    {
        require(metmodel::recovered_qem_part_face_budget(9820U, 9820U, 5000U) == 5051U,
                "Shoe recovered QEM part budget mismatch");
        require(metmodel::recovered_qem_part_face_budget(2014823U, 2014823U, 200000U) == 202000U,
                "South recovered QEM part budget mismatch");
        require(metmodel::recovered_qem_part_face_budget(9820U, 9820U, 5000U, true) == 5501U,
                "expanded recovered QEM part budget mismatch");
        bool rejected = false;
        try
        {
            (void)metmodel::recovered_qem_part_face_budget(1U, 0U, 1U);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        require(rejected, "recovered QEM part budget accepted zero total faces");
    }
    TEST(RecoveredModelReference, recovered_qem_part_budget)
    {
        EXPECT_NO_THROW(test_recovered_qem_part_budget());
    }
    void test_recovered_mesh_trim()
    {
        std::vector<metmodel::Face> faces(1);
        faces[0].vertices = {0U, 1U, 2U};
        const std::vector<metmodel::RecoveredMeshTrimAttribute> attributes{
            {1.0F, 3.0F, 10.0F, 1.0F}, {1.0F, 3.0F, 10.0F, 1.0F}, {1.0F, 0.0F, 4.0F, 1.0F}};
        const auto trace = metmodel::recover_mesh_trim_mask(faces, attributes, {1, 0.1F});
        require(trace.initial_keep == std::vector<std::uint8_t>({1U, 1U, 0U}),
                "recovered trim initial support gate mismatch");
        require(trace.frontier_inputs.size() == 2U && trace.frontier_outputs[0] == std::vector<std::uint32_t>({2U}) &&
                    trace.frontier_outputs[1].empty(),
                "recovered trim source-gated frontier mismatch");
        require(trace.after_frontier == std::vector<std::uint8_t>({1U, 1U, 1U}),
                "recovered trim frontier keep mask mismatch");
        require(trace.morphology_auxiliary_faces.size() == 12U && trace.before_morphology_expanded.size() == 15U,
                "recovered trim boundary padding mismatch");
        require(trace.final_keep == std::vector<std::uint8_t>({0U, 0U, 0U}), "recovered trim morphology mismatch");

        metmodel::Mesh mesh;
        mesh.vertices.resize(3U);
        metmodel::assign_recovered_vertex_confidence(mesh, attributes);
        require(mesh.vertices[0].confidence == 3.0F && mesh.vertices[1].confidence == 3.0F &&
                    mesh.vertices[2].confidence == 0.0F,
                "recovered confidence channel1/channel3 normalization mismatch");
        mesh.faces = faces;
        auto compact_attributes = attributes;
        const std::array<std::uint8_t, 3> keep{1U, 0U, 1U};
        metmodel::compact_mesh_recovered_trim(mesh, compact_attributes, keep);
        require(mesh.vertices.size() == 2U && mesh.faces.empty() && compact_attributes.size() == 2U &&
                    compact_attributes[1] == attributes[2],
                "recovered trim compaction mismatch");

        metmodel::Mesh color_mesh;
        color_mesh.vertices.resize(2U);
        const std::array<metmodel::RecoveredVertexColorAccumulator, 2> color_accumulator{
            {{std::bit_cast<float>(0x4122a745U),
              std::bit_cast<float>(0x4122a745U),
              std::bit_cast<float>(0x4122a745U),
              std::bit_cast<float>(0x3d234a90U)},
             {0.0F, 0.0F, 0.0F, 0.0F}}};
        metmodel::assign_recovered_vertex_colors_from_accumulator(color_mesh, color_accumulator);
        require(color_mesh.vertices[0].color == std::array<std::uint8_t, 3>{255U, 255U, 255U} &&
                    color_mesh.vertices[1].color == std::array<std::uint8_t, 3>{0U, 0U, 0U},
                "recovered reciprocal-multiply RGB normalization mismatch");

        const std::array<metmodel::RecoveredVertexColorAccumulator, 2> normalized_color_accumulator{
            {{0.5F, 1.0F, 0.25F, 1.0F}, {0.0F, 0.0F, 0.0F, 0.0F}}};
        metmodel::assign_recovered_vertex_colors_from_normalized_accumulator(color_mesh, normalized_color_accumulator);
        require(color_mesh.vertices[0].color == std::array<std::uint8_t, 3>{127U, 255U, 63U} &&
                    color_mesh.vertices[1].color == std::array<std::uint8_t, 3>{0U, 0U, 0U},
                "recovered Vulkan normalized RGB finalizer mismatch");

        metmodel::Mesh extrapolation_mesh;
        extrapolation_mesh.vertices.resize(4U);
        extrapolation_mesh.faces.push_back({{0U, 1U, 2U}});
        std::vector<metmodel::RecoveredVertexColorAccumulator> extrapolation_accumulator{
            {10.0F, 20.0F, 30.0F, 2.0F}, {1.0F, 2.0F, 3.0F, 0.01F}, {9.0F, 8.0F, 7.0F, 0.0F}, {4.0F, 5.0F, 6.0F, 0.0F}};
        const auto distance =
            metmodel::extrapolate_recovered_vertex_color_accumulator(extrapolation_mesh, extrapolation_accumulator);
        require(
            distance == std::vector<std::int32_t>({0, 1, 1, -1}) &&
                extrapolation_accumulator[0] == metmodel::RecoveredVertexColorAccumulator{10.0F, 20.0F, 30.0F, 2.0F} &&
                extrapolation_accumulator[1] == metmodel::RecoveredVertexColorAccumulator{5.0F, 10.0F, 15.0F, 1.0F} &&
                extrapolation_accumulator[2] == metmodel::RecoveredVertexColorAccumulator{5.0F, 10.0F, 15.0F, 1.0F} &&
                extrapolation_accumulator[3] == metmodel::RecoveredVertexColorAccumulator{},
            "recovered BFS color extrapolation mismatch");
    }
    TEST(RecoveredModelReference, recovered_mesh_trim)
    {
        EXPECT_NO_THROW(test_recovered_mesh_trim());
    }
    void test_recovered_mesh_region_clip()
    {
        std::vector<metmodel::RecoveredMeshClipVertex> vertices{{-1.0F, -1.0F, 0.0F, 11.0F},
                                                                {1.0F, -1.0F, 0.0F, 12.0F},
                                                                {1.0F, 1.0F, 0.0F, 13.0F},
                                                                {-1.0F, 1.0F, 0.0F, 14.0F}};
        std::vector<metmodel::RecoveredMeshClipFace> faces{{0U, 1U, 2U}, {0U, 2U, 3U}};
        const auto stats = metmodel::clip_mesh_to_plane_recovered(vertices, faces, {{{1.0, 0.0, 0.0}}, 0.0, 0.0});
        require(stats.input_vertices == 4U && stats.input_faces == 2U && stats.inserted_vertices == 3U &&
                    stats.removed_vertices == 2U && stats.appended_faces == 1U && stats.removed_faces == 0U &&
                    stats.output_vertices == 5U && stats.output_faces == 3U,
                "recovered region clip counters mismatch");
        const std::vector<metmodel::RecoveredMeshClipVertex> expected_vertices{{1.0F, -1.0F, 0.0F, 12.0F},
                                                                               {1.0F, 1.0F, 0.0F, 13.0F},
                                                                               {0.0F, -1.0F, 0.0F, 0.0F},
                                                                               {0.0F, 0.0F, 0.0F, 0.0F},
                                                                               {0.0F, 1.0F, 0.0F, 0.0F}};
        const std::vector<metmodel::RecoveredMeshClipFace> expected_faces{{2U, 0U, 1U}, {3U, 1U, 4U}, {2U, 1U, 3U}};
        require(vertices == expected_vertices && faces == expected_faces, "recovered region clip topology mismatch");

        const std::array<double, 15> identity_region{
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0};
        const auto planes = metmodel::make_recovered_region_clip_planes(identity_region, 0.125);
        require(planes[0].normal == std::array<double, 3>{-1.0, 0.0, 0.0} && planes[0].offset == -1.0 &&
                    planes[1].normal == std::array<double, 3>{0.0, -1.0, 0.0} && planes[1].offset == -2.0 &&
                    planes[2].normal == std::array<double, 3>{0.0, 0.0, -1.0} && planes[2].offset == -3.0 &&
                    planes[3].normal == std::array<double, 3>{1.0, 0.0, 0.0} && planes[3].offset == -1.0 &&
                    planes[5].normal == std::array<double, 3>{0.0, 0.0, 1.0} && planes[5].offset == -3.0 &&
                    planes[5].tolerance == 0.125,
                "recovered oriented-region plane construction mismatch");

        metmodel::Mesh production_mesh;
        production_mesh.vertices.resize(3U);
        production_mesh.vertices[0].position = {0.0, 0.0, 0.0};
        production_mesh.vertices[1].position = {0.25, 0.0, 0.0};
        production_mesh.vertices[2].position = {0.0, 0.25, 0.0};
        production_mesh.vertices[0].confidence = 2.0F;
        production_mesh.vertices[1].confidence = 3.0F;
        production_mesh.vertices[2].confidence = 4.0F;
        production_mesh.faces.push_back({{0U, 1U, 2U}});
        (void)metmodel::clip_mesh_to_region_recovered(production_mesh, identity_region, 0.0);
        require(production_mesh.vertices.size() == 3U && production_mesh.vertices[0].confidence == 2.0F &&
                    production_mesh.vertices[1].confidence == 3.0F && production_mesh.vertices[2].confidence == 4.0F,
                "recovered production region clip discarded confidence");
    }
    TEST(RecoveredModelReference, recovered_mesh_region_clip)
    {
        EXPECT_NO_THROW(test_recovered_mesh_region_clip());
    }
    void test_ooc_histogram_vote_tail()
    {
        const std::array<float, 8> pyramid{1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F, 4.0F, 40.0F};
        const metmodel::OocHistogramPyramidSelection selection{1.0F, 1.0F, 2, 2, 1U, 0U};
        const auto accumulation = metmodel::accumulate_ooc_histogram_pyramid_sample(selection, pyramid);
        require(accumulation.valid() && accumulation.depth_sum == 2.5F && accumulation.auxiliary_sum == 25.0F &&
                    accumulation.valid_weight == 1.0F,
                "OOC histogram float2 bilinear accumulation mismatch");

        const metmodel::OocHistogramAccumulatedSample accepted{10.0F, 0.5F, 2.0F, 1.0F, 1.0F, 1U, 4.0F, 6.0F, 1U};
        const auto accepted_vote = metmodel::evaluate_ooc_histogram_sample(accepted);
        require(accepted_vote.accepted() && std::abs(accepted_vote.normalized_residual - (2.0F / 3.0F)) < 1.0e-6F &&
                    accepted_vote.raw_vote_weight == 3.0F,
                "OOC histogram accumulator-to-vote replay mismatch");

        auto rejected = accepted;
        rejected.projected_depth = 25.0F;
        require(metmodel::evaluate_ooc_histogram_sample(rejected).decision ==
                    metmodel::OocHistogramSampleDecision::NegativeDepthGate,
                "OOC histogram negative-depth gate mismatch");
        rejected = accepted;
        rejected.geometry_distance = 2.0F;
        require(metmodel::evaluate_ooc_histogram_sample(rejected).decision ==
                    metmodel::OocHistogramSampleDecision::GeometryGate,
                "OOC histogram geometry gate mismatch");

        metmodel::OocHistogramVoxel voxel;
        metmodel::accumulate_ooc_histogram_vote(voxel, -1.0F, 3.0F);
        require(voxel.histogram[0] == 3U && voxel.histogram[1] == 0U, "OOC histogram negative endpoint split mismatch");
        metmodel::accumulate_ooc_histogram_vote(voxel, -1.0F, 1.0F);
        require(voxel.histogram[0] == 4U, "OOC histogram direct-bin contribution mismatch");

        metmodel::accumulate_ooc_histogram_vote(voxel, 0.0F, 3.0F);
        require(voxel.histogram[4] == 2U && voxel.histogram[5] == 1U, "OOC histogram central two-bin split mismatch");
        metmodel::accumulate_ooc_histogram_vote(voxel, 1.0F, 3.0F);
        require(voxel.histogram[8] == 0U && voxel.histogram[9] == 3U, "OOC histogram positive endpoint split mismatch");

        voxel.histogram[4] = 255U;
        metmodel::accumulate_ooc_histogram_vote(voxel, 0.0F, 3.0F);
        require(voxel.histogram[4] == 255U && voxel.histogram[5] == 2U, "OOC histogram locked-bin sentinel mismatch");
        voxel.histogram[9] = 253U;
        metmodel::accumulate_ooc_histogram_vote(voxel, 1.0F, 12.0F);
        require(voxel.histogram[9] == 254U, "OOC histogram saturation mismatch");
    }
    TEST(RecoveredModelReference, ooc_histogram_vote_tail)
    {
        EXPECT_NO_THROW(test_ooc_histogram_vote_tail());
    }
    void test_ooc_histogram_row_major_region_and_leaf_records()
    {
        std::vector<metmodel::OocWeightedNodeRecord> nodes(9U);
        nodes[0].level = 0U;
        nodes[0].weight_denominator = 1U;
        for (std::uint32_t child = 0U; child != 8U; ++child)
        {
            nodes[child + 1U].morton_words[0] = child << 29U;
            nodes[child + 1U].level = 1U;
            nodes[child + 1U].weight_denominator = 1U;
        }
        metmodel::OocHistogramVoxelBuildInput input;
        input.balanced_records = nodes;
        input.region_rotation = {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        input.region_center = {10.0, 20.0, 30.0};
        input.region_size = {2.0, 4.0, 6.0};
        input.root_scale = 6.0F;
        auto voxels = metmodel::build_ooc_histogram_voxels_mode0(input);
        require(voxels.size() == nodes.size() + 1U && voxels[1].position[0] == 9.0F && voxels[1].position[1] == 22.0F &&
                    voxels[1].position[2] == 30.0F,
                "OOC histogram region transform is not target row-major");
        for (std::size_t index = 1U; index != voxels.size(); ++index)
            voxels[index].histogram[index % 10U] = static_cast<std::uint8_t>(index);
        const auto records = metmodel::build_ooc_octree_records_from_histogram_mode0(nodes, voxels);
        require(records.size() == nodes.size() && records.back().level == 1U,
                "OOC histogram bridge omitted current support-level leaves");
    }
    TEST(RecoveredModelReference, ooc_histogram_row_major_region_and_leaf_records)
    {
        EXPECT_NO_THROW(test_ooc_histogram_row_major_region_and_leaf_records());
    }
    void test_ooc_histogram_projection_fail_closed()
    {
        metmodel::OocHistogramVoxel voxel;
        voxel.scalar_lut_index = 0U;
        metmodel::OocHistogramProjectionInput input;
        input.camera = {640, 480, 500.0, 0.0, 0.0, 0.0, 0.0};
        input.world_to_camera = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        input.camera_to_world = {
            1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
        const std::array<float, 1> scalar_lut{1.0F};
        input.scalar_lut = scalar_lut;
        input.pyramid_levels = 1U;

        const auto rejects = [&voxel](const metmodel::OocHistogramProjectionInput& value)
        {
            try
            {
                (void)metmodel::prepare_ooc_histogram_pyramid_selection_mode0(voxel, value);
                return false;
            }
            catch (const std::runtime_error&)
            {
                return true;
            }
        };

        auto invalid = input;
        invalid.camera.width = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
        require(rejects(invalid), "OOC histogram projection accepted an overflowing width");
        invalid = input;
        invalid.camera.height = 0;
        require(rejects(invalid), "OOC histogram projection accepted a zero height");
        invalid = input;
        invalid.pyramid_levels = 32U;
        require(rejects(invalid), "OOC histogram projection accepted too many pyramid levels");
        invalid = input;
        invalid.scalar_lut = {};
        require(rejects(invalid), "OOC histogram projection accepted a missing scalar LUT entry");
        auto invalid_voxel = voxel;
        invalid_voxel.metadata[2] = 33U;
        bool invalid_scale_rejected = false;
        try
        {
            (void)metmodel::prepare_ooc_histogram_pyramid_selection_mode0(invalid_voxel, input);
        }
        catch (const std::runtime_error&)
        {
            invalid_scale_rejected = true;
        }
        require(invalid_scale_rejected, "OOC histogram projection accepted an unsupported scale code");

        metmodel::OocHistogramVoxel accepted_voxel;
        accepted_voxel.position = {0.0F, 0.0F, 1.0F};
        accepted_voxel.metadata[3] = 1U;
        auto accepted_input = input;
        accepted_input.camera.width = 2;
        accepted_input.camera.height = 2;
        accepted_input.mode_b = 1U;
        accepted_input.threshold_a = 1.0F;
        const std::array<float, 8> pyramid{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        const auto accepted_chain =
            metmodel::accumulate_ooc_histogram_camera_vote_mode0(accepted_voxel, accepted_input, 1.0F, pyramid);
        require(accepted_chain.accepted() && accepted_voxel.histogram[4] == 2U && accepted_voxel.histogram[5] == 1U,
                "OOC histogram camera-vote composition mismatch");

        metmodel::OocHistogramVoxel truncated_voxel;
        truncated_voxel.position = {0.0F, 0.0F, 1.0F};
        bool truncated_rejected = false;
        try
        {
            (void)metmodel::accumulate_ooc_histogram_camera_vote_mode0(
                truncated_voxel, accepted_input, 1.0F, std::span<const float>(pyramid).first(7U));
        }
        catch (const std::runtime_error&)
        {
            truncated_rejected = true;
        }
        require(truncated_rejected, "OOC histogram camera-vote composition accepted a truncated pyramid");
    }
    TEST(RecoveredModelReference, ooc_histogram_projection_fail_closed)
    {
        EXPECT_NO_THROW(test_ooc_histogram_projection_fail_closed());
    }
    void test_ooc_histogram_cuda_camera_pack()
    {
        metmodel::OocHistogramProjectionInput input;
        input.camera.width = 3072;
        input.camera.height = 2304;
        input.camera.focal_length = 2500.25;
        input.camera.principal_x = -3.5;
        input.camera.principal_y = 7.25;
        input.camera.additive_focal = 0.75;
        input.camera.shear = -0.125;
        input.camera_to_world = {
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
            7.0F,
            8.0F,
            9.0F,
            10.0F,
            11.0F,
            12.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
        };
        const auto packed = metmodel::make_ooc_histogram_cuda_camera_parameters_mode0(input);
        const auto load_u32 = [](const auto& bytes, std::size_t offset)
        {
            std::uint32_t value = 0U;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return value;
        };
        const auto load_f32 = [](const auto& bytes, std::size_t offset)
        {
            float value = 0.0F;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            return value;
        };
        require(load_f32(packed.calibration.bytes, 32U) == 1.0e9F &&
                    std::isinf(load_f32(packed.calibration.bytes, 64U)) &&
                    std::isinf(load_f32(packed.calibration.bytes, 72U)) &&
                    std::signbit(load_f32(packed.calibration.bytes, 72U)),
                "OOC CUDA calibration sentinel pack mismatch");
        require(load_u32(packed.calibration.bytes, 80U) == 1U && load_u32(packed.calibration.bytes, 84U) == 3072U &&
                    load_u32(packed.calibration.bytes, 88U) == 2304U,
                "OOC CUDA calibration dimension pack mismatch");
        require(load_f32(packed.calibration.bytes, 92U) == 2500.25F &&
                    load_f32(packed.calibration.bytes, 96U) == -3.5F &&
                    load_f32(packed.calibration.bytes, 100U) == 7.25F &&
                    load_f32(packed.calibration.bytes, 104U) == 0.75F &&
                    load_f32(packed.calibration.bytes, 108U) == -0.125F,
                "OOC CUDA calibration intrinsic pack mismatch");
        require(std::all_of(packed.rolling_shutter.values.begin(),
                            packed.rolling_shutter.values.end(),
                            [](float value) { return value == 0.0F; }),
                "OOC CUDA ordinary-camera rolling shutter is not zero");
        const std::array<float, 12> identity{
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
        };
        require(packed.before_rotation.values == identity && packed.after_rotation.values == identity,
                "OOC CUDA camera identity matrix pack mismatch");
        const std::array<float, 16> exterior{
            1.0F,
            2.0F,
            3.0F,
            0.0F,
            5.0F,
            6.0F,
            7.0F,
            0.0F,
            9.0F,
            10.0F,
            11.0F,
            0.0F,
            4.0F,
            8.0F,
            12.0F,
            1.0F,
        };
        require(packed.exterior_transform.values == exterior, "OOC CUDA exterior transform pack mismatch");

        bool rejected = false;
        input.camera.focal_length = std::numeric_limits<double>::quiet_NaN();
        try
        {
            (void)metmodel::make_ooc_histogram_cuda_camera_parameters_mode0(input);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        require(rejected, "OOC CUDA camera packer accepted non-finite intrinsics");

        std::vector<metmodel::OocHistogramVoxel> voxels(10U);
        for (std::size_t index = 0; index != voxels.size(); ++index)
        {
            voxels[index].position[0] = static_cast<float>(index);
        }
        const auto partitions = metmodel::partition_ooc_histogram_voxels_four_way(voxels);
        require(partitions[0].size() == 3U && partitions[1].size() == 3U && partitions[2].size() == 3U &&
                    partitions[3].size() == 1U,
                "OOC histogram four-way partition sizes mismatch");
        std::size_t next = 0U;
        for (const auto& partition : partitions)
        {
            for (const auto& voxel : partition)
            {
                require(voxel.position[0] == static_cast<float>(next),
                        "OOC histogram four-way partition order mismatch");
                ++next;
            }
        }
        require(next == voxels.size(), "OOC histogram four-way partition lost records");

        std::array<std::uint64_t, 33> level_counts{};
        require(metmodel::select_ooc_pyramid_maximum_level(level_counts) == 32U,
                "OOC maximum-level empty histogram mismatch");
        level_counts[31] = 10U;
        require(metmodel::select_ooc_pyramid_maximum_level(level_counts) == 31U,
                "OOC maximum-level upper-tail selection mismatch");
        level_counts = {};
        level_counts[7] = 11U;
        level_counts[0] = 89U;
        require(metmodel::select_ooc_pyramid_maximum_level(level_counts) == 7U,
                "OOC maximum-level ten-percent selection mismatch");
        level_counts = {};
        level_counts[0] = 100U;
        require(metmodel::select_ooc_pyramid_maximum_level(level_counts) == 1U,
                "OOC maximum-level lower clamp mismatch");

        // Shoe camera 0, build 22956, captured at 0x1EC553B and followed in the
        // same run to registry+0x438 at 0x17EA6BE.
        level_counts = {};
        level_counts[3] = 33U;
        level_counts[4] = 87U;
        level_counts[5] = 468U;
        level_counts[6] = 14940U;
        level_counts[7] = 34439U;
        level_counts[8] = 968U;
        require(metmodel::select_ooc_pyramid_maximum_level(level_counts) == 7U,
                "OOC maximum-level real capture mismatch");

        const std::array<std::uint64_t, 32> captured_selected_counts{
            366,  1009, 1372, 1620, 1722, 1846, 2111, 2079, 2149, 2144, 1993, 1799, 1715, 1599, 1355, 1236,
            1067, 887,  887,  806,  752,  687,  631,  619,  600,  578,  543,  543,  511,  513,  502,  420,
        };
        const std::size_t selected_total = 36661U;
        std::vector<float> metadata_depth(selected_total, 1.0F);
        std::vector<std::uint8_t> metadata_levels(selected_total, 7U);
        std::vector<std::uint8_t> metadata_codes;
        metadata_codes.reserve(selected_total);
        for (std::size_t code = 0; code != captured_selected_counts.size(); ++code)
        {
            metadata_codes.insert(
                metadata_codes.end(), captured_selected_counts[code], static_cast<std::uint8_t>(code));
        }
        const auto registry_metadata =
            metmodel::make_ooc_pyramid_registry_level_metadata(metadata_depth, metadata_levels, metadata_codes);
        require(registry_metadata.maximum_level == 7U, "OOC registry metadata maximum level mismatch");
        for (std::size_t code = 0; code != captured_selected_counts.size(); ++code)
        {
            require(std::bit_cast<std::uint32_t>(registry_metadata.selected_level_histogram[code]) ==
                        std::bit_cast<std::uint32_t>(static_cast<float>(captured_selected_counts[code])),
                    "OOC registry selected-level histogram mismatch");
        }
        const std::array<metmodel::OocPyramidRegistryLevelMetadata, 1> adaptive_metadata{registry_metadata};
        const auto adaptive = metmodel::derive_ooc_adaptive_root_mode0(adaptive_metadata, {10.0, 2.0, 3.0});
        require(adaptive.maximum_level == 7U && adaptive.winning_bin == 7U &&
                    adaptive.total_samples == selected_total && adaptive.scale_factor == 1.2153846153846153 &&
                    adaptive.root_scale == static_cast<float>(10.0 * 1.2153846153846153),
                "OOC adaptive-root selected-level reduction mismatch");

        bool invalid_metadata_rejected = false;
        metadata_levels[0] = 33U;
        try
        {
            (void)metmodel::make_ooc_pyramid_registry_level_metadata(metadata_depth, metadata_levels, metadata_codes);
        }
        catch (const std::invalid_argument&)
        {
            invalid_metadata_rejected = true;
        }
        require(invalid_metadata_rejected, "OOC registry metadata accepted an out-of-range level code");

        const auto quantized_low = metmodel::quantize_ooc_pyramid_metadata_codes(0.75F, 1.0F);
        require(quantized_low.level == 0U && quantized_low.histogram_code == 0U,
                "OOC pyramid metadata lower endpoint mismatch");
        const auto quantized_high = metmodel::quantize_ooc_pyramid_metadata_codes(1.5F, 1.0F);
        require(quantized_high.level == 0U && quantized_high.histogram_code == 31U,
                "OOC pyramid metadata upper clamp mismatch");
        const auto quantized_level = metmodel::quantize_ooc_pyramid_metadata_codes(0.375F, 1.0F);
        require(quantized_level.level == 1U && quantized_level.histogram_code == 0U,
                "OOC pyramid metadata level selection mismatch");

        std::array<metmodel::OocPyramidRegistryRawRecord, 2> raw_registry{};
        for (std::size_t index = 0; index != raw_registry[0].size(); ++index)
        {
            raw_registry[0][index] = static_cast<std::byte>(index & 0xFFU);
            raw_registry[1][index] = static_cast<std::byte>((index * 17U + 3U) & 0xFFU);
        }
        std::fill(raw_registry[0].begin() + 0x4BCU, raw_registry[0].end(), std::byte{});
        const auto serialized_registry = metmodel::serialize_ooc_pyramid_registry(raw_registry);
        require(serialized_registry.size() == 8U + 2U * 0x4D4U &&
                    std::to_integer<unsigned>(serialized_registry[0]) == 2U,
                "OOC pyramid registry envelope mismatch");
        const auto decoded_registry = metmodel::deserialize_ooc_pyramid_registry(serialized_registry);
        require(decoded_registry ==
                    std::vector<metmodel::OocPyramidRegistryRawRecord>(raw_registry.begin(), raw_registry.end()),
                "OOC pyramid registry roundtrip mismatch");
        const auto empty_location = metmodel::ooc_pyramid_registry_payload_location(decoded_registry[0]);
        require(empty_location.shard_index == 0U && empty_location.byte_offset == 0U && empty_location.byte_size == 0U,
                "OOC pyramid registry payload location mismatch");
        const std::array<std::byte, 96> synthetic_pyramid = {
            std::byte{6},  std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},
            std::byte{},   std::byte{3},  std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},
            std::byte{},   std::byte{},   std::byte{5},  std::byte{},   std::byte{},   std::byte{},   std::byte{},
            std::byte{},   std::byte{},   std::byte{},   std::byte{2},  std::byte{},   std::byte{},   std::byte{},
            std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{3},  std::byte{},   std::byte{},
            std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{7},  std::byte{},
            std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{11},
            std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},
            std::byte{32}, std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},   std::byte{},
            std::byte{},   std::byte{1},  std::byte{2},  std::byte{3},  std::byte{4},  std::byte{5},  std::byte{6},
            std::byte{7},  std::byte{8},  std::byte{9},  std::byte{10}, std::byte{11}, std::byte{12}, std::byte{13},
            std::byte{14}, std::byte{15}, std::byte{16}, std::byte{17}, std::byte{18}, std::byte{19}, std::byte{20},
            std::byte{21}, std::byte{22}, std::byte{23}, std::byte{24}, std::byte{25}, std::byte{26}, std::byte{27},
            std::byte{28}, std::byte{29}, std::byte{30}, std::byte{31}, std::byte{32}};
        const auto pyramid_layout = metmodel::parse_ooc_pyramid_payload_layout(synthetic_pyramid, 2U);
        require(
            pyramid_layout.table_count == 6U && pyramid_layout.encoded_blob_offset == 64U &&
                pyramid_layout.encoded_blob_size == 32U && pyramid_layout.levels.size() == 2U &&
                pyramid_layout.levels[0].first_exr_offset == 64U && pyramid_layout.levels[0].first_exr_size == 3U &&
                pyramid_layout.levels[0].second_exr_offset == 67U && pyramid_layout.levels[0].second_exr_size == 5U &&
                pyramid_layout.levels[0].interlevel_plane_offset == 72U &&
                pyramid_layout.levels[0].interlevel_plane_size == 6U && pyramid_layout.levels[0].next_width == 2U &&
                pyramid_layout.levels[0].next_height == 3U && pyramid_layout.levels[1].first_exr_offset == 78U &&
                pyramid_layout.levels[1].first_exr_size == 7U && pyramid_layout.levels[1].second_exr_offset == 85U &&
                pyramid_layout.levels[1].second_exr_size == 11U,
            "OOC pyramid segment envelope mismatch");
        auto registry_with_tail = serialized_registry;
        registry_with_tail.push_back(std::byte{});
        bool registry_tail_rejected = false;
        try
        {
            (void)metmodel::deserialize_ooc_pyramid_registry(registry_with_tail);
        }
        catch (const std::invalid_argument&)
        {
            registry_tail_rejected = true;
        }
        require(registry_tail_rejected, "OOC pyramid registry accepted a trailing byte");

        const std::array<float, 16> reduction_depth{
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
            7.0F,
            8.0F,
            9.0F,
            10.0F,
            11.0F,
            12.0F,
            13.0F,
            14.0F,
            15.0F,
            16.0F,
        };
        const std::array<float, 16> reduction_temporary{
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
        };
        const std::array<float, 16> reduction_scale{
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
            0.5F,
        };
        const std::array<std::uint8_t, 4> reduction_gate{1U, 0U, 1U, 1U};
        const auto reduction = metmodel::reduce_ooc_pyramid_level({
            4U,
            4U,
            reduction_depth,
            reduction_temporary,
            reduction_scale,
            reduction_gate,
            true,
        });
        require(reduction.width == 2U && reduction.height == 2U && reduction.depth.size() == 4U &&
                    reduction.depth[0] == 3.5F && std::bit_cast<std::uint32_t>(reduction.depth[1]) == 0xD3800000U &&
                    reduction.depth[2] == 11.5F && reduction.depth[3] == 13.5F &&
                    reduction.temporary == std::vector<float>({4.0F, 0.0F, 4.0F, 4.0F}) &&
                    reduction.sample_scale == std::vector<float>({1.0F, 0.0F, 1.0F, 1.0F}) &&
                    reduction.temporary_u8 == std::vector<std::uint8_t>({4U, 0U, 4U, 4U}),
                "OOC pyramid hidden-state reduction mismatch");

        const std::array<float, 1> finer_depth{1.0F};
        std::array<metmodel::OocSampleScalePointRecord, 1> finer_records{};
        finer_records[0].point = {0.0F, 0.0F, 1.0F};
        finer_records[0].direction = {0.0F, 0.0F, 1.0F};
        metmodel::OocPyramidFinerLevelInput finer_input;
        finer_input.width = 1U;
        finer_input.height = 1U;
        finer_input.depth = finer_depth;
        finer_input.point_records = finer_records;
        finer_input.camera.width = 1;
        finer_input.camera.height = 1;
        finer_input.camera.focal_length = 1.0;
        finer_input.camera_to_record = {
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        };
        const auto finer = metmodel::build_ooc_pyramid_finer_level(finer_input);
        require(finer.depth == std::vector<float>({1.0F}) && finer.temporary == std::vector<float>({1.0F}) &&
                    finer.sample_scale == std::vector<float>({0.5F * 0.8695652484893799F}) &&
                    finer.temporary_u8 == std::vector<std::uint8_t>({1U}) && finer.accepted == 1U &&
                    finer.maximum_depth == 1.0F,
                "OOC pyramid finer-level geometry mismatch");
        const std::array<std::uint8_t, 1> finer_gate{0U};
        finer_input.gate = finer_gate;
        finer_input.special_invalid_depth = true;
        const auto gated_finer = metmodel::build_ooc_pyramid_finer_level(finer_input);
        require(std::bit_cast<std::uint32_t>(gated_finer.depth[0]) == 0xD3800000U && gated_finer.temporary[0] == 0.0F &&
                    gated_finer.sample_scale[0] == 0.0F && gated_finer.temporary_u8[0] == 0U &&
                    gated_finer.accepted == 0U && gated_finer.maximum_depth == 0.0F,
                "OOC pyramid finer-level gate mismatch");

        const std::array<float, 4> roi_depth{1.0F, std::bit_cast<float>(0x80000000U), 3.0F, 4.0F};
        const auto expanded_roi = metmodel::expand_ooc_depth_roi({4U, 3U, 1U, 1U, 3U, 3U, roi_depth});
        const std::array<float, 12> expected_roi{
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            std::bit_cast<float>(0x80000000U),
            0.0F,
            0.0F,
            3.0F,
            4.0F,
            0.0F,
        };
        require(expanded_roi.size() == expected_roi.size() &&
                    std::memcmp(expanded_roi.data(), expected_roi.data(), sizeof(expected_roi)) == 0,
                "OOC depth ROI expansion mismatch");
        const auto extracted_roi = metmodel::extract_ooc_depth_roi(4U, 3U, {1, 1, 3, 3}, expanded_roi);
        require(extracted_roi.size() == roi_depth.size() &&
                    std::memcmp(extracted_roi.data(), roi_depth.data(), sizeof(roi_depth)) == 0,
                "OOC depth ROI extraction mismatch");
        bool invalid_roi_rejected = false;
        try
        {
            (void)metmodel::expand_ooc_depth_roi({4U, 3U, 1U, 1U, 4U, 3U, roi_depth});
        }
        catch (const std::invalid_argument&)
        {
            invalid_roi_rejected = true;
        }
        require(invalid_roi_rejected, "OOC depth ROI accepted a truncated payload");

        const std::array<double, 16> projective_camera_transform{
            2.0,
            0.0,
            0.0,
            6.0,
            0.0,
            2.0,
            0.0,
            8.0,
            0.0,
            0.0,
            2.0,
            10.0,
            3.0,
            4.0,
            5.0,
            2.0,
        };
        const auto conditioned_camera_transform =
            metmodel::condition_ooc_camera_transform_mode0(projective_camera_transform);
        const std::array<double, 16> expected_conditioned_camera_transform{
            1.0,
            0.0,
            0.0,
            3.0,
            0.0,
            1.0,
            0.0,
            4.0,
            0.0,
            0.0,
            1.0,
            5.0,
            0.0,
            0.0,
            0.0,
            1.0,
        };
        require(std::memcmp(conditioned_camera_transform.data(),
                            expected_conditioned_camera_transform.data(),
                            sizeof(expected_conditioned_camera_transform)) == 0,
                "OOC camera-transform SVD conditioning mismatch");

        metmodel::OocDepthRoiBoundsMode0Input bounds_input;
        bounds_input.camera = {100, 80, 20.0, 0.0, 0.0, 0.0, 0.0};
        bounds_input.world_to_camera = {
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        };
        bounds_input.region_rotation = {
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        };
        bounds_input.region_center = {0.0, 0.0, 5.0};
        bounds_input.region_size = {2.0, 2.0, 2.0};
        const auto computed_bounds = metmodel::compute_ooc_depth_roi_bounds_mode0(bounds_input);
        require(computed_bounds.valid_samples == 64U && computed_bounds.bounds.x_begin == 45 &&
                    computed_bounds.bounds.y_begin == 35 && computed_bounds.bounds.x_end == 55 &&
                    computed_bounds.bounds.y_end == 45 &&
                    computed_bounds.samples.front().world == std::array<double, 3>{-1.0, -1.0, 4.0} &&
                    computed_bounds.samples.front().projected == std::array<double, 2>{45.0, 35.0},
                "OOC autonomous ROI projection/bounds mismatch");
        const auto half_bounds = metmodel::downsample_ooc_depth_roi_bounds(computed_bounds.bounds);
        require(half_bounds.x_begin == 22 && half_bounds.y_begin == 17 && half_bounds.x_end == 28 &&
                    half_bounds.y_end == 23,
                "OOC ROI lower-floor/upper-ceil transition mismatch");

        const std::vector<float> pyramid_depth_0(8U * 8U, 1.0F);
        const std::vector<float> pyramid_depth_1(4U * 4U, 1.0F);
        const std::vector<float> pyramid_depth_2(2U * 2U, 1.0F);
        metmodel::OocSampleScalePyramidInput pyramid_input;
        pyramid_input.seeds[0] = {8U, 8U, pyramid_depth_0, {8, 8, 8.0, 0.0, 0.0, 0.0, 0.0}, {}};
        pyramid_input.seeds[1] = {4U, 4U, pyramid_depth_1, {4, 4, 4.0, 0.0, 0.0, 0.0, 0.0}, {}};
        pyramid_input.seeds[2] = {2U, 2U, pyramid_depth_2, {2, 2, 2.0, 0.0, 0.0, 0.0, 0.0}, {}};
        pyramid_input.camera_to_record = finer_input.camera_to_record;
        pyramid_input.maximum_sample_scale = 10.0F;
        const auto pyramid = metmodel::build_ooc_sample_scale_pyramid_mode0(pyramid_input);
        require(pyramid.levels.size() == 4U && pyramid.levels[0].width == 8U && pyramid.levels[1].width == 4U &&
                    pyramid.levels[2].width == 2U && pyramid.levels[3].width == 1U && pyramid.levels[3].height == 1U &&
                    pyramid.levels[0].temporary_u8.empty() && pyramid.levels[1].temporary_u8.size() == 16U &&
                    pyramid.levels[2].temporary_u8.size() == 4U && pyramid.levels[3].temporary_u8.size() == 1U &&
                    pyramid.initial_level.size() == 64U && pyramid.initial_level_weight.size() == 64U,
                "OOC sample-scale pyramid orchestration mismatch");

        metmodel::OocSampleScalePyramidRoiInput roi_pyramid_input;
        const std::array<std::span<const float>, 3> pyramid_depths{pyramid_depth_0, pyramid_depth_1, pyramid_depth_2};
        for (std::size_t seed = 0U; seed != pyramid_depths.size(); ++seed)
        {
            const auto& source = pyramid_input.seeds[seed];
            roi_pyramid_input.seeds[seed] = {
                {source.width, source.height, 0U, 0U, source.width, source.height, pyramid_depths[seed]},
                source.camera,
                {},
            };
        }
        roi_pyramid_input.camera_to_record = pyramid_input.camera_to_record;
        roi_pyramid_input.maximum_sample_scale = pyramid_input.maximum_sample_scale;
        const auto roi_pyramid = metmodel::build_ooc_sample_scale_pyramid_from_rois_mode0(roi_pyramid_input);
        require(roi_pyramid.levels.size() == pyramid.levels.size() &&
                    roi_pyramid.initial_level == pyramid.initial_level &&
                    roi_pyramid.initial_level_weight == pyramid.initial_level_weight,
                "OOC ROI-to-pyramid orchestration mismatch");
        for (std::size_t level = 0U; level != pyramid.levels.size(); ++level)
        {
            require(roi_pyramid.levels[level].depth == pyramid.levels[level].depth &&
                        roi_pyramid.levels[level].temporary == pyramid.levels[level].temporary &&
                        roi_pyramid.levels[level].sample_scale == pyramid.levels[level].sample_scale &&
                        roi_pyramid.levels[level].temporary_u8 == pyramid.levels[level].temporary_u8,
                    "OOC ROI-to-pyramid level mismatch");
        }

        metmodel::OocSampleScalePyramidRegionInput region_pyramid_input;
        region_pyramid_input.bounds.camera = pyramid_input.seeds[0].camera;
        region_pyramid_input.bounds.camera_to_world = {
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        };
        region_pyramid_input.bounds.region_rotation = bounds_input.region_rotation;
        region_pyramid_input.bounds.region_center = {0.0, 0.0, 4.0};
        // The target derives the sample-scale ceiling from the largest region
        // dimension, not specifically from X. Keep Y largest to exercise that
        // distinction while preserving the 10.0f pyramid ceiling above.
        region_pyramid_input.bounds.region_size = {8.0, 10.0, 2.0};
        region_pyramid_input.roi_depths = pyramid_depths;
        for (std::size_t seed = 0U; seed != pyramid_depths.size(); ++seed)
        {
            region_pyramid_input.cameras[seed] = pyramid_input.seeds[seed].camera;
        }
        region_pyramid_input.camera_to_record = pyramid_input.camera_to_record;
        region_pyramid_input.maximum_sample_scale = pyramid_input.maximum_sample_scale;
        const auto region_pyramid = metmodel::build_ooc_sample_scale_pyramid_from_region_mode0(region_pyramid_input);
        require(region_pyramid.levels.size() == pyramid.levels.size() &&
                    region_pyramid.initial_level == pyramid.initial_level &&
                    region_pyramid.initial_level_weight == pyramid.initial_level_weight,
                "OOC autonomous region-to-pyramid orchestration mismatch");
        for (std::size_t level = 0U; level != pyramid.levels.size(); ++level)
        {
            require(region_pyramid.levels[level].depth == pyramid.levels[level].depth &&
                        region_pyramid.levels[level].temporary == pyramid.levels[level].temporary &&
                        region_pyramid.levels[level].sample_scale == pyramid.levels[level].sample_scale &&
                        region_pyramid.levels[level].temporary_u8 == pyramid.levels[level].temporary_u8,
                    "OOC autonomous region-to-pyramid level mismatch");
        }

        metmodel::OocSampleScalePyramidVotedDepthInput voted_pyramid_input;
        voted_pyramid_input.bounds = region_pyramid_input.bounds;
        voted_pyramid_input.voted_depths = pyramid_depths;
        voted_pyramid_input.cameras = region_pyramid_input.cameras;
        require(metmodel::derive_ooc_maximum_sample_scale(voted_pyramid_input.bounds.region_size) ==
                    pyramid_input.maximum_sample_scale,
                "OOC project maximum sample-scale derivation mismatch");
        const auto voted_pyramid =
            metmodel::build_ooc_sample_scale_pyramid_from_voted_depths_mode0(voted_pyramid_input);
        require(voted_pyramid.levels.size() == region_pyramid.levels.size() &&
                    voted_pyramid.initial_level == region_pyramid.initial_level &&
                    voted_pyramid.initial_level_weight == region_pyramid.initial_level_weight,
                "OOC voted-depth-to-pyramid orchestration mismatch");
        for (std::size_t level = 0U; level != region_pyramid.levels.size(); ++level)
        {
            require(voted_pyramid.levels[level].depth == region_pyramid.levels[level].depth &&
                        voted_pyramid.levels[level].temporary == region_pyramid.levels[level].temporary &&
                        voted_pyramid.levels[level].sample_scale == region_pyramid.levels[level].sample_scale &&
                        voted_pyramid.levels[level].temporary_u8 == region_pyramid.levels[level].temporary_u8,
                    "OOC voted-depth-to-pyramid level mismatch");
        }

        metmodel::Scene scene_adapter_input;
        scene_adapter_input.region.specified = true;
        scene_adapter_input.region.rotation = region_pyramid_input.bounds.region_rotation;
        scene_adapter_input.region.center = {0.0, 0.0, 4.0};
        scene_adapter_input.region.size = {8.0, 10.0, 2.0};
        metmodel::Camera scene_adapter_camera;
        scene_adapter_camera.aligned = true;
        scene_adapter_camera.image.width = 8U;
        scene_adapter_camera.image.height = 8U;
        scene_adapter_camera.model.f = 8.0;
        scene_adapter_camera.model.cx = 4.0;
        scene_adapter_camera.model.cy = 4.0;
        scene_adapter_camera.center = {0.0, 0.0, 0.0};
        scene_adapter_input.cameras.push_back(scene_adapter_camera);
        const auto scene_project = metmodel::make_ooc_depth_roi_project_mode0_input(scene_adapter_input, 0U, 1U);
        require(scene_project.camera.width == 8 && scene_project.camera.height == 8 &&
                    scene_project.camera.focal_length == 8.0 && scene_project.camera.principal_x == 0.0 &&
                    scene_project.camera.principal_y == 0.0 &&
                    scene_project.camera_to_world == region_pyramid_input.bounds.camera_to_world &&
                    scene_project.region_rotation == region_pyramid_input.bounds.region_rotation &&
                    scene_project.region_center == region_pyramid_input.bounds.region_center &&
                    scene_project.region_size == region_pyramid_input.bounds.region_size,
                "OOC Scene project adapter mismatch");

        // Shoe camera 0: public alignment.json values mapped to the exact mode-0
        // camera row captured at the target OOC worker boundary.
        metmodel::Scene shoe_scene_adapter_input;
        shoe_scene_adapter_input.region.specified = true;
        shoe_scene_adapter_input.region.size = {4.0, 2.0, 1.0};
        metmodel::Camera shoe_camera;
        shoe_camera.aligned = true;
        shoe_camera.image.width = 4928U;
        shoe_camera.image.height = 3264U;
        shoe_camera.model.f = 0x1.e631d9fbfc068p+11;
        shoe_camera.model.cx = 0x1.35a123acd0071p+11;
        shoe_camera.model.cy = 0x1.9d580b61543a9p+10;
        shoe_camera.pose.rotation.v = {
            0x1.c411d87972194p-2,
            -0x1.3fafc8b41644cp-1,
            0x1.49eda5618ae21p-1,
            0x1.60aa360a8db2cp-1,
            0x1.64641104724f4p-1,
            0x1.9edd36c86e25ap-3,
            -0x1.266a0916fcf68p-1,
            0x1.6aeeb0d46d175p-2,
            0x1.7989829b3315cp-1,
        };
        shoe_camera.center = {
            0x1.59c0903186a25p+1,
            -0x1.88353908ea4efp+0,
            0x1.686608564d269p+0,
        };
        shoe_scene_adapter_input.cameras.push_back(shoe_camera);
        const auto shoe_project = metmodel::make_ooc_depth_roi_project_mode0_input(shoe_scene_adapter_input, 0U, 8U);
        const std::array<double, 16> shoe_expected_camera_to_world{
            0x1.c411d87972194p-2,
            0x1.60aa360a8db2cp-1,
            -0x1.266a0916fcf68p-1,
            0x1.59c0903186a25p+1,
            -0x1.3fafc8b41644cp-1,
            0x1.64641104724f4p-1,
            0x1.6aeeb0d46d175p-2,
            -0x1.88353908ea4efp+0,
            0x1.49eda5618ae21p-1,
            0x1.9edd36c86e25ap-3,
            0x1.7989829b3315cp-1,
            0x1.686608564d269p+0,
            0.0,
            0.0,
            0.0,
            1.0,
        };
        require(shoe_project.camera.width == 616 && shoe_project.camera.height == 408 &&
                    shoe_project.camera.focal_length == 0x1.e631d9fbfc068p+8 &&
                    shoe_project.camera.principal_x == 0x1.a123acd007100p+0 &&
                    shoe_project.camera.principal_y == 0x1.5602d8550ea40p+1 &&
                    shoe_project.camera.additive_focal == 0.0 && shoe_project.camera.shear == 0.0 &&
                    shoe_project.camera_to_world == shoe_expected_camera_to_world,
                "OOC Shoe public Scene camera adapter mismatch");
        const std::array<std::vector<float>, 3> scene_voted_depths{pyramid_depth_0, pyramid_depth_1, pyramid_depth_2};
        const auto scene_voted_pyramid = metmodel::build_ooc_sample_scale_pyramid_from_scene_voted_depths_mode0(
            scene_adapter_input, 0U, scene_voted_depths, 1U, false);
        require(scene_voted_pyramid.levels.size() == voted_pyramid.levels.size() &&
                    scene_voted_pyramid.initial_level == voted_pyramid.initial_level &&
                    scene_voted_pyramid.initial_level_weight == voted_pyramid.initial_level_weight,
                "OOC Scene voted-depth orchestration mismatch");
        for (std::size_t level = 0U; level != voted_pyramid.levels.size(); ++level)
        {
            require(scene_voted_pyramid.levels[level].depth == voted_pyramid.levels[level].depth &&
                        scene_voted_pyramid.levels[level].temporary == voted_pyramid.levels[level].temporary &&
                        scene_voted_pyramid.levels[level].sample_scale == voted_pyramid.levels[level].sample_scale &&
                        scene_voted_pyramid.levels[level].temporary_u8 == voted_pyramid.levels[level].temporary_u8,
                    "OOC Scene voted-depth level mismatch");
        }

        const std::array<std::span<const float>, 3> scene_voted_views{
            scene_voted_depths[0], scene_voted_depths[1], scene_voted_depths[2]};
        const std::array<metmodel::OocSceneVotedDepthMode0View, 1> scene_bundle_views{{{0U, scene_voted_views, false}}};
        const auto scene_bundle = metmodel::build_ooc_pyramid_bundle_from_scene_voted_depths_mode0(
            scene_adapter_input, scene_bundle_views, 1U, 20U, 100U);
        const std::array<metmodel::OocPyramidBundleMode0Item, 1> expected_scene_items{
            {{static_cast<std::uint32_t>(scene_adapter_input.cameras[0].index),
              metmodel::make_ooc_depth_roi_project_mode0_input(scene_adapter_input, 0U, 1U)}}};
        const std::array<metmodel::OocSampleScalePyramidOutput, 1> expected_scene_pyramids{{scene_voted_pyramid}};
        const auto expected_scene_bundle =
            metmodel::serialize_ooc_pyramid_bundle_single_shard_mode0(expected_scene_items, expected_scene_pyramids);
        require(scene_bundle.camera_groups.size() == 1U && scene_bundle.camera_groups[0].begin_index == 0U &&
                    scene_bundle.camera_groups[0].item_count == 1U && scene_bundle.items.size() == 1U &&
                    scene_bundle.pyramids.size() == 1U && scene_bundle.bundle.payload_shards.size() == 1U &&
                    scene_bundle.bundle.registry == expected_scene_bundle.registry &&
                    scene_bundle.bundle.payload_shards[0] == expected_scene_bundle.payload,
                "OOC Scene voted-depth bundle orchestration mismatch");

        const metmodel::OocPyramidRegistryPayloadLocation registry_location{2U, 1234U, 5678U};
        const auto registry_record = metmodel::make_ooc_pyramid_registry_record_mode0(
            {17U, voted_pyramid_input.bounds, registry_location}, voted_pyramid);
        const auto read_u32 = [&](std::size_t offset)
        {
            std::uint32_t value = 0U;
            std::memcpy(&value, registry_record.data() + offset, sizeof(value));
            return value;
        };
        const auto read_u64 = [&](std::size_t offset)
        {
            std::uint64_t value = 0U;
            std::memcpy(&value, registry_record.data() + offset, sizeof(value));
            return value;
        };
        const auto registry_bounds =
            metmodel::compute_ooc_depth_roi_bounds_from_project_mode0(voted_pyramid_input.bounds).bounds;
        require(read_u32(0x000U) == 6U && read_u32(0x004U) == 17U && read_u64(0x088U) == 1U && read_u64(0x090U) == 8U &&
                    read_u64(0x098U) == 8U && read_u32(0x12CU) == 0x41CDCD65U && read_u32(0x400U) == 3U &&
                    read_u32(0x404U) == voted_pyramid.levels.size() && read_u64(0x408U) == 64U &&
                    read_u32(0x410U) == 1U && read_u32(0x414U) == 0x7F7FFFFFU &&
                    read_u64(0x418U) == static_cast<std::uint64_t>(registry_bounds.x_begin) &&
                    read_u64(0x420U) == static_cast<std::uint64_t>(registry_bounds.y_begin) &&
                    read_u64(0x428U) == static_cast<std::uint64_t>(registry_bounds.x_end) &&
                    read_u64(0x430U) == static_cast<std::uint64_t>(registry_bounds.y_end) &&
                    read_u64(0x4BCU) == registry_location.shard_index &&
                    read_u64(0x4C4U) == registry_location.byte_offset &&
                    read_u64(0x4CCU) == registry_location.byte_size,
                "OOC mode-0 registry record layout mismatch");

        const std::array<metmodel::OocPyramidBundleMode0Item, 2> bundle_items{{
            {23U, voted_pyramid_input.bounds},
            {19U, voted_pyramid_input.bounds},
        }};
        const std::array<metmodel::OocSampleScalePyramidOutput, 2> bundle_pyramids{{voted_pyramid, voted_pyramid}};
        const auto bundle = metmodel::serialize_ooc_pyramid_bundle_single_shard_mode0(bundle_items, bundle_pyramids);
        const auto bundle_records = metmodel::deserialize_ooc_pyramid_registry(bundle.registry);
        require(bundle_records.size() == 2U, "OOC single-shard bundle registry count mismatch");
        const auto first_location = metmodel::ooc_pyramid_registry_payload_location(bundle_records[0]);
        const auto second_location = metmodel::ooc_pyramid_registry_payload_location(bundle_records[1]);
        require(
            read_u32(0x004U) == 17U &&
                [&]
                    {
                        std::uint32_t id = 0U;
                        std::memcpy(&id, bundle_records[0].data() + 4U, sizeof(id));
                        return id;
                    }() == 23U &&
                [&]
                    {
                        std::uint32_t id = 0U;
                        std::memcpy(&id, bundle_records[1].data() + 4U, sizeof(id));
                        return id;
                    }() == 19U &&
                first_location.shard_index == 0U && first_location.byte_offset == 0U && first_location.byte_size > 0U &&
                second_location.shard_index == 0U && second_location.byte_offset == first_location.byte_size &&
                second_location.byte_size == first_location.byte_size &&
                bundle.payload.size() == first_location.byte_size + second_location.byte_size,
            "OOC single-shard bundle ordering/location mismatch");

        const auto south_group_ranges = metmodel::partition_ooc_pyramid_camera_groups(128U, 20U, 100U);
        require(south_group_ranges.size() == 7U && south_group_ranges[0].begin_index == 0U &&
                    south_group_ranges[0].item_count == 19U && south_group_ranges[1].begin_index == 19U &&
                    south_group_ranges[1].item_count == 19U && south_group_ranges[2].begin_index == 38U &&
                    south_group_ranges[2].item_count == 18U && south_group_ranges[6].begin_index == 110U &&
                    south_group_ranges[6].item_count == 18U,
                "OOC South camera group partition mismatch");
        const auto capped_group_ranges = metmodel::partition_ooc_pyramid_camera_groups(6U, 1U, 2U);
        require(capped_group_ranges.size() == 2U && capped_group_ranges[0].begin_index == 0U &&
                    capped_group_ranges[0].item_count == 3U && capped_group_ranges[1].begin_index == 3U &&
                    capped_group_ranges[1].item_count == 3U,
                "OOC maximum workgroup cap mismatch");

        const auto shoe_worker_plan = metmodel::plan_ooc_pyramid_workers(67023597568ULL, 8U, 251328U, 6U);
        require(shoe_worker_plan.target_memory_bytes == 33511798784ULL &&
                    shoe_worker_plan.requested_outer_threads == 8U && shoe_worker_plan.active_camera_slots == 6U &&
                    shoe_worker_plan.per_camera_inner_threads == 2U && !shoe_worker_plan.memory_limited,
                "OOC Shoe worker plan mismatch");
        const auto memory_limited_worker_plan = metmodel::plan_ooc_pyramid_workers(67023597568ULL, 8U, 200000000U, 6U);
        require(memory_limited_worker_plan.target_memory_bytes == 33511798784ULL &&
                    memory_limited_worker_plan.requested_outer_threads == 4U &&
                    memory_limited_worker_plan.active_camera_slots == 4U &&
                    memory_limited_worker_plan.per_camera_inner_threads == 2U &&
                    memory_limited_worker_plan.memory_limited,
                "OOC memory-limited worker plan mismatch");
        const auto minimum_memory_worker_plan = metmodel::plan_ooc_pyramid_workers(0x400000001ULL, 16U, 100000000U, 2U);
        require(minimum_memory_worker_plan.target_memory_bytes == 0x200000000ULL &&
                    minimum_memory_worker_plan.requested_outer_threads == 2U &&
                    minimum_memory_worker_plan.active_camera_slots == 2U &&
                    minimum_memory_worker_plan.per_camera_inner_threads == 8U,
                "OOC minimum-memory worker plan mismatch");
        bool invalid_worker_plan_rejected = false;
        try
        {
            (void)metmodel::plan_ooc_pyramid_workers(1U, 0U, 1U, 1U);
        }
        catch (const std::invalid_argument&)
        {
            invalid_worker_plan_rejected = true;
        }
        require(invalid_worker_plan_rejected, "OOC zero-thread worker plan was not rejected");

        const std::array<metmodel::OocPyramidBundleMode0Item, 1> second_group_items{
            {{31U, voted_pyramid_input.bounds}}};
        const std::array<metmodel::OocSampleScalePyramidOutput, 1> second_group_pyramids{{voted_pyramid}};
        const std::array<metmodel::OocPyramidBundleMode0Group, 2> bundle_groups{{
            {bundle_items, bundle_pyramids},
            {second_group_items, second_group_pyramids},
        }};
        const auto multi_bundle = metmodel::serialize_ooc_pyramid_bundle_mode0(bundle_groups);
        const auto multi_records = metmodel::deserialize_ooc_pyramid_registry(multi_bundle.registry);
        require(multi_bundle.payload_shards.size() == 2U && multi_records.size() == 3U &&
                    metmodel::ooc_pyramid_registry_payload_location(multi_records[0]).shard_index == 0U &&
                    metmodel::ooc_pyramid_registry_payload_location(multi_records[1]).shard_index == 0U &&
                    metmodel::ooc_pyramid_registry_payload_location(multi_records[2]).shard_index == 1U &&
                    metmodel::ooc_pyramid_registry_payload_location(multi_records[2]).byte_offset == 0U &&
                    multi_bundle.payload_shards[0].size() == first_location.byte_size + second_location.byte_size &&
                    multi_bundle.payload_shards[1].size() == first_location.byte_size,
                "OOC multi-group bundle location/order mismatch");

        auto invalid_pyramid_input = pyramid_input;
        invalid_pyramid_input.seeds[1].width = 3U;
        bool invalid_pyramid_rejected = false;
        try
        {
            (void)metmodel::build_ooc_sample_scale_pyramid_mode0(invalid_pyramid_input);
        }
        catch (const std::invalid_argument&)
        {
            invalid_pyramid_rejected = true;
        }
        require(invalid_pyramid_rejected, "OOC sample-scale pyramid accepted non-adjacent seeds");
    }
    TEST(RecoveredModelReference, ooc_histogram_cuda_camera_pack)
    {
        if (!metmodel::ooc_pyramid_exact_encoder_available())
        {
            GTEST_SKIP() << "target-exact OOC serialization requires OpenEXR 3.2.2 and zlib 1.3.2";
        }
        EXPECT_NO_THROW(test_ooc_histogram_cuda_camera_pack());
    }
    void test_ooc_marching_bridge()
    {
        const auto cache_quantize_bits = [](std::uint32_t bits)
        { return std::bit_cast<std::uint32_t>(metmodel::quantize_ooc_sample_scale_cache(std::bit_cast<float>(bits))); };
        require(cache_quantize_bits(0x3CA47379U) == 0x3CA47300U, "sample-scale cache round-down mismatch");
        require(cache_quantize_bits(0x3CA47381U) == 0x3CA47400U, "sample-scale cache round-up mismatch");
        require(cache_quantize_bits(0x3CA47280U) == 0x3CA47300U, "sample-scale cache even-low-bit midpoint mismatch");
        require(cache_quantize_bits(0x3CA47380U) == 0x3CA47400U, "sample-scale cache odd-low-bit midpoint mismatch");
        std::array<float, 3> cache_plane{std::bit_cast<float>(0x3CA47379U), std::bit_cast<float>(0x3CA47381U), 0.0F};
        metmodel::quantize_ooc_sample_scale_cache(cache_plane);
        require(std::bit_cast<std::uint32_t>(cache_plane[0]) == 0x3CA47300U &&
                    std::bit_cast<std::uint32_t>(cache_plane[1]) == 0x3CA47400U &&
                    std::bit_cast<std::uint32_t>(cache_plane[2]) == 0U,
                "sample-scale cache plane quantization mismatch");

        require(metmodel::quantize_ooc_initial_weight(4.0F, 3.0F) == 0U, "initial OOC weight lower boundary mismatch");
        require(metmodel::quantize_ooc_initial_weight(4.0F, 4.0F) == 85U, "initial OOC weight interior mismatch");
        require(metmodel::quantize_ooc_initial_weight(4.0F, 6.0F) == 255U,
                "initial OOC weight upper boundary mismatch");
        require(metmodel::quantize_ooc_initial_weight(std::numeric_limits<float>::quiet_NaN(), 1.0F) == 255U,
                "initial OOC weight unordered branch mismatch");

        const auto limited_selection = metmodel::select_ooc_initial_weight(2.0F, 2.0F, 0.005F, 7U);
        require(limited_selection.level == 7U && limited_selection.cell_scale == 0.0078125F &&
                    limited_selection.sample_scale == 0.005F && !limited_selection.direct_zero,
                "initial OOC limited-level selection mismatch");
        const auto direct_zero_selection = metmodel::select_ooc_initial_weight(2.0F, 2.0F, 0.005F, 0U);
        require(direct_zero_selection.level == 0U && direct_zero_selection.direct_zero,
                "initial OOC direct-zero selection mismatch");
        const auto clamped_selection = metmodel::select_ooc_initial_weight(2.0F, 2.0F, 0.01F, 32U);
        require(clamped_selection.level == 7U && clamped_selection.sample_scale == 0.01F &&
                    !clamped_selection.direct_zero,
                "initial OOC adaptive clamp selection mismatch");

        metmodel::OocWeightedNodeRecord initial_weight;
        metmodel::set_initial_ooc_weight(initial_weight, 0x102U, 255U);
        require(initial_weight.weight_denominator == 2U, "initial OOC denominator truncation mismatch");
        require(metmodel::ooc_weight_sum(initial_weight) == 0x00FEU, "initial OOC weighted-sum truncation mismatch");

        const std::array<float, 4> marching_scalars{2.0F, -1.0F, 4.0F, 0.5F};
        const std::array<float, 4> marching_weights{0.5F, 2.0F, 1.0F, 0.5F};
        require(metmodel::aggregate_ooc_marching_scalar(marching_scalars, marching_weights) == 0.8125F,
                "marching scalar aggregation mismatch");
        const std::array<float, 2> zero_scalars{-1.0F, 1.0F};
        const std::array<float, 2> unit_weights{1.0F, 1.0F};
        require(metmodel::aggregate_ooc_marching_scalar(zero_scalars, unit_weights) == 1.0e-6F,
                "marching scalar zero replacement mismatch");
        require(std::isnan(metmodel::aggregate_ooc_marching_scalar({}, {})),
                "empty marching scalar aggregation should preserve NaN");

        std::array<metmodel::OocMarchingExtractNode, 9> planner_tree{};
        planner_tree[0].child_group = 0U;
        for (std::size_t index = 1U; index != planner_tree.size(); ++index)
        {
            planner_tree[index].child_group = std::numeric_limits<std::uint32_t>::max();
        }
        const auto leaf_frontier = metmodel::plan_ooc_marching_capacity_frontier(planner_tree, 1U);
        require(leaf_frontier.size() == 8U, "marching capacity planner did not split an oversized root");
        for (std::size_t slot = 0U; slot != leaf_frontier.size(); ++slot)
        {
            require(leaf_frontier[slot].node_index == slot + 1U && leaf_frontier[slot].x == (slot & 1U) &&
                        leaf_frontier[slot].y == ((slot >> 1U) & 1U) && leaf_frontier[slot].z == ((slot >> 2U) & 1U) &&
                        leaf_frontier[slot].level == 1U && leaf_frontier[slot].subtree_node_count == 1U,
                    "marching capacity planner child coordinate mismatch");
        }
        const auto root_frontier = metmodel::plan_ooc_marching_capacity_frontier(planner_tree, 9U);
        require(root_frontier.size() == 1U && root_frontier[0].node_index == 0U &&
                    root_frontier[0].subtree_node_count == 9U,
                "marching capacity planner root threshold mismatch");
        bool zero_capacity_rejected = false;
        try
        {
            (void)metmodel::plan_ooc_marching_capacity_frontier(planner_tree, 0U);
        }
        catch (const std::invalid_argument&)
        {
            zero_capacity_rejected = true;
        }
        require(zero_capacity_rejected, "marching capacity planner accepted a zero threshold");

        metmodel::OocMarchingEdgeInput edge_input;
        edge_input.edge = 0U;
        edge_input.child_slot = 0U;
        edge_input.cell = {1, 2, 3};
        edge_input.level = 1U;
        edge_input.corner_scalar = {-1.0F, 3.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F};
        edge_input.root_scale = 8.0;
        edge_input.transform = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        edge_input.source_node_index = 123U;
        const auto edge_output = metmodel::make_ooc_marching_edge_vertex(edge_input);
        require(edge_output.vertex.position == std::array<float, 3>{4.5F, 8.0F, 12.0F},
                "marching edge zero-crossing mismatch");
        require(edge_output.vertex.trailing_word == 0U && edge_output.vertex_scale == 1.0F &&
                    edge_output.source_node_index == 123U,
                "marching edge attribute mismatch");

        const std::array<metmodel::OocMarchingVertex, 4> square{{
            {{{0.0F, 0.0F, 0.0F}}, 0U},
            {{{1.0F, 0.0F, 0.0F}}, 0U},
            {{{1.0F, 1.0F, 0.0F}}, 0U},
            {{{0.0F, 1.0F, 0.0F}}, 0U},
        }};
        const std::array<std::uint32_t, 4> square_indices{10U, 11U, 12U, 13U};
        const auto square_dp = metmodel::triangulate_ooc_marching_polygon(square, square_indices, true);
        require(square_dp.split[3] == 1 && square_dp.triangles.size() == 2U, "marching polygon DP split mismatch");
        require(square_dp.triangles[0].vertices == std::array<std::uint32_t, 3>{10U, 11U, 13U} &&
                    square_dp.triangles[1].vertices == std::array<std::uint32_t, 3>{11U, 12U, 13U},
                "marching polygon backtracking mismatch");

        metmodel::OocMarchingCellState contour_cell;
        contour_cell.corner_scalar = {-1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F};
        contour_cell.edge_vertex.fill(-1);
        contour_cell.child_state.fill(-1);
        contour_cell.edge_vertex[0] = 10;
        contour_cell.edge_vertex[4] = 11;
        contour_cell.edge_vertex[8] = 12;
        const auto contour = metmodel::trace_ooc_marching_local_contour(contour_cell, 0U);
        require(contour.edges.size() == 3U && contour.vertex_indices.size() == 3U && contour.vertex_indices[0] == 10U,
                "marching local contour trace mismatch");

        metmodel::OocWeightedNodeRecord weighted_a;
        weighted_a.level = 5;
        weighted_a.morton_words = {1, 2, 3};
        weighted_a.weight_denominator = 200;
        metmodel::set_ooc_weight_sum(weighted_a, 40000);
        auto weighted_b = weighted_a;
        weighted_b.weight_denominator = 100;
        metmodel::set_ooc_weight_sum(weighted_b, 20000);
        metmodel::merge_ooc_weighted_node(weighted_a, weighted_b);
        require(weighted_a.weight_denominator == 127U, "weighted-node denominator renormalization mismatch");
        require(metmodel::ooc_weight_sum(weighted_a) == 25400U, "weighted-node sum renormalization mismatch");
        require(metmodel::normalized_ooc_weight(weighted_a) == 200U, "weighted-node quotient mismatch");

        std::vector<metmodel::OocOctreeRecord> records(9);
        records[0].level = 0;
        records[0].weight = 64;
        records[0].histogram = {0, 0, 0, 1, 2, 3, 4, 0, 0, 0};
        records[0].trailing_field = metmodel::float_to_half(0.0F);
        for (std::uint32_t child = 0; child < 8U; ++child)
        {
            records[child + 1U].morton_words[0] = child << 29U;
            records[child + 1U].level = 1;
            records[child + 1U].weight = static_cast<std::uint8_t>(child);
            records[child + 1U].scalar_lut_index = static_cast<std::uint16_t>(child + 1U);
            records[child + 1U].trailing_field =
                metmodel::float_to_half(child == 0U ? -0.0625F : static_cast<float>(child + 1U) / 16.0F);
        }
        std::vector<std::uint32_t> selected(9);
        for (std::uint32_t index = 0; index < selected.size(); ++index)
        {
            selected[index] = index;
        }
        const std::vector<std::uint8_t> denominators{200, 1, 2, 3, 4, 5, 6, 7, 8};
        const auto nodes = metmodel::make_ooc_marching_nodes(records, selected, denominators);
        require(nodes[0].weight_denominator == 127U, "marching denominator cap mismatch");
        require(nodes[0].central_histogram_sum == 10U, "marching central histogram sum mismatch");
        require(nodes[0].scalar == 1.0e-6F, "marching zero scalar replacement mismatch");

        std::vector<metmodel::OocWeightedNodeRecord> upstream(records.size());
        for (std::size_t index = 0; index != records.size(); ++index)
        {
            upstream[index].morton_words = records[index].morton_words;
            upstream[index].level = records[index].level;
            upstream[index].weight_denominator = denominators[index];
        }
        const auto identity_nodes = metmodel::make_ooc_marching_nodes(
            records, selected, std::span<const metmodel::OocWeightedNodeRecord>(upstream));
        require(std::memcmp(identity_nodes.data(), nodes.data(), nodes.size() * sizeof(nodes.front())) == 0,
                "marching same-index denominator projection mismatch");
        // The producer table and persistent table need not have the same order.
        std::rotate(upstream.begin(), upstream.begin() + 3, upstream.end());
        const auto producer_nodes = metmodel::make_ooc_marching_nodes(
            records, selected, std::span<const metmodel::OocWeightedNodeRecord>(upstream));
        require(std::memcmp(producer_nodes.data(), nodes.data(), nodes.size() * sizeof(nodes.front())) == 0,
                "marching upstream-denominator join mismatch");
        auto missing_upstream = upstream;
        missing_upstream.pop_back();
        bool missing_denominator_rejected = false;
        try
        {
            (void)metmodel::make_ooc_marching_nodes(
                records, selected, std::span<const metmodel::OocWeightedNodeRecord>(missing_upstream));
        }
        catch (const std::exception&)
        {
            missing_denominator_rejected = true;
        }
        require(missing_denominator_rejected, "marching accepted a missing upstream denominator");

        auto extract = metmodel::build_ooc_marching_extract(nodes);
        require(extract[0].child_group == 0U, "marching child group mismatch");
        for (std::size_t index = 1; index < extract.size(); ++index)
        {
            require(extract[index].child_group == std::numeric_limits<std::uint32_t>::max(),
                    "marching leaf unexpectedly has children");
        }
        metmodel::reorder_ooc_marching_child_groups(extract);
        constexpr std::array<std::size_t, 8> expected{0, 4, 2, 6, 1, 5, 3, 7};
        for (std::size_t slot = 0; slot < expected.size(); ++slot)
        {
            require(extract[slot + 1U].scalar == nodes[expected[slot] + 1U].scalar, "marching child reorder mismatch");
        }

        const std::vector<std::uint8_t> active(selected.size(), 1U);
        std::vector<float> scalar_lut(65536U, 0.0F);
        for (const auto& record : records)
        {
            scalar_lut[record.scalar_lut_index] = metmodel::half_to_float(record.trailing_field);
        }
        metmodel::OocFusionParameters no_iterations;
        no_iterations.iterations = 0U;
        const auto continuous =
            metmodel::run_recovered_ooc_continuous_part_cpu(records,
                                                            selected,
                                                            active,
                                                            scalar_lut,
                                                            std::span<const metmodel::OocWeightedNodeRecord>(upstream),
                                                            {},
                                                            {},
                                                            no_iterations);
        require(continuous.records.size() == records.size() && continuous.fusion.size() == selected.size() &&
                    continuous.packed_scalars.size() == selected.size() &&
                    continuous.marching_nodes.size() == selected.size() &&
                    continuous.marching_extract.size() == selected.size(),
                "continuous OOC part output size mismatch");
        auto continuous_expected = metmodel::build_ooc_marching_extract(metmodel::make_ooc_marching_nodes(
            continuous.records, selected, std::span<const metmodel::OocWeightedNodeRecord>(upstream)));
        metmodel::reorder_ooc_marching_child_groups(continuous_expected);
        require(std::memcmp(continuous.marching_extract.data(),
                            continuous_expected.data(),
                            continuous_expected.size() * sizeof(continuous_expected.front())) == 0,
                "continuous OOC part marching bridge mismatch");

        auto support_tree = upstream;
        support_tree.reserve(9U + 64U);
        for (std::uint32_t parent = 0U; parent != 8U; ++parent)
        {
            for (std::uint32_t child = 0U; child != 8U; ++child)
            {
                metmodel::OocWeightedNodeRecord support;
                support.morton_words[0] = (parent << 29U) | (child << 26U);
                support.level = 2U;
                support.weight_denominator = 1U;
                support_tree.push_back(support);
            }
        }
        const std::array<double, 16> identity{
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
        const std::array<std::uint32_t, 1> model_stages{2U};
        const auto multilevel = metmodel::run_recovered_ooc_multilevel_model_cpu(
            records, support_tree, scalar_lut, model_stages, 1.0, identity, no_iterations);
        require(multilevel.stages.size() == 1U && multilevel.stages[0].support_level == 2U &&
                    multilevel.stages[0].balanced_node_count == 73U &&
                    multilevel.stages[0].persistent_node_count == 9U && multilevel.records.size() == records.size() &&
                    multilevel.marching_maximum_level == 1U,
                "multilevel OOC model stage contract mismatch");
        metmodel::validate_ooc_marching_raw_output(multilevel.raw_mesh);
    }
    TEST(RecoveredModelReference, ooc_marching_bridge)
    {
        EXPECT_NO_THROW(test_ooc_marching_bridge());
    }
    void test_recovered_d4_voting_to_ooc_bridge()
    {
        metmodel::Scene scene;
        scene.region.specified = true;
        scene.region.center = {0.0, 0.0, 4.0};
        scene.region.size = {8.0, 10.0, 2.0};
        metmodel::Camera camera;
        camera.index = 17U;
        camera.aligned = true;
        camera.image.width = 32U;
        camera.image.height = 32U;
        camera.model.f = 32.0;
        camera.model.cx = 16.0;
        camera.model.cy = 16.0;
        camera.pose.rotation.v = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        scene.cameras.push_back(camera);

        metmodel::RecoveredPatchMatchD4SceneOutput recovered;
        recovered.cameras.resize(1U);
        recovered.cameras[0].patchmatch.camera_index = 0U;
        recovered.cameras[0].voting.depth_after_components = {
            std::vector<float>(64U, 4.0F),
            std::vector<float>(16U, 4.0F),
            std::vector<float>(4U, 4.0F),
        };
        const std::array<std::size_t, 1> references{0U};
        metmodel::RecoveredD4VotingToOocMode0Input input;
        input.reference_camera_indices = references;
        input.diagonal_policy = metmodel::RecoveredOocDiagonalPolicy::DeterministicDisabled;
        input.workitem_size_cameras = 20U;
        input.max_workgroup_size = 100U;

        const auto bridged = metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0(scene, recovered, input);
        const std::array<std::span<const float>, 3> depth_views{
            recovered.cameras[0].voting.depth_after_components[0],
            recovered.cameras[0].voting.depth_after_components[1],
            recovered.cameras[0].voting.depth_after_components[2],
        };
        const std::array<metmodel::OocSceneVotedDepthMode0View, 1> direct_views{{{0U, depth_views, false}}};
        const auto direct =
            metmodel::build_ooc_pyramid_bundle_from_scene_voted_depths_mode0(scene, direct_views, 4U, 20U, 100U);
        require(bridged.ooc.bundle.registry == direct.bundle.registry &&
                    bridged.ooc.bundle.payload_shards == direct.bundle.payload_shards &&
                    bridged.manifest.abi_version == 1U &&
                    bridged.manifest.reference_camera_indices == std::vector<std::size_t>{0U} &&
                    bridged.manifest.stable_camera_ids == std::vector<std::uint32_t>{17U} &&
                    bridged.manifest.diagonal_pixel_scale_used == std::vector<std::uint8_t>{0U} &&
                    bridged.manifest.workitem_size_cameras == 20U && bridged.manifest.max_workgroup_size == 100U &&
                    bridged.manifest.voted_depth_hashes.size() == 1U && bridged.manifest.registry_hash != 0U &&
                    bridged.manifest.payload_shard_hashes.size() == 1U,
                "recovered d4 voting-to-OOC continuous bridge mismatch");

        // The preserving path is the serial oracle. The consuming production
        // path may build independent camera pyramids in parallel, but every
        // ordinal product and the serialized bundle must remain byte-identical.
        auto parallel_scene = scene;
        auto second_camera = camera;
        second_camera.index = 29U;
        parallel_scene.cameras.push_back(second_camera);
        auto parallel_recovered = recovered;
        parallel_recovered.cameras.push_back(recovered.cameras[0]);
        parallel_recovered.cameras[1].patchmatch.camera_index = 1U;
        const std::array<std::size_t, 2> parallel_references{0U, 1U};
        auto parallel_input = input;
        parallel_input.reference_camera_indices = parallel_references;
        const auto serial_oracle =
            metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0(parallel_scene, parallel_recovered, parallel_input);
        auto parallel_consumed_depth = parallel_recovered;
        const auto parallel_output = metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0_consuming(
            parallel_scene, parallel_consumed_depth, parallel_input);
        auto serial_input = parallel_input;
        serial_input.camera_pyramid_execution = metmodel::RecoveredOocCameraPyramidExecution::Serial;
        auto serial_consumed_depth = parallel_recovered;
        const auto serial_output = metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0_consuming(
            parallel_scene, serial_consumed_depth, serial_input);

        const auto equal_bytes = [](const auto& left, const auto& right)
        {
            return left.size() == right.size() &&
                   (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(left[0])) == 0);
        };
        require(serial_oracle.ooc.items.size() == 2U &&
                    serial_oracle.ooc.items.size() == parallel_output.ooc.items.size() &&
                    serial_oracle.ooc.pyramids.size() == parallel_output.ooc.pyramids.size(),
                "consuming OOC camera slots do not match serial oracle");
        for (std::size_t ordinal = 0U; ordinal != serial_oracle.ooc.items.size(); ++ordinal)
        {
            const auto& expected_item = serial_oracle.ooc.items[ordinal];
            const auto& actual_item = parallel_output.ooc.items[ordinal];
            require(expected_item.camera_id == actual_item.camera_id &&
                        std::memcmp(&expected_item.project, &actual_item.project, sizeof(expected_item.project)) == 0,
                    "parallel OOC item differs from serial oracle");
            const auto& expected_pyramid = serial_oracle.ooc.pyramids[ordinal];
            const auto& actual_pyramid = parallel_output.ooc.pyramids[ordinal];
            require(equal_bytes(expected_pyramid.initial_level, actual_pyramid.initial_level) &&
                        equal_bytes(expected_pyramid.initial_level_weight, actual_pyramid.initial_level_weight) &&
                        expected_pyramid.finer_accepted == actual_pyramid.finer_accepted &&
                        std::memcmp(expected_pyramid.finer_maximum_depth.data(),
                                    actual_pyramid.finer_maximum_depth.data(),
                                    sizeof(expected_pyramid.finer_maximum_depth)) == 0 &&
                        expected_pyramid.levels.size() == actual_pyramid.levels.size(),
                    "parallel OOC pyramid header differs from serial oracle");
            for (std::size_t level = 0U; level != expected_pyramid.levels.size(); ++level)
            {
                const auto& expected = expected_pyramid.levels[level];
                const auto& actual = actual_pyramid.levels[level];
                require(expected.width == actual.width && expected.height == actual.height &&
                            equal_bytes(expected.depth, actual.depth) &&
                            equal_bytes(expected.temporary, actual.temporary) &&
                            equal_bytes(expected.sample_scale, actual.sample_scale) &&
                            equal_bytes(expected.temporary_u8, actual.temporary_u8),
                        "parallel OOC pyramid level differs from serial oracle");
            }
        }
        const auto equal_camera_groups = [](const auto& left, const auto& right)
        {
            if (left.size() != right.size())
                return false;
            for (std::size_t index = 0U; index != left.size(); ++index)
            {
                if (left[index].begin_index != right[index].begin_index ||
                    left[index].item_count != right[index].item_count)
                    return false;
            }
            return true;
        };
        require(equal_camera_groups(serial_oracle.ooc.camera_groups, parallel_output.ooc.camera_groups) &&
                    serial_oracle.ooc.bundle.registry == parallel_output.ooc.bundle.registry &&
                    serial_oracle.ooc.bundle.payload_shards == parallel_output.ooc.bundle.payload_shards &&
                    serial_oracle.manifest.registry_hash == parallel_output.manifest.registry_hash &&
                    serial_oracle.manifest.payload_shard_hashes == parallel_output.manifest.payload_shard_hashes &&
                    serial_oracle.ooc.bundle.registry == serial_output.ooc.bundle.registry &&
                    serial_oracle.ooc.bundle.payload_shards == serial_output.ooc.bundle.payload_shards &&
                    parallel_output.camera_pyramid_execution ==
                        metmodel::RecoveredOocCameraPyramidExecution::Parallel &&
                    serial_output.camera_pyramid_execution == metmodel::RecoveredOocCameraPyramidExecution::Serial &&
                    parallel_output.camera_pyramid_wall_seconds >= 0.0 &&
                    serial_output.camera_pyramid_wall_seconds >= 0.0 &&
                    parallel_output.bundle_serialization_wall_seconds >= 0.0 &&
                    serial_output.bundle_serialization_wall_seconds >= 0.0 &&
                    parallel_consumed_depth.cameras[0].voting.depth_after_components[0].empty() &&
                    parallel_consumed_depth.cameras[1].voting.depth_after_components[0].empty() &&
                    serial_consumed_depth.cameras[0].voting.depth_after_components[0].empty() &&
                    serial_consumed_depth.cameras[1].voting.depth_after_components[0].empty(),
                "parallel or serial consuming OOC output differs from serial oracle");

        const auto weighted = metmodel::build_recovered_ooc_weighted_nodes_mode0(scene, bridged);
        require(weighted.root_scale > 10.0F && weighted.root_scale <= static_cast<float>(10.0 * 1.953846153846154) &&
                    weighted.alternate_scale == 10.0F && weighted.adaptive_maximum_level >= 1U &&
                    weighted.adaptive_maximum_level <= 32U && weighted.adaptive_winning_bin <= 31U &&
                    weighted.adaptive_total_samples != 0U,
                "recovered OOC weighted-node root scale mismatch");
        require(weighted.cameras.size() == 1U && weighted.cameras[0].camera_index == 0U &&
                    weighted.cameras[0].stable_camera_id == 17U,
                "recovered OOC weighted-node camera identity mismatch");
        require(weighted.cameras[0].candidate_count != 0U && weighted.cameras[0].local_node_count != 0U &&
                    !weighted.merged_nodes.empty(),
                "recovered OOC weighted-node candidate generation is empty");
        require(weighted.multi_camera_nodes.size() <= weighted.merged_nodes.size() &&
                    weighted.records_before_histogram.size() == weighted.balanced_nodes.size() &&
                    (weighted.multi_camera_nodes.empty() || !weighted.balanced_nodes.empty()),
                "recovered OOC weighted-node reduction sizes are inconsistent");

        const std::array<std::uint8_t, 1> captured_diagonal{1U};
        input.diagonal_policy = metmodel::RecoveredOocDiagonalPolicy::StrictCaptured;
        input.captured_diagonal_pixel_scale = captured_diagonal;
        const auto captured = metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0(scene, recovered, input);
        require(captured.manifest.diagonal_pixel_scale_used == std::vector<std::uint8_t>{1U},
                "strict-captured OOC diagonal policy mismatch");

        input.diagonal_policy = metmodel::RecoveredOocDiagonalPolicy::DeterministicEnabled;
        input.captured_diagonal_pixel_scale = {};
        const auto deterministic_enabled =
            metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0(scene, recovered, input);
        require(deterministic_enabled.manifest.diagonal_pixel_scale_used == std::vector<std::uint8_t>{1U},
                "deterministic-enabled OOC diagonal policy mismatch");

        input.diagonal_policy = metmodel::RecoveredOocDiagonalPolicy::StrictCaptured;
        input.captured_diagonal_pixel_scale = captured_diagonal;

        const auto rejected = [&](auto mutate)
        {
            auto invalid = input;
            mutate(invalid);
            try
            {
                (void)metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0(scene, recovered, invalid);
                return false;
            }
            catch (const std::exception&)
            {
                return true;
            }
        };
        require(rejected([](auto& value) { value.captured_diagonal_pixel_scale = {}; }),
                "strict-captured OOC policy accepted a missing capture value");
        const std::array<std::uint8_t, 1> invalid_captured_diagonal{2U};
        require(rejected([&](auto& value) { value.captured_diagonal_pixel_scale = invalid_captured_diagonal; }),
                "strict-captured OOC policy accepted a non-boolean value");
        require(rejected(
                    [&](auto& value)
                    {
                        value.diagonal_policy = metmodel::RecoveredOocDiagonalPolicy::DeterministicDisabled;
                        value.captured_diagonal_pixel_scale = captured_diagonal;
                    }),
                "deterministic OOC policy silently ignored a capture value");
        require(rejected([](auto& value) { value.volumetric_masks = true; }),
                "recovered OOC bridge accepted volumetric masks");
        require(rejected([](auto& value) { value.depth_downscale = 8U; }),
                "recovered d4 voting bridge accepted a different downscale");

        auto wrong_depth_size = recovered;
        wrong_depth_size.cameras[0].voting.depth_after_components[1].pop_back();
        bool wrong_depth_size_rejected = false;
        try
        {
            (void)metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0(scene, wrong_depth_size, input);
        }
        catch (const std::exception&)
        {
            wrong_depth_size_rejected = true;
        }
        require(wrong_depth_size_rejected, "recovered voting-to-OOC bridge accepted a truncated depth level");

        auto wrong_order = recovered;
        wrong_order.cameras[0].patchmatch.camera_index = 1U;
        bool wrong_order_rejected = false;
        try
        {
            (void)metmodel::build_recovered_d4_voting_to_ooc_bundle_mode0(scene, wrong_order, input);
        }
        catch (const std::exception&)
        {
            wrong_order_rejected = true;
        }
        require(wrong_order_rejected, "recovered voting-to-OOC bridge accepted a camera-order mismatch");
    }
    TEST(RecoveredModelReference, recovered_d4_voting_to_ooc_bridge)
    {
        if (!metmodel::ooc_pyramid_exact_encoder_available())
        {
            GTEST_SKIP() << "target-exact OOC serialization requires OpenEXR 3.2.2 and zlib 1.3.2";
        }
        EXPECT_NO_THROW(test_recovered_d4_voting_to_ooc_bridge());
    }
} // namespace
