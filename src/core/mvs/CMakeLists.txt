cmake_minimum_required(VERSION 3.16)
project(mvs LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)

# ── CUDA（可选）──────────────────────────────────────────────────────────────
# CUDA 已在根 CMakeLists 中全局启用, 直接判断即可
if(DEFINED CMAKE_CUDA_COMPILER AND CMAKE_CUDA_COMPILER)
    set(MVS_HAS_CUDA TRUE)
    message(STATUS "mvs: CUDA 已启用 (${CMAKE_CUDA_COMPILER})")
else()
    set(MVS_HAS_CUDA FALSE)
    message(STATUS "mvs: CUDA 不可用，将使用 CPU 回退路径")
endif()

if(PLASCAN_ENABLE_OPENCL AND TARGET OpenCL::OpenCL)
    set(MVS_HAS_OPENCL TRUE)
    message(STATUS "mvs: OpenCL GPU backend enabled")
else()
    set(MVS_HAS_OPENCL FALSE)
    message(STATUS "mvs: OpenCL GPU backend unavailable")
endif()

# ── 源文件 ────────────────────────────────────────────────────────────────────
set(MVS_CXX_SOURCES
    AdaptiveGeometryEvidencePolicy.cpp
    DenseCloudArtifactValidation.cpp
    DenseCloudRefinementService.cpp
    DenseCloudQualityFilter.cpp
    DepthCompletenessMetrics.cpp
    DepthGapTargetedRecovery.cpp
    DepthResidualReestimation.cpp
    DepthMissingReason.cpp
    DepthMemoryPolicy.cpp
    DepthProvenance.cpp
    DepthConsistencyCache.cpp
    DepthConsistencyEvidencePolicy.cpp
    DepthConsensusBiasPolicy.cpp
    DepthAnchoredHoleInterpolator.cpp
    DepthCrossViewHoleRepair.cpp
    DepthGeometryConsistency.cpp
    DepthPoseAlignmentRefiner.cpp
    DepthPoseRefinementStage.cpp
    DepthMapFusion.cpp
    DepthFrameQualityGate.cpp
    DepthPyramidPolicy.cpp
    DepthPyramidEstimator.cpp
    DepthPyramidPropagation.cpp
    DepthFrameUtils.cpp
    DepthComputeScheduler.cpp
    GpuDeviceLease.cpp
    MvsSourcePlanner.cpp
    MvsVisibilityGraphBuilder.cpp
    MvsQualityReport.cpp
    MvsSceneClassifier.cpp
    MvsImageCache.cpp
    MvsImageFrame.cpp
    MvsImageMetadataProbe.cpp
    MvsImagePreprocessor.cpp
    MvsWorkspaceManifest.cpp
    MvsWorkspaceReplay.cpp
    PatchMatchCPU.cpp
    PatchMatchEstimator.cpp
    DenseCloudBuilder.cpp
    DepthMapGenerator.cpp
    EpipolarRectifier.cpp
    SparseCloudPreprocessor.cpp
    DisparityTriangulator.cpp
    PointCloudArtifactIO.cpp
    StreamingDepthFusionService.cpp
)

set(MVS_HEADERS
    AdaptiveGeometryEvidencePolicy.h
    DenseCloudArtifactValidation.h
    DenseCloudRefinementService.h
    DenseCloudQualityFilter.h
    DepthCompletenessMetrics.h
    DepthGapTargetedRecovery.h
    DepthResidualReestimation.h
    DepthMissingReason.h
    DepthMemoryPolicy.h
    DepthProvenance.h
    DepthConsistencyCache.h
    DepthConsistencyEvidencePolicy.h
    DepthConsensusBiasPolicy.h
    DepthAnchoredHoleInterpolator.h
    DepthCrossViewHoleRepair.h
    DepthGeometryConsistency.h
    DepthPoseAlignmentRefiner.h
    DepthPoseRefinementStage.h
    MvsTypes.h
    MvsSourcePlanner.h
    MvsVisibilityGraphBuilder.h
    MvsQualityReport.h
    MvsSceneClassifier.h
    MvsImageCache.h
    MvsImageMetadataProbe.h
    MvsImagePreprocessor.h
    MvsViewSelection.h
    MvsWorkspaceManifest.h
    MvsWorkspaceReplay.h
    PatchMatchCUDA.h
    DepthMapFusion.h
    DepthFrameQualityGate.h
    DepthPyramidPolicy.h
    DepthPyramidEstimator.h
    DepthPyramidPropagation.h
    DepthFrameUtils.h
    DepthComputeScheduler.h
    GpuDeviceLease.h
    DenseCloudBuilder.h
    DepthMapGenerator.h
    PatchMatchPhotometricCost.h
    DensePointCloudCUDA.h
    EpipolarRectifier.h
    SparseCloudPreprocessor.h
    DisparityTriangulator.h
    PointCloudArtifactIO.h
    StreamingDepthFusionService.h
)

set(MVS_QOBJECT_HEADERS
    DepthMapGenerator.h
)

qt6_wrap_cpp(MVS_MOC_SOURCES
    ${MVS_QOBJECT_HEADERS}
)

if(MVS_HAS_CUDA)
    set(MVS_CUDA_SOURCES
        PatchMatchCUDA.cu
        DensePointCloudCUDA.cu
    )
else()
    set(MVS_CUDA_SOURCES
        PatchMatchNoCUDA.cpp
    )
endif()

if(MVS_HAS_OPENCL)
    list(APPEND MVS_CXX_SOURCES PatchMatchOpenCL.cpp)
else()
    list(APPEND MVS_CXX_SOURCES PatchMatchNoOpenCL.cpp)
endif()

add_library(mvs STATIC
    ${MVS_CXX_SOURCES}
    ${MVS_CUDA_SOURCES}
    ${MVS_HEADERS}
    ${MVS_MOC_SOURCES}
)

if(MVS_HAS_CUDA)
    set_target_properties(mvs PROPERTIES
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_definitions(mvs PUBLIC MVS_ENABLE_CUDA=1)
    target_compile_options(mvs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    )
    if(MSVC)
        # NVCC misdiagnoses Qt6 comparison conversion operators and exported
        # value members while parsing the host side of these CUDA units.
        target_compile_options(mvs PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=940,1394>
        )
    endif()
endif()

if(MVS_HAS_OPENCL)
    target_compile_definitions(mvs PUBLIC MVS_ENABLE_OPENCL=1)
    target_link_libraries(mvs PRIVATE OpenCL::OpenCL)
endif()

target_include_directories(mvs PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/src/common
)

if(GDAL_INCLUDE_DIRS)
    target_include_directories(mvs PRIVATE ${GDAL_INCLUDE_DIRS})
endif()

target_link_libraries(mvs
    PUBLIC
        camera
        plascan_common_log
        plascan_common_io
        plapoint::plapoint
        Qt6::Core
    PRIVATE
        plascan_common_concurrency
        plascan_common_string_utils
        Qt6::Concurrent
        ${OpenCV_LIBS}
        ${PLASCAN_GDAL_TARGET}
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(mvs PRIVATE HAS_OPENMP)
endif()

if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        include(GoogleTest)
        add_executable(test_mvs_rectifier_unit
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_rectifier.cpp
        )
        target_link_libraries(test_mvs_rectifier_unit PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_rectifier_unit PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
            ${CMAKE_SOURCE_DIR}/src/core/camera
        )
        gtest_discover_tests(test_mvs_rectifier_unit)

        add_executable(test_mvs_image_preprocessor
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_image_preprocessor.cpp
        )
        target_link_libraries(test_mvs_image_preprocessor PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_image_preprocessor PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_image_preprocessor)

        add_executable(test_mvs_image_cache
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_image_cache.cpp
        )
        set_target_properties(test_mvs_image_cache PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_image_cache PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_image_cache PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_image_cache DISCOVERY_MODE PRE_TEST)

        add_executable(test_mvs_depth_pyramid
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_depth_pyramid.cpp
        )
        target_link_libraries(test_mvs_depth_pyramid PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_depth_pyramid PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_depth_pyramid)

        add_executable(test_mvs_depth_completeness
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_depth_completeness.cpp
        )
        set_target_properties(test_mvs_depth_completeness PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_depth_completeness PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_depth_completeness PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_depth_completeness)

        add_executable(test_mvs_depth_gap_targeted_recovery
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_depth_gap_targeted_recovery.cpp
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_depth_residual_reestimation.cpp
        )
        set_target_properties(test_mvs_depth_gap_targeted_recovery PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_depth_gap_targeted_recovery PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_depth_gap_targeted_recovery PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_depth_gap_targeted_recovery)

        add_executable(test_mvs_depth_missing_reason
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_depth_missing_reason.cpp
        )
        set_target_properties(test_mvs_depth_missing_reason PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_depth_missing_reason PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_depth_missing_reason PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_depth_missing_reason)

        add_executable(test_mvs_depth_provenance
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_depth_provenance.cpp
        )
        set_target_properties(test_mvs_depth_provenance PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_depth_provenance PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_depth_provenance PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_depth_provenance)

        add_executable(test_mvs_depth_consistency_evidence
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_depth_consistency_evidence.cpp
        )
        set_target_properties(test_mvs_depth_consistency_evidence PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_depth_consistency_evidence PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_depth_consistency_evidence PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_depth_consistency_evidence)

        add_executable(test_mvs_adaptive_geometry_evidence_policy
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_adaptive_geometry_evidence_policy.cpp
        )
        set_target_properties(test_mvs_adaptive_geometry_evidence_policy PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_adaptive_geometry_evidence_policy PRIVATE
            mvs
            GTest::gtest_main
        )
        target_include_directories(test_mvs_adaptive_geometry_evidence_policy PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_adaptive_geometry_evidence_policy)

        add_executable(test_mvs_depth_memory_policy
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_depth_memory_policy.cpp
        )
        set_target_properties(test_mvs_depth_memory_policy PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_depth_memory_policy PRIVATE
            mvs
            GTest::gtest_main
        )
        target_include_directories(test_mvs_depth_memory_policy PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_depth_memory_policy)

        add_executable(test_mvs_depth_pose_alignment
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_depth_pose_alignment.cpp
        )
        set_target_properties(test_mvs_depth_pose_alignment PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_depth_pose_alignment PRIVATE
            mvs
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_depth_pose_alignment PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_depth_pose_alignment DISCOVERY_MODE PRE_TEST)

        add_executable(test_mvs_cross_view_hole_repair
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_cross_view_hole_repair.cpp
        )
        set_target_properties(test_mvs_cross_view_hole_repair PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_cross_view_hole_repair PRIVATE
            mvs
            camera
            ${OpenCV_LIBS}
            GTest::gtest_main
        )
        target_include_directories(test_mvs_cross_view_hole_repair PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
            ${CMAKE_SOURCE_DIR}/src/core/camera
        )
        gtest_discover_tests(test_mvs_cross_view_hole_repair DISCOVERY_MODE PRE_TEST)

        add_executable(test_mvs_types
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_types.cpp
        )
        set_target_properties(test_mvs_types PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_mvs_types PRIVATE
            mvs
            camera
            ${OpenCV_LIBS}
            GTest::gtest_main
        )
        target_include_directories(test_mvs_types PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
            ${CMAKE_SOURCE_DIR}/src/core/camera
            ${CMAKE_SOURCE_DIR}/src/core
        )
        target_compile_definitions(test_mvs_types PRIVATE
            TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
        )
        gtest_discover_tests(test_mvs_types)

        add_executable(test_depth_compute_scheduler
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_depth_compute_scheduler.cpp
        )
        set_target_properties(test_depth_compute_scheduler PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
        target_link_libraries(test_depth_compute_scheduler PRIVATE
            mvs
            GTest::gtest_main
        )
        target_include_directories(test_depth_compute_scheduler PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_depth_compute_scheduler DISCOVERY_MODE PRE_TEST)

        add_executable(test_mvs_pipeline
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_pipeline.cpp
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_depth_frame_quality_gate.cpp
        )
        if(CMAKE_CUDA_COMPILER)
            set_target_properties(test_mvs_pipeline PROPERTIES
                CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
            )
        else()
            set_target_properties(test_mvs_pipeline PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
            )
        endif()
        target_link_libraries(test_mvs_pipeline PRIVATE
            mvs
            camera
            Qt6::Core
            ${OpenCV_LIBS}
            GTest::gtest_main
        )
        target_include_directories(test_mvs_pipeline PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
            ${CMAKE_SOURCE_DIR}/src/core/camera
            ${CMAKE_SOURCE_DIR}/src/core
        )
        gtest_discover_tests(test_mvs_pipeline)

        add_executable(test_mvs_workspace_manifest
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_workspace_manifest.cpp
        )
        target_link_libraries(test_mvs_workspace_manifest PRIVATE
            mvs
            GTest::gtest_main
            Qt6::Core
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_workspace_manifest PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_workspace_manifest)

        add_executable(test_mvs_source_planner
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_source_planner.cpp
        )
        target_link_libraries(test_mvs_source_planner PRIVATE
            mvs
            GTest::gtest_main
            Qt6::Core
            ${OpenCV_LIBS}
        )
        target_include_directories(test_mvs_source_planner PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_source_planner)

        add_executable(test_mvs_visibility_graph_builder
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_visibility_graph_builder.cpp
        )
        target_link_libraries(test_mvs_visibility_graph_builder PRIVATE
            mvs
            GTest::gtest_main
        )
        target_include_directories(test_mvs_visibility_graph_builder PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_visibility_graph_builder)

        add_executable(test_dense_cloud_quality_filter
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_dense_cloud_quality_filter.cpp
        )
        target_link_libraries(test_dense_cloud_quality_filter PRIVATE
            mvs
            GTest::gtest_main
        )
        target_include_directories(test_dense_cloud_quality_filter PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_dense_cloud_quality_filter)

        add_executable(test_mvs_dense_cloud_refinement
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_dense_cloud_refinement.cpp
        )
        target_link_libraries(test_mvs_dense_cloud_refinement PRIVATE
            mvs
            GTest::gtest_main
            Qt6::Core
        )
        target_include_directories(test_mvs_dense_cloud_refinement PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_dense_cloud_refinement DISCOVERY_MODE PRE_TEST)

        add_executable(test_mvs_streaming_depth_fusion
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_mvs_streaming_depth_fusion.cpp
        )
        target_link_libraries(test_mvs_streaming_depth_fusion PRIVATE
            mvs
            GTest::gtest_main
        )
        target_include_directories(test_mvs_streaming_depth_fusion PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_mvs_streaming_depth_fusion DISCOVERY_MODE PRE_TEST)

        add_executable(test_patchmatch_photometric_cost
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_patchmatch_photometric_cost.cpp
        )
        target_link_libraries(test_patchmatch_photometric_cost PRIVATE
            GTest::gtest_main
        )
        target_include_directories(test_patchmatch_photometric_cost PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_patchmatch_photometric_cost)

        add_executable(test_patchmatch_mask_aware
            ${CMAKE_SOURCE_DIR}/src/core/mvs/tests/test_patchmatch_mask_aware.cpp
        )
        target_link_libraries(test_patchmatch_mask_aware PRIVATE
            mvs
            GTest::gtest_main
        )
        target_include_directories(test_patchmatch_mask_aware PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/mvs
        )
        gtest_discover_tests(test_patchmatch_mask_aware DISCOVERY_MODE PRE_TEST)
    endif()
endif()
