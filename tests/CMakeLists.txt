if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

include(CheckLanguage)

# ── GTest ────────────────────────────────────────────────────────────────────
find_package(GTest REQUIRED)
include(GoogleTest)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Test Widgets Concurrent)
find_package(Python3 COMPONENTS Interpreter)

# ── 依赖 ─────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "5.0.0")
    find_package(OpenCV REQUIRED COMPONENTS core geometry stereo imgproc)
else()
    find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc)
endif()

# ── 测试数据根目录──────────────────────────────────────────────────────────────
set(TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/testData")
add_definitions(-DTEST_DATA_DIR="${TEST_DATA_DIR}")

add_executable(test_common_unicode_path_io test_common_unicode_path_io.cpp)
target_link_libraries(test_common_unicode_path_io PRIVATE
    plascan_common_io
    Qt6::Core
    ${OpenCV_LIBS}
    GTest::gtest_main
)
target_include_directories(test_common_unicode_path_io PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common
)
gtest_discover_tests(test_common_unicode_path_io)

add_executable(test_torchscript_model_resolver test_torchscript_model_resolver.cpp)
target_link_libraries(test_torchscript_model_resolver PRIVATE
    plascan_common_model
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_torchscript_model_resolver PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common
)
gtest_discover_tests(test_torchscript_model_resolver)

add_executable(test_mask_generation test_mask_generation.cpp)
target_link_libraries(test_mask_generation PRIVATE
    mask
    ${OpenCV_LIBS}
    GTest::gtest_main
)
target_include_directories(test_mask_generation PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/mask
)
gtest_discover_tests(test_mask_generation)

add_executable(test_source_contracts test_source_contracts.cpp)
target_link_libraries(test_source_contracts PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_compile_definitions(test_source_contracts PRIVATE
    PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
gtest_discover_tests(test_source_contracts)

add_executable(test_cli_contracts test_cli_contracts.cpp)
target_link_libraries(test_cli_contracts PRIVATE
    Qt6::Core
    ${PLASCAN_LIBZIP_TARGET}
    GTest::gtest_main
)
target_compile_definitions(test_cli_contracts PRIVATE
    PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
if(TARGET reconstruct_pipeline_cli)
    target_compile_definitions(test_cli_contracts PRIVATE
        PLASCAN_RECONSTRUCT_PIPELINE_CLI_PATH="$<TARGET_FILE:reconstruct_pipeline_cli>"
    )
    add_dependencies(test_cli_contracts reconstruct_pipeline_cli)
endif()
if(TARGET camera_convert_cli)
    target_compile_definitions(test_cli_contracts PRIVATE
        PLASCAN_CAMERA_CONVERT_CLI_PATH="$<TARGET_FILE:camera_convert_cli>"
    )
    add_dependencies(test_cli_contracts camera_convert_cli)
endif()
if(TARGET bundle_adjust_cli)
    target_compile_definitions(test_cli_contracts PRIVATE
        PLASCAN_BUNDLE_ADJUST_CLI_PATH="$<TARGET_FILE:bundle_adjust_cli>"
    )
    add_dependencies(test_cli_contracts bundle_adjust_cli)
endif()
if(TARGET feature_match_cli)
    target_compile_definitions(test_cli_contracts PRIVATE
        PLASCAN_FEATURE_MATCH_CLI_PATH="$<TARGET_FILE:feature_match_cli>"
    )
    add_dependencies(test_cli_contracts feature_match_cli)
endif()
if(TARGET dense_cloud_refine_cli)
    target_compile_definitions(test_cli_contracts PRIVATE
        PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH="$<TARGET_FILE:dense_cloud_refine_cli>"
    )
    add_dependencies(test_cli_contracts dense_cloud_refine_cli)
endif()
gtest_discover_tests(test_cli_contracts)

# ==============================================================================
# test_sfm_filter — 稀疏点云过滤器
# ==============================================================================
add_executable(test_sfm_filter test_sfm_filter.cpp)
target_link_libraries(test_sfm_filter PRIVATE
    sfm
    camera
    intersection
    bundle_adjust
    ${OpenCV_LIBS}
    GTest::gtest_main
)
target_include_directories(test_sfm_filter PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
)
gtest_discover_tests(test_sfm_filter)

# ==============================================================================
# test_sparse_point_cloud_processor — 稀疏点云后处理统一内核
# ==============================================================================
add_executable(test_sparse_point_cloud_processor test_sparse_point_cloud_processor.cpp)
target_link_libraries(test_sparse_point_cloud_processor PRIVATE
    sfm
    GTest::gtest_main
)
target_include_directories(test_sparse_point_cloud_processor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
)
gtest_discover_tests(test_sparse_point_cloud_processor)

# ==============================================================================
# test_sparse_point_cloud_workspace — SfM 与 plapoint 点云桥接
# ==============================================================================
add_executable(test_sparse_point_cloud_workspace test_sparse_point_cloud_workspace.cpp)
target_link_libraries(test_sparse_point_cloud_workspace PRIVATE
    sfm
    GTest::gtest_main
)
target_include_directories(test_sparse_point_cloud_workspace PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
)
gtest_discover_tests(test_sparse_point_cloud_workspace)

# ==============================================================================
# test_mvs_types — MVS 类型与参数验证
# ==============================================================================
add_executable(test_mvs_types test_mvs_types.cpp)
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
gtest_discover_tests(test_mvs_types)

# ==============================================================================
# test_patchmatch_cpu — CPU PatchMatch 深度恢复回归测试
# ==============================================================================
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(test_patchmatch_cpu test_patchmatch_cpu.cpp)
    set_target_properties(test_patchmatch_cpu PROPERTIES
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(test_patchmatch_cpu PRIVATE
        mvs
        camera
        ${OpenCV_LIBS}
        GTest::gtest_main
    )
    target_include_directories(test_patchmatch_cpu PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/mvs
        ${CMAKE_SOURCE_DIR}/src/core/camera
        ${CMAKE_SOURCE_DIR}/src/core
    )
    gtest_discover_tests(test_patchmatch_cpu)
else()
    message(WARNING "tests: CUDA compiler not found, skipping test_patchmatch_cpu")
endif()

# ==============================================================================
# test_sfm_params — SfM 参数收紧验证
# ==============================================================================
add_executable(test_sfm_params test_sfm_params.cpp)
target_link_libraries(test_sfm_params PRIVATE
    sfm
    camera
    intersection
    bundle_adjust
    ${OpenCV_LIBS}
    GTest::gtest_main
)
target_include_directories(test_sfm_params PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
)
gtest_discover_tests(test_sfm_params)

# ==============================================================================
# test_sfm_quality_report — SfM 稀疏重建质量报告
# ==============================================================================
add_executable(test_sfm_quality_report test_sfm_quality_report.cpp)
target_link_libraries(test_sfm_quality_report PRIVATE
    sfm
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_sfm_quality_report PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
)
gtest_discover_tests(test_sfm_quality_report)

# ==============================================================================
# test_sfm_pipeline — SFM 管线与 BA 离群点过滤集成测试
# ==============================================================================
add_executable(test_sfm_pipeline test_sfm_pipeline.cpp)
target_link_libraries(test_sfm_pipeline PRIVATE
    sfm
    camera
    intersection
    bundle_adjust
    GTest::gtest_main
)
target_include_directories(test_sfm_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
)
gtest_discover_tests(test_sfm_pipeline)

# ==============================================================================
# test_ba_input_builder — BA 输入构建与测绘控制点观测接入
# ==============================================================================
add_executable(test_ba_input_builder test_ba_input_builder.cpp)
target_link_libraries(test_ba_input_builder PRIVATE
    sfm
    camera
    intersection
    bundle_adjust
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_ba_input_builder PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/common
)
gtest_discover_tests(test_ba_input_builder)

# ==============================================================================
# test_sfm_pair_planner — SFM 已知相机匹配配对规划
# ==============================================================================
add_executable(test_sfm_pair_planner test_sfm_pair_planner.cpp)
target_link_libraries(test_sfm_pair_planner PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_sfm_pair_planner PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
    ${CMAKE_SOURCE_DIR}/src/core/pipeline
)
gtest_discover_tests(test_sfm_pair_planner)

add_executable(test_reconstruction_prerequisites
    test_reconstruction_prerequisites.cpp
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation/ReconstructionPrerequisiteReport.cpp
)
target_link_libraries(test_reconstruction_prerequisites PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_reconstruction_prerequisites PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
    ${CMAKE_SOURCE_DIR}/src/core/pipeline
)
gtest_discover_tests(test_reconstruction_prerequisites)

add_executable(test_match_result_catalog
    test_match_result_catalog.cpp
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation/MatchResultCatalog.cpp
)
target_link_libraries(test_match_result_catalog PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_match_result_catalog PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
    ${CMAKE_SOURCE_DIR}/src/core/pipeline
)
gtest_discover_tests(test_match_result_catalog)

add_executable(test_aerial_triangulation_workflow
    test_aerial_triangulation_workflow.cpp
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation/AerialTriangulationWorkflow.cpp
)
target_link_libraries(test_aerial_triangulation_workflow PRIVATE
    Qt6::Core
    sfm
    camera
    intersection
    bundle_adjust
    GTest::gtest_main
)
target_include_directories(test_aerial_triangulation_workflow PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
    ${CMAKE_SOURCE_DIR}/src/core/pipeline
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
)
gtest_discover_tests(test_aerial_triangulation_workflow)

add_executable(test_lightglue_feature_budget
    test_lightglue_feature_budget.cpp
    ${CMAKE_SOURCE_DIR}/src/core/feature_match/match.cpp
)
target_link_libraries(test_lightglue_feature_budget PRIVATE
    Qt6::Core
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
target_include_directories(test_lightglue_feature_budget PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/pipeline
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
)
gtest_discover_tests(test_lightglue_feature_budget)

add_executable(test_guided_rematch_service
    test_guided_rematch_service.cpp
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation/GuidedRematchService.cpp
)
target_link_libraries(test_guided_rematch_service PRIVATE
    Qt6::Core
    ${OpenCV_LIBS}
    GTest::gtest_main
)
target_include_directories(test_guided_rematch_service PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
    ${CMAKE_SOURCE_DIR}/src/core/pipeline
    ${CMAKE_SOURCE_DIR}/src/core/sfm
)
gtest_discover_tests(test_guided_rematch_service)

add_executable(test_feature_pair_planner
    test_feature_pair_planner.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/FeaturePairPlanner.cpp
)
target_link_libraries(test_feature_pair_planner PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_feature_pair_planner PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
    ${CMAKE_SOURCE_DIR}/src/core/pipeline
)
gtest_discover_tests(test_feature_pair_planner)

# ==============================================================================
# test_match_photos_task — Metashape-like 匹配照片任务框架
# ==============================================================================
add_executable(test_matchphotos_algorithm_selector
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/tests/test_algorithm_selector.cpp
)
target_link_libraries(test_matchphotos_algorithm_selector PRIVATE
    matchphototask
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_matchphotos_algorithm_selector PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/algorithm
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/task
)
gtest_discover_tests(test_matchphotos_algorithm_selector)

add_executable(test_matchphotos_pair_selector
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/tests/test_pair_selector.cpp
)
target_link_libraries(test_matchphotos_pair_selector PRIVATE
    matchphototask
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_matchphotos_pair_selector PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/pair_selection
)
gtest_discover_tests(test_matchphotos_pair_selector)

add_executable(test_match_photos_task
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/tests/test_match_photos_task.cpp
)
target_link_libraries(test_match_photos_task PRIVATE
    matchphototask
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_match_photos_task PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/task
)
gtest_discover_tests(test_match_photos_task)

add_executable(test_tie_point_track_manager
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/tests/test_tie_point_track_manager.cpp
)
target_link_libraries(test_tie_point_track_manager PRIVATE
    matchphototask
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_tie_point_track_manager PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/tie_points
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/task
)
gtest_discover_tests(test_tie_point_track_manager)

# ==============================================================================
# test_initial_sparse_triangulator — core/sfm 初始稀疏点云三角化器
# ==============================================================================
add_executable(test_initial_sparse_triangulator test_initial_sparse_triangulator.cpp)
target_link_libraries(test_initial_sparse_triangulator PRIVATE
    sfm
    camera
    intersection
    bundle_adjust
    GTest::gtest_main
)
target_include_directories(test_initial_sparse_triangulator PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
)
gtest_discover_tests(test_initial_sparse_triangulator)

# ==============================================================================
# test_vocabulary_overlap_retriever — 基于特征词汇的重叠对预检索
# ==============================================================================
add_executable(test_vocabulary_overlap_retriever test_vocabulary_overlap_retriever.cpp)
target_link_libraries(test_vocabulary_overlap_retriever PRIVATE
    overlap
    ${OpenCV_LIBS}
    GTest::gtest_main
)
target_include_directories(test_vocabulary_overlap_retriever PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/overlap
)
gtest_discover_tests(test_vocabulary_overlap_retriever)

# ==============================================================================
# test_overlap_pair_graph_planner — 无相机 Generic/BoW 候选图连通性规划
# ==============================================================================
add_executable(test_overlap_pair_graph_planner test_overlap_pair_graph_planner.cpp)
target_link_libraries(test_overlap_pair_graph_planner PRIVATE
    overlap
    GTest::gtest_main
)
target_include_directories(test_overlap_pair_graph_planner PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/overlap
)
gtest_discover_tests(test_overlap_pair_graph_planner)

# test_overlap_analyzer — 基于相机模型的重叠对估计

add_executable(test_overlap_analyzer test_overlap_analyzer.cpp)
target_link_libraries(test_overlap_analyzer PRIVATE
    overlap
    GTest::gtest_main
)
target_include_directories(test_overlap_analyzer PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/overlap
)
gtest_discover_tests(test_overlap_analyzer)

# ==============================================================================
# test_lidar_constraint_map — LiDAR 平面约束地图读取与索引
# ==============================================================================
add_executable(test_lidar_constraint_map test_lidar_constraint_map.cpp)
target_link_libraries(test_lidar_constraint_map PRIVATE
    lidar
    GTest::gtest_main
)
target_include_directories(test_lidar_constraint_map PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/lidar
    ${CMAKE_SOURCE_DIR}/src/core/overlap
)
gtest_discover_tests(test_lidar_constraint_map)

# ==============================================================================
# test_lidar_constraint_association — BA track 与 LiDAR 平面约束关联
# ==============================================================================
add_executable(test_lidar_constraint_association test_lidar_constraint_association.cpp)
target_link_libraries(test_lidar_constraint_association PRIVATE
    lidar
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_lidar_constraint_association PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/lidar
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_lidar_constraint_association)

# ==============================================================================
# test_bundle_adjust_lidar_constraints — BA LiDAR 点到面残差
# ==============================================================================
add_executable(test_bundle_adjust_lidar_constraints test_bundle_adjust_lidar_constraints.cpp)
target_link_libraries(test_bundle_adjust_lidar_constraints PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_lidar_constraints PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_lidar_constraints)

# ==============================================================================
# test_bundle_adjust_projection — BA AutoDiff 投影模型与 Camera 投影一致性
# ==============================================================================
add_executable(test_bundle_adjust_projection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_projection.cpp
)
target_link_libraries(test_bundle_adjust_projection PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_projection PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_projection)

# ==============================================================================
# test_bundle_adjust_ceres_backend — BA Ceres/GPU 后端选择与回退
# ==============================================================================
if(NOT TARGET Ceres::ceres)
    find_package(Ceres CONFIG QUIET)
endif()
if(TARGET Ceres::ceres)
    add_executable(test_bundle_adjust_ceres_backend
        ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_ceres_backend.cpp
    )
    target_link_libraries(test_bundle_adjust_ceres_backend PRIVATE
        bundle_adjust
        camera
        GTest::gtest_main
    )
    target_include_directories(test_bundle_adjust_ceres_backend PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
        ${CMAKE_SOURCE_DIR}/src/core/camera
    )
    gtest_discover_tests(test_bundle_adjust_ceres_backend)
else()
    message(STATUS "tests: Ceres 未找到，跳过 test_bundle_adjust_ceres_backend")
endif()

add_executable(test_bundle_adjust_backend_selection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp
)
target_link_libraries(test_bundle_adjust_backend_selection PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_backend_selection PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_backend_selection)

add_executable(test_bundle_adjust_native_cuda_workset
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_workset.cpp
)
target_link_libraries(test_bundle_adjust_native_cuda_workset PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_native_cuda_workset PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_native_cuda_workset)

add_executable(test_bundle_adjust_native_cuda_math
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_math.cpp
)
target_link_libraries(test_bundle_adjust_native_cuda_math PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_native_cuda_math PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_native_cuda_math)

add_executable(test_bundle_adjust_quality_gate
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_quality_gate.cpp
)
target_link_libraries(test_bundle_adjust_quality_gate PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_quality_gate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_quality_gate)

# ==============================================================================
# test_reference_terrain_prior — DEM/LiDAR height surface soft priors for BA
# ==============================================================================
add_executable(test_reference_terrain_prior test_reference_terrain_prior.cpp)
target_link_libraries(test_reference_terrain_prior PRIVATE
    sfm
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_reference_terrain_prior PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_reference_terrain_prior)

# ==============================================================================
# test_bundle_adjust_service_lidar — 服务层 LiDAR 约束接入与 JSON 输出
# ==============================================================================
add_executable(test_bundle_adjust_service_lidar
    test_bundle_adjust_service_lidar.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/services/BundleAdjustService.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectSupportUtils.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/io/ProjectIO.cpp
    ${CMAKE_SOURCE_DIR}/src/common/log/Logger.cpp
)
target_link_libraries(test_bundle_adjust_service_lidar PRIVATE
    lidar
    bundle_adjust
    camera
    Qt6::Core
    Qt6::Gui
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_service_lidar PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/project/services
    ${CMAKE_SOURCE_DIR}/src/gui/project/support
    ${CMAKE_SOURCE_DIR}/src/gui/project/io
    ${CMAKE_SOURCE_DIR}/src/core/lidar
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/common/log
    ${CMAKE_SOURCE_DIR}/src/common
)
gtest_discover_tests(test_bundle_adjust_service_lidar)

set(PLASCAN_QT_PLATFORM_PLUGIN_PATH
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/Qt6/plugins/platforms")

# ==============================================================================
# test_bundle_adjust_dialog_lidar — GUI LiDAR 约束参数入口
# ==============================================================================
add_executable(test_bundle_adjust_dialog_lidar
    test_bundle_adjust_dialog_lidar.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/BundleAdjustDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/BundleAdjustDialog.ui
)
set_target_properties(test_bundle_adjust_dialog_lidar PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
)
target_link_libraries(test_bundle_adjust_dialog_lidar PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test
    Qt6::Widgets
    Qt6::Concurrent
    GTest::gtest
)
target_compile_definitions(test_bundle_adjust_dialog_lidar PRIVATE
    PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_include_directories(test_bundle_adjust_dialog_lidar PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs
)
gtest_discover_tests(test_bundle_adjust_dialog_lidar
    DISCOVERY_TIMEOUT 60
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        ENVIRONMENT
            "QT_QPA_PLATFORM=offscreen;QT_QPA_PLATFORM_PLUGIN_PATH=${PLASCAN_QT_PLATFORM_PLUGIN_PATH}")

# ==============================================================================
# test_project_data — ProjectData 持久化契约与 core/results 分域
# ==============================================================================
add_executable(test_project_data
    test_project_data.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/archive/PlascanArchive.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/data/ProjectData.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/data/ProjectFilesManager.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/io/ProjectIO.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/JsonMergeUtil.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/ProjectConfigManager.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/ProjectUiConfigManager.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/ProjectWorkflowConfigManager.cpp
    ${CMAKE_SOURCE_DIR}/src/common/log/Logger.cpp
)
set_target_properties(test_project_data PROPERTIES
    AUTOMOC ON
)
target_link_libraries(test_project_data PRIVATE
    Qt6::Core
    ${PLASCAN_LIBZIP_TARGET}
    GTest::gtest_main
)
target_include_directories(test_project_data PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/log
    ${CMAKE_SOURCE_DIR}/src/gui/config
    ${CMAKE_SOURCE_DIR}/src/gui/project/archive
    ${CMAKE_SOURCE_DIR}/src/gui/project/data
    ${CMAKE_SOURCE_DIR}/src/gui/project/io
)
gtest_discover_tests(test_project_data)

# ==============================================================================
# test_gui_project_utils — GUI project 工具与三角化服务
# ==============================================================================
add_executable(test_gui_project_utils
    test_gui_project_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/FeatureExtractionDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/FeatureExtractionDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/FeaturePairPlanner.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/FeatureMatchingDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/FeatureMatchingDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/InitCameraPoseDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/InitCameraPoseDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/AerialTriangulationDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/AerialTriangulationDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/ThreeDReconstructionDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/ThreeDReconstructionDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/SparseCloudPostProcessDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/SparseCloudPostProcessDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/MapProjectDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/MapProjectDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/BundleAdjustDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/BundleAdjustDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/AboutDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/GenerateMaskDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/SurveyControlDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/CreateTiePointsDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/ThinTiePointsDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/CleanTiePointsDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/menu/MainMenu.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/main_window/ModelDropSupport.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/archive/PlascanArchive.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/data/ProjectData.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/data/ProjectFilesManager.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/io/ProjectIO.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectDashboardSummary.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectDenseWorkflowConfig.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectMetadataOperations.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectSupportUtils.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectWorkflowUtils.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectReferenceDatasets.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectReferenceTerrainBa.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectSurveyControl.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectWorkflowReports.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/services/ProjectCameraImportService.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectResultRecords.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/JsonMergeUtil.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/PythonRuntimeBinding.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/ProjectConfigManager.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/ProjectUiConfigManager.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/ProjectWorkflowConfigManager.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerImageLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/PhotoStripWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/ProjectDashboardWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/TaskStatusWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/WindowPanel.cpp
)
set_target_properties(test_gui_project_utils PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
)
if(MSVC)
    target_compile_options(test_gui_project_utils PRIVATE /bigobj)
endif()
target_link_libraries(test_gui_project_utils PRIVATE
    mvs
    sfm
    camera
    intersection
    bundle_adjust
    qc
    terrain
    mask
    Qt6::Core
    Qt6::Concurrent
    Qt6::Gui
    Qt6::Test
    Qt6::Widgets
    ${OpenCV_LIBS}
    ${PLASCAN_LIBZIP_TARGET}
    ${PLASCAN_GDAL_TARGET}
    plascan_common_io
    plascan_common_model
    GTest::gtest
)
target_compile_definitions(test_gui_project_utils PRIVATE
    PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_include_directories(test_gui_project_utils PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/config
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs
    ${CMAKE_SOURCE_DIR}/src/gui/menu
    ${CMAKE_SOURCE_DIR}/src/gui/main_window
    ${CMAKE_SOURCE_DIR}/src/gui/project
    ${CMAKE_SOURCE_DIR}/src/gui/project/archive
    ${CMAKE_SOURCE_DIR}/src/gui/project/data
    ${CMAKE_SOURCE_DIR}/src/gui/project/io
    ${CMAKE_SOURCE_DIR}/src/gui/project/manager
    ${CMAKE_SOURCE_DIR}/src/gui/project/services
    ${CMAKE_SOURCE_DIR}/src/gui/project/support
    ${CMAKE_SOURCE_DIR}/src/gui/views
    ${CMAKE_SOURCE_DIR}/src/gui/widgets
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/feature_match
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/qc
    ${CMAKE_SOURCE_DIR}/src/core/terrain
    ${CMAKE_SOURCE_DIR}/src/core/mvs
    ${CMAKE_SOURCE_DIR}/src/core/mask
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
    ${CMAKE_SOURCE_DIR}/src/core/pipeline
    ${CMAKE_SOURCE_DIR}/src/core
    ${OpenCV_INCLUDE_DIRS}
)
set(PLASCAN_QT_PLATFORM_PLUGIN_PATH
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/Qt6/plugins/platforms")
gtest_discover_tests(test_gui_project_utils
    DISCOVERY_TIMEOUT 60
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        ENVIRONMENT
            "QT_QPA_PLATFORM=offscreen;QT_QPA_PLATFORM_PLUGIN_PATH=${PLASCAN_QT_PLATFORM_PLUGIN_PATH}")

add_executable(test_layer_renderer_batched_overlay
    test_layer_renderer_batched_overlay.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerFeatureLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerImageLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerOverlayItems.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerRenderer.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerStitchedDebug.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/io/ProjectIO.cpp
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/FeatureData.cpp
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/FeatureFileIO.cpp
    ${CMAKE_SOURCE_DIR}/src/common/log/Logger.cpp
)
set_target_properties(test_layer_renderer_batched_overlay PROPERTIES
    AUTOMOC ON
)
target_link_libraries(test_layer_renderer_batched_overlay PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${PLASCAN_GDAL_TARGET}
    plascan_common_io
    mask
    GTest::gtest
)
target_include_directories(test_layer_renderer_batched_overlay PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/views
    ${CMAKE_SOURCE_DIR}/src/gui/project/io
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/common/log
    ${CMAKE_SOURCE_DIR}/src/core/mask
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/superpoint
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
if(GDAL_INCLUDE_DIRS)
    target_include_directories(test_layer_renderer_batched_overlay PRIVATE ${GDAL_INCLUDE_DIRS})
endif()
add_test(NAME test_layer_renderer_batched_overlay
         COMMAND test_layer_renderer_batched_overlay)

add_executable(test_multiview_track_builder test_multiview_track_builder.cpp)
target_link_libraries(test_multiview_track_builder PRIVATE sfm GTest::gtest_main)
target_include_directories(test_multiview_track_builder PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/sfm
)
gtest_discover_tests(test_multiview_track_builder)

if(Python3_Interpreter_FOUND)
    add_test(
        NAME FullPipelineScriptTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_full_pipeline_script
    )
    set_tests_properties(FullPipelineScriptTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME FullPipelineEntrypointTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_full_pipeline_entrypoint
    )
    set_tests_properties(FullPipelineEntrypointTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME RepoHygieneTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_repo_hygiene
    )
    set_tests_properties(RepoHygieneTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME PythonRuntimeSetupTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_setup_python_runtime
    )
    set_tests_properties(PythonRuntimeSetupTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME InstallSam21ModelScriptTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_install_sam21_model
    )
    set_tests_properties(InstallSam21ModelScriptTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME DownloadPhotogrammetryTestDataTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_download_photogrammetry_testdata
    )
    set_tests_properties(DownloadPhotogrammetryTestDataTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME PreparePhotogrammetryBenchmarksTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_prepare_photogrammetry_benchmarks
    )
    set_tests_properties(PreparePhotogrammetryBenchmarksTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME ValidateLidarBaInputsTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_validate_lidar_ba_inputs
    )
    set_tests_properties(ValidateLidarBaInputsTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME PrepareLidarBaAbBenchmarkTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_prepare_lidar_ba_ab_benchmark
    )
    set_tests_properties(PrepareLidarBaAbBenchmarkTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME CompareLidarBaAbResultsTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_compare_lidar_ba_ab_results
    )
    set_tests_properties(CompareLidarBaAbResultsTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME PhotogrammetryBenchmarkRunnerTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_photogrammetry_benchmark_runner
    )
    set_tests_properties(PhotogrammetryBenchmarkRunnerTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME RunLightGlueScriptTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_run_lightglue_script
    )
    set_tests_properties(RunLightGlueScriptTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME DiskTorchscriptModelTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_disk_torchscript_model
    )
    set_tests_properties(DiskTorchscriptModelTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME LightGlueTorchscriptModelTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_lightglue_torchscript_models
    )
    set_tests_properties(LightGlueTorchscriptModelTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME PrepareMunFrlLidarBaProjectTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_prepare_mun_frl_lidar_ba_project
    )
    set_tests_properties(PrepareMunFrlLidarBaProjectTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_test(
        NAME EstimateLidarNormalsTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_estimate_lidar_normals
    )
    set_tests_properties(EstimateLidarNormalsTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

if(TARGET triangulate_cli)
    add_executable(test_triangulate_cli test_triangulate_cli.cpp)
    target_link_libraries(test_triangulate_cli PRIVATE
        Qt6::Core
        ${OpenCV_LIBS}
        GTest::gtest_main
    )
    target_compile_definitions(test_triangulate_cli PRIVATE
        PLASCAN_TRIANGULATE_CLI_PATH="$<TARGET_FILE:triangulate_cli>"
    )
    add_dependencies(test_triangulate_cli triangulate_cli)
    gtest_discover_tests(test_triangulate_cli)
endif()

# ==============================================================================
# test_mvs_pipeline — MVS 管道集成测试（CPU PatchMatch → 融合 → 反投影）
# ==============================================================================
if(CMAKE_CUDA_COMPILER)
    add_executable(test_mvs_pipeline test_mvs_pipeline.cpp)
    set_target_properties(test_mvs_pipeline PROPERTIES
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
    )
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
else()
    message(WARNING "tests: CUDA compiler not found, skipping test_mvs_pipeline")
endif()

# ==============================================================================
# test_mesh_reconstructor — 点云到网格重建降级路径
# ==============================================================================
add_executable(test_mesh_reconstructor test_mesh_reconstructor.cpp)
target_link_libraries(test_mesh_reconstructor PRIVATE
    meshing
    plapoint::plapoint
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_mesh_reconstructor PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/mesh
)
gtest_discover_tests(test_mesh_reconstructor)

# pointcloud 模块测试迁移到 src/core/pointcloud/tests，
# 不再在全局 tests 目录内维护该测试源码。

# ==============================================================================
# test_feature_extractor_suffix — 多提取器后缀映射 + FeatureFileIO + 工厂
# ==============================================================================
add_executable(test_feature_extractor_suffix test_feature_extractor_suffix.cpp)
target_link_libraries(test_feature_extractor_suffix PRIVATE
    feature_extractors_traditional
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
target_include_directories(test_feature_extractor_suffix PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/compat
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/superpoint
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/disk
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/aliked
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/tradition
)
gtest_discover_tests(test_feature_extractor_suffix)

add_executable(test_feature_file_io
    test_feature_file_io.cpp
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/FeatureData.cpp
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/FeatureFileIO.cpp
)
target_link_libraries(test_feature_file_io PRIVATE
    Qt6::Core
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
target_include_directories(test_feature_file_io PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
)
gtest_discover_tests(test_feature_file_io)
