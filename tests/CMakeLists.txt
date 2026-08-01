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

add_executable(test_source_contracts test_source_contracts.cpp)
target_link_libraries(test_source_contracts PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_compile_definitions(test_source_contracts PRIVATE
    PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
gtest_discover_tests(test_source_contracts)

add_executable(test_project_file_integration
    test_project_file_integration.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/platform/ProjectFileIntegration.cpp
)
target_link_libraries(test_project_file_integration PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_project_file_integration PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/platform
)
if(WIN32)
    target_link_libraries(test_project_file_integration PRIVATE advapi32 shell32)
endif()
gtest_discover_tests(test_project_file_integration)

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

add_executable(test_lightglue_feature_budget
    test_lightglue_feature_budget.cpp
)
target_link_libraries(test_lightglue_feature_budget PRIVATE
    image_matching
    Qt6::Core
    GTest::gtest_main
    ${OpenCV_LIBS}
)
target_include_directories(test_lightglue_feature_budget PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/image_matching
)
gtest_discover_tests(test_lightglue_feature_budget)

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

add_executable(test_matchphotos_runtime
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/tests/test_matchphotos_runtime.cpp
)
target_link_libraries(test_matchphotos_runtime PRIVATE
    matchphototask
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_matchphotos_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/runtime
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/task
)
gtest_discover_tests(test_matchphotos_runtime)

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

add_executable(test_match_photos_parallelism
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/tests/test_match_photos_parallelism.cpp
)
target_link_libraries(test_match_photos_parallelism PRIVATE
    matchphototask
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_match_photos_parallelism PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/runtime
)
gtest_discover_tests(test_match_photos_parallelism)

add_executable(test_feature_preparation_queue
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/tests/test_feature_preparation_queue.cpp
)
target_link_libraries(test_feature_preparation_queue PRIVATE
    matchphototask
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_feature_preparation_queue PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask
    ${CMAKE_SOURCE_DIR}/src/core/matchphototask/stages
)
gtest_discover_tests(test_feature_preparation_queue)

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
find_package(Ceres CONFIG QUIET)
if(Ceres_FOUND AND TARGET Ceres::ceres)
    target_link_libraries(test_bundle_adjust_projection PRIVATE Ceres::ceres)
    target_compile_definitions(test_bundle_adjust_projection PRIVATE PLASCAN_TEST_HAS_CERES=1)
endif()
gtest_discover_tests(test_bundle_adjust_projection)

# ==============================================================================
# test_bundle_adjust_shared_focal — BA 无相机自标定的共享焦距优化
# ==============================================================================
add_executable(test_bundle_adjust_shared_focal
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_shared_focal.cpp
)
target_link_libraries(test_bundle_adjust_shared_focal PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_shared_focal PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_shared_focal)

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

add_executable(test_bundle_adjust_native_cuda_backend
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_backend.cpp
)
target_link_libraries(test_bundle_adjust_native_cuda_backend PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_native_cuda_backend PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_native_cuda_backend)

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
# test_bundle_adjust_service_lidar — 服务层 LiDAR 约束接入与 JSON 输出
# ==============================================================================
add_executable(test_bundle_adjust_service_lidar
    test_bundle_adjust_service_lidar.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/services/BundleAdjustService.cpp
    ${CMAKE_SOURCE_DIR}/src/common/log/Logger.cpp
)
target_link_libraries(test_bundle_adjust_service_lidar PRIVATE
    plascan_common_project
    lidar
    bundle_adjust
    camera
    control_points
    Qt6::Core
    Qt6::Gui
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_service_lidar PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/project/services
    ${CMAKE_SOURCE_DIR}/src/gui/project/support
    ${CMAKE_SOURCE_DIR}/src/core/lidar
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/control_points
    ${CMAKE_SOURCE_DIR}/src/common/log
    ${CMAKE_SOURCE_DIR}/src/common
)
gtest_discover_tests(test_bundle_adjust_service_lidar)

set(PLASCAN_QT_PLATFORM_PLUGIN_PATH
    "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/Qt6/plugins/platforms")

# ==============================================================================
# test_project_data — ProjectData 持久化契约与 core/results 分域
# ==============================================================================
add_executable(test_project_data
    test_project_data.cpp
)
set_target_properties(test_project_data PROPERTIES
    AUTOMOC ON
)
target_link_libraries(test_project_data PRIVATE
    plascan_common_project
    plascan_common_json
    Qt6::Core
    Qt6::Concurrent
    ${PLASCAN_LIBZIP_TARGET}
    GTest::gtest_main
)
target_include_directories(test_project_data PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/log
    ${CMAKE_SOURCE_DIR}/src/gui/config
    ${CMAKE_SOURCE_DIR}/src/gui/project/archive
    ${CMAKE_SOURCE_DIR}/src/gui/project/data
)
gtest_discover_tests(test_project_data)

# ==============================================================================
# test_gui_project_utils — GUI project 工具与三角化服务
# ==============================================================================
set(MARKER_PANEL_TEST_SOURCES)
set(MARKER_PANEL_REQUIRED_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_marker_panels.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerProjectionPanel.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerReferencePanel.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerFocusMeasurementDialog.cpp
)
set(MARKER_PANEL_FILES_COMPLETE TRUE)
foreach(marker_panel_file IN LISTS MARKER_PANEL_REQUIRED_FILES)
    if(NOT EXISTS "${marker_panel_file}")
        set(MARKER_PANEL_FILES_COMPLETE FALSE)
    endif()
endforeach()
if(MARKER_PANEL_FILES_COMPLETE)
    list(APPEND MARKER_PANEL_TEST_SOURCES ${MARKER_PANEL_REQUIRED_FILES})
    list(APPEND MARKER_PANEL_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/src/gui/markers/DetectMarkersDialog.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerTaskRunner.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerDetectionJobBuilder.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerDetectionReviewDialog.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/markers/PrintMarkersDialog.cpp
    )
else()
    message(STATUS "plascan tests: marker panel tests skipped until all panel sources are present")
endif()

add_executable(test_gui_project_utils
    test_gui_project_utils.cpp
    test_marker_canvas_interaction.cpp
    ${MARKER_PANEL_TEST_SOURCES}
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/reconstruction/AerialTriangulationDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/reconstruction/AerialTriangulationDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/application/AboutDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/image/GenerateMaskDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/reconstruction/GenerateModelDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/shared/WorkflowParameterDialogStyle.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/camera/SurveyControlDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/tie_points/CreateTiePointsDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/tie_points/ThinTiePointsDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/tie_points/CleanTiePointsDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/ObjRenderPreparation.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/menu/ToolbarButton.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/menu/MainMenu.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/main_window/ModelDropSupport.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/main_window/WorkspacePanelController.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/main_window/ProjectUiHydrator.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectDashboardSummary.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectDenseWorkflowConfig.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectModelWorkflowPolicy.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectMetadataOperations.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectReferenceDatasets.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectReferenceTerrainBa.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectSurveyControl.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectWorkflowReports.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/services/ProjectCameraImportService.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/services/ProjectResourceCleanupService.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/services/ProjectTiePointResultService.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectResultRecords.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/ImageViewRotationSettings.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/settings/ProjectDialogJsonSettingBase.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/config/settings/DialogSettingStore.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/DepthOverlayData.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DepthOverlayController.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerImageLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerFeatureLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/FeatureResidualLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerOverlayItems.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerRenderer.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerStitchedDebug.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/CanvasWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerOverlayItems.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerWorkspaceController.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerWorkspaceDetection.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerWorkspaceView.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerUndoCommand.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/ProjectMarkerRepository.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DualImageViewer.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DualImageViewer.ui
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/ImageViewWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/ImageViewWidget.ui
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/MatchLineOverlay.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DisparityHeatmapOverlay.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeContextMenu.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeModel.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreePopulation.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeResourceAlignment.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeResourceMetadata.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/WorkspaceSectionIcons.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/PhotoStripWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/ProjectDashboardWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/TaskStatusWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/HenuBrandWidget.cpp
)
set_target_properties(test_gui_project_utils PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
)
if(MSVC)
    target_compile_options(test_gui_project_utils PRIVATE /bigobj)
endif()
target_link_libraries(test_gui_project_utils PRIVATE
    plascan_common_project
    mvs
    sfm
    camera
    intersection
    bundle_adjust
    qc
    control_points
    terrain
    project_workflows
    mask
    image_matching
    matchphototask
    Qt6::Core
    Qt6::Concurrent
    Qt6::Gui
    Qt6::Test
    Qt6::Widgets
    ${OpenCV_LIBS}
    ${PLASCAN_LIBZIP_TARGET}
    ${PLASCAN_GDAL_TARGET}
    plascan_common_io
    plascan_common_json
    plascan_common_runtime
    plascan_common_string_utils
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
    ${CMAKE_SOURCE_DIR}/src/gui/markers
    ${CMAKE_SOURCE_DIR}/src/gui/tasks
    ${CMAKE_SOURCE_DIR}/src/gui/project
    ${CMAKE_SOURCE_DIR}/src/gui/project/archive
    ${CMAKE_SOURCE_DIR}/src/gui/project/data
    ${CMAKE_SOURCE_DIR}/src/gui/project/manager
    ${CMAKE_SOURCE_DIR}/src/gui/project/services
    ${CMAKE_SOURCE_DIR}/src/gui/project/support
    ${CMAKE_SOURCE_DIR}/src/gui/config/settings
    ${CMAKE_SOURCE_DIR}/src/gui/views
    ${CMAKE_SOURCE_DIR}/src/gui/widgets
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/image_matching
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/qc
    ${CMAKE_SOURCE_DIR}/src/core/control_points
    ${CMAKE_SOURCE_DIR}/src/core/terrain
    ${CMAKE_SOURCE_DIR}/src/core/mvs
    ${CMAKE_SOURCE_DIR}/src/core/mask
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
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

add_executable(test_map_project_dialog
    test_map_project_dialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/reconstruction/MapProjectDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/reconstruction/MapProjectDialogSettings.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/reconstruction/MapProjectDialogEstimate.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/reconstruction/MapProjectDialogLayout.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/shared/WorkflowParameterDialogStyle.cpp
)
set_target_properties(test_map_project_dialog PROPERTIES
    AUTOMOC ON
)
target_link_libraries(test_map_project_dialog PRIVATE
    terrain
    Qt6::Core
    Qt6::Gui
    Qt6::Test
    Qt6::Widgets
    ${OpenCV_LIBS}
    GTest::gtest
)
target_include_directories(test_map_project_dialog PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/terrain
    ${CMAKE_SOURCE_DIR}/src/common
    ${OpenCV_INCLUDE_DIRS}
)
gtest_discover_tests(test_map_project_dialog
    DISCOVERY_TIMEOUT 60
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        ENVIRONMENT
            "QT_QPA_PLATFORM=offscreen;QT_QPA_PLATFORM_PLUGIN_PATH=${PLASCAN_QT_PLATFORM_PLUGIN_PATH}")

add_executable(test_workflow_settings_dialog
    test_workflow_settings_dialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/application/WorkflowSettingsDialog.cpp
)
target_link_libraries(test_workflow_settings_dialog PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    GTest::gtest
)
target_include_directories(test_workflow_settings_dialog PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs
)
gtest_discover_tests(test_workflow_settings_dialog
    DISCOVERY_TIMEOUT 60
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        ENVIRONMENT
            "QT_QPA_PLATFORM=offscreen;QT_QPA_PLATFORM_PLUGIN_PATH=${PLASCAN_QT_PLATFORM_PLUGIN_PATH}")

add_executable(test_workspace_section_icons
    test_workspace_section_icons.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeWidget.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeContextMenu.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeModel.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreePopulation.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeResourceAlignment.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeResourceMetadata.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/DataTreeWidget.ui
    ${CMAKE_SOURCE_DIR}/src/gui/widgets/WorkspaceSectionIcons.cpp
)
set_target_properties(test_workspace_section_icons PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
)
target_link_libraries(test_workspace_section_icons PRIVATE
    plascan_common_project
    project_workflows
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    GTest::gtest
)
target_include_directories(test_workspace_section_icons PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/project/support
    ${CMAKE_SOURCE_DIR}/src/gui/widgets
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/common
)
gtest_discover_tests(test_workspace_section_icons
    DISCOVERY_TIMEOUT 60
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        ENVIRONMENT
            "QT_QPA_PLATFORM=offscreen;QT_QPA_PLATFORM_PLUGIN_PATH=${PLASCAN_QT_PLATFORM_PLUGIN_PATH}")

add_executable(test_workflow_parameter_dialog_style
    test_workflow_parameter_dialog_style.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/reconstruction/GenerateModelDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/shared/WorkflowParameterDialogStyle.cpp
)
set_target_properties(test_workflow_parameter_dialog_style PROPERTIES
    AUTOMOC ON
)
target_link_libraries(test_workflow_parameter_dialog_style PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test
    Qt6::Widgets
    GTest::gtest
)
target_include_directories(test_workflow_parameter_dialog_style PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs
)
target_compile_definitions(test_workflow_parameter_dialog_style PRIVATE
    PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
gtest_discover_tests(test_workflow_parameter_dialog_style
    DISCOVERY_TIMEOUT 60
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
        ENVIRONMENT
            "QT_QPA_PLATFORM=offscreen;QT_QPA_PLATFORM_PLUGIN_PATH=${PLASCAN_QT_PLATFORM_PLUGIN_PATH}")

add_test(
    NAME WorkflowParameterDialogStyleDpi200
    COMMAND $<TARGET_FILE:test_workflow_parameter_dialog_style>
            --gtest_filter=WorkflowParameterDialogStyleTest.AdvancedSectionAndRegionControlsKeepBehavior
)
set_tests_properties(WorkflowParameterDialogStyleDpi200 PROPERTIES
    ENVIRONMENT
        "QT_QPA_PLATFORM=offscreen;QT_QPA_PLATFORM_PLUGIN_PATH=${PLASCAN_QT_PLATFORM_PLUGIN_PATH};QT_SCALE_FACTOR=2.0"
)

add_executable(test_layer_renderer_batched_overlay
    test_layer_renderer_batched_overlay.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerFeatureLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerImageLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerOverlayItems.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerRenderer.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/LayerStitchedDebug.cpp
    ${CMAKE_SOURCE_DIR}/src/common/log/Logger.cpp
)
set_target_properties(test_layer_renderer_batched_overlay PROPERTIES
    AUTOMOC ON
)
target_link_libraries(test_layer_renderer_batched_overlay PRIVATE
    plascan_common_project
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    ${OpenCV_LIBS}
    ${PLASCAN_GDAL_TARGET}
    plascan_common_io
    plascan_common_string_utils
    mask
    image_matching
    GTest::gtest
)
target_include_directories(test_layer_renderer_batched_overlay PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/views
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/common/log
    ${CMAKE_SOURCE_DIR}/src/core/mask
    ${CMAKE_SOURCE_DIR}/src/core/image_matching
    ${OpenCV_INCLUDE_DIRS}
)
if(GDAL_INCLUDE_DIRS)
    target_include_directories(test_layer_renderer_batched_overlay PRIVATE ${GDAL_INCLUDE_DIRS})
endif()
add_test(NAME test_layer_renderer_batched_overlay
         COMMAND test_layer_renderer_batched_overlay)

if(Python3_Interpreter_FOUND)
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

# ==============================================================================
# test_mesh_reconstructor — 点云到网格重建降级路径
# ==============================================================================
add_executable(test_mesh_reconstructor
    test_mesh_reconstructor.cpp
    test_robust_surface_displacement_solver.cpp
    test_depth_ray_metric.cpp
    test_depth_tsdf_narrow_band_activation.cpp
    test_binary_grid_min_cut_solver.cpp
    test_visibility_occupancy_distance_field.cpp
    test_visibility_occupancy_boundary_extractor.cpp
    test_visibility_occupancy_carrier_subdivider.cpp
    test_visibility_occupancy_carrier_fairer.cpp
    test_visibility_occupancy_carrier_field_projector.cpp
    test_visibility_occupancy_well_composed_repair.cpp
    test_visibility_occupancy_surface_builder.cpp
    test_mesh_boundary_attribution.cpp
    test_consistent_isosurface_extractor.cpp
    test_mc33_isosurface_extractor.cpp
    test_mesh_face_orientation.cpp
    test_openmesh_simplifier.cpp
)
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
# test_camera_scene_view_math — 三维相机视角选择、深度排序与校准投影
# ==============================================================================
add_executable(test_camera_scene_view_math
    test_camera_scene_view_math.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/CameraSceneViewMath.cpp
)
target_link_libraries(test_camera_scene_view_math PRIVATE
    Qt6::Core
    Qt6::Gui
    GTest::gtest_main
)
target_include_directories(test_camera_scene_view_math PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/views
)
gtest_discover_tests(test_camera_scene_view_math)

add_executable(test_camera_scene_render_contracts
    test_camera_scene_render_contracts.cpp
)
target_link_libraries(test_camera_scene_render_contracts PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_compile_definitions(test_camera_scene_render_contracts PRIVATE
    PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
gtest_discover_tests(test_camera_scene_render_contracts)

add_executable(test_tie_point_visualization
    test_tie_point_visualization.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/TiePointVisualization.cpp
)
target_link_libraries(test_tie_point_visualization PRIVATE
    GTest::gtest_main
    Qt6::Core
    Qt6::Gui
)
target_include_directories(test_tie_point_visualization PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/views
)
gtest_discover_tests(test_tie_point_visualization)

add_executable(test_model_visualization
    test_model_visualization.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/ModelVisualization.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/views/TiePointVisualization.cpp
)
target_link_libraries(test_model_visualization PRIVATE
    GTest::gtest_main
    Qt6::Core
    Qt6::Gui
)
target_include_directories(test_model_visualization PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/views
)
gtest_discover_tests(test_model_visualization)

# ==============================================================================
# test_marker_task_runner — 标靶后台检测、真实进度与取消
# ==============================================================================
add_executable(test_marker_task_runner
    test_marker_task_runner.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerTaskRunner.cpp
)
set_target_properties(test_marker_task_runner PROPERTIES AUTOMOC ON)
target_link_libraries(test_marker_task_runner PRIVATE
    control_points
    Qt6::Core
    Qt6::Gui
    Qt6::Concurrent
    GTest::gtest
)
target_include_directories(test_marker_task_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/markers
    ${CMAKE_SOURCE_DIR}/src/core/control_points
)
gtest_discover_tests(test_marker_task_runner DISCOVERY_MODE PRE_TEST)

# ==============================================================================
# test_marker_detection_job_builder — 将项目影像/蒙版元数据转换为自动检测任务
# ==============================================================================
add_executable(test_marker_detection_job_builder
    test_marker_detection_job_builder.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/markers/MarkerDetectionJobBuilder.cpp
)
set_target_properties(test_marker_detection_job_builder PROPERTIES AUTOMOC ON)
target_link_libraries(test_marker_detection_job_builder PRIVATE
    plascan_common_project
    plascan_common_json
    control_points
    Qt6::Core
    Qt6::Gui
    Qt6::Concurrent
    ${PLASCAN_LIBZIP_TARGET}
    GTest::gtest_main
)
target_include_directories(test_marker_detection_job_builder PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/log
    ${CMAKE_SOURCE_DIR}/src/gui/config
    ${CMAKE_SOURCE_DIR}/src/gui/markers
    ${CMAKE_SOURCE_DIR}/src/gui/project/archive
    ${CMAKE_SOURCE_DIR}/src/gui/project/data
    ${CMAKE_SOURCE_DIR}/src/core/control_points
)
gtest_discover_tests(test_marker_detection_job_builder DISCOVERY_MODE PRE_TEST)

# ============================================================================
# test_project_session_context — 后台任务结果仅写回启动时的项目与 Chunk 会话
# ============================================================================
add_executable(test_project_session_context
    test_project_session_context.cpp
)
target_link_libraries(test_project_session_context PRIVATE
    Qt6::Core
    GTest::gtest_main
)
target_include_directories(test_project_session_context PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/project/support
    ${CMAKE_SOURCE_DIR}/src/core/terrain
)
gtest_discover_tests(test_project_session_context)
