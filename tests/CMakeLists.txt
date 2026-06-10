if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

include(CheckLanguage)

# ── GTest ────────────────────────────────────────────────────────────────────
find_package(GTest REQUIRED)
include(GoogleTest)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(Python3 COMPONENTS Interpreter)

# ── 依赖 ─────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc)

# ── 测试数据根目录──────────────────────────────────────────────────────────────
set(TEST_DATA_DIR "${CMAKE_SOURCE_DIR}/testData")
add_definitions(-DTEST_DATA_DIR="${TEST_DATA_DIR}")

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
# test_mvs_types — MVS 类型与参数验证
# ==============================================================================
add_executable(test_mvs_types test_mvs_types.cpp)
target_link_libraries(test_mvs_types PRIVATE
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
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
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
# test_gui_project_utils — GUI project 工具与三角化服务
# ==============================================================================
add_executable(test_gui_project_utils
    test_gui_project_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/FeatureExtractionDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/FeatureExtractionDialog.ui
    ${CMAKE_SOURCE_DIR}/src/gui/main_window/ModelDropSupport.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/io/ProjectIO.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectSupportUtils.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/services/ProjectBaInputBuilder.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/project/services/ProjectTriangulationService.cpp
)
set_target_properties(test_gui_project_utils PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
)
target_link_libraries(test_gui_project_utils PRIVATE
    sfm
    camera
    intersection
    bundle_adjust
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    GTest::gtest
)
target_include_directories(test_gui_project_utils PRIVATE
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs
    ${CMAKE_SOURCE_DIR}/src/gui/main_window
    ${CMAKE_SOURCE_DIR}/src/gui/project
    ${CMAKE_SOURCE_DIR}/src/gui/project/archive
    ${CMAKE_SOURCE_DIR}/src/gui/project/data
    ${CMAKE_SOURCE_DIR}/src/gui/project/io
    ${CMAKE_SOURCE_DIR}/src/gui/project/manager
    ${CMAKE_SOURCE_DIR}/src/gui/project/services
    ${CMAKE_SOURCE_DIR}/src/gui/project/support
    ${CMAKE_SOURCE_DIR}/src/core/sfm
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
)
gtest_discover_tests(test_gui_project_utils)

if(Python3_Interpreter_FOUND)
    add_test(
        NAME FullPipelineScriptTest
        COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_full_pipeline_script
    )
    set_tests_properties(FullPipelineScriptTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

if(TARGET triangulate_cli)
    add_executable(test_triangulate_cli test_triangulate_cli.cpp)
    target_link_libraries(test_triangulate_cli PRIVATE
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
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(test_mvs_pipeline PRIVATE
        mvs
        camera
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
