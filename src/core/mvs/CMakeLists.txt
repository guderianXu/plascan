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

# ── 源文件 ────────────────────────────────────────────────────────────────────
set(MVS_CXX_SOURCES
    DenseCloudQualityFilter.cpp
    DepthConsistencyCache.cpp
    DepthMapFusion.cpp
    DepthFrameQualityGate.cpp
    DepthPyramidPolicy.cpp
    DepthPyramidEstimator.cpp
    DepthPyramidPropagation.cpp
    DepthFrameUtils.cpp
    MvsSourcePlanner.cpp
    MvsQualityReport.cpp
    MvsSceneClassifier.cpp
    MvsImagePreprocessor.cpp
    MvsWorkspaceManifest.cpp
    DenseCloudBuilder.cpp
    DepthMapGenerator.cpp
    EpipolarRectifier.cpp
    SparseCloudValidator.cpp
    SparseCloudPreprocessor.cpp
    SubpixelRefiner.cpp
    DisparityFilter.cpp
    DisparityTriangulator.cpp
    PointCloudTifIO.cpp
    StereoDenseCloudPipelineOutput.cpp
    StereoDenseCloudPipelinePaths.cpp
    AspPointCloudMetrics.cpp
    StereoDenseCloudPipeline.cpp
)

set(MVS_HEADERS
    DenseCloudQualityFilter.h
    DepthConsistencyCache.h
    MvsTypes.h
    MvsSourcePlanner.h
    MvsQualityReport.h
    MvsSceneClassifier.h
    MvsImagePreprocessor.h
    MvsViewSelection.h
    MvsWorkspaceManifest.h
    PatchMatchCUDA.h
    DepthMapFusion.h
    DepthFrameQualityGate.h
    DepthPyramidPolicy.h
    DepthPyramidEstimator.h
    DepthPyramidPropagation.h
    DepthFrameUtils.h
    DenseCloudBuilder.h
    DepthMapGenerator.h
    PatchMatchPhotometricCost.h
    DensePointCloudCUDA.h
    EpipolarRectifier.h
    SparseCloudValidator.h
    SparseCloudPreprocessor.h
    SubpixelRefiner.h
    DisparityFilter.h
    DisparityTriangulator.h
    PointCloudTifIO.h
    StereoDenseCloudPipelineOutput.h
    StereoDenseCloudPipelinePaths.h
    AspPointCloudMetrics.h
    StereoDenseCloudPipeline.h
)

set(MVS_QOBJECT_HEADERS
    DepthMapGenerator.h
    StereoDenseCloudPipeline.h
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
    )
    target_compile_definitions(mvs PUBLIC MVS_ENABLE_CUDA=1)
    target_compile_options(mvs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    )
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
            ${CMAKE_SOURCE_DIR}/src/core/mvs/EpipolarRectifier_tests.cpp
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
    endif()
endif()
