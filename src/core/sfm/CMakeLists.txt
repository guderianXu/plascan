cmake_minimum_required(VERSION 3.16)
project(plascan_sfm LANGUAGES CXX)

# ============================================================
# SfM 模块（增量式运动恢复结构）
# 依赖：plascan_camera, plascan_intersection, plascan_bundle_adjust
# OpenCV/Qt6 由根 CMakeLists.txt 统一查找
# ============================================================

set(SFM_SOURCES
    # ---- 通用类型 ----

    # ---- 图结构 ----
    graph/CorrespondenceGraph.cpp
    graph/ObservationNetworkBuilder.cpp

    # ---- 重建状态 ----
    reconstruction/SfmReconstruction.cpp

    # ---- 位姿估计 ----
    pose/PnpSolver.cpp

    # ---- 三角化 ----
    triangulation/Triangulator.cpp
    triangulation/InitialSparsePointCloudTriangulator.cpp

    # ---- 流水线 ----
    pipeline/IncrementalSfm.cpp
    BaInputBuilder.cpp
    TriangulationService.cpp

    # ---- 点云过滤 ----
    filtering/SfmPointCloudFilter.cpp
    filtering/SparsePointCloudProcessor.cpp
)

set(SFM_HEADERS
    common/SfmTypes.h
    graph/CorrespondenceGraph.h
    graph/ObservationNetworkBuilder.h
    reconstruction/SfmReconstruction.h
    pose/PnpSolver.h
    triangulation/Triangulator.h
    triangulation/InitialSparsePointCloudTriangulator.h
    pipeline/IncrementalSfm.h
    BaInputBuilder.h
    TriangulationService.h
    filtering/SfmPointCloudFilter.h
    filtering/SparsePointCloudProcessor.h
)

add_library(sfm STATIC ${SFM_SOURCES} ${SFM_HEADERS})

target_include_directories(sfm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
    ${CMAKE_CURRENT_SOURCE_DIR}/../intersection
    ${CMAKE_CURRENT_SOURCE_DIR}/../bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(sfm
    PUBLIC
        camera
        plapoint::plapoint
    PRIVATE
        intersection
        bundle_adjust
        Qt6::Core
        Qt6::Gui
        plascan_common_log
        ${OpenCV_LIBS}
)

target_compile_features(sfm PUBLIC cxx_std_17)
