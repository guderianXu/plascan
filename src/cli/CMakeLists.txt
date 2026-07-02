cmake_minimum_required(VERSION 3.16)
project(plascan_cli LANGUAGES CXX)

# CLI 工具 — 基于 CLI11, 零 Qt 依赖

function(add_plascan_cli name src_file libs include_dirs)
    add_executable(${name} ${src_file})
    target_link_libraries(${name} PRIVATE ${libs} ${OpenCV_LIBS})
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${include_dirs}
    )
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

# ── 通用相机格式转换 ──
if(TARGET camera)
    add_plascan_cli(camera_convert_cli cli_camera_convert.cpp
        "camera"
        "${CMAKE_SOURCE_DIR}/src/core/camera")
endif()

# ── 密集匹配 ──
if(TARGET dense_match)
    add_plascan_cli(dense_match_cli cli_dense_match.cpp
        "dense_match"
        "${CMAKE_SOURCE_DIR}/src/core/dense_match")
endif()

# ── 极线校正 ──
if(TARGET mvs)
    add_plascan_cli(rectify_cli cli_epipolar_rectify.cpp
        "mvs;camera"
        "${CMAKE_SOURCE_DIR}/src/core/mvs;${CMAKE_SOURCE_DIR}/src/core/camera")
endif()

# ── 视差三角化 → 密集点云 ──
if(TARGET mvs)
    add_plascan_cli(triangulate_cli cli_disparity_triangulate.cpp
        "mvs;camera"
        "${CMAKE_SOURCE_DIR}/src/core/mvs;${CMAKE_SOURCE_DIR}/src/core/camera")
endif()

# ── 密集点云质量细化：raw dense cloud -> refined dense cloud ──
if(TARGET mvs)
    add_plascan_cli(dense_cloud_refine_cli cli_dense_cloud_refine.cpp
        "mvs"
        "${CMAKE_SOURCE_DIR}/src/core/mvs")
endif()

# ── 特征提取 ──
if(TARGET superpoint)
    add_plascan_cli(feature_extract_cli cli_feature_extract.cpp
        "superpoint;feature_extractors_disk;feature_extractors_aliked;feature_extractors_traditional;Qt6::Core;${TORCH_LIBRARIES}"
        "${CMAKE_SOURCE_DIR}/src/core/feature_extractors/superpoint;${CMAKE_SOURCE_DIR}/src/core/feature_extractors;${CMAKE_SOURCE_DIR}/src/core/feature_extractors/tradition;${CMAKE_SOURCE_DIR}/src/core/feature_extractors/disk;${CMAKE_SOURCE_DIR}/src/core/feature_extractors/aliked")
endif()

# ── 特征匹配 ──
if(TARGET superglue_matcher)
    add_plascan_cli(feature_match_cli cli_feature_match.cpp
        "feature_match_factory;superglue_matcher;lightglue_matcher;feature_match_traditional;loftr_matcher;Qt6::Core;${TORCH_LIBRARIES}"
        "${CMAKE_SOURCE_DIR}/src/core/feature_match/superglue;${CMAKE_SOURCE_DIR}/src/core/feature_match/lightglue;${CMAKE_SOURCE_DIR}/src/core/feature_match/tradition;${CMAKE_SOURCE_DIR}/src/core/feature_match/loftr;${CMAKE_SOURCE_DIR}/src/core/feature_match;${CMAKE_SOURCE_DIR}/src/core/feature_extractors")
endif()

# ── 光束法平差：可选 LiDAR 点到面约束与 A/B 对比 ──
if(TARGET sfm AND TARGET bundle_adjust AND TARGET lidar)
    add_executable(bundle_adjust_cli
        cli_bundle_adjust.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/services/BundleAdjustService.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/archive/PlascanArchive.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/io/ProjectIO.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectSupportUtils.cpp
    )
    target_link_libraries(bundle_adjust_cli PRIVATE
        sfm
        bundle_adjust
        lidar
        camera
        intersection
        plascan_common_log
        plascan_common_project
        Qt6::Core
        Qt6::Gui
        ${PLASCAN_LIBZIP_TARGET}
        ${OpenCV_LIBS}
    )
    target_include_directories(bundle_adjust_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
        ${CMAKE_SOURCE_DIR}/src/core/camera
        ${CMAKE_SOURCE_DIR}/src/core/intersection
        ${CMAKE_SOURCE_DIR}/src/core/lidar
        ${CMAKE_SOURCE_DIR}/src/core/sfm
        ${CMAKE_SOURCE_DIR}/src/gui/project/archive
        ${CMAKE_SOURCE_DIR}/src/gui/project/data
        ${CMAKE_SOURCE_DIR}/src/gui/project/io
        ${CMAKE_SOURCE_DIR}/src/gui/project/services
        ${CMAKE_SOURCE_DIR}/src/gui/project/support
        ${OpenCV_INCLUDE_DIRS}
    )
    set_target_properties(bundle_adjust_cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# ── GUI 等价一键重建：SFM -> MVS -> Mesh -> DEM/DOM ──
if(TARGET feature_extractors_traditional AND TARGET feature_match_traditional AND TARGET lightglue_matcher AND TARGET superglue_matcher AND TARGET sfm AND TARGET mvs AND TARGET meshing AND TARGET terrain)
    add_executable(reconstruct_pipeline_cli
        cli_reconstruct_pipeline.cpp
        ${CMAKE_SOURCE_DIR}/src/core/pipeline/SFMService.cpp
        ${CMAKE_SOURCE_DIR}/src/core/pipeline/GuidedRematchService.cpp
        ${CMAKE_SOURCE_DIR}/src/core/pipeline/MatchResultCatalog.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/io/ProjectIO.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectSupportUtils.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectDenseWorkflowConfig.cpp
    )
    target_link_libraries(reconstruct_pipeline_cli PRIVATE
        sfm
        mvs
        meshing
        terrain
        superglue_matcher
        lightglue_matcher
        feature_match_traditional
        feature_extractors_traditional
        feature_match_common
        camera
        overlap
        Qt6::Core
        Qt6::Concurrent
        Qt6::Gui
        plascan_common_log
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
    )
    target_include_directories(reconstruct_pipeline_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_SOURCE_DIR}/src/core/camera
        ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
        ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/superpoint
        ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/disk
        ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/aliked
        ${CMAKE_SOURCE_DIR}/src/core/feature_match
        ${CMAKE_SOURCE_DIR}/src/core/feature_match/superglue
        ${CMAKE_SOURCE_DIR}/src/core/feature_match/lightglue
        ${CMAKE_SOURCE_DIR}/src/core/feature_match/tradition
        ${CMAKE_SOURCE_DIR}/src/core/mesh
        ${CMAKE_SOURCE_DIR}/src/core/mvs
        ${CMAKE_SOURCE_DIR}/src/core/overlap
        ${CMAKE_SOURCE_DIR}/src/core/pipeline
        ${CMAKE_SOURCE_DIR}/src/core/sfm
        ${CMAKE_SOURCE_DIR}/src/core/terrain
        ${CMAKE_SOURCE_DIR}/src/gui
        ${CMAKE_SOURCE_DIR}/src/gui/compat
        ${CMAKE_SOURCE_DIR}/src/gui/project/io
        ${CMAKE_SOURCE_DIR}/src/gui/project/support
        ${TORCH_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_definitions(reconstruct_pipeline_cli PRIVATE PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    set_target_properties(reconstruct_pipeline_cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    if(PLASCAN_EXTRA_NVRTC_RPATHS)
        set_property(TARGET reconstruct_pipeline_cli APPEND PROPERTY BUILD_RPATH "${PLASCAN_EXTRA_NVRTC_RPATHS}")
    endif()
endif()

# ── GUI 三维重建等价 CLI：SFM -> MVS -> Mesh（不生成 DEM/DOM） ──
if(TARGET feature_extractors_traditional AND TARGET feature_match_traditional AND TARGET lightglue_matcher AND TARGET superglue_matcher AND TARGET sfm AND TARGET mvs AND TARGET meshing)
    add_executable(three_d_reconstruction_cli
        cli_reconstruct_pipeline.cpp
        ${CMAKE_SOURCE_DIR}/src/core/pipeline/SFMService.cpp
        ${CMAKE_SOURCE_DIR}/src/core/pipeline/GuidedRematchService.cpp
        ${CMAKE_SOURCE_DIR}/src/core/pipeline/MatchResultCatalog.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/io/ProjectIO.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectSupportUtils.cpp
        ${CMAKE_SOURCE_DIR}/src/gui/project/support/ProjectDenseWorkflowConfig.cpp
    )
    target_link_libraries(three_d_reconstruction_cli PRIVATE
        sfm
        mvs
        meshing
        superglue_matcher
        lightglue_matcher
        feature_match_traditional
        feature_extractors_traditional
        feature_match_common
        camera
        overlap
        Qt6::Core
        Qt6::Concurrent
        Qt6::Gui
        plascan_common_log
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
    )
    target_include_directories(three_d_reconstruction_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_SOURCE_DIR}/src/core/camera
        ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
        ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/superpoint
        ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/disk
        ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/aliked
        ${CMAKE_SOURCE_DIR}/src/core/feature_match
        ${CMAKE_SOURCE_DIR}/src/core/feature_match/superglue
        ${CMAKE_SOURCE_DIR}/src/core/feature_match/lightglue
        ${CMAKE_SOURCE_DIR}/src/core/mesh
        ${CMAKE_SOURCE_DIR}/src/core/mvs
        ${CMAKE_SOURCE_DIR}/src/core/overlap
        ${CMAKE_SOURCE_DIR}/src/core/pipeline
        ${CMAKE_SOURCE_DIR}/src/core/sfm
        ${CMAKE_SOURCE_DIR}/src/gui
        ${CMAKE_SOURCE_DIR}/src/gui/compat
        ${CMAKE_SOURCE_DIR}/src/gui/project/io
        ${CMAKE_SOURCE_DIR}/src/gui/project/support
        ${TORCH_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_definitions(three_d_reconstruction_cli PRIVATE
        PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        PLASCAN_THREE_D_ONLY
    )
    set_target_properties(three_d_reconstruction_cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    if(PLASCAN_EXTRA_NVRTC_RPATHS)
        set_property(TARGET three_d_reconstruction_cli APPEND PROPERTY BUILD_RPATH "${PLASCAN_EXTRA_NVRTC_RPATHS}")
    endif()
endif()
