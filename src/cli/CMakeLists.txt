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

# ── 特征提取 ──
if(TARGET superpoint)
    add_plascan_cli(feature_extract_cli cli_feature_extract.cpp
        "superpoint;feature_extractors_traditional;Qt6::Core;${TORCH_LIBRARIES}"
        "${CMAKE_SOURCE_DIR}/src/core/feature_extractors/superpoint;${CMAKE_SOURCE_DIR}/src/core/feature_extractors;${CMAKE_SOURCE_DIR}/src/core/feature_extractors/tradition")
endif()

# ── 特征匹配 ──
if(TARGET superglue_matcher)
    add_plascan_cli(feature_match_cli cli_feature_match.cpp
        "superglue_matcher;lightglue_matcher;feature_match_traditional;loftr_matcher;Qt6::Core;${TORCH_LIBRARIES}"
        "${CMAKE_SOURCE_DIR}/src/core/feature_match/superglue;${CMAKE_SOURCE_DIR}/src/core/feature_match/lightglue;${CMAKE_SOURCE_DIR}/src/core/feature_match/tradition;${CMAKE_SOURCE_DIR}/src/core/feature_match/loftr;${CMAKE_SOURCE_DIR}/src/core/feature_match;${CMAKE_SOURCE_DIR}/src/core/feature_extractors")
endif()

# ── 待实现 ──
# sfm_cli: 需 headless SFMService (当前 SFMService 耦合到 GUI ProjectIO)
