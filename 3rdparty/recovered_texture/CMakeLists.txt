cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(plascan_recovered_texture LANGUAGES CXX)
endif()

# Clean-room Natural UV and CPU-kernel boundary recovered from the authorized
# reference study.  The Vulkan/page-codec compatibility executable is kept out
# of PlaScan until its external float32 ABI can be represented explicitly.
add_library(plascan_recovered_texture_core STATIC
    src/candidate_unary.cpp
    src/camera_model.cpp
    src/focus_grayscale.cpp
    src/focus_score.cpp
    src/graph_partition.cpp
    src/natural_uv_parameterization.cpp
    src/natural_uv_partition.cpp
    src/natural_uv_worker.cpp
    src/recovered_kernels.cpp
    src/target_svd.cpp
)

target_include_directories(plascan_recovered_texture_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_features(plascan_recovered_texture_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(plascan_recovered_texture_core PRIVATE /W4 /permissive- /EHsc)
    target_compile_definitions(plascan_recovered_texture_core PRIVATE _CRT_SECURE_NO_WARNINGS)
    set(_plascan_recovered_texture_strict_fp /fp:strict)
else()
    target_compile_options(plascan_recovered_texture_core PRIVATE -Wall -Wextra -Wpedantic)
    set(_plascan_recovered_texture_strict_fp -ffp-contract=off)
endif()

set_source_files_properties(
    src/candidate_unary.cpp
    src/focus_grayscale.cpp
    src/focus_score.cpp
    src/graph_partition.cpp
    PROPERTIES COMPILE_OPTIONS "${_plascan_recovered_texture_strict_fp}"
)
