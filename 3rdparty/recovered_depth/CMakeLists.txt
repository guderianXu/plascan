cmake_minimum_required(VERSION 3.20)

if(NOT PLASCAN_ENABLE_CUDA OR NOT CMAKE_CUDA_COMPILER OR NOT CUDAToolkit_FOUND)
    add_library(recovered_depth INTERFACE)
    target_include_directories(recovered_depth INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_definitions(recovered_depth INTERFACE PLASCAN_RECOVERED_DEPTH_AVAILABLE=0)
    return()
endif()

add_library(recovered_depth STATIC
    src/metalign_compat.cpp
    src/patchmatch.cpp
    src/patchmatch_orchestrator.cpp
    src/patchmatch_store.cpp
    src/gpu_cuda.cu
)

target_include_directories(recovered_depth PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(recovered_depth PUBLIC ${OpenCV_LIBS} CUDA::cudart CUDA::cuda_driver Threads::Threads)
target_compile_definitions(recovered_depth
    PUBLIC PLASCAN_RECOVERED_DEPTH_AVAILABLE=1
    PRIVATE
        METMODEL_HAS_CUDA=1
        METMODEL_PM_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/pm_refinement_cuda_sm30.ptx"
        METMODEL_PM_UNDISTORT_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/pm_undistort_cuda_sm30.ptx"
        METMODEL_PM_FINAL_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/pm_final_refinement_cuda_sm30.ptx"
        METMODEL_PM_WTA_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/pm_wta_cuda_sm30.ptx"
        METMODEL_PM_PROPAGATION_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/pm_propagation_cuda_sm30.ptx"
        METMODEL_PM_COARSE_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/pm_coarse_cuda_sm30.ptx"
        METMODEL_PM_COST_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/pm_cost_cuda_sm30.ptx"
        METMODEL_PM_FILTER_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/pm_filter_cuda_sm30.ptx"
        METMODEL_PM_BILATERAL_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/pm_bilateral_cuda_sm30.ptx"
        METMODEL_DEPTH_RADIUS_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/depth_radius_cuda_sm30.ptx"
        METMODEL_DEPTH_NEIGHBOR_VOTES_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/depth_neighbor_votes_cuda_sm30.ptx"
        METMODEL_DEPTH_VOTING_CUDA_PTX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/depth_voting_cuda_sm30.ptx"
        METMODEL_OOC_HISTOGRAM_CUDA_PTX=""
        METMODEL_OOC_UPDATE_U_CUDA_PTX=""
        METMODEL_OOC_UPDATE_P_CUDA_PTX=""
)

set_source_files_properties(src/patchmatch.cpp PROPERTIES COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/fp:strict>;$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ffp-contract=off>")
set_target_properties(recovered_depth PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
)
