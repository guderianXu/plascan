cmake_minimum_required(VERSION 3.16)

add_library(plascan_cudasift STATIC
  cudaImage.cu
  cudaImage.h
  cudaSift.h
  cudaSiftD.h
  cudaSiftH.cu
  cudaSiftH.h
  cudautils.h
)

target_include_directories(plascan_cudasift
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(plascan_cudasift
  PRIVATE
    _CRT_SECURE_NO_WARNINGS
)

target_compile_options(plascan_cudasift
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

set_target_properties(plascan_cudasift PROPERTIES
  CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)
