cmake_minimum_required(VERSION 3.16)

# SuperPoint
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/superpoint/CMakeLists.txt)
  set(SUPERPOINT_INSTALL OFF CACHE BOOL "Disable SuperPoint standalone install/export in PlaScan build" FORCE)
  add_subdirectory(superpoint)
  message(STATUS "plascan_core: building feature extractor - SuperPoint")
else()
  message(WARNING "plascan_core: SuperPoint feature extractor not found")
endif()

add_library(feature_extractors_traditional STATIC
  FeatureData.cpp
  FeatureData.h
  ExtractorFactory.cpp
  ExtractorFactory.h
  tradition/CudaSiftFeatureExtractor.cpp
  tradition/CudaSiftFeatureExtractor.h
  tradition/TraditionalFeatureExtractor.cpp
  tradition/TraditionalFeatureExtractor.h
)

target_include_directories(feature_extractors_traditional
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tradition
    ${CMAKE_CURRENT_SOURCE_DIR}/superpoint
    ${CMAKE_CURRENT_SOURCE_DIR}/disk
    ${CMAKE_CURRENT_SOURCE_DIR}/aliked
    ${CMAKE_SOURCE_DIR}/3rdparty/cudasift
    ${CMAKE_SOURCE_DIR}/src/common
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(feature_extractors_traditional
  PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

if(TARGET superpoint)
  target_link_libraries(feature_extractors_traditional PUBLIC superpoint)
endif()

set(_plascan_cuda_sift_default OFF)
if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/cudasift/CMakeLists.txt")
  set(_plascan_cuda_sift_default ON)
endif()
option(PLASCAN_ENABLE_CUDA_SIFT "Build optional CUDA SIFT extractor backend" ${_plascan_cuda_sift_default})
if(PLASCAN_ENABLE_CUDA_SIFT AND CMAKE_CUDA_COMPILER)
  add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/cudasift" "${CMAKE_BINARY_DIR}/3rdparty/cudasift")
  target_link_libraries(feature_extractors_traditional PUBLIC plascan_cudasift)
  target_compile_definitions(feature_extractors_traditional PUBLIC PLASCAN_HAS_CUDA_SIFT=1)
  message(STATUS "plascan_core: building feature extractor - CUDA SIFT")
else()
  message(STATUS "plascan_core: CUDA SIFT disabled")
endif()

# ── DISK ──
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/disk/CMakeLists.txt)
  add_subdirectory(disk)
  message(STATUS "plascan_core: building feature extractor - DISK")
endif()

# ── ALIKED ──
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/aliked/CMakeLists.txt)
  add_subdirectory(aliked)
  message(STATUS "plascan_core: building feature extractor - ALIKED")
endif()

# ExtractorFactory 引用 DiskExtractor/AlikedExtractor, 链接对应库
if(TARGET feature_extractors_disk)
  target_link_libraries(feature_extractors_traditional PUBLIC feature_extractors_disk)
endif()
if(TARGET feature_extractors_aliked)
  target_link_libraries(feature_extractors_traditional PUBLIC feature_extractors_aliked)
endif()
