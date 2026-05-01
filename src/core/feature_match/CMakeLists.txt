cmake_minimum_required(VERSION 3.16)
project(FeatureMatch LANGUAGES CXX)

add_library(feature_match_common STATIC
  match.cpp
  match.h
)

target_include_directories(feature_match_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../feature_extractors
    ${CMAKE_CURRENT_SOURCE_DIR}/../feature_extractors/superpoint
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(feature_match_common
  PUBLIC
    feature_extractors_traditional
  PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lightglue/CMakeLists.txt)
    add_subdirectory(lightglue)
    message(STATUS "plascan_core: building feature matcher - LightGlue")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/superglue/CMakeLists.txt)
    add_subdirectory(superglue)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tradition/CMakeLists.txt)
    add_subdirectory(tradition)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/loftr/CMakeLists.txt)
    add_subdirectory(loftr)
endif()
