cmake_minimum_required(VERSION 3.16)
project(FeatureMatch LANGUAGES CXX)

add_library(feature_match_common STATIC
  match.cpp
  match.h
  MatcherFactory.h
  IMatcher.h
)

target_include_directories(feature_match_common

  PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}

    ${CMAKE_CURRENT_SOURCE_DIR}/superglue

    ${CMAKE_CURRENT_SOURCE_DIR}/lightglue

    ${CMAKE_CURRENT_SOURCE_DIR}/tradition

    ${CMAKE_CURRENT_SOURCE_DIR}/loftr

    ${CMAKE_CURRENT_SOURCE_DIR}/../feature_extractors

    ${CMAKE_CURRENT_SOURCE_DIR}/../feature_extractors/superpoint

    ${CMAKE_CURRENT_SOURCE_DIR}/../feature_extractors/disk

    ${CMAKE_CURRENT_SOURCE_DIR}/../feature_extractors/aliked

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

# MatcherFactory 需要引用各匹配器具体类, 必须放在子目录之后
add_library(feature_match_factory STATIC
    MatcherFactory.cpp
    MatcherFactory.h
)
target_include_directories(feature_match_factory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/superglue
    ${CMAKE_CURRENT_SOURCE_DIR}/lightglue
    ${CMAKE_CURRENT_SOURCE_DIR}/tradition
    ${CMAKE_CURRENT_SOURCE_DIR}/loftr
    ${CMAKE_CURRENT_SOURCE_DIR}/../feature_extractors
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(feature_match_factory PUBLIC
    feature_match_common
)
if(TARGET lightglue_matcher)
    target_link_libraries(feature_match_factory PUBLIC lightglue_matcher)
endif()
if(TARGET superglue_matcher)
    target_link_libraries(feature_match_factory PUBLIC superglue_matcher)
endif()
if(TARGET feature_match_traditional)
    target_link_libraries(feature_match_factory PUBLIC feature_match_traditional)
endif()
if(TARGET loftr_matcher)
    target_link_libraries(feature_match_factory PUBLIC loftr_matcher)
endif()
