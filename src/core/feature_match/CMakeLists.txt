cmake_minimum_required(VERSION 3.16)
project(FeatureMatch LANGUAGES CXX)

add_library(feature_match_common STATIC
  match.cpp
  match.h
  MatchExportIO.cpp
  MatchExportIO.h
  MatchFileIO.cpp
  MatchFileIO.h
  MatchGeometryFilter.cpp
  MatchGeometryFilter.h
  MatchVisualization.cpp
  MatchVisualization.h
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

    ${CMAKE_SOURCE_DIR}/src/common

    ${TORCH_INCLUDE_DIRS}

    ${OpenCV_INCLUDE_DIRS}

)

target_link_libraries(feature_match_common
  PUBLIC
    feature_extractors_traditional
    plascan_common_io
    Qt6::Core
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
target_compile_definitions(feature_match_factory PRIVATE PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
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

if(BUILD_TESTS)
  add_executable(test_algorithm_compat tests/test_algorithm_compat.cpp)
  target_link_libraries(test_algorithm_compat PRIVATE GTest::gtest GTest::gtest_main Qt6::Core)
  target_include_directories(test_algorithm_compat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME AlgorithmCompatTest COMMAND test_algorithm_compat)

  add_executable(test_match_file_io tests/test_match_file_io.cpp)
  target_link_libraries(test_match_file_io PRIVATE GTest::gtest GTest::gtest_main feature_match_common Qt6::Core)
  target_include_directories(test_match_file_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME MatchFileIOTest COMMAND test_match_file_io)
endif()
