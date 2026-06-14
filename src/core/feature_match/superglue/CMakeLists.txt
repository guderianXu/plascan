cmake_minimum_required(VERSION 3.10)
project(SuperGlueCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 包含目录
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..                   # match.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../..                 # 访问 feature_extractors/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../feature_extractors
    ${CMAKE_CURRENT_SOURCE_DIR}/../../feature_extractors/superpoint
    ${CMAKE_SOURCE_DIR}/src/common                   # plascan_common_log
)

# 查找 Qt6（用于 SuperGlueMatchIO）
find_package(Qt6 COMPONENTS Core REQUIRED)

# SuperGlue 库
add_library(superglue_matcher STATIC
    SuperGlueMatcher.cpp
    SuperGlueMatchIO.cpp
    MatchOutlierRejector.cpp
)

target_link_libraries(superglue_matcher
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Qt6::Core
    plascan_common_log
    feature_match_common
    feature_extractors_traditional
)


# 设置 RPATH 以便找到共享库
set_target_properties(superglue_matcher PROPERTIES
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(PLASCAN_EXTRA_NVRTC_RPATHS)
    set_property(TARGET superglue_matcher APPEND PROPERTY BUILD_RPATH "${PLASCAN_EXTRA_NVRTC_RPATHS}")
endif()

# Build usage_examples executable
add_executable(usage_examples
    usage_examples.cpp
)

target_include_directories(usage_examples PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(usage_examples
    superglue_matcher
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

set_target_properties(usage_examples PROPERTIES
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(PLASCAN_EXTRA_NVRTC_RPATHS)
    set_property(TARGET usage_examples APPEND PROPERTY BUILD_RPATH "${PLASCAN_EXTRA_NVRTC_RPATHS}")
endif()
