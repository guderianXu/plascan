cmake_minimum_required(VERSION 3.16)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_library(lightglue_matcher STATIC
    LightGlueMatcher.cpp
    LightGlueMatcher.h
)

target_include_directories(lightglue_matcher
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..              # match.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../..           # 访问 feature_extractors/FeatureData.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../../feature_extractors
        ${CMAKE_CURRENT_SOURCE_DIR}/../../feature_extractors/superpoint
        ${TORCH_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(lightglue_matcher
    PUBLIC
        feature_match_common
        feature_extractors_traditional
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)
