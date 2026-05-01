cmake_minimum_required(VERSION 3.16)

add_library(loftr_matcher STATIC
    LoFTRMatcher.cpp
    LoFTRMatcher.h
)

target_link_libraries(loftr_matcher PUBLIC
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_include_directories(loftr_matcher PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
)
