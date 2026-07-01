cmake_minimum_required(VERSION 3.16)

find_package(OpenCV REQUIRED)

add_library(feature_match_traditional STATIC
    CudaSiftMatcher.cpp
    TraditionalFeatureMatcher.cpp
)

target_include_directories(feature_match_traditional
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/common
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(feature_match_traditional
    PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)

add_executable(traditional_feature_matcher_diagnostic
    test_traditional_feature_matcher.cpp
)

target_link_libraries(traditional_feature_matcher_diagnostic
    PRIVATE
        feature_match_traditional
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)

target_include_directories(traditional_feature_matcher_diagnostic
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
)

target_compile_definitions(traditional_feature_matcher_diagnostic
    PRIVATE
        TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
)
