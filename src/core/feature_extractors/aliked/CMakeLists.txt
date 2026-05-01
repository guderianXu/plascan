cmake_minimum_required(VERSION 3.16)

add_library(feature_extractors_aliked STATIC
    AlikedExtractor.cpp
    AlikedExtractor.h
)

target_link_libraries(feature_extractors_aliked PUBLIC
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_include_directories(feature_extractors_aliked PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        include(GoogleTest)
        add_executable(test_aliked_unit tests/test_aliked.cpp)
        target_link_libraries(test_aliked_unit PRIVATE
            feature_extractors_aliked
            GTest::gtest_main
            ${OpenCV_LIBS}
            ${TORCH_LIBRARIES}
        )
        target_compile_definitions(test_aliked_unit PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
        target_include_directories(test_aliked_unit PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
        )
        target_compile_definitions(test_aliked_unit PRIVATE
            TESTDATA_DIR="${CMAKE_SOURCE_DIR}/src/core/feature_extractors/testdata"
            MODEL_DIR="${CMAKE_SOURCE_DIR}/resources/models"
        )
        gtest_discover_tests(test_aliked_unit)
    endif()
endif()
