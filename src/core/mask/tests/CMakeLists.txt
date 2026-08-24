include(PlascanTestRuntime)

add_executable(test_mask_generation
    test_interactive_mask_algorithms.cpp
    test_mask_generation.cpp
)

target_link_libraries(test_mask_generation PRIVATE
    mask
    ${OpenCV_LIBS}
    GTest::gtest_main
)

target_include_directories(test_mask_generation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_definitions(test_mask_generation PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
)

plascan_gtest_discover_tests(test_mask_generation)
