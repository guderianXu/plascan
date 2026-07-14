find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_common_string_parsing StringParsing_tests.cpp)
target_link_libraries(test_common_string_parsing PRIVATE
    plascan_common_string_utils
    GTest::gtest_main
)
gtest_discover_tests(test_common_string_parsing)

add_executable(test_common_string_transform StringTransform_tests.cpp)
target_link_libraries(test_common_string_transform PRIVATE
    plascan_common_string_utils
    GTest::gtest_main
)
gtest_discover_tests(test_common_string_transform)
