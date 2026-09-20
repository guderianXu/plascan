find_package(GTest REQUIRED)
include(GoogleTest)
add_executable(test_standard_file_io test_file_io.cpp)
target_link_libraries(test_standard_file_io PRIVATE plascan_common_json_io GTest::gtest_main)
target_compile_definitions(test_standard_file_io PRIVATE
    PLASCAN_STANDARD_IO_TMP_DIR="${CMAKE_SOURCE_DIR}/build/tmp/aerial-standard-io-tests")
gtest_discover_tests(test_standard_file_io DISCOVERY_MODE PRE_TEST)
