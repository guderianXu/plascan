find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_plafs test_plafs.cpp)
target_link_libraries(test_plafs PRIVATE plascan_common_plafs GTest::gtest_main)
target_compile_definitions(test_plafs PRIVATE
    PLASCAN_PLAFS_TMP_DIR="${CMAKE_SOURCE_DIR}/build/tmp/plafs-tests")
gtest_discover_tests(test_plafs DISCOVERY_MODE PRE_TEST)
