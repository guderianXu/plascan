find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_common_logger test_logger.cpp)
target_link_libraries(test_common_logger PRIVATE
  plascan_common_log
  GTest::gtest
)
gtest_discover_tests(test_common_logger DISCOVERY_MODE PRE_TEST)
