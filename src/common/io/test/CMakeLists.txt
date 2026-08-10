find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_common_image_io test_image_io.cpp)
target_link_libraries(test_common_image_io PRIVATE
  plascan_common_io
  GTest::gtest_main
)
gtest_discover_tests(test_common_image_io DISCOVERY_MODE PRE_TEST)

add_executable(test_common_path_io test_path_io.cpp)
target_link_libraries(test_common_path_io PRIVATE
  plascan_common_io
  GTest::gtest_main
)
gtest_discover_tests(test_common_path_io DISCOVERY_MODE PRE_TEST)
