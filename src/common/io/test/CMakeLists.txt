find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_common_image_io test_image_io.cpp)
target_link_libraries(test_common_image_io PRIVATE
  plascan_common_io
  GTest::gtest_main
)
target_compile_definitions(test_common_image_io PRIVATE
  PLASCAN_IMAGE_IO_TEST_TMP_DIR="${CMAKE_SOURCE_DIR}/build/tmp/sfm-color-tests")
gtest_discover_tests(test_common_image_io DISCOVERY_MODE PRE_TEST)

add_executable(test_image_size_reader test_image_size_reader.cpp)
target_link_libraries(test_image_size_reader PRIVATE plascan_common_io GTest::gtest_main)
if(GDAL_INCLUDE_DIRS)
  target_include_directories(test_image_size_reader PRIVATE ${GDAL_INCLUDE_DIRS})
endif()
target_compile_definitions(test_image_size_reader PRIVATE
  PLASCAN_AERIAL_IO_TEST_TMP_DIR="${CMAKE_SOURCE_DIR}/build/tmp/aerial-image-io-tests")
gtest_discover_tests(test_image_size_reader DISCOVERY_MODE PRE_TEST)

add_executable(test_common_path_io test_path_io.cpp)
target_link_libraries(test_common_path_io PRIVATE
  plascan_common_io
  GTest::gtest_main
)
gtest_discover_tests(test_common_path_io DISCOVERY_MODE PRE_TEST)
