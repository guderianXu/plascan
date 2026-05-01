cmake_minimum_required(VERSION 3.16)

add_library(plascan_common_log STATIC
  log/Logger.cpp
)

add_library(plascan_common_math INTERFACE)

add_library(plascan_common_spatial INTERFACE)

target_include_directories(plascan_common_log PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/log
)

target_include_directories(plascan_common_math INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(plascan_common_spatial INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(plascan_common_log PUBLIC cxx_std_17)
target_compile_features(plascan_common_math INTERFACE cxx_std_17)
target_compile_features(plascan_common_spatial INTERFACE cxx_std_17)

if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  include(GoogleTest)

  add_executable(test_common_kdtree
    spatial/tests/test_kdtree.cpp
  )

  target_link_libraries(test_common_kdtree PRIVATE
    plascan_common_spatial
    GTest::gtest_main
  )

  target_include_directories(test_common_kdtree PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  gtest_discover_tests(test_common_kdtree)
endif()
