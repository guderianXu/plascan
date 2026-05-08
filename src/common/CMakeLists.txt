cmake_minimum_required(VERSION 3.16)

add_library(plascan_common_log STATIC
  log/Logger.cpp
)

add_library(plascan_common_math INTERFACE)

target_include_directories(plascan_common_log PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/log
)

target_include_directories(plascan_common_math INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(plascan_common_log PUBLIC cxx_std_17)
target_compile_features(plascan_common_math INTERFACE cxx_std_17)
