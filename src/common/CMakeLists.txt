cmake_minimum_required(VERSION 3.16)

add_library(plascan_common_log STATIC
  log/Logger.cpp
)

add_library(plascan_common_io STATIC
  io/PathIO.cpp
)

add_library(plascan_common_math INTERFACE)

add_library(plascan_common_project STATIC
  project/SparseResultQuality.cpp
)

target_include_directories(plascan_common_log PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/log
)

target_include_directories(plascan_common_io PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(GDAL_INCLUDE_DIRS)
  target_include_directories(plascan_common_io PRIVATE ${GDAL_INCLUDE_DIRS})
endif()

target_include_directories(plascan_common_math INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(plascan_common_project PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(plascan_common_log PUBLIC cxx_std_17)
target_compile_features(plascan_common_io PUBLIC cxx_std_17)
target_compile_features(plascan_common_math INTERFACE cxx_std_17)
target_compile_features(plascan_common_project PUBLIC cxx_std_17)

target_link_libraries(plascan_common_io PUBLIC
  Qt6::Core
  ${OpenCV_LIBS}
  ${PLASCAN_GDAL_TARGET}
)

target_link_libraries(plascan_common_project PUBLIC Qt6::Core)
