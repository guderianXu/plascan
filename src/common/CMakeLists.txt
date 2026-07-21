cmake_minimum_required(VERSION 3.16)

add_library(plascan_common_log STATIC
  log/Logger.cpp
)

add_library(plascan_common_io STATIC
  io/PathIO.cpp
)

add_library(plascan_common_string_utils STATIC
  string_utils/StringParsing.cpp
  string_utils/StringTransform.cpp
)

add_library(plascan_common_model STATIC
  model/FeatureExtractorModelCatalog.cpp
  model/Sam21ModelCatalog.cpp
  model/TorchScriptModelResolver.cpp
  model/U2NetModelCatalog.cpp
)

add_library(plascan_common_project STATIC
  project/ProjectCameraIO.cpp
  project/ProjectIO.cpp
  project/ProjectMatchCatalog.cpp
  project/ProjectMetadata.cpp
  project/SparseResultQuality.cpp
)

target_include_directories(plascan_common_log PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/log
)

target_include_directories(plascan_common_io PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(plascan_common_string_utils PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(plascan_common_model PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(GDAL_INCLUDE_DIRS)
  target_include_directories(plascan_common_io PRIVATE ${GDAL_INCLUDE_DIRS})
endif()

target_include_directories(plascan_common_project PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(plascan_common_log PUBLIC cxx_std_17)
target_compile_features(plascan_common_io PUBLIC cxx_std_17)
target_compile_features(plascan_common_string_utils PUBLIC cxx_std_17)
target_compile_features(plascan_common_model PUBLIC cxx_std_17)
target_compile_features(plascan_common_project PUBLIC cxx_std_17)

target_link_libraries(plascan_common_io PUBLIC
  Qt6::Core
  ${OpenCV_LIBS}
  ${PLASCAN_GDAL_TARGET}
)

target_link_libraries(plascan_common_model PUBLIC Qt6::Core)
target_compile_definitions(plascan_common_model PRIVATE PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_link_libraries(plascan_common_project PUBLIC Qt6::Core camera)

if(BUILD_TESTS)
  add_subdirectory(string_utils/test)
  add_subdirectory(model/test)
  add_subdirectory(project/test)
endif()
