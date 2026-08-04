cmake_minimum_required(VERSION 3.16)

add_library(plascan_common_log STATIC
  log/Logger.cpp
)

add_library(plascan_common_io STATIC
  io/ImageIO.cpp
  io/ImageExifMetadata.cpp
  io/JsonObjectFile.cpp
  io/PathIO.cpp
)

add_library(plascan_common_json STATIC
  json/JsonObjectMerge.cpp
)

add_library(plascan_common_runtime STATIC
  runtime/PythonRuntimeLocator.cpp
)

add_library(plascan_common_string_utils STATIC
  string_utils/StringParsing.cpp
  string_utils/StringTransform.cpp
)

add_library(plascan_common_model STATIC
  model/ModelAssetCatalog.cpp
  model/ModelFileResolver.cpp
  model/U2NetModelCatalog.cpp
)

add_library(plascan_common_project STATIC
  project/AspMatchPointReader.cpp
  project/PlascanArchive.cpp
  project/ProjectArtifactIO.cpp
  project/ProjectAssetInspection.cpp
  project/ProjectAssetImporter.cpp
  project/ProjectArchivePath.cpp
  project/ProjectChunkStore.cpp
  project/ProjectChunkIndex.cpp
  project/ProjectDocumentModel.cpp
  project/ProjectConfigManager.cpp
  project/ProjectIO.cpp
  project/ProjectLock.cpp
  project/ProjectMatchCatalog.cpp
  project/ProjectMetadata.cpp
  project/ProjectPackageLayout.cpp
  project/ProjectResourceStore.cpp
  project/ProjectSession.cpp
  project/ProjectSessionModel.cpp
  project/ProjectSharedImageStore.cpp
  project/ProjectWorkspaceStore.cpp
  project/ProjectUiConfigManager.cpp
  project/ProjectWorkflowConfigManager.cpp
  project/PortableProjectFormat.cpp
  project/SparseResultQuality.cpp
)

target_include_directories(plascan_common_log PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/log
)

target_include_directories(plascan_common_io PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(plascan_common_json PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(plascan_common_runtime PUBLIC
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

target_compile_features(plascan_common_log PUBLIC cxx_std_20)
target_compile_features(plascan_common_io PUBLIC cxx_std_20)
target_compile_features(plascan_common_json PUBLIC cxx_std_20)
target_compile_features(plascan_common_runtime PUBLIC cxx_std_20)
target_compile_features(plascan_common_string_utils PUBLIC cxx_std_20)
target_compile_features(plascan_common_model PUBLIC cxx_std_20)
target_compile_features(plascan_common_project PUBLIC cxx_std_20)

target_link_libraries(plascan_common_io PUBLIC
  Qt6::Core
  ${OpenCV_LIBS}
  ${PLASCAN_GDAL_TARGET}
)

target_link_libraries(plascan_common_json PUBLIC Qt6::Core)
target_link_libraries(plascan_common_runtime PUBLIC Qt6::Core)
target_link_libraries(plascan_common_model PUBLIC Qt6::Core)
target_compile_definitions(plascan_common_model PRIVATE PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_link_libraries(plascan_common_project PUBLIC
  Qt6::Core
  Qt6::Concurrent
  plascan_common_json
  plascan_common_log
  ${PLASCAN_LIBZIP_TARGET}
)

set_target_properties(plascan_common_project PROPERTIES AUTOMOC ON)

if(BUILD_TESTS)
  add_subdirectory(log/test)
  add_subdirectory(io/test)
  add_subdirectory(string_utils/test)
  add_subdirectory(project/test)
endif()
