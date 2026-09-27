cmake_minimum_required(VERSION 3.16)

find_package(nlohmann_json CONFIG REQUIRED)

add_library(plascan_common_file STATIC
  file/FileIO.cpp
  file/AtomicFile.cpp
)
target_include_directories(plascan_common_file PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(plascan_common_file PUBLIC cxx_std_20)

add_library(plascan_common_json_io STATIC file/JsonFile.cpp)
target_include_directories(plascan_common_json_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(plascan_common_json_io PUBLIC plascan_common_file nlohmann_json::nlohmann_json)
target_compile_features(plascan_common_json_io PUBLIC cxx_std_20)

add_library(plascan_common_plafs STATIC
  plafs/PlaChunkLayout.cpp
  plafs/PlaDir.cpp
  plafs/PlaFile.cpp
  plafs/PlaProjectLayout.cpp
)
target_include_directories(plascan_common_plafs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(plascan_common_plafs PUBLIC plascan_common_file)
target_compile_features(plascan_common_plafs PUBLIC cxx_std_20)

add_library(plascan_common_log STATIC
  log/Logger.cpp
)

add_library(plascan_common_io STATIC
  io/ImageIO.cpp
  io/ImageSizeReader.cpp
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

add_library(plascan_common_concurrency INTERFACE)

add_library(plascan_common_string_utils STATIC
  string_utils/StringParsing.cpp
  string_utils/StringTransform.cpp
)

add_library(plascan_common_model STATIC
  model/BiRefNetModelCatalog.cpp
  model/ModelAssetCatalog.cpp
  model/ModelFileResolver.cpp
  model/U2NetModelCatalog.cpp
)

add_library(plascan_common_project STATIC
  project/camera_reference/CameraReferenceSet.cpp
  project/camera_reference/CameraReferenceSetJson.cpp
  project/camera_reference/CameraReferenceSetJsonDecode.cpp
  project/camera_reference/CameraReferenceSetStore.cpp
  project/AspMatchPointReader.cpp
  project/PlascanArchive.cpp
  project/ProjectArtifactIO.cpp
  project/ProjectAssetInspection.cpp
  project/ProjectAssetImporter.cpp
  project/ProjectArchivePath.cpp
  project/ProjectChunkStore.cpp
  project/ProjectChunkIndex.cpp
  project/ProjectDocumentModel.cpp
  project/ProjectExporter.cpp
  project/ProjectCameraIO.cpp
  project/ProjectCameraIO.h
  project/ProjectFramePinholeMetadataIO.cpp
  project/ProjectFramePinholeMetadataIO.h
  project/ProjectConfigManager.cpp
  project/ProjectIO.cpp
  project/ProjectLock.cpp
  project/ProjectMatchCatalog.cpp
  project/ProjectMetadata.cpp
  project/ProjectPackageLayout.cpp
  project/ProjectResourceStore.cpp
  project/ProjectSession.cpp
  project/ProjectSessionModel.cpp
  project/ProjectSharedImageGarbageCollector.cpp
  project/ProjectSharedImageStore.cpp
  project/ProjectSharedImageStoreCoordinator.cpp
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

target_include_directories(plascan_common_concurrency INTERFACE
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
  ${CMAKE_SOURCE_DIR}/src/core
)

target_compile_features(plascan_common_log PUBLIC cxx_std_20)
target_compile_features(plascan_common_io PUBLIC cxx_std_20)
target_compile_features(plascan_common_json PUBLIC cxx_std_20)
target_compile_features(plascan_common_runtime PUBLIC cxx_std_20)
target_compile_features(plascan_common_concurrency INTERFACE cxx_std_20)
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
  plascan_common_plafs
  plascan_common_json
  plascan_common_log
  plascan::placamera_runtime
  placamera::placamera
  placamera::reference
  ${PLASCAN_LIBZIP_TARGET}
)

set_target_properties(plascan_common_project PROPERTIES AUTOMOC ON)

if(BUILD_TESTS)
  add_subdirectory(concurrency/test)
  add_subdirectory(log/test)
  add_subdirectory(io/test)
  add_subdirectory(file/test)
  add_subdirectory(plafs/test)
  add_subdirectory(string_utils/test)
  add_subdirectory(project/test)
endif()
