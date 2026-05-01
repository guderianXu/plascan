cmake_minimum_required(VERSION 3.16)

# SuperPoint
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/superpoint/CMakeLists.txt)
  set(SUPERPOINT_INSTALL OFF CACHE BOOL "Disable SuperPoint standalone install/export in PlaScan build" FORCE)
  add_subdirectory(superpoint)
  message(STATUS "plascan_core: building feature extractor - SuperPoint")
else()
  message(WARNING "plascan_core: SuperPoint feature extractor not found")
endif()

add_library(feature_extractors_traditional STATIC
  FeatureData.cpp
  FeatureData.h
  tradition/TraditionalFeatureExtractor.cpp
  tradition/TraditionalFeatureExtractor.h
)

target_include_directories(feature_extractors_traditional
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tradition
    ${CMAKE_CURRENT_SOURCE_DIR}/superpoint
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(feature_extractors_traditional
  PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

if(TARGET superpoint)
  target_link_libraries(feature_extractors_traditional PUBLIC superpoint)
endif()

add_executable(traditional_feature_extractor_orb_diagnostic
  tradition/test_traditional_feature_extractor_orb.cpp
)

target_link_libraries(traditional_feature_extractor_orb_diagnostic
  PRIVATE
    feature_extractors_traditional
    ${OpenCV_LIBS}
)

target_compile_definitions(traditional_feature_extractor_orb_diagnostic
  PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
)

# ── DISK extractor ──
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/disk/CMakeLists.txt)
  add_subdirectory(disk)
  message(STATUS "plascan_core: building feature extractor - DISK")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/aliked/CMakeLists.txt)
  add_subdirectory(aliked)
  message(STATUS "plascan_core: building feature extractor - ALIKED")
endif()
