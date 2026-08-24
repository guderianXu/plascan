include_guard(GLOBAL)

include(FetchContent)

set(PLASCAN_ONNXRUNTIME_VERSION "1.29.0" CACHE STRING
  "Pinned ONNX Runtime version used by portable neural-network inference" FORCE)
set(PLASCAN_ONNXRUNTIME_API_VERSION "29" CACHE INTERNAL
  "C API level corresponding to the pinned ONNX Runtime release" FORCE)
set(PLASCAN_ONNXRUNTIME_ROOT "" CACHE PATH
  "Optional preinstalled ONNX Runtime root containing include/ and lib/")
set(PLASCAN_ONNXRUNTIME_ARCHIVE "" CACHE FILEPATH
  "Optional local ONNX Runtime release archive; its SHA-256 is still verified")
set(PLASCAN_ONNXRUNTIME_CACHE_DIR
  "${CMAKE_SOURCE_DIR}/build/env/downloads/onnxruntime/${PLASCAN_ONNXRUNTIME_VERSION}"
  CACHE PATH "Reusable download cache for the pinned ONNX Runtime archive")

if(NOT PLASCAN_ONNXRUNTIME_VERSION STREQUAL "1.29.0")
  message(FATAL_ERROR
    "PlaScan currently pins ONNX Runtime 1.29.0; unsupported override: "
    "${PLASCAN_ONNXRUNTIME_VERSION}")
endif()

set(_plascan_ort_platform "")
set(_plascan_ort_archive_hash "")
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(_plascan_ort_platform "osx-arm64")
  set(_plascan_ort_archive_hash
    "d0706fc34f315d8c88639d0a8c81f2e09e815f282cabed3493c06a054352cf92")
elseif(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_plascan_ort_platform "win-x64")
  set(_plascan_ort_archive_hash
    "c9b4b7086b529ad814f428c1bad028e20a25d7dc0699836775faace4ab5b78b2")
elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_plascan_ort_platform "linux-x64")
  set(_plascan_ort_archive_hash
    "c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba")
endif()

if(PLASCAN_ONNXRUNTIME_ROOT)
  get_filename_component(_plascan_ort_root
    "${PLASCAN_ONNXRUNTIME_ROOT}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
else()
  if(NOT _plascan_ort_platform)
    message(FATAL_ERROR
      "No pinned ONNX Runtime binary is available for "
      "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. Set PLASCAN_ONNXRUNTIME_ROOT.")
  endif()
  if(WIN32)
    set(_plascan_ort_archive_extension "zip")
  else()
    set(_plascan_ort_archive_extension "tgz")
  endif()
  string(CONCAT _plascan_ort_archive_url
    "https://github.com/microsoft/onnxruntime/releases/download/"
    "v${PLASCAN_ONNXRUNTIME_VERSION}/onnxruntime-${_plascan_ort_platform}-"
    "${PLASCAN_ONNXRUNTIME_VERSION}.${_plascan_ort_archive_extension}")
  if(PLASCAN_ONNXRUNTIME_ARCHIVE)
    get_filename_component(_plascan_ort_archive_url
      "${PLASCAN_ONNXRUNTIME_ARCHIVE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    if(NOT EXISTS "${_plascan_ort_archive_url}")
      message(FATAL_ERROR
        "PLASCAN_ONNXRUNTIME_ARCHIVE does not exist: ${_plascan_ort_archive_url}")
    endif()
  endif()
  FetchContent_Declare(plascan_onnxruntime_prebuilt
    URL "${_plascan_ort_archive_url}"
    URL_HASH "SHA256=${_plascan_ort_archive_hash}"
    DOWNLOAD_DIR "${PLASCAN_ONNXRUNTIME_CACHE_DIR}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(plascan_onnxruntime_prebuilt)
  set(_plascan_ort_root "${plascan_onnxruntime_prebuilt_SOURCE_DIR}")
endif()

set(_plascan_ort_include_dir "${_plascan_ort_root}/include")
if(WIN32)
  set(_plascan_ort_link_library "${_plascan_ort_root}/lib/onnxruntime.lib")
  set(_plascan_ort_runtime_library "${_plascan_ort_root}/lib/onnxruntime.dll")
elseif(APPLE)
  set(_plascan_ort_link_library
    "${_plascan_ort_root}/lib/libonnxruntime.${PLASCAN_ONNXRUNTIME_VERSION}.dylib")
  set(_plascan_ort_runtime_library "${_plascan_ort_link_library}")
else()
  set(_plascan_ort_link_library
    "${_plascan_ort_root}/lib/libonnxruntime.so.${PLASCAN_ONNXRUNTIME_VERSION}")
  set(_plascan_ort_runtime_library "${_plascan_ort_link_library}")
endif()

foreach(_plascan_ort_required IN ITEMS
    "${_plascan_ort_include_dir}/onnxruntime_cxx_api.h"
    "${_plascan_ort_link_library}"
    "${_plascan_ort_runtime_library}")
  if(NOT EXISTS "${_plascan_ort_required}")
    message(FATAL_ERROR
      "The ONNX Runtime package is incomplete; missing: ${_plascan_ort_required}")
  endif()
endforeach()

add_library(plascan_onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(plascan_onnxruntime PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_plascan_ort_include_dir}")
if(WIN32)
  set_target_properties(plascan_onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${_plascan_ort_link_library}"
    IMPORTED_LOCATION "${_plascan_ort_runtime_library}")
else()
  set_target_properties(plascan_onnxruntime PROPERTIES
    IMPORTED_LOCATION "${_plascan_ort_link_library}")
endif()

set(PLASCAN_ONNXRUNTIME_RUNTIME_LIBRARY "${_plascan_ort_runtime_library}"
  CACHE INTERNAL "ONNX Runtime shared library deployed with PlaScan")
set(PLASCAN_ONNXRUNTIME_LICENSE_FILE "${_plascan_ort_root}/LICENSE"
  CACHE INTERNAL "ONNX Runtime license deployed with PlaScan")

function(plascan_deploy_onnxruntime target_name)
  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR
      "plascan_deploy_onnxruntime: target does not exist: ${target_name}")
  endif()
  add_custom_command(TARGET "${target_name}" POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${PLASCAN_ONNXRUNTIME_RUNTIME_LIBRARY}"
      "$<TARGET_FILE_DIR:${target_name}>/"
    VERBATIM
    COMMENT "Deploying ONNX Runtime for ${target_name}")
endfunction()

message(STATUS
  "plascan: ONNX Runtime ${PLASCAN_ONNXRUNTIME_VERSION} (${_plascan_ort_platform})")
if(NOT PLASCAN_ONNXRUNTIME_ROOT)
  message(STATUS "plascan: ONNX Runtime archive cache: ${PLASCAN_ONNXRUNTIME_CACHE_DIR}")
endif()
