include_guard(GLOBAL)

include(FetchContent)

set(PLASCAN_ONNXRUNTIME_VERSION "1.20.1" CACHE STRING
  "Pinned ONNX Runtime version used by portable neural-network inference")
set(PLASCAN_ONNXRUNTIME_ROOT "" CACHE PATH
  "Optional preinstalled ONNX Runtime root containing include/ and lib/")

if(NOT PLASCAN_ONNXRUNTIME_VERSION STREQUAL "1.20.1")
  message(FATAL_ERROR
    "PlaScan currently pins ONNX Runtime 1.20.1; unsupported override: "
    "${PLASCAN_ONNXRUNTIME_VERSION}")
endif()

set(_plascan_ort_platform "")
set(_plascan_ort_archive_hash "")
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(_plascan_ort_platform "osx-arm64")
  set(_plascan_ort_archive_hash
    "b678fc3c2354c771fea4fba420edeccfba205140088334df801e7fc40e83a57a")
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_plascan_ort_platform "osx-x86_64")
  set(_plascan_ort_archive_hash
    "0f73006813af2a1a5d1723ed7dfb694fc629d15037124081bb61b7bf7d99fc78")
elseif(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_plascan_ort_platform "win-x64")
  set(_plascan_ort_archive_hash
    "78d447051e48bd2e1e778bba378bec4ece11191c9e538cf7b2c4a4565e8f5581")
elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_plascan_ort_platform "linux-x64")
  set(_plascan_ort_archive_hash
    "67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105")
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
  FetchContent_Declare(plascan_onnxruntime_prebuilt
    URL "${_plascan_ort_archive_url}"
    URL_HASH "SHA256=${_plascan_ort_archive_hash}"
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
