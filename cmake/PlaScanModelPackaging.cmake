include_guard(GLOBAL)

function(plascan_verify_bundled_model)
  set(_one_value_args LABEL FILE EXPECTED_SIZE EXPECTED_SHA256)
  cmake_parse_arguments(MODEL "" "${_one_value_args}" "" ${ARGN})

  foreach(_required_arg IN ITEMS LABEL FILE EXPECTED_SIZE EXPECTED_SHA256)
    if(NOT DEFINED MODEL_${_required_arg} OR "${MODEL_${_required_arg}}" STREQUAL "")
      message(FATAL_ERROR
        "plascan_verify_bundled_model requires ${_required_arg}")
    endif()
  endforeach()

  if(NOT EXISTS "${MODEL_FILE}" OR IS_DIRECTORY "${MODEL_FILE}")
    message(FATAL_ERROR
      "Bundled ${MODEL_LABEL} model is missing: ${MODEL_FILE}\n"
      "Download the models-v1.1.0 asset or configure its PLASCAN_*_ONNX_PATH "
      "before running cmake --install or CPack.")
  endif()

  file(SIZE "${MODEL_FILE}" _actual_size)
  if(NOT "${_actual_size}" STREQUAL "${MODEL_EXPECTED_SIZE}")
    message(FATAL_ERROR
      "Bundled ${MODEL_LABEL} model has an unexpected size: ${MODEL_FILE}\n"
      "Expected ${MODEL_EXPECTED_SIZE} bytes, got ${_actual_size} bytes.")
  endif()

  file(SHA256 "${MODEL_FILE}" _actual_sha256)
  string(TOLOWER "${MODEL_EXPECTED_SHA256}" _expected_sha256)
  string(TOLOWER "${_actual_sha256}" _actual_sha256)
  if(NOT "${_actual_sha256}" STREQUAL "${_expected_sha256}")
    message(FATAL_ERROR
      "Bundled ${MODEL_LABEL} model failed SHA-256 verification: ${MODEL_FILE}\n"
      "Expected ${_expected_sha256}, got ${_actual_sha256}.")
  endif()

  message(STATUS
    "Verified bundled ${MODEL_LABEL} model: ${MODEL_FILE} (${_actual_size} bytes)")
endfunction()
