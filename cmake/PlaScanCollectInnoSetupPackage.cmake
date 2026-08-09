cmake_minimum_required(VERSION 3.27)

if(NOT CPACK_GENERATOR STREQUAL "INNOSETUP")
  return()
endif()

foreach(_required_variable
    CPACK_PACKAGE_DIRECTORY
    CPACK_PACKAGE_FILE_NAME
    CPACK_PACKAGE_FILES)
  if(NOT DEFINED ${_required_variable} OR "${${_required_variable}}" STREQUAL "")
    message(FATAL_ERROR
      "PlaScan INNOSETUP post-build: ${_required_variable} is not set")
  endif()
endforeach()

set(_package_base "${CPACK_PACKAGE_FILE_NAME}")
set(_setup_candidates "")
foreach(_package_file IN LISTS CPACK_PACKAGE_FILES)
  if(_package_file MATCHES "\\.exe$")
    list(APPEND _setup_candidates "${_package_file}")
  endif()
endforeach()
list(LENGTH _setup_candidates _setup_candidate_count)
if(NOT _setup_candidate_count EQUAL 1)
  message(FATAL_ERROR
    "PlaScan INNOSETUP post-build: expected exactly one setup executable in "
    "CPACK_PACKAGE_FILES, got ${_setup_candidate_count}: ${CPACK_PACKAGE_FILES}")
endif()
list(GET _setup_candidates 0 _setup_executable)
get_filename_component(_setup_name "${_setup_executable}" NAME)
if(NOT _setup_name STREQUAL "${_package_base}.exe")
  message(FATAL_ERROR
    "PlaScan INNOSETUP post-build: unexpected setup executable name: "
    "${_setup_name} (expected ${_package_base}.exe)")
endif()
get_filename_component(_inno_output_directory
  "${_setup_executable}" DIRECTORY)

file(GLOB _package_slices LIST_DIRECTORIES FALSE
  "${_inno_output_directory}/${_package_base}-*.bin")
if(NOT _package_slices)
  message(FATAL_ERROR
    "PlaScan INNOSETUP post-build: disk spanning is enabled, but no "
    "${_package_base}-*.bin slices were generated")
endif()
list(SORT _package_slices COMPARE NATURAL ORDER ASCENDING)

file(MAKE_DIRECTORY "${CPACK_PACKAGE_DIRECTORY}")
set(_slice_names "")
set(_expected_slice_index 1)
foreach(_package_slice IN LISTS _package_slices)
  get_filename_component(_slice_name "${_package_slice}" NAME)
  string(LENGTH "${_package_base}-" _slice_prefix_length)
  string(SUBSTRING "${_slice_name}" ${_slice_prefix_length} -1 _slice_suffix)
  if(NOT _slice_suffix MATCHES "^[0-9]+\\.bin$")
    message(FATAL_ERROR
      "PlaScan INNOSETUP post-build: unexpected slice name: ${_slice_name}")
  endif()
  string(REGEX REPLACE "\\.bin$" "" _slice_index "${_slice_suffix}")
  if(NOT _slice_index EQUAL _expected_slice_index)
    message(FATAL_ERROR
      "PlaScan INNOSETUP post-build: slice numbering is not continuous; "
      "expected ${_expected_slice_index}, got ${_slice_index} (${_slice_name})")
  endif()
  math(EXPR _expected_slice_index "${_expected_slice_index} + 1")
  list(APPEND _slice_names "${_slice_name}")
endforeach()

set(_package_files "${_setup_executable}" ${_package_slices})
set(_output_package_files "")
set(_temporary_package_files "")
set(_manifest_path
  "${CPACK_PACKAGE_DIRECTORY}/${_package_base}-INNOSETUP.sha256")
file(REMOVE "${_manifest_path}")

foreach(_package_file IN LISTS _package_files)
  get_filename_component(_package_name "${_package_file}" NAME)
  set(_output_package_file
    "${CPACK_PACKAGE_DIRECTORY}/${_package_name}")
  set(_temporary_package_file "${_output_package_file}.cpack-tmp")
  file(REMOVE "${_temporary_package_file}")
  file(COPY_FILE
    "${_package_file}"
    "${_temporary_package_file}"
    ONLY_IF_DIFFERENT
    INPUT_MAY_BE_RECENT)
  list(APPEND _output_package_files "${_output_package_file}")
  list(APPEND _temporary_package_files "${_temporary_package_file}")
endforeach()

set(_checksum_manifest "")
list(LENGTH _output_package_files _output_package_count)
math(EXPR _last_package_index "${_output_package_count} - 1")
foreach(_package_index RANGE 0 ${_last_package_index})
  list(GET _output_package_files
    ${_package_index} _output_package_file)
  list(GET _temporary_package_files
    ${_package_index} _temporary_package_file)
  file(SIZE "${_temporary_package_file}" _package_size)
  if(_package_size GREATER_EQUAL 2147483648)
    message(FATAL_ERROR
      "PlaScan INNOSETUP post-build: release asset must be smaller than "
      "2 GiB, but ${_output_package_file} is ${_package_size} bytes")
  endif()

  file(SHA256 "${_temporary_package_file}" _package_sha256)
  get_filename_component(_package_name "${_output_package_file}" NAME)
  string(APPEND _checksum_manifest
    "${_package_sha256}  ${_package_name}\n")
endforeach()

# Only publish files after every temporary copy has passed all release gates.
foreach(_package_index RANGE 0 ${_last_package_index})
  list(GET _output_package_files
    ${_package_index} _output_package_file)
  list(GET _temporary_package_files
    ${_package_index} _temporary_package_file)
  file(RENAME "${_temporary_package_file}" "${_output_package_file}")
endforeach()

# Remove slices left by an older package with the same version and a different
# slice count. Only strict numeric -N.bin files are in scope.
file(GLOB _old_output_slices LIST_DIRECTORIES FALSE
  "${CPACK_PACKAGE_DIRECTORY}/${_package_base}-*.bin")
foreach(_old_output_slice IN LISTS _old_output_slices)
  get_filename_component(_old_output_name "${_old_output_slice}" NAME)
  string(LENGTH "${_package_base}-" _slice_prefix_length)
  string(SUBSTRING "${_old_output_name}"
    ${_slice_prefix_length} -1 _old_output_suffix)
  if(_old_output_suffix MATCHES "^[0-9]+\\.bin$"
      AND NOT _old_output_name IN_LIST _slice_names)
    file(REMOVE "${_old_output_slice}")
  endif()
endforeach()

set(_temporary_manifest_path "${_manifest_path}.tmp")
file(WRITE "${_temporary_manifest_path}" "${_checksum_manifest}")
file(RENAME "${_temporary_manifest_path}" "${_manifest_path}")
list(LENGTH _package_slices _slice_count)
message(STATUS
  "PlaScan INNOSETUP package: copied ${_slice_count} slices and wrote "
  "${_manifest_path}")
