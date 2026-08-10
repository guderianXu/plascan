find_program(_dpkg_deb_executable NAMES dpkg-deb REQUIRED)

set(_deb_packages "")
foreach(_package_file IN LISTS CPACK_PACKAGE_FILES)
  if(_package_file MATCHES "[.]deb$")
    list(APPEND _deb_packages "${_package_file}")
  endif()
endforeach()
list(LENGTH _deb_packages _deb_count)
if(NOT _deb_count EQUAL 1)
  message(FATAL_ERROR
    "Expected exactly one PlaScan DEB package, found ${_deb_count}: ${_deb_packages}")
endif()
list(GET _deb_packages 0 _deb_package)

execute_process(
  COMMAND "${_dpkg_deb_executable}" --field "${_deb_package}"
  RESULT_VARIABLE _field_result
  OUTPUT_VARIABLE _fields
  ERROR_VARIABLE _field_error)
if(NOT _field_result EQUAL 0)
  message(FATAL_ERROR
    "dpkg-deb --field failed for ${_deb_package}: ${_field_error}")
endif()

foreach(_required_field IN ITEMS
    "Architecture: amd64"
    "Section: science"
    "Priority: optional"
    "Depends:")
  string(FIND "${_fields}" "${_required_field}" _field_index)
  if(_field_index EQUAL -1)
    message(FATAL_ERROR
      "DEB metadata is missing '${_required_field}':\n${_fields}")
  endif()
endforeach()
if(NOT _fields MATCHES
   "(^|\n)Package: ${PLASCAN_EXPECTED_DEB_PACKAGE_NAME}(\n|$)")
  message(FATAL_ERROR "Unexpected DEB package name:\n${_fields}")
endif()
if(NOT _fields MATCHES
   "(^|\n)Version: ${PLASCAN_EXPECTED_DEB_PACKAGE_VERSION}(\n|$)")
  message(FATAL_ERROR "Unexpected DEB package version:\n${_fields}")
endif()

set(_expected_dependencies "${PLASCAN_EXPECTED_DEB_DEPENDS}")
string(REPLACE ", " ";" _expected_dependencies "${_expected_dependencies}")
foreach(_expected_dependency IN LISTS _expected_dependencies)
  string(FIND "${_fields}" "${_expected_dependency}" _dependency_index)
  if(_dependency_index EQUAL -1)
    message(FATAL_ERROR
      "DEB metadata is missing dependency '${_expected_dependency}':\n${_fields}")
  endif()
endforeach()

if(PLASCAN_EXPECTED_DEB_PACKAGE_VARIANT STREQUAL "cuda")
  foreach(_cuda_field IN ITEMS
      "Provides: plascan"
      "Conflicts: plascan"
      "Replaces: plascan")
    string(FIND "${_fields}" "${_cuda_field}" _cuda_field_index)
    if(_cuda_field_index EQUAL -1)
      message(FATAL_ERROR
        "CUDA DEB metadata is missing '${_cuda_field}':\n${_fields}")
    endif()
  endforeach()
  set(_expected_doc_path "./usr/share/doc/plascan-cuda/copyright")
else()
  string(FIND "${_fields}" "Conflicts: plascan-cuda" _conflicts_index)
  if(_conflicts_index EQUAL -1)
    message(FATAL_ERROR
      "Portable DEB metadata is missing 'Conflicts: plascan-cuda':\n${_fields}")
  endif()
  set(_expected_doc_path "./usr/share/doc/plascan/copyright")
endif()

execute_process(
  COMMAND "${_dpkg_deb_executable}" --contents "${_deb_package}"
  RESULT_VARIABLE _contents_result
  OUTPUT_VARIABLE _contents
  ERROR_VARIABLE _contents_error)
if(NOT _contents_result EQUAL 0)
  message(FATAL_ERROR
    "dpkg-deb --contents failed for ${_deb_package}: ${_contents_error}")
endif()

set(_required_paths
    "./opt/plascan/bin/plascan_gui.bin"
    "./opt/plascan/resources/models/U2Net_v1.onnx"
    "./opt/plascan/resources/models/lightglue_tensorrt/lightglue_sift_bucket4096.onnx"
    "./usr/bin/plascan"
    "./usr/share/applications/plascan.desktop"
    "${_expected_doc_path}")
set(_birefnet_paths
    "./opt/plascan/resources/models/birefnet_dynamic/BiRefNet_dynamic_1024.onnx"
    "./opt/plascan/resources/models/birefnet_dynamic/BiRefNet_dynamic_1024.provenance.json"
    "./opt/plascan/share/plascan/models/BiRefNet_NOTICE.md"
    "./opt/plascan/share/plascan/models/BiRefNet-MIT.txt"
    "./opt/plascan/share/plascan/models/models-v1.2.0.sha256")
if(PLASCAN_EXPECTED_DEB_PACKAGE_VARIANT STREQUAL "cuda")
  list(APPEND _required_paths ${_birefnet_paths})
else()
  foreach(_unexpected_birefnet_path IN LISTS _birefnet_paths)
    string(FIND "${_contents}" "${_unexpected_birefnet_path}"
      _unexpected_birefnet_index)
    if(NOT _unexpected_birefnet_index EQUAL -1)
      message(FATAL_ERROR
        "Portable CPU DEB must not contain BiRefNet Dynamic: "
        "${_unexpected_birefnet_path}")
    endif()
  endforeach()
endif()

foreach(_required_path IN LISTS _required_paths)
  string(FIND "${_contents}" "${_required_path}" _path_index)
  if(_path_index EQUAL -1)
    message(FATAL_ERROR
      "DEB package is missing required path: ${_required_path}")
  endif()
endforeach()

if(_contents MATCHES "[ ]+[.]/opt/plascan/include/" OR
   _contents MATCHES "[.](a|la)( -> [^\n]+)?\n" OR
   _contents MATCHES "/cmake/")
  message(FATAL_ERROR
    "DEB package contains development headers, static libraries or CMake exports")
endif()

if(NOT CPACK_TOPLEVEL_DIRECTORY)
  message(FATAL_ERROR "CPACK_TOPLEVEL_DIRECTORY is unavailable for DEB control verification")
endif()
set(_control_directory
  "${CPACK_TOPLEVEL_DIRECTORY}/_plascan_deb_control_verification")
file(REMOVE_RECURSE "${_control_directory}")
file(MAKE_DIRECTORY "${_control_directory}")
execute_process(
  COMMAND "${_dpkg_deb_executable}" --control
    "${_deb_package}" "${_control_directory}"
  RESULT_VARIABLE _control_result
  ERROR_VARIABLE _control_error)
if(NOT _control_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to extract DEB control archive: ${_control_error}")
endif()
find_program(_stat_executable NAMES stat REQUIRED)
foreach(_control_script IN ITEMS postinst postrm)
  set(_control_script_path "${_control_directory}/${_control_script}")
  if(NOT EXISTS "${_control_script_path}")
    message(FATAL_ERROR "DEB control script is missing: ${_control_script}")
  endif()
  execute_process(
    COMMAND "${_stat_executable}" -c "%a" "${_control_script_path}"
    OUTPUT_VARIABLE _control_mode
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  if(NOT _control_mode STREQUAL "755")
    message(FATAL_ERROR
      "DEB control script ${_control_script} must be mode 755, got ${_control_mode}")
  endif()
endforeach()

file(SIZE "${_deb_package}" _deb_size)
message(STATUS
  "Verified PlaScan DEB package: ${_deb_package} (${_deb_size} bytes)")
