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
if(NOT _fields MATCHES "(^|\n)Package: plascan(-cuda)?(\n|$)")
  message(FATAL_ERROR "Unexpected DEB package name:\n${_fields}")
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

foreach(_required_path IN ITEMS
    "./opt/plascan/bin/plascan_gui.bin"
    "./opt/plascan/resources/models/U2Net_v1.onnx"
    "./opt/plascan/resources/models/lightglue_tensorrt/lightglue_sift_bucket4096.onnx"
    "./usr/bin/plascan"
    "./usr/share/applications/plascan.desktop")
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

file(SIZE "${_deb_package}" _deb_size)
message(STATUS
  "Verified PlaScan DEB package: ${_deb_package} (${_deb_size} bytes)")
