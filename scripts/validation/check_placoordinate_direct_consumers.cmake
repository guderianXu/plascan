if(NOT PLASCAN_SOURCE_ROOT)
    message(FATAL_ERROR "PLASCAN_SOURCE_ROOT is required")
endif()

file(GLOB_RECURSE _source_files LIST_DIRECTORIES FALSE
    "${PLASCAN_SOURCE_ROOT}/src/*.h"
    "${PLASCAN_SOURCE_ROOT}/src/*.hpp"
    "${PLASCAN_SOURCE_ROOT}/src/*.cpp"
    "${PLASCAN_SOURCE_ROOT}/src/*.cc"
    "${PLASCAN_SOURCE_ROOT}/src/*.cu"
    "${PLASCAN_SOURCE_ROOT}/src/*.cuh"
    "${PLASCAN_SOURCE_ROOT}/src/*.cmake"
    "${PLASCAN_SOURCE_ROOT}/src/CMakeLists.txt"
    "${PLASCAN_SOURCE_ROOT}/tests/*.h"
    "${PLASCAN_SOURCE_ROOT}/tests/*.hpp"
    "${PLASCAN_SOURCE_ROOT}/tests/*.cpp"
    "${PLASCAN_SOURCE_ROOT}/tests/*.cc"
    "${PLASCAN_SOURCE_ROOT}/tests/*.cu"
    "${PLASCAN_SOURCE_ROOT}/tests/*.cuh"
    "${PLASCAN_SOURCE_ROOT}/tests/*.cmake"
    "${PLASCAN_SOURCE_ROOT}/tests/CMakeLists.txt"
)

set(_legacy_tokens
    "coordinate_system/"
    "xjw::coordinate_system"
    "coordinate_system::"
    "coordinate_system_types"
    "coordinate_system_transform"
    "coordinate_system_gdal"
    "coordinate_system_json"
)

set(_legacy_files)
foreach(_source_file IN LISTS _source_files)
    file(READ "${_source_file}" _contents)
    foreach(_token IN LISTS _legacy_tokens)
        string(FIND "${_contents}" "${_token}" _offset)
        if(NOT _offset EQUAL -1)
            file(RELATIVE_PATH _relative_path "${PLASCAN_SOURCE_ROOT}" "${_source_file}")
            list(APPEND _legacy_files "${_relative_path}: ${_token}")
        endif()
    endforeach()
endforeach()

if(_legacy_files)
    list(JOIN _legacy_files "\n  " _legacy_report)
    message(FATAL_ERROR "PlaScan code still uses removed coordinate-system API entry points:\n  ${_legacy_report}")
endif()
