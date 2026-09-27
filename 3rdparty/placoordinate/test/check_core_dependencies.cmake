if(NOT PLACOORDINATE_PUBLIC_INCLUDE_DIR)
    message(FATAL_ERROR "PLACOORDINATE_PUBLIC_INCLUDE_DIR is required")
endif()

file(GLOB_RECURSE _placoordinate_core_headers
    "${PLACOORDINATE_PUBLIC_INCLUDE_DIR}/context/*.h"
    "${PLACOORDINATE_PUBLIC_INCLUDE_DIR}/transform/*.h"
    "${PLACOORDINATE_PUBLIC_INCLUDE_DIR}/types/*.h"
)
list(APPEND _placoordinate_core_headers
    "${PLACOORDINATE_PUBLIC_INCLUDE_DIR}/placoordinate.h"
)

set(_placoordinate_forbidden_tokens
    "xjw::"
    "coordinate_system/"
    "#include <Q"
    "#include <opencv"
    "#include <gdal"
    "#include <nlohmann"
)

foreach(_placoordinate_header IN LISTS _placoordinate_core_headers)
    file(READ "${_placoordinate_header}" _placoordinate_contents)
    foreach(_placoordinate_token IN LISTS _placoordinate_forbidden_tokens)
        string(FIND "${_placoordinate_contents}" "${_placoordinate_token}" _placoordinate_index)
        if(NOT _placoordinate_index EQUAL -1)
            message(FATAL_ERROR
                "PlaCoordinate core header ${_placoordinate_header} contains forbidden dependency token: "
                "${_placoordinate_token}")
        endif()
    endforeach()
endforeach()
