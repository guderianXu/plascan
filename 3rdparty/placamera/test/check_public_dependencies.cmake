file(GLOB_RECURSE _placamera_public_headers
    "${PLACAMERA_PUBLIC_INCLUDE_DIR}/*.h"
    "${PLACAMERA_PUBLIC_INCLUDE_DIR}/*.hpp"
)
if(NOT _placamera_public_headers)
    message(FATAL_ERROR "No PlaCamera public headers found in ${PLACAMERA_PUBLIC_INCLUDE_DIR}")
endif()

foreach(_header IN LISTS _placamera_public_headers)
    file(READ "${_header}" _contents)
    if(_contents MATCHES "#[ \t]*include[ \t]*[<\"]Q")
        message(FATAL_ERROR "Qt include leaked into public header: ${_header}")
    endif()
    foreach(_forbidden
            "xjw::" "PlaScan" "plascan" "PlaPoint" "plapoint"
            "PlaMatrix" "plamatrix" "PlaBundle" "plabundle"
            "OpenCV" "opencv" "GDAL" "gdal" "nlohmann" "Eigen"
            "/home/" "/Users/" "/workspace/")
        string(FIND "${_contents}" "${_forbidden}" _position)
        if(NOT _position EQUAL -1)
            message(FATAL_ERROR "Forbidden token '${_forbidden}' in public header: ${_header}")
        endif()
    endforeach()
    if(_contents MATCHES "[A-Za-z]:\\\\")
        message(FATAL_ERROR "Absolute Windows path leaked into public header: ${_header}")
    endif()
endforeach()
