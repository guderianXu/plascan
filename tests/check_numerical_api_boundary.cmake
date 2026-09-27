cmake_minimum_required(VERSION 3.18)
if(NOT IS_DIRECTORY "${PLASCAN_SOURCE_DIR}/src")
    message(FATAL_ERROR "PLASCAN_SOURCE_DIR must name the PlaScan source tree")
endif()
file(GLOB_RECURSE _sources
    "${PLASCAN_SOURCE_DIR}/src/*.h" "${PLASCAN_SOURCE_DIR}/src/*.hpp"
    "${PLASCAN_SOURCE_DIR}/src/*.cpp" "${PLASCAN_SOURCE_DIR}/src/*.cu" "${PLASCAN_SOURCE_DIR}/src/*.cuh"
    "${PLASCAN_SOURCE_DIR}/tests/*.h" "${PLASCAN_SOURCE_DIR}/tests/*.hpp"
    "${PLASCAN_SOURCE_DIR}/tests/*.cpp" "${PLASCAN_SOURCE_DIR}/tests/*.cu" "${PLASCAN_SOURCE_DIR}/tests/*.cuh")
set(_forbidden
    "(plamatrix|plapoint)::([A-Za-z_][A-Za-z_0-9]*::)*internal::"
    "plamatrix::(DenseMatrix|DeviceMatrix|CSRMatrix|COOMatrix|Vec3)[ \t]*<"
    "plapoint::(PointCloud|search::KdTree)[ \t]*<[ \t]*(float|double)([ \t]*[,>])"
    "plamatrix::Matrix[ \t]*<[^<>,]+>")
foreach(_source IN LISTS _sources)
    file(READ "${_source}" _content)
    foreach(_pattern IN LISTS _forbidden)
        if(_content MATCHES "${_pattern}")
            message(FATAL_ERROR "Non-public or obsolete numerical API in ${_source}: ${CMAKE_MATCH_0}")
        endif()
    endforeach()
    # Match directives, not literal source assertions in the tests.
    if(_content MATCHES "(^|\n)[ \t]*#[ \t]*include[ \t]*[<\"](plamatrix/internal/|plapoint/(internal|gpu|opencl)/)")
        message(FATAL_ERROR "Backend implementation header included by ${_source}: ${CMAKE_MATCH_0}")
    endif()
endforeach()
message(STATUS "PlaScan public numerical API boundary passed")
