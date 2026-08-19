if(NOT PLASCAN_SOURCE_DIR)
    message(FATAL_ERROR "PLASCAN_SOURCE_DIR is required")
endif()

set(_testRoot "${CMAKE_CURRENT_BINARY_DIR}/plascan-dependency-paths-test")
set(_testSource "${_testRoot}/source")
set(_validBuild "${_testRoot}/valid-build")
set(_invalidBuild "${_testRoot}/invalid-build")
set(_poisonPrefix "${_testRoot}/poison-conda")
set(_fakeToolchain "${_testRoot}/vcpkg.cmake")

file(REMOVE_RECURSE "${_testRoot}")
file(MAKE_DIRECTORY "${_testSource}" "${_poisonPrefix}")
file(WRITE "${_fakeToolchain}" [=[
set(VCPKG_MANIFEST_MODE ON CACHE BOOL "" FORCE)
set(VCPKG_TARGET_TRIPLET "x64-test" CACHE STRING "" FORCE)
]=])
file(WRITE "${_testSource}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.25)
project(PlascanDependencyPathsTest LANGUAGES NONE)

include("${PLASCAN_SOURCE_DIR}/cmake/PlascanDependencyPaths.cmake")
plascan_configure_dependency_paths()

if(NOT PLASCAN_DEPENDENCY_PROVIDER_SUMMARY STREQUAL
   "vcpkg(manifest,x64-test)")
    message(FATAL_ERROR
        "Unexpected provider summary: ${PLASCAN_DEPENDENCY_PROVIDER_SUMMARY}")
endif()
]=])

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "CONDA_PREFIX=${_poisonPrefix}"
        "${CMAKE_COMMAND}"
        -S "${_testSource}"
        -B "${_validBuild}"
        -DPLASCAN_SOURCE_DIR=${PLASCAN_SOURCE_DIR}
        -DCMAKE_TOOLCHAIN_FILE=${_fakeToolchain}
    RESULT_VARIABLE _validResult
    OUTPUT_VARIABLE _validOutput
    ERROR_VARIABLE _validError)
if(NOT _validResult EQUAL 0)
    message(FATAL_ERROR
        "vcpkg manifest configure failed:\n${_validOutput}\n${_validError}")
endif()

file(READ "${_validBuild}/CMakeCache.txt" _validCache)
string(FIND "${_validCache}" "${_poisonPrefix}" _poisonPosition)
if(NOT _poisonPosition EQUAL -1)
    message(FATAL_ERROR "CONDA_PREFIX leaked into the vcpkg-only CMake cache")
endif()
foreach(_removedVariable IN ITEMS
        PLASCAN_ENABLE_CONDA
        PLASCAN_CONDA_PREFIX
        PLASCAN_EFFECTIVE_CONDA_PREFIX
        PLASCAN_ENABLE_VCPKG)
    if(_validCache MATCHES "(^|\n)${_removedVariable}:")
        message(FATAL_ERROR
            "Removed dependency option remains in CMake cache: ${_removedVariable}")
    endif()
endforeach()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${_testSource}"
        -B "${_invalidBuild}"
        -DPLASCAN_SOURCE_DIR=${PLASCAN_SOURCE_DIR}
    RESULT_VARIABLE _invalidResult
    OUTPUT_VARIABLE _invalidOutput
    ERROR_VARIABLE _invalidError)
if(_invalidResult EQUAL 0)
    message(FATAL_ERROR "Configure without the vcpkg toolchain unexpectedly passed")
endif()
set(_invalidLog "${_invalidOutput}\n${_invalidError}")
if(NOT _invalidLog MATCHES "requires the vcpkg toolchain")
    message(FATAL_ERROR
        "Missing-toolchain failure was not actionable:\n${_invalidLog}")
endif()

file(REMOVE_RECURSE "${_testRoot}")
