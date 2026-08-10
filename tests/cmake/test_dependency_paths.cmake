if(NOT PLASCAN_SOURCE_DIR)
    message(FATAL_ERROR "PLASCAN_SOURCE_DIR is required")
endif()

include("${PLASCAN_SOURCE_DIR}/cmake/PlascanDependencyPaths.cmake")

set(_testRoot "${CMAKE_CURRENT_BINARY_DIR}/plascan-dependency-paths-test")
set(_oldCudaRoot "${_testRoot}/old-cuda")
set(_newCudaRoot "${_testRoot}/new-cuda")
file(REMOVE_RECURSE "${_testRoot}")
file(MAKE_DIRECTORY
    "${_oldCudaRoot}/bin"
    "${_oldCudaRoot}/include"
    "${_newCudaRoot}/include"
    "${_newCudaRoot}/lib")
file(WRITE "${_oldCudaRoot}/bin/nvcc" "old")
file(WRITE "${_oldCudaRoot}/include/cuda.h" "old")
file(WRITE "${_newCudaRoot}/include/cuda.h" "new")
file(WRITE "${_newCudaRoot}/lib/libcudart.so" "new")

set(PLASCAN_EFFECTIVE_CONDA_PREFIX "${_newCudaRoot}")
set(PLASCAN_AUTO_CONDA_CUDA_ROOT "${_oldCudaRoot}"
    CACHE INTERNAL "test old CUDA root" FORCE)
set(CMAKE_CUDA_COMPILER "${_oldCudaRoot}/bin/nvcc"
    CACHE FILEPATH "test stale CUDA compiler" FORCE)
set(CUDAToolkit_ROOT "${_oldCudaRoot}"
    CACHE PATH "test stale CUDA root" FORCE)
set(CMAKE_CUDA_FLAGS "-I${_oldCudaRoot}/include"
    CACHE STRING "test stale CUDA flags" FORCE)
if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()

plascan_configure_dependency_paths()

if(NOT PLASCAN_AUTO_CONDA_CUDA_ROOT STREQUAL _newCudaRoot)
    message(FATAL_ERROR
        "Expected current inferred CUDA root '${_newCudaRoot}', got "
        "'${PLASCAN_AUTO_CONDA_CUDA_ROOT}'")
endif()
if(NOT CUDAToolkit_ROOT STREQUAL _newCudaRoot)
    message(FATAL_ERROR
        "Expected CUDAToolkit_ROOT '${_newCudaRoot}', got '${CUDAToolkit_ROOT}'")
endif()
string(FIND "${CMAKE_CUDA_FLAGS}" "${_oldCudaRoot}" _staleFlagPosition)
if(NOT _staleFlagPosition EQUAL -1)
    message(FATAL_ERROR "Stale conda CUDA include remained in CMAKE_CUDA_FLAGS")
endif()
string(FIND "${CMAKE_CUDA_FLAGS}" "-I${_newCudaRoot}/include" _newFlagPosition)
if(_newFlagPosition EQUAL -1)
    message(FATAL_ERROR "Current conda CUDA include was not added to CMAKE_CUDA_FLAGS")
endif()
if(UNIX AND NOT APPLE AND EXISTS "/usr/bin/ar")
    string(FIND "${CMAKE_CXX_ARCHIVE_CREATE}" "/usr/bin/ar" _archiveToolPosition)
    if(_archiveToolPosition EQUAL -1)
        message(FATAL_ERROR "Mixed-toolchain archive rule did not reach the caller scope")
    endif()

    set(_reconfigureSource "${_testRoot}/reconfigure-source")
    set(_reconfigureBuild "${_testRoot}/reconfigure-build")
    file(MAKE_DIRECTORY "${_reconfigureSource}" "${_testRoot}/fake-conda")
    file(WRITE "${_reconfigureSource}/library.cpp"
        "int plascan_dependency_paths_reconfigure_test() { return 0; }\n")
    file(WRITE "${_reconfigureSource}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.25)
project(PlascanDependencyPathsReconfigure LANGUAGES CXX)

if(ENABLE_TEST_CONDA)
    set(PLASCAN_EFFECTIVE_CONDA_PREFIX "${TEST_CONDA_PREFIX}")
else()
    set(PLASCAN_EFFECTIVE_CONDA_PREFIX "")
endif()
include("${PLASCAN_SOURCE_DIR}/cmake/PlascanDependencyPaths.cmake")
plascan_configure_dependency_paths()

if(NOT ENABLE_TEST_CONDA)
    foreach(_requiredTool IN ITEMS CMAKE_LINKER CMAKE_AR CMAKE_RANLIB CMAKE_NM)
        if(NOT DEFINED ${_requiredTool} OR "${${_requiredTool}}" STREQUAL "")
            message(FATAL_ERROR
                "${_requiredTool} was cleared when mixed-toolchain mode was disabled")
        endif()
    endforeach()
    if(CMAKE_EXE_LINKER_FLAGS MATCHES "(^| )-B/usr/bin($| )" OR
       CMAKE_SHARED_LINKER_FLAGS MATCHES "(^| )-B/usr/bin($| )" OR
       CMAKE_MODULE_LINKER_FLAGS MATCHES "(^| )-B/usr/bin($| )")
        message(FATAL_ERROR "Mixed-toolchain linker flags survived reconfiguration")
    endif()
endif()

add_library(reconfigure_static STATIC library.cpp)
]=])

    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            -S "${_reconfigureSource}"
            -B "${_reconfigureBuild}"
            -DPLASCAN_SOURCE_DIR=${PLASCAN_SOURCE_DIR}
            -DTEST_CONDA_PREFIX=${_testRoot}/fake-conda
            -DENABLE_TEST_CONDA=ON
        RESULT_VARIABLE _mixedConfigureResult
        OUTPUT_VARIABLE _mixedConfigureOutput
        ERROR_VARIABLE _mixedConfigureError)
    if(NOT _mixedConfigureResult EQUAL 0)
        message(FATAL_ERROR
            "Initial mixed-toolchain configure failed:\n"
            "${_mixedConfigureOutput}\n${_mixedConfigureError}")
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            -S "${_reconfigureSource}"
            -B "${_reconfigureBuild}"
            -DPLASCAN_SOURCE_DIR=${PLASCAN_SOURCE_DIR}
            -DENABLE_TEST_CONDA=OFF
        RESULT_VARIABLE _systemConfigureResult
        OUTPUT_VARIABLE _systemConfigureOutput
        ERROR_VARIABLE _systemConfigureError)
    if(NOT _systemConfigureResult EQUAL 0)
        message(FATAL_ERROR
            "System-only reconfigure failed:\n"
            "${_systemConfigureOutput}\n${_systemConfigureError}")
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${_reconfigureBuild}"
        RESULT_VARIABLE _staticBuildResult
        OUTPUT_VARIABLE _staticBuildOutput
        ERROR_VARIABLE _staticBuildError)
    if(NOT _staticBuildResult EQUAL 0)
        message(FATAL_ERROR
            "Static library build after mixed-toolchain reconfigure failed:\n"
            "${_staticBuildOutput}\n${_staticBuildError}")
    endif()
endif()

# Reproduce the second configure after a mixed-toolchain build without relying
# on this test host to have a conda installation.  CMake has already enabled
# its languages at this point in a normal project, so these valid tools must
# remain available while only the mixed-mode flags are removed.
set(PLASCAN_USE_SYSTEM_BINUTILS_FOR_MIXED_TOOLCHAIN ON
    CACHE BOOL "test previous mixed-toolchain state" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "-B/usr/bin -Wl,test-exe"
    CACHE STRING "test executable linker flags" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,test-shared -B/usr/bin"
    CACHE STRING "test shared linker flags" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS "-B/usr/bin"
    CACHE STRING "test module linker flags" FORCE)
set(CMAKE_CUDA_FLAGS "-Xcompiler=-B/usr/bin --test-cuda-flag"
    CACHE STRING "test CUDA flags" FORCE)
foreach(_toolVarAndValue IN ITEMS
        "CMAKE_LINKER;/usr/bin/ld"
        "CMAKE_AR;/usr/bin/ar"
        "CMAKE_RANLIB;/usr/bin/ranlib"
        "CMAKE_NM;/usr/bin/nm"
        "CMAKE_CUDA_HOST_COMPILER;/usr/bin/g++")
    list(GET _toolVarAndValue 0 _toolVar)
    list(GET _toolVarAndValue 1 _toolValue)
    set(${_toolVar} "${_toolValue}" CACHE FILEPATH "test system tool" FORCE)
endforeach()

plascan_configure_mixed_toolchain_binutils("")

foreach(_toolVarAndValue IN ITEMS
        "CMAKE_LINKER;/usr/bin/ld"
        "CMAKE_AR;/usr/bin/ar"
        "CMAKE_RANLIB;/usr/bin/ranlib"
        "CMAKE_NM;/usr/bin/nm"
        "CMAKE_CUDA_HOST_COMPILER;/usr/bin/g++")
    list(GET _toolVarAndValue 0 _toolVar)
    list(GET _toolVarAndValue 1 _expectedToolValue)
    if(NOT "${${_toolVar}}" STREQUAL "${_expectedToolValue}")
        message(FATAL_ERROR
            "${_toolVar} was cleared while disabling mixed-toolchain mode")
    endif()
endforeach()
foreach(_flagVar IN ITEMS
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_SHARED_LINKER_FLAGS
        CMAKE_MODULE_LINKER_FLAGS
        CMAKE_CUDA_FLAGS)
    string(FIND "${${_flagVar}}" "-B/usr/bin" _mixedFlagPosition)
    if(NOT _mixedFlagPosition EQUAL -1)
        message(FATAL_ERROR
            "Mixed-toolchain flag survived in ${_flagVar}: ${${_flagVar}}")
    endif()
endforeach()

file(REMOVE_RECURSE "${_testRoot}")
