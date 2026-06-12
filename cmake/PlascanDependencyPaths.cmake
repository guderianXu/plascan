include_guard(GLOBAL)

# ==============================================================================
# PlaScan 依赖查找辅助
#
# 目标：
# 1. 保持 vcpkg toolchain 的优先级不变。
# 2. 当用户启用了 conda 环境时，把 conda 前缀加入 CMake 查找路径。
# 3. 为通过 conda/pip 安装的 PyTorch 自动补充 Torch_DIR 候选路径。
#
# 说明：
# - vcpkg 仍建议通过 CMAKE_TOOLCHAIN_FILE 使用。
# - conda 作为额外前缀参与 find_package 查找，可补足 Torch/OpenCV/GDAL/Qt 等。
# - 对于 pip/conda 中的 torch，常见的 TorchConfig.cmake 位于：
#   <prefix>/lib/pythonX.Y/site-packages/torch/share/cmake/Torch
# ==============================================================================

option(PLASCAN_ENABLE_VCPKG "Allow dependency discovery through vcpkg toolchain" ON)
option(PLASCAN_ENABLE_CONDA "Allow dependency discovery through conda prefix" ON)

set(PLASCAN_CONDA_PREFIX "" CACHE PATH
    "Optional conda prefix for dependency lookup; defaults to ENV{CONDA_PREFIX} when available")
set(PLASCAN_TORCH_DIR "" CACHE PATH
    "Optional explicit TorchConfig.cmake directory; overrides auto-detection when set")

function(plascan_append_prefix_if_exists prefix)
    if(NOT prefix)
        return()
    endif()

    if(EXISTS "${prefix}")
        list(APPEND CMAKE_PREFIX_PATH "${prefix}")
        list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
    endif()
endfunction()

function(plascan_find_torch_dir_from_prefix prefix out_var)
    if(NOT prefix OR NOT EXISTS "${prefix}")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    set(candidates
        "${prefix}/lib/cmake/Torch"
        "${prefix}/share/cmake/Torch"
    )

    file(GLOB pythonTorchDirs LIST_DIRECTORIES true
        "${prefix}/lib/python3.[0-9][0-9]*/site-packages/torch/share/cmake/Torch")
    list(APPEND candidates ${pythonTorchDirs})

    foreach(candidate IN LISTS candidates)
        if(EXISTS "${candidate}/TorchConfig.cmake")
            set(${out_var} "${candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(plascan_find_cuda_root_from_prefix prefix out_var)
    if(NOT prefix OR NOT EXISTS "${prefix}")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    set(candidates
        "${prefix}/targets/x86_64-linux"
        "${prefix}"
    )

    file(GLOB cudaTargetDirs LIST_DIRECTORIES true "${prefix}/targets/*")
    list(APPEND candidates ${cudaTargetDirs})

    foreach(candidate IN LISTS candidates)
        if(EXISTS "${candidate}/include/cuda.h" AND (
               EXISTS "${candidate}/bin/nvcc" OR EXISTS "${candidate}/lib/libcudart.so"))
            set(${out_var} "${candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(plascan_path_is_under_prefix path prefix out_var)
    if(NOT path OR NOT prefix)
        set(${out_var} FALSE PARENT_SCOPE)
        return()
    endif()

    file(REAL_PATH "${path}" _realPath BASE_DIRECTORY "${CMAKE_SOURCE_DIR}")
    file(REAL_PATH "${prefix}" _realPrefix BASE_DIRECTORY "${CMAKE_SOURCE_DIR}")
    string(FIND "${_realPath}" "${_realPrefix}/" _prefixPos)
    if(_prefixPos EQUAL 0 OR _realPath STREQUAL _realPrefix)
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(plascan_append_unique_flag var_name flag)
    if(NOT "${${var_name}}" MATCHES "(^| )${flag}($| )")
        set(${var_name} "${flag} ${${var_name}}" CACHE STRING "" FORCE)
    endif()
endfunction()

function(plascan_configure_mixed_toolchain_binutils conda_prefix)
    if(APPLE OR NOT conda_prefix OR NOT EXISTS "${conda_prefix}")
        set(PLASCAN_USE_SYSTEM_BINUTILS_FOR_MIXED_TOOLCHAIN OFF CACHE BOOL
            "Use system binutils when system compilers are combined with conda dependencies" FORCE)
        return()
    endif()

    plascan_path_is_under_prefix("${CMAKE_CXX_COMPILER}" "${conda_prefix}" _compilerInConda)
    if(_compilerInConda)
        set(PLASCAN_USE_SYSTEM_BINUTILS_FOR_MIXED_TOOLCHAIN OFF CACHE BOOL
            "Use system binutils when system compilers are combined with conda dependencies" FORCE)
        return()
    endif()

    if(EXISTS "/usr/bin/ld")
        set(CMAKE_LINKER "/usr/bin/ld" CACHE FILEPATH "Linker" FORCE)
    endif()
    if(EXISTS "/usr/bin/ar")
        set(CMAKE_AR "/usr/bin/ar" CACHE FILEPATH "Archiver" FORCE)
    endif()
    if(EXISTS "/usr/bin/ranlib")
        set(CMAKE_RANLIB "/usr/bin/ranlib" CACHE FILEPATH "Ranlib" FORCE)
    endif()
    if(EXISTS "/usr/bin/nm")
        set(CMAKE_NM "/usr/bin/nm" CACHE FILEPATH "nm" FORCE)
    endif()
    if(EXISTS "/usr/bin/g++")
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE FILEPATH "CUDA host compiler" FORCE)
    elseif(CMAKE_CXX_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "CUDA host compiler" FORCE)
    endif()

    # The system compiler can still locate conda's ld through an activated
    # environment. Prefer system binutils explicitly for every link step.
    plascan_append_unique_flag(CMAKE_EXE_LINKER_FLAGS "-B/usr/bin")
    plascan_append_unique_flag(CMAKE_SHARED_LINKER_FLAGS "-B/usr/bin")
    plascan_append_unique_flag(CMAKE_MODULE_LINKER_FLAGS "-B/usr/bin")
    plascan_append_unique_flag(CMAKE_CUDA_FLAGS "-Xcompiler=-B/usr/bin")

    set(PLASCAN_USE_SYSTEM_BINUTILS_FOR_MIXED_TOOLCHAIN ON CACHE BOOL
        "Use system binutils when system compilers are combined with conda dependencies" FORCE)
    message(STATUS
        "plascan: system C++ compiler detected; using system binutils to avoid conda sysroot/glibc mixing")
endfunction()

function(plascan_apply_mixed_toolchain_binutils_to_scope)
    if(NOT PLASCAN_USE_SYSTEM_BINUTILS_FOR_MIXED_TOOLCHAIN)
        return()
    endif()

    set(CMAKE_LINKER "/usr/bin/ld")
    set(CMAKE_AR "/usr/bin/ar")
    set(CMAKE_RANLIB "/usr/bin/ranlib")
    set(CMAKE_NM "/usr/bin/nm")
    if(EXISTS "/usr/bin/g++")
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")
    elseif(CMAKE_CXX_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    endif()

    foreach(_flagVar IN ITEMS
            CMAKE_EXE_LINKER_FLAGS
            CMAKE_SHARED_LINKER_FLAGS
            CMAKE_MODULE_LINKER_FLAGS)
        if(NOT "${${_flagVar}}" MATCHES "(^| )-B/usr/bin($| )")
            set(${_flagVar} "-B/usr/bin ${${_flagVar}}" CACHE STRING "" FORCE)
        endif()
    endforeach()

    if(NOT "${CMAKE_CUDA_FLAGS}" MATCHES "(^| )-Xcompiler=-B/usr/bin($| )")
        set(CMAKE_CUDA_FLAGS "-Xcompiler=-B/usr/bin ${CMAKE_CUDA_FLAGS}"
            CACHE STRING "CUDA compiler flags" FORCE)
    endif()

    foreach(_lang IN ITEMS C CXX CUDA)
        set(CMAKE_${_lang}_ARCHIVE_CREATE
            "/usr/bin/ar qc <TARGET> <LINK_FLAGS> <OBJECTS>")
        set(CMAKE_${_lang}_ARCHIVE_APPEND
            "/usr/bin/ar q <TARGET> <LINK_FLAGS> <OBJECTS>")
        set(CMAKE_${_lang}_ARCHIVE_FINISH
            "/usr/bin/ranlib <TARGET>")

        set(CMAKE_${_lang}_ARCHIVE_CREATE
            "${CMAKE_${_lang}_ARCHIVE_CREATE}" PARENT_SCOPE)
        set(CMAKE_${_lang}_ARCHIVE_APPEND
            "${CMAKE_${_lang}_ARCHIVE_APPEND}" PARENT_SCOPE)
        set(CMAKE_${_lang}_ARCHIVE_FINISH
            "${CMAKE_${_lang}_ARCHIVE_FINISH}" PARENT_SCOPE)
    endforeach()

    foreach(_toolVar IN ITEMS
            CMAKE_LINKER
            CMAKE_AR
            CMAKE_RANLIB
            CMAKE_NM
            CMAKE_CUDA_HOST_COMPILER
            CMAKE_CUDA_FLAGS
            CMAKE_EXE_LINKER_FLAGS
            CMAKE_SHARED_LINKER_FLAGS
            CMAKE_MODULE_LINKER_FLAGS)
        set(${_toolVar} "${${_toolVar}}" PARENT_SCOPE)
    endforeach()
endfunction()

function(plascan_configure_dependency_paths)
    set(_providerSummary "")

    if(PLASCAN_ENABLE_VCPKG)
        if(CMAKE_TOOLCHAIN_FILE)
            list(APPEND _providerSummary "vcpkg(toolchain)")
        elseif(DEFINED ENV{VCPKG_ROOT})
            list(APPEND _providerSummary "vcpkg(env-only)")
        endif()
    endif()

    set(_condaPrefix "${PLASCAN_CONDA_PREFIX}")
    if(NOT _condaPrefix AND DEFINED ENV{CONDA_PREFIX})
        set(_condaPrefix "$ENV{CONDA_PREFIX}")
    endif()

    plascan_configure_mixed_toolchain_binutils("${_condaPrefix}")
    plascan_apply_mixed_toolchain_binutils_to_scope()

    if(PLASCAN_ENABLE_CONDA AND _condaPrefix)
        plascan_append_prefix_if_exists("${_condaPrefix}")
        plascan_append_prefix_if_exists("${_condaPrefix}/lib/cmake")
        plascan_append_prefix_if_exists("${_condaPrefix}/share/cmake")
        plascan_append_prefix_if_exists("${_condaPrefix}/lib")
        list(APPEND _providerSummary "conda:${_condaPrefix}")

        plascan_find_cuda_root_from_prefix("${_condaPrefix}" _autoCudaRoot)
        if(_autoCudaRoot)
            set(ENV{CUDA_HOME} "${_autoCudaRoot}")
            set(ENV{CUDA_PATH} "${_autoCudaRoot}")
            set(ENV{CUDA_BIN_PATH} "${_autoCudaRoot}/bin")
            set(ENV{PATH} "${_autoCudaRoot}/bin:${_autoCudaRoot}/nvvm/bin:$ENV{PATH}")

            set(CUDAToolkit_ROOT "${_autoCudaRoot}" CACHE PATH "CUDA toolkit root" FORCE)
            set(CUDA_TOOLKIT_ROOT_DIR "${_autoCudaRoot}" CACHE PATH "CUDA toolkit root for legacy CMake modules" FORCE)
            set(CUDAToolkit_INCLUDE_DIR "${_autoCudaRoot}/include" CACHE PATH "CUDA toolkit include directory" FORCE)
            set(CUDA_INCLUDE_DIRS "${_autoCudaRoot}/include" CACHE PATH "CUDA include directories" FORCE)

            if(NOT CMAKE_CUDA_FLAGS MATCHES "(^| )-I${_autoCudaRoot}/include( |$)")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${_autoCudaRoot}/include"
                    CACHE STRING "CUDA compiler flags" FORCE)
            endif()

            if(EXISTS "${_condaPrefix}/bin/nvcc")
                set(CMAKE_CUDA_COMPILER "${_condaPrefix}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
            elseif(EXISTS "${_autoCudaRoot}/bin/nvcc")
                set(CMAKE_CUDA_COMPILER "${_autoCudaRoot}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
            endif()

            list(APPEND _providerSummary "CUDA:${_autoCudaRoot}")
        endif()
    endif()

    if(PLASCAN_TORCH_DIR)
        if(EXISTS "${PLASCAN_TORCH_DIR}/TorchConfig.cmake")
            set(Torch_DIR "${PLASCAN_TORCH_DIR}" CACHE PATH "Torch config directory" FORCE)
        endif()
    elseif(DEFINED ENV{TORCH_DIR} AND EXISTS "$ENV{TORCH_DIR}/TorchConfig.cmake")
        set(Torch_DIR "$ENV{TORCH_DIR}" CACHE PATH "Torch config directory" FORCE)
    elseif(PLASCAN_ENABLE_CONDA AND _condaPrefix)
        plascan_find_torch_dir_from_prefix("${_condaPrefix}" _autoTorchDir)
        if(_autoTorchDir)
            set(Torch_DIR "${_autoTorchDir}" CACHE PATH "Torch config directory" FORCE)
        endif()
    endif()

    if(Torch_DIR)
        list(APPEND _providerSummary "Torch:${Torch_DIR}")
    endif()

    list(REMOVE_DUPLICATES _providerSummary)
    set(PLASCAN_DEPENDENCY_PROVIDER_SUMMARY "${_providerSummary}" PARENT_SCOPE)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
endfunction()
