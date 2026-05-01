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

function(plascan_configure_dependency_paths)
    set(_providerSummary "")

    # 强制使用系统 binutils 避免 conda ld 与系统 glibc 版本冲突
    if(DEFINED ENV{CONDA_PREFIX} AND NOT CMAKE_LINKER)
        if(EXISTS "/usr/bin/ld")
            set(CMAKE_LINKER "/usr/bin/ld" CACHE FILEPATH "Linker" FORCE)
        endif()
    endif()

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