cmake_minimum_required(VERSION 3.16)

# 必须在 project() 之前设置, 覆盖 conda gcc 的旧 ABI 默认值
# conda OpenCV/GTest 用新 ABI, PyTorch 2.5.1 也跟随
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_CUDA_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")

project(PlaScan VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# ── conda/PyTorch 前缀 (可选, Docker/系统构建不需要) ──
if(NOT DEFINED PLASCAN_CONDA_PREFIX)
  set(_p "$ENV{CONDA_PREFIX}")
  if(_p AND EXISTS "${_p}/targets/x86_64-linux/include")
    set(PLASCAN_CONDA_PREFIX "${_p}" CACHE PATH "Conda environment prefix")
  elseif(EXISTS "$ENV{HOME}/anaconda3/envs/plascan")
    set(PLASCAN_CONDA_PREFIX "$ENV{HOME}/anaconda3/envs/plascan" CACHE PATH "Conda env prefix")
  else()
    set(PLASCAN_CONDA_PREFIX "" CACHE PATH "Conda env prefix (empty = system build)")
  endif()
endif()
message(STATUS "plascan: CONDA_PREFIX = ${PLASCAN_CONDA_PREFIX}")

# 全局启用 CUDA (各子模块不需要单独 enable_language)

# 全局启用 CUDA (各子模块不需要单独 enable_language)
# conda nvcc 不在默认 PATH 中, 显式指定路径
set(PLASCAN_CONDA_NVCC "${PLASCAN_CONDA_PREFIX}/bin/nvcc")
if(NOT DEFINED CMAKE_CUDA_COMPILER OR NOT CMAKE_CUDA_COMPILER)
  if(EXISTS "${PLASCAN_CONDA_NVCC}")
    set(CMAKE_CUDA_COMPILER "${PLASCAN_CONDA_NVCC}" CACHE FILEPATH "CUDA compiler")
  endif()
endif()
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "plascan: CUDA enabled (${CMAKE_CUDA_COMPILER})")
else()
  message(STATUS "plascan: CUDA compiler not found, GPU modules will fall back to CPU")
endif()

# 系统库路径必须在 conda sysroot 之前, 避免 glibc 版本冲突
if(EXISTS "/usr/lib/x86_64-linux-gnu")
  link_directories(BEFORE "/usr/lib/x86_64-linux-gnu")
  set(CMAKE_EXE_LINKER_FLAGS "-L/usr/lib/x86_64-linux-gnu ${CMAKE_EXE_LINKER_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "-L/usr/lib/x86_64-linux-gnu ${CMAKE_SHARED_LINKER_FLAGS}")
endif()

set(PLASCAN_CUDA_ARCHITECTURES "75;86;89" CACHE STRING "Default CUDA architectures for PlaScan targets")
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  unset(CMAKE_CUDA_ARCHITECTURES CACHE)
endif()

# 统一使用 C++11 ABI (与 conda OpenCV/GTest 一致)
# conda gcc 默认旧 ABI, 直接替换 CXX_FLAGS 中的旧值为新值
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=0" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
message(STATUS "plascan: C++11 ABI enabled")

# CUDA runtime + toolkit 头文件 (conda 环境才需要, 系统构建跳过)
if(PLASCAN_CONDA_PREFIX AND EXISTS "${PLASCAN_CONDA_PREFIX}")
  # targets 必须在 cu13 之前, 否则 nvcc 13.1 会找到 cu13 的 13.0 版 cuda_runtime_api.h
  include_directories("${PLASCAN_CONDA_PREFIX}/targets/x86_64-linux/include")

  file(GLOB _plascan_cu13_dirs LIST_DIRECTORIES true
    "${PLASCAN_CONDA_PREFIX}/lib/python3.*/site-packages/nvidia/cu13/include")
  if(_plascan_cu13_dirs)
    list(GET _plascan_cu13_dirs 0 _plascan_cu13_include)
    include_directories("${_plascan_cu13_include}")
  endif()
else()
  message(STATUS "plascan: system build (no conda), assuming deps from find_package")
endif()

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  enable_testing()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(PlascanDependencyPaths)
plascan_configure_dependency_paths()
include(PlascanPackages)

# 为运行时 NVRTC builtins 解析补充 conda/pip CUDA 库目录。
# 典型问题：libnvrtc.so 能加载，但 dlopen("libnvrtc-builtins.so.*") 在默认搜索路径中找不到。
set(PLASCAN_EXTRA_NVRTC_RPATHS "")
set(_plascan_runtime_prefix "${PLASCAN_CONDA_PREFIX}")

if(_plascan_runtime_prefix AND EXISTS "${_plascan_runtime_prefix}")
  set(_plascan_nvrtc_candidates
    "${_plascan_runtime_prefix}/targets/x86_64-linux/lib"
  )

  # 只匹配两位及以上版本号（python3.12），排除 python3.1 这类单位数目录
  file(GLOB _plascan_nvidia_cuda_dirs LIST_DIRECTORIES true
    "${_plascan_runtime_prefix}/lib/python3.[0-9][0-9]*/site-packages/nvidia/cu[0-9]*/lib")
  list(APPEND _plascan_nvrtc_candidates ${_plascan_nvidia_cuda_dirs})

  foreach(_nvrtc_dir IN LISTS _plascan_nvrtc_candidates)
    if(EXISTS "${_nvrtc_dir}")
      list(APPEND PLASCAN_EXTRA_NVRTC_RPATHS "${_nvrtc_dir}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES PLASCAN_EXTRA_NVRTC_RPATHS)

  # 若 targets/x86_64-linux/lib 里只有 .so.13.1 而缺少 .so.13.0，
  # 在 configure 阶段自动创建符号链接，避免 PyTorch NVRTC 运行时找不到。
  set(_nvrtc_builtins_dir "${_plascan_runtime_prefix}/targets/x86_64-linux/lib")
  if(EXISTS "${_nvrtc_builtins_dir}/libnvrtc-builtins.so.13.1" AND
     NOT EXISTS "${_nvrtc_builtins_dir}/libnvrtc-builtins.so.13.0")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        libnvrtc-builtins.so.13.1
        "${_nvrtc_builtins_dir}/libnvrtc-builtins.so.13.0"
    )
    message(STATUS "plascan: created libnvrtc-builtins.so.13.0 symlink in ${_nvrtc_builtins_dir}")
  endif()
endif()

if(PLASCAN_EXTRA_NVRTC_RPATHS)
  message(STATUS "plascan: extra NVRTC runtime paths = ${PLASCAN_EXTRA_NVRTC_RPATHS}")
  list(APPEND CMAKE_BUILD_RPATH ${PLASCAN_EXTRA_NVRTC_RPATHS})
endif()

#==============================================================================
# PlaScan 依赖配置
#==============================================================================

message(STATUS "========================================")
message(STATUS "PlaScan 配置:")
if(CMAKE_TOOLCHAIN_FILE)
  message(STATUS "  CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
else()
  message(WARNING "  CMAKE_TOOLCHAIN_FILE 未设置。将使用系统/conda 前缀查找依赖；推荐优先使用 vcpkg preset。")
endif()
if(VCPKG_TARGET_TRIPLET)
  message(STATUS "  VCPKG_TARGET_TRIPLET = ${VCPKG_TARGET_TRIPLET}")
endif()
if(PLASCAN_CONDA_PREFIX)
  message(STATUS "  PLASCAN_CONDA_PREFIX = ${PLASCAN_CONDA_PREFIX}")
elseif(DEFINED ENV{CONDA_PREFIX})
  message(STATUS "  CONDA_PREFIX = $ENV{CONDA_PREFIX}")
endif()
if(CUDAToolkit_ROOT)
  message(STATUS "  CUDAToolkit_ROOT = ${CUDAToolkit_ROOT}")
endif()
if(CUDA_TOOLKIT_ROOT_DIR)
  message(STATUS "  CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
endif()
if(CMAKE_CUDA_COMPILER)
  message(STATUS "  CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
endif()
if(Torch_DIR)
  message(STATUS "  Torch_DIR = ${Torch_DIR}")
endif()
if(PLASCAN_DEPENDENCY_PROVIDER_SUMMARY)
  message(STATUS "  Dependency providers = ${PLASCAN_DEPENDENCY_PROVIDER_SUMMARY}")
endif()
message(STATUS "========================================")

# 运行时库路径交由各目标基于真实链接结果自动生成，避免系统 Qt 与 conda Qt 混装。
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#==============================================================================
# 构建类型
#==============================================================================

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#==============================================================================
# 子模块
#==============================================================================

add_subdirectory(src/common)

# 核心模块（仅当存在时加入）
if(EXISTS ${CMAKE_SOURCE_DIR}/src/core/CMakeLists.txt)
  add_subdirectory(src/core)
else()
  message(WARNING "src/core missing - skipping core module (you removed core components)")
endif()

# GUI模块
add_subdirectory(src/gui)

# CLI 工具
add_subdirectory(src/cli)

#==============================================================================
# 单元测试
#==============================================================================
if(BUILD_TESTS)
  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
    message(STATUS "plascan: building tests")
  endif()
endif()

#==============================================================================
# 安装和打包(可选)
#==============================================================================

include(GNUInstallDirs)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/opt/plascan" CACHE PATH "Install path prefix" FORCE)
endif()

set(CPACK_PACKAGE_NAME "PlaScan")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "PlaScan Team")
set(CPACK_PACKAGE_CONTACT "plascan@example.com")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "PlaScan Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Planetary Scanning Tool")
set(CPACK_GENERATOR "DEB;TGZ")
set(CPACK_PACKAGE_FILE_NAME "plascan-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_SET_DESTDIR ON)
set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/plascan")

# 离线安装优先：不依赖目标机在线拉取 Qt 等运行时依赖。
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OFF)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
set(CPACK_DEBIAN_COMPRESSION_TYPE "xz")

include(CPack)

message(STATUS "========================================")
message(STATUS "PlaScan 配置完成")
message(STATUS "========================================")
