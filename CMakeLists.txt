cmake_minimum_required(VERSION 3.25)

project(PlaScan VERSION 1.1.7 LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
  set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
endif()

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
option(PLASCAN_ENABLE_CUDA "Build PlaScan CUDA backends when a CUDA compiler is available" ON)
option(PLASCAN_ENABLE_OPENCL "Build PlaScan OpenCL GPU backends when OpenCL is available" ON)
option(PLASCAN_ENABLE_CERES_REFERENCE
       "Build the optional Ceres comparison backend used by BA parity tests" OFF)
if(PLASCAN_ENABLE_CERES_REFERENCE)
  find_package(Ceres CONFIG REQUIRED)
endif()

if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/permissive->
  )
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(PlascanDependencyPaths)
plascan_configure_dependency_paths()

# 全局启用 CUDA（各子模块不需要单独 enable_language）。
# PLASCAN_ENABLE_CUDA=OFF 用于持续验证真实 CPU 后端，避免因开发机安装了 CUDA 而无法覆盖该路径。
if(PLASCAN_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "plascan: CUDA enabled (${CMAKE_CUDA_COMPILER})")
  else()
    add_compile_definitions(PLAMATRIX_NO_CUDA=1)
    message(STATUS "plascan: CUDA compiler not found, GPU modules will fall back to CPU")
  endif()
else()
  unset(CMAKE_CUDA_COMPILER CACHE)
  unset(CMAKE_CUDA_COMPILER)
  add_compile_definitions(PLAMATRIX_NO_CUDA=1)
  message(STATUS "plascan: CUDA backends disabled by PLASCAN_ENABLE_CUDA=OFF")
endif()

set(PLASCAN_CUDA_ARCHITECTURES "75;86;89;120" CACHE STRING "Default CUDA architectures for PlaScan targets")
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  unset(CMAKE_CUDA_ARCHITECTURES CACHE)
endif()

option(BUILD_TESTS "Build unit tests" ON)
option(PLASCAN_BUILD_GUI "Build the Qt GUI application" ON)
option(PLASCAN_BUILD_GUI_TESTS "Build tests that require the Qt GUI stack" ON)
if(BUILD_TESTS)
  enable_testing()
endif()

include(PlascanPackages)
include(PlascanQtRuntime)
include(PlascanTensorRtRuntime)
include(PlascanWindowsRuntime)

#==============================================================================
# PlaScan 依赖配置
#==============================================================================

message(STATUS "========================================")
message(STATUS "PlaScan 配置:")
if(CMAKE_TOOLCHAIN_FILE)
  message(STATUS "  CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()
if(VCPKG_TARGET_TRIPLET)
  message(STATUS "  VCPKG_TARGET_TRIPLET = ${VCPKG_TARGET_TRIPLET}")
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
if(PLASCAN_DEPENDENCY_PROVIDER_SUMMARY)
  message(STATUS "  Dependency providers = ${PLASCAN_DEPENDENCY_PROVIDER_SUMMARY}")
endif()
message(STATUS "========================================")

# Linux 发布包必须只依赖可迁移的相对 RPATH。把 vcpkg 的构建机绝对
# 路径写进 RUNPATH 会让包在构建机上假通过、换机后立即失效。
if(WIN32)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

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

# 核心模块
add_subdirectory(src/core)

# GUI模块
if(PLASCAN_BUILD_GUI)
  add_subdirectory(src/gui)
endif()

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
  if(WIN32)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix" FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX "/opt/plascan" CACHE PATH "Install path prefix" FORCE)
  endif()
endif()

if(TARGET plascan_gui)
  if(WIN32)
    set(PLASCAN_PACKAGE_SMOKE_DIRECTORY
      "${CMAKE_BINARY_DIR}/package-smoke/PlaScan"
      CACHE PATH "Incremental PlaScan runtime staging directory")
    add_custom_target(plascan_package_smoke
      COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
        --config "$<CONFIG>"
        --component Runtime
        --prefix "${PLASCAN_PACKAGE_SMOKE_DIRECTORY}"
      DEPENDS plascan_gui
      COMMENT "Updating the incremental PlaScan runtime staging directory"
      USES_TERMINAL
      VERBATIM)
  elseif(UNIX AND NOT APPLE)
    set(PLASCAN_PACKAGE_SMOKE_DIRECTORY
      "${CMAKE_BINARY_DIR}/package-smoke/rootfs"
      CACHE PATH "Incremental PlaScan Linux runtime rootfs")
    add_custom_target(plascan_package_smoke
      COMMAND "${CMAKE_COMMAND}" -E env
        "DESTDIR=${PLASCAN_PACKAGE_SMOKE_DIRECTORY}"
        "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
          --config "$<CONFIG>"
          --component Runtime
          --prefix "/opt/plascan"
      DEPENDS plascan_gui
      COMMENT "Updating the incremental PlaScan Linux runtime rootfs"
      USES_TERMINAL
      VERBATIM)
  endif()
endif()

set(CPACK_PACKAGE_NAME "PlaScan")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "PlaScan Team")
set(CPACK_PACKAGE_CONTACT "guderianXu <guderian_xu@henu.edu.cn>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Planetary Scanning Tool")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "PlaScan")
set(CPACK_COMPONENTS_ALL Runtime)
set(CPACK_INSTALL_CMAKE_PROJECTS
  "${CMAKE_BINARY_DIR};${CMAKE_PROJECT_NAME};Runtime;/")
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "PlaScan Runtime")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION
  "PlaScan application, runtime libraries, plugins and bundled models")
set(CPACK_COMPONENT_RUNTIME_HIDDEN ON)
set(CPACK_COMPONENT_RUNTIME_REQUIRED ON)
if(NOT CPACK_GENERATOR)
  if(WIN32)
    set(CPACK_GENERATOR "ZIP")
  elseif(APPLE)
    set(CPACK_GENERATOR "TGZ")
  else()
    set(CPACK_GENERATOR "DEB;TGZ")
  endif()
endif()
set(CPACK_PACKAGE_FILE_NAME "plascan-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/packages" CACHE PATH "CPack package output directory")
if(WIN32)
  set(PLASCAN_INNO_LZMA_BLOCK_THREADS "4" CACHE STRING
    "Parallel LZMA2 block threads used by the Inno Setup release package")
  if(NOT PLASCAN_INNO_LZMA_BLOCK_THREADS MATCHES "^[1-9][0-9]*$" OR
      PLASCAN_INNO_LZMA_BLOCK_THREADS GREATER 256)
    message(FATAL_ERROR
      "PLASCAN_INNO_LZMA_BLOCK_THREADS must be an integer from 1 to 256")
  endif()
  set(CPACK_SET_DESTDIR OFF)
  # CPack concatenates this value with its staging directory, so the leading
  # slash is required. PlaScanCPackOptions.cmake overrides installer-specific
  # layout after CPack selects the active generator.
  set(CPACK_PACKAGING_INSTALL_PREFIX "/PlaScan")
  set(CPACK_PACKAGE_EXECUTABLES "plascan" "PlaScan")
  set(CPACK_CREATE_DESKTOP_LINKS "plascan")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

  set(CPACK_INNOSETUP_ARCHITECTURE "x64")
  set(CPACK_INNOSETUP_USE_MODERN_WIZARD ON)
  set(CPACK_INNOSETUP_CREATE_UNINSTALL_LINK ON)
  set(CPACK_INNOSETUP_RUN_EXECUTABLES "plascan")
  set(CPACK_INNOSETUP_SETUP_ChangesAssociations ON)
  set(CPACK_INNOSETUP_SETUP_UninstallDisplayIcon
    "{app}\\\\bin\\\\plascan.exe")
  set(CPACK_INNOSETUP_EXTRA_SCRIPTS
    "${CMAKE_SOURCE_DIR}/src/gui/packaging/PlaScanInnoSetup.iss")

  find_program(PLASCAN_INNOSETUP_EXECUTABLE
    NAMES ISCC.exe ISCC
    HINTS
      "$ENV{LOCALAPPDATA}/Programs/Inno Setup 6"
      "$ENV{ProgramFiles}/Inno Setup 6"
      "$ENV{ProgramFiles\(x86\)}/Inno Setup 6")
  if(PLASCAN_INNOSETUP_EXECUTABLE)
    set(CPACK_INNOSETUP_EXECUTABLE
      "${PLASCAN_INNOSETUP_EXECUTABLE}")
  endif()
else()
  set(CPACK_SET_DESTDIR ON)
  set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/plascan")
endif()

if(UNIX AND NOT APPLE)
  set(PLASCAN_DEBIAN_PACKAGE_VARIANT "portable" CACHE STRING
    "Linux DEB package variant: portable or cuda")
  set_property(CACHE PLASCAN_DEBIAN_PACKAGE_VARIANT PROPERTY STRINGS
    portable cuda)
  set(PLASCAN_DEBIAN_ACCELERATOR_DEPENDS "" CACHE STRING
    "Additional apt dependencies required by a CUDA/TensorRT DEB build")
  if(PLASCAN_DEBIAN_PACKAGE_VARIANT STREQUAL "cuda")
    set(CPACK_DEBIAN_PACKAGE_NAME "plascan-cuda")
    set(CPACK_DEBIAN_PACKAGE_PROVIDES "plascan")
    set(CPACK_DEBIAN_PACKAGE_CONFLICTS "plascan")
    set(CPACK_DEBIAN_PACKAGE_REPLACES "plascan")
  elseif(PLASCAN_DEBIAN_PACKAGE_VARIANT STREQUAL "portable")
    set(CPACK_DEBIAN_PACKAGE_NAME "plascan")
    set(CPACK_DEBIAN_PACKAGE_CONFLICTS "plascan-cuda")
  else()
    message(FATAL_ERROR
      "Unsupported PLASCAN_DEBIAN_PACKAGE_VARIANT: ${PLASCAN_DEBIAN_PACKAGE_VARIANT}")
  endif()
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER
    "guderianXu <guderian_xu@henu.edu.cn>")
  set(CPACK_DEBIAN_PACKAGE_SECTION "science")
  set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
  set(CPACK_DEBIAN_PACKAGE_HOMEPAGE
    "https://github.com/guderianXu/plascan")
  set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
  set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/plascan")
  set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION ON)
  set(CPACK_DEB_COMPONENT_INSTALL OFF)
  # vcpkg 动态库随包部署，系统基础库和桌面栈由 apt 管理。
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OFF)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "ca-certificates, libc6 (>= 2.39), libstdc++6 (>= 13.1), libgcc-s1, libgomp1, libgfortran5, libblas3, liblapack3, libapriltag3t64, ocl-icd-libopencl1, libvulkan1, libgl1, libegl1, libdbus-1-3, libfontconfig1, libglib2.0-0t64, libx11-6, libx11-xcb1, libxext6, libxrender1, libxi6, libxrandr2, libxfixes3, libxcursor1, libxcb1, libxcb-cursor0, libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-randr0, libxcb-render-util0, libxcb-render0, libxcb-shape0, libxcb-shm0, libxcb-sync1, libxcb-util1, libxcb-xfixes0, libxcb-xinput0, libxcb-xkb1, libxkbcommon0, libxkbcommon-x11-0")
  if(PLASCAN_DEBIAN_ACCELERATOR_DEPENDS)
    string(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS
      ", ${PLASCAN_DEBIAN_ACCELERATOR_DEPENDS}")
  endif()
  set(CPACK_DEBIAN_COMPRESSION_TYPE "xz")
  set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${CMAKE_SOURCE_DIR}/cmake/debian/postinst"
    "${CMAKE_SOURCE_DIR}/cmake/debian/postrm")
endif()

set(_plascan_cpack_project_config
  "${CMAKE_BINARY_DIR}/PlaScanCPackOptions.cmake")
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/PlaScanCPackOptions.cmake.in"
  "${_plascan_cpack_project_config}"
  @ONLY)
set(CPACK_PROJECT_CONFIG_FILE "${_plascan_cpack_project_config}")

if(WIN32)
  install(DIRECTORY resources/ DESTINATION share/plascan
      COMPONENT Runtime
      FILES_MATCHING
      PATTERN "*.png"
      PATTERN "*.svg"
      PATTERN "models" EXCLUDE)
endif()

include(CPack)

message(STATUS "========================================")
message(STATUS "PlaScan 配置完成")
message(STATUS "========================================")
