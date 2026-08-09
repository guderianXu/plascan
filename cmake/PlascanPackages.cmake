include_guard(GLOBAL)

# ==============================================================================
# PlaScan 统一依赖查找
#
# 所有 find_package 调用集中在此，子模块直接使用已找到的 target，
# 不再重复搜索。此文件由根 CMakeLists.txt 在 add_subdirectory 之前 include。
# ==============================================================================

# ── Qt6 ───────────────────────────────────────────────────────────────────────
# 合并所有模块所需组件（Network 用于异步下载可选模型资源）。
set(PLASCAN_QT_COMPONENTS
  Core Gui Widgets Network Concurrent ShaderTools ShaderToolsTools)
if(WIN32)
  # vcpkg exposes private Qt modules as explicit Qt6 components.
  list(APPEND PLASCAN_QT_COMPONENTS GuiPrivate)
endif()
find_package(Qt6 6.7 REQUIRED COMPONENTS ${PLASCAN_QT_COMPONENTS})
if(NOT TARGET Qt6::GuiPrivate)
  message(FATAL_ERROR
    "PlaScan requires the Qt GuiPrivate target. Install the Qt base private development package.")
endif()
message(STATUS "plascan: found Qt6 ${Qt6_VERSION}")

get_target_property(_PLASCAN_QT_GUI_PUBLIC_FEATURES Qt6::Gui QT_ENABLED_PUBLIC_FEATURES)
list(FIND _PLASCAN_QT_GUI_PUBLIC_FEATURES vulkan _PLASCAN_QT_GUI_VULKAN_FEATURE_INDEX)
if(_PLASCAN_QT_GUI_VULKAN_FEATURE_INDEX EQUAL -1)
  message(FATAL_ERROR
    "PlaScan Vulkan rendering requires QtGui built with Vulkan support. "
    "Install Vulkan SDK/loader/headers and rebuild the vcpkg Qt package.")
endif()

# ── OpenCV ────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "5.0.0")
  # OpenCV 5 将 calib3d/features2d 能力拆到 geometry/features/stereo 模块。
  set(PLASCAN_OPENCV_COMPONENTS core imgproc geometry stereo features imgcodecs flann xfeatures2d ximgproc)
else()
  set(PLASCAN_OPENCV_COMPONENTS core imgproc calib3d features2d imgcodecs flann)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${PLASCAN_OPENCV_COMPONENTS})
message(STATUS "plascan: found OpenCV ${OpenCV_VERSION} (${PLASCAN_OPENCV_COMPONENTS})")

# ── OpenCL（可选，用于 AMD/Intel/NVIDIA 通用 GPU 计算）─────────────────────────────
if(PLASCAN_ENABLE_OPENCL)
  find_package(OpenCL 1.2 QUIET)
  if(OpenCL_FOUND)
    message(STATUS "plascan: found OpenCL ${OpenCL_VERSION_STRING}")
  else()
    message(STATUS "plascan: OpenCL 1.2 development files not found, OpenCL GPU backend disabled")
  endif()
endif()

# ── 平台检测 ──────────────────────────────────────────────────────────────────
set(PLASCAN_APPLE_SILICON OFF)
set(PLASCAN_CUDA_AVAILABLE OFF)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(PLASCAN_APPLE_SILICON ON)
  message(STATUS "plascan: Apple Silicon (M-series) detected — CUDA disabled")
endif()

# ── CUDA architecture configuration ───────────────────────────────────────────
if(PLASCAN_ENABLE_CUDA AND NOT PLASCAN_APPLE_SILICON)
  if(NOT DEFINED PLASCAN_CUDA_ARCHITECTURES)
    set(PLASCAN_CUDA_ARCHITECTURES "75;86;89;120" CACHE STRING "Target CUDA architectures")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
    CACHE STRING "CUDA architectures for PlaScan CUDA targets" FORCE)
  set(PLAMATRIX_CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
    CACHE STRING "CUDA architectures for PlaMatrix" FORCE)
  set(PLAPOINT_CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
    CACHE STRING "CUDA architectures for PlaPoint" FORCE)
  message(STATUS "plascan: CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ── GDAL ──────────────────────────────────────────────────────────────────────
find_package(GDAL REQUIRED)
if(TARGET GDAL::GDAL)
  set(PLASCAN_GDAL_TARGET GDAL::GDAL CACHE INTERNAL "GDAL CMake target")
else()
  set(PLASCAN_GDAL_TARGET ${GDAL_LIBRARIES} CACHE INTERNAL "GDAL CMake target")
endif()
message(STATUS "plascan: found GDAL, target=${PLASCAN_GDAL_TARGET}")

# ── AprilTag ─────────────────────────────────────────────────────────────────
if(WIN32)
  find_package(apriltag CONFIG REQUIRED)
else()
  # Ubuntu 24.04's apriltag CMake package exports an invalid duplicated
  # /usr/lib/lib/<triplet> path. Its pkg-config metadata points at the real
  # multiarch library location and is also installed by upstream on Unix.
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(APRILTAG REQUIRED IMPORTED_TARGET GLOBAL apriltag)
  add_library(apriltag::apriltag ALIAS PkgConfig::APRILTAG)
endif()
message(STATUS "plascan: found AprilTag")

# ── libtiff ───────────────────────────────────────────────────────────────────
find_package(TIFF REQUIRED)
if(TARGET TIFF::TIFF)
  set(PLASCAN_TIFF_TARGET TIFF::TIFF CACHE INTERNAL "TIFF CMake target")
else()
  set(PLASCAN_TIFF_TARGET ${TIFF_LIBRARIES} CACHE INTERNAL "TIFF CMake target")
endif()
message(STATUS "plascan: found TIFF, target=${PLASCAN_TIFF_TARGET}")

# ── libzip ────────────────────────────────────────────────────────────────────
find_package(libzip CONFIG REQUIRED)
if(TARGET libzip::zip)
  set(PLASCAN_LIBZIP_TARGET libzip::zip CACHE INTERNAL "libzip CMake target")
elseif(TARGET zip)
  set(PLASCAN_LIBZIP_TARGET zip CACHE INTERNAL "libzip CMake target")
else()
  message(FATAL_ERROR "plascan: libzip found but no usable CMake target was exported")
endif()
message(STATUS "plascan: found libzip, target=${PLASCAN_LIBZIP_TARGET}")

# ── plamatrix (submodule) ──────────────────────────────────────────────────────
if(PLASCAN_ENABLE_CUDA AND CMAKE_CUDA_COMPILER AND NOT PLASCAN_APPLE_SILICON)
  set(PLAMATRIX_WITH_CUDA ON CACHE BOOL "Build PlaMatrix with CUDA acceleration" FORCE)
  set(PLAPOINT_WITH_CUDA ON CACHE BOOL "Build PlaPoint with CUDA acceleration" FORCE)
else()
  set(PLAMATRIX_WITH_CUDA OFF CACHE BOOL "Build PlaMatrix with CUDA acceleration" FORCE)
  set(PLAPOINT_WITH_CUDA OFF CACHE BOOL "Build PlaPoint with CUDA acceleration" FORCE)
endif()
if(PLASCAN_ENABLE_OPENCL AND TARGET OpenCL::OpenCL)
  set(PLAMATRIX_WITH_OPENCL ON CACHE BOOL "Build PlaMatrix OpenCL infrastructure" FORCE)
else()
  set(PLAMATRIX_WITH_OPENCL OFF CACHE BOOL "Build PlaMatrix OpenCL infrastructure" FORCE)
endif()
set(PLAPOINT_WITH_OPENCL ${PLAMATRIX_WITH_OPENCL}
  CACHE BOOL "Build PlaPoint OpenCL acceleration on PlaMatrix" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/plamatrix ${CMAKE_BINARY_DIR}/3rdparty/plamatrix)
message(STATUS "plascan: using plamatrix from 3rdparty/")

# ── plapoint (submodule) ───────────────────────────────────────────────────────
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/plapoint ${CMAKE_BINARY_DIR}/3rdparty/plapoint)
message(STATUS "plascan: using plapoint from 3rdparty/")

# ── OpenMP ────────────────────────────────────────────────────────────────────
if(PLASCAN_APPLE_SILICON)
  # macOS 不自带 OpenMP, 尝试 Homebrew libomp
  find_package(OpenMP QUIET)
  if(NOT OpenMP_CXX_FOUND)
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(LIBOMP_PREFIX)
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_omp_LIBRARY ${LIBOMP_PREFIX}/lib/libomp.dylib)
      set(OpenMP_CXX_FOUND TRUE)
      message(STATUS "plascan: found OpenMP via Homebrew (${LIBOMP_PREFIX})")
    else()
      message(STATUS "plascan: OpenMP not found, install: brew install libomp")
    endif()
  endif()
else()
  find_package(OpenMP QUIET)
endif()
if(OpenMP_CXX_FOUND)
  message(STATUS "plascan: found OpenMP ${OpenMP_CXX_VERSION}")
else()
  message(STATUS "plascan: OpenMP not found, single-thread fallback")
endif()

# ── GTest ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  message(STATUS "plascan: found GTest")
endif()
