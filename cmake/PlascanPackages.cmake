include_guard(GLOBAL)

# ==============================================================================
# PlaScan 统一依赖查找
#
# 所有 find_package 调用集中在此，子模块直接使用已找到的 target，
# 不再重复搜索。此文件由根 CMakeLists.txt 在 add_subdirectory 之前 include。
# ==============================================================================

# ── Qt6 ───────────────────────────────────────────────────────────────────────
# 合并所有模块所需组件（Core/Gui/Widgets/Concurrent/OpenGL/OpenGLWidgets）
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Concurrent OpenGL OpenGLWidgets)
message(STATUS "plascan: found Qt6 ${Qt6_VERSION}")

# ── OpenCV ────────────────────────────────────────────────────────────────────
# 合并所有模块所需组件（core/imgproc/calib3d/imgcodecs）
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d imgcodecs)
message(STATUS "plascan: found OpenCV ${OpenCV_VERSION}")

# ── 平台检测 ──────────────────────────────────────────────────────────────────
set(PLASCAN_APPLE_SILICON OFF)
set(PLASCAN_CUDA_AVAILABLE OFF)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(PLASCAN_APPLE_SILICON ON)
  message(STATUS "plascan: Apple Silicon (M-series) detected — CUDA disabled, MPS used via PyTorch")
endif()

# ── LibTorch ──────────────────────────────────────────────────────────────────
if(NOT PLASCAN_APPLE_SILICON)
  # CUDA 架构 (仅 Linux/Windows NVIDIA GPU)
  if(NOT DEFINED PLASCAN_CUDA_ARCHITECTURES)
    set(PLASCAN_CUDA_ARCHITECTURES "75;86;89" CACHE STRING "Target CUDA architectures")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${PLASCAN_CUDA_ARCHITECTURES})
  message(STATUS "plascan: CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "plascan: Apple Silicon — skipping CUDA, using MPS acceleration")
endif()

# conda linker fix (Linux only)
if(DEFINED ENV{CONDA_PREFIX} AND NOT APPLE)
  set(PLASCAN_ORIGINAL_CMAKE_LINKER ${CMAKE_LINKER})
  set(CMAKE_LINKER "/usr/bin/ld" CACHE FILEPATH "System linker" FORCE)
  message(STATUS "plascan: Using system linker")
endif()

find_package(Torch REQUIRED)
message(STATUS "plascan: found LibTorch")

if(DEFINED PLASCAN_ORIGINAL_CMAKE_LINKER AND NOT APPLE)
  set(CMAKE_LINKER ${PLASCAN_ORIGINAL_CMAKE_LINKER} CACHE FILEPATH "Linker" FORCE)
endif()

# ── GDAL ──────────────────────────────────────────────────────────────────────
find_package(GDAL REQUIRED)
if(TARGET GDAL::GDAL)
  set(PLASCAN_GDAL_TARGET GDAL::GDAL CACHE INTERNAL "GDAL CMake target")
else()
  set(PLASCAN_GDAL_TARGET ${GDAL_LIBRARIES} CACHE INTERNAL "GDAL CMake target")
endif()
message(STATUS "plascan: found GDAL, target=${PLASCAN_GDAL_TARGET}")

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
