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
find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "5.0.0")
  # OpenCV 5 将 calib3d/features2d 能力拆到 geometry/features/stereo 模块。
  set(PLASCAN_OPENCV_COMPONENTS core imgproc geometry stereo features imgcodecs flann xfeatures2d ximgproc)
else()
  set(PLASCAN_OPENCV_COMPONENTS core imgproc calib3d features2d imgcodecs flann)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${PLASCAN_OPENCV_COMPONENTS})
message(STATUS "plascan: found OpenCV ${OpenCV_VERSION} (${PLASCAN_OPENCV_COMPONENTS})")

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
  # 防止 Caffe2 自动检测, 避免 nvcc 13.1 不支持的 compute_50
  # TORCH_CUDA_ARCH_LIST 格式为 "7.5;8.6;8.9" (带小数点)
  set(TORCH_CUDA_ARCH_LIST "7.5;8.6;8.9" CACHE STRING "Torch CUDA architectures")
  set(ENV{TORCH_CUDA_ARCH_LIST} "7.5;8.6;8.9")
  message(STATUS "plascan: CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "plascan: Apple Silicon — skipping CUDA, using MPS acceleration")
endif()

# conda linker fix (Linux only). Keep this opt-in because full conda toolchains
# must not mix the system linker with the conda sysroot.
option(PLASCAN_USE_SYSTEM_LINKER_FOR_TORCH
  "Temporarily use /usr/bin/ld while finding LibTorch in mixed system/conda builds"
  ON)
if(PLASCAN_USE_SYSTEM_LINKER_FOR_TORCH AND DEFINED ENV{CONDA_PREFIX} AND NOT APPLE)
  set(PLASCAN_ORIGINAL_CMAKE_LINKER ${CMAKE_LINKER})
  set(CMAKE_LINKER "/usr/bin/ld" CACHE FILEPATH "System linker" FORCE)
  message(STATUS "plascan: Using system linker")
endif()

find_package(Torch REQUIRED)
message(STATUS "plascan: found LibTorch")

if(DEFINED PLASCAN_ORIGINAL_CMAKE_LINKER AND NOT APPLE)
  if(PLASCAN_USE_SYSTEM_BINUTILS_FOR_MIXED_TOOLCHAIN AND EXISTS "/usr/bin/ld")
    set(CMAKE_LINKER "/usr/bin/ld" CACHE FILEPATH "Linker" FORCE)
  else()
    set(CMAKE_LINKER ${PLASCAN_ORIGINAL_CMAKE_LINKER} CACHE FILEPATH "Linker" FORCE)
  endif()
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

# ── plamatrix (submodule) ──────────────────────────────────────────────────────
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
