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

# ── LibTorch ──────────────────────────────────────────────────────────────────
# 预先设置 CUDA 架构，避免 PyTorch 的自动检测失败
# 如果用户通过 -DPLASCAN_CUDA_ARCHITECTURES 指定了架构，使用用户指定的值
# 否则使用默认值（支持 RTX 20/30/40 系列：75=Turing, 86=Ampere, 89=Ada）
if(NOT DEFINED PLASCAN_CUDA_ARCHITECTURES)
  set(PLASCAN_CUDA_ARCHITECTURES "75;86;89" CACHE STRING "Target CUDA architectures")
endif()

# 设置 CMAKE_CUDA_ARCHITECTURES，让 PyTorch 跳过自动检测
set(CMAKE_CUDA_ARCHITECTURES ${PLASCAN_CUDA_ARCHITECTURES})
message(STATUS "plascan: CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

# 如果在 conda 环境中，使用系统链接器而不是 conda 的链接器
# 这可以避免 conda ld 与系统 glibc 不兼容的问题
if(DEFINED ENV{CONDA_PREFIX})
  # 保存原始的 CMAKE_LINKER
  set(PLASCAN_ORIGINAL_CMAKE_LINKER ${CMAKE_LINKER})
  # 使用系统链接器
  set(CMAKE_LINKER "/usr/bin/ld" CACHE FILEPATH "System linker" FORCE)
  message(STATUS "plascan: Using system linker to avoid conda ld/glibc conflicts")
endif()

find_package(Torch REQUIRED)
message(STATUS "plascan: found LibTorch")

# 恢复原始的链接器设置
if(DEFINED PLASCAN_ORIGINAL_CMAKE_LINKER)
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
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "plascan: found OpenMP ${OpenMP_CXX_VERSION}")
else()
  message(STATUS "plascan: OpenMP not found, parallel modules will use single-thread fallback")
endif()

# ── GTest ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  message(STATUS "plascan: found GTest")
endif()
