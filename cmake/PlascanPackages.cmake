include_guard(GLOBAL)

# ==============================================================================
# PlaScan 统一依赖查找
#
# 所有 find_package 调用集中在此，子模块直接使用已找到的 target，
# 不再重复搜索。此文件由根 CMakeLists.txt 在 add_subdirectory 之前 include。
# ==============================================================================

# ── Qt6 ───────────────────────────────────────────────────────────────────────
# 核心库和 CLI 只依赖公开 Qt 组件；GUI 关闭时不触发 Widgets、
# ShaderTools 或私有 Gui 模块的发现。
set(PLASCAN_QT_COMPONENTS Core Gui Network Concurrent)
if(PLASCAN_BUILD_GUI)
  list(APPEND PLASCAN_QT_COMPONENTS Widgets ShaderTools ShaderToolsTools)
  if(BUILD_TESTS AND PLASCAN_BUILD_GUI_TESTS)
    list(APPEND PLASCAN_QT_COMPONENTS Test)
  endif()
endif()
if(WIN32 AND PLASCAN_BUILD_GUI)
  # vcpkg exposes private Qt modules as explicit Qt6 components.
  list(APPEND PLASCAN_QT_COMPONENTS GuiPrivate)
endif()
list(REMOVE_DUPLICATES PLASCAN_QT_COMPONENTS)
if(PLASCAN_BUILD_GUI)
  # CameraSceneWidget intentionally uses QRhi private APIs and PlaScan ships
  # the matching Qt runtime. Confirm that version lock once at this boundary
  # instead of repeating Qt's private-module warning on every configure.
  set(QT_NO_PRIVATE_MODULE_WARNING ON)
endif()
find_package(Qt6 6.7 REQUIRED COMPONENTS ${PLASCAN_QT_COMPONENTS})
if(PLASCAN_BUILD_GUI AND NOT TARGET Qt6::GuiPrivate)
  # Qt's vcpkg package exports private modules as standalone package configs on
  # Linux instead of loading them with the public Qt6 component set.
  find_package(Qt6GuiPrivate ${Qt6_VERSION} CONFIG QUIET)
endif()
if(PLASCAN_BUILD_GUI AND NOT TARGET Qt6::GuiPrivate)
  message(FATAL_ERROR
    "PlaScan requires the Qt GuiPrivate target. Install the Qt base private development package.")
endif()
message(STATUS "plascan: found Qt6 ${Qt6_VERSION}")

get_target_property(_PLASCAN_QT_GUI_PUBLIC_FEATURES Qt6::Gui QT_ENABLED_PUBLIC_FEATURES)
list(FIND _PLASCAN_QT_GUI_PUBLIC_FEATURES vulkan _PLASCAN_QT_GUI_VULKAN_FEATURE_INDEX)
set(PLASCAN_QT_HAS_VULKAN ON)
if(_PLASCAN_QT_GUI_VULKAN_FEATURE_INDEX EQUAL -1)
  set(PLASCAN_QT_HAS_VULKAN OFF)
  message(WARNING
    "QtGui was built without Vulkan support. PlaScan will use the platform "
    "QRhi fallback backend (D3D11 on Windows or OpenGL elsewhere).")
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
# PlaMatrix CPU linear algebra is self-contained and native-only.
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
if(MSVC)
  # PlaMatrix has a portable OpenMP collapse pragma that MSVC treats as an
  # ignored-clause warning. Keep the suppression scoped to the third-party
  # target; PlaScan sources still report C4849 normally.
  target_compile_options(plamatrix PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/wd4849>
  )
endif()
message(STATUS "plascan: using plamatrix from 3rdparty/")

# ── plapoint (submodule) ───────────────────────────────────────────────────────
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/plapoint ${CMAKE_BINARY_DIR}/3rdparty/plapoint)
if(MSVC AND PLAPOINT_WITH_CUDA)
  # PlaPoint deliberately uses host constexpr limits in device helpers. NVCC
  # also reports #550 for variables consumed in discarded if-constexpr paths.
  # Keep both compatibility settings confined to this third-party CUDA target.
  target_compile_options(plapoint PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -diag-suppress=550>
  )
endif()
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
