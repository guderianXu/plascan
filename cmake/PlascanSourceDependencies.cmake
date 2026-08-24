include_guard(GLOBAL)

include(ExternalProject)
include(${CMAKE_CURRENT_LIST_DIR}/PlascanSourceDependencyVersions.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/PlascanVerifySourceCheckout.cmake)

set(PLASCAN_SOURCE_DEPENDENCY_PREFIX
    "${CMAKE_BINARY_DIR}/install"
    CACHE PATH "Install prefix shared by the pinned Qt, OpenCV, GDAL, and AprilTag source builds")
option(PLASCAN_SOURCE_OPENCV_WITH_CUDA
       "Build the source OpenCV dependency with CUDA support" OFF)
option(PLASCAN_SOURCE_OPENCV_WITH_CUDNN
       "Build the source OpenCV DNN backend with cuDNN support" OFF)
set(PLASCAN_SOURCE_CUDA_ARCHITECTURES ""
    CACHE STRING "CUDA architectures used by the source OpenCV build")

if(PLASCAN_SOURCE_OPENCV_WITH_CUDNN AND NOT PLASCAN_SOURCE_OPENCV_WITH_CUDA)
  message(FATAL_ERROR
    "PLASCAN_SOURCE_OPENCV_WITH_CUDNN requires PLASCAN_SOURCE_OPENCV_WITH_CUDA=ON")
endif()

set(_plascan_qt_source "${CMAKE_SOURCE_DIR}/3rdparty/qt")
set(_plascan_opencv_source "${CMAKE_SOURCE_DIR}/3rdparty/opencv")
set(_plascan_apriltag_source "${CMAKE_SOURCE_DIR}/3rdparty/apriltag")
set(_plascan_gdal_source "${CMAKE_SOURCE_DIR}/3rdparty/gdal")
set(_plascan_poisson_recon_source "${CMAKE_SOURCE_DIR}/3rdparty/PoissonRecon")

plascan_verify_source_checkout(
  "Qt ${PLASCAN_SOURCE_QT_VERSION}"
  "${_plascan_qt_source}"
  "${PLASCAN_SOURCE_QT_COMMIT}"
  "configure.bat")
plascan_verify_source_checkout(
  "Qt Base ${PLASCAN_SOURCE_QT_VERSION}"
  "${_plascan_qt_source}/qtbase"
  "${PLASCAN_SOURCE_QTBASE_COMMIT}"
  "CMakeLists.txt")
plascan_verify_source_checkout(
  "Qt Shader Tools ${PLASCAN_SOURCE_QT_VERSION}"
  "${_plascan_qt_source}/qtshadertools"
  "${PLASCAN_SOURCE_QTSHADERTOOLS_COMMIT}"
  "CMakeLists.txt")
plascan_verify_source_checkout(
  "OpenCV ${PLASCAN_SOURCE_OPENCV_VERSION}"
  "${_plascan_opencv_source}"
  "${PLASCAN_SOURCE_OPENCV_COMMIT}"
  "CMakeLists.txt")
plascan_verify_source_checkout(
  "AprilTag ${PLASCAN_SOURCE_APRILTAG_VERSION}"
  "${_plascan_apriltag_source}"
  "${PLASCAN_SOURCE_APRILTAG_COMMIT}"
  "apriltag.c")
plascan_verify_source_checkout(
  "GDAL ${PLASCAN_SOURCE_GDAL_VERSION}"
  "${_plascan_gdal_source}"
  "${PLASCAN_SOURCE_GDAL_COMMIT}"
  "gdal.cmake")
plascan_verify_source_checkout(
  "PoissonRecon"
  "${_plascan_poisson_recon_source}"
  "${PLASCAN_SOURCE_POISSON_RECON_COMMIT}"
  "Src/Reconstructors.h")

set(_plascan_external_cmake_args
  "-DCMAKE_BUILD_TYPE:STRING=Release"
  "-DCMAKE_INSTALL_PREFIX:PATH=${PLASCAN_SOURCE_DEPENDENCY_PREFIX}"
  "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON")
foreach(_plascan_forwarded_variable IN ITEMS
    CMAKE_TOOLCHAIN_FILE
    VCPKG_INSTALLED_DIR
    VCPKG_TARGET_TRIPLET
    VCPKG_OVERLAY_PORTS
    CUDAToolkit_ROOT
    CMAKE_CUDA_COMPILER
    CMAKE_CUDA_HOST_COMPILER)
  if(DEFINED ${_plascan_forwarded_variable} AND
     NOT "${${_plascan_forwarded_variable}}" STREQUAL "")
    list(APPEND _plascan_external_cmake_args
      "-D${_plascan_forwarded_variable}:STRING=${${_plascan_forwarded_variable}}")
  endif()
endforeach()

set(_plascan_qt_configure "${_plascan_qt_source}/configure")
if(WIN32)
  set(_plascan_qt_configure "${_plascan_qt_source}/configure.bat")
endif()

set(_plascan_qt_configure_args
  -prefix "${PLASCAN_SOURCE_DEPENDENCY_PREFIX}"
  -release
  -shared
  -opensource
  -confirm-license
  -nomake examples
  -nomake tests
  -submodules qtbase,qtshadertools
  -cmake-generator Ninja)
if(WIN32)
  list(APPEND _plascan_qt_configure_args -schannel)
endif()
list(APPEND _plascan_qt_configure_args -- ${_plascan_external_cmake_args})

ExternalProject_Add(plascan_qt_source
  SOURCE_DIR "${_plascan_qt_source}"
  BINARY_DIR "${CMAKE_BINARY_DIR}/qt-build"
  CONFIGURE_COMMAND "${_plascan_qt_configure}" ${_plascan_qt_configure_args}
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel
  INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR>
  BUILD_BYPRODUCTS
    "${PLASCAN_SOURCE_DEPENDENCY_PREFIX}/lib/cmake/Qt6/Qt6Config.cmake"
  STEP_TARGETS configure
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE
  USES_TERMINAL_INSTALL TRUE)

set(_plascan_opencv_build_list
  core,imgproc,imgcodecs,flann,dnn,features,geometry,stereo)
set(_plascan_opencv_cmake_args
  ${_plascan_external_cmake_args}
  "-UOPENCV_EXTRA_MODULES_PATH"
  "-UBUILD_opencv_x*"
  "-DOPENCV_CMAKE_HOOKS_DIR:PATH=${CMAKE_SOURCE_DIR}/cmake/opencv-hooks"
  "-DBUILD_LIST:STRING=${_plascan_opencv_build_list}"
  "-DBUILD_SHARED_LIBS:BOOL=ON"
  "-DBUILD_TESTS:BOOL=OFF"
  "-DBUILD_PERF_TESTS:BOOL=OFF"
  "-DBUILD_EXAMPLES:BOOL=OFF"
  "-DBUILD_opencv_apps:BOOL=OFF"
  "-DBUILD_JAVA:BOOL=OFF"
  "-DBUILD_opencv_python2:BOOL=OFF"
  "-DBUILD_opencv_python3:BOOL=OFF"
  "-DBUILD_opencv_world:BOOL=OFF"
  "-DWITH_QT:BOOL=OFF"
  "-DWITH_CUDA:BOOL=${PLASCAN_SOURCE_OPENCV_WITH_CUDA}"
  "-DWITH_CUDNN:BOOL=${PLASCAN_SOURCE_OPENCV_WITH_CUDNN}"
  "-DOPENCV_DNN_CUDA:BOOL=${PLASCAN_SOURCE_OPENCV_WITH_CUDNN}")
if(PLASCAN_SOURCE_CUDA_ARCHITECTURES)
  list(APPEND _plascan_opencv_cmake_args
    "-DCMAKE_CUDA_ARCHITECTURES:STRING=${PLASCAN_SOURCE_CUDA_ARCHITECTURES}")
endif()

ExternalProject_Add(plascan_opencv_source
  SOURCE_DIR "${_plascan_opencv_source}"
  BINARY_DIR "${CMAKE_BINARY_DIR}/opencv-build"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS ${_plascan_opencv_cmake_args}
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel
  INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR>
  DEPENDS plascan_qt_source
  STEP_TARGETS configure
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE
  USES_TERMINAL_INSTALL TRUE)

ExternalProject_Add(plascan_apriltag_source
  SOURCE_DIR "${_plascan_apriltag_source}"
  BINARY_DIR "${CMAKE_BINARY_DIR}/apriltag-build"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS
    ${_plascan_external_cmake_args}
    "-DBUILD_SHARED_LIBS:BOOL=ON"
    "-DBUILD_EXAMPLES:BOOL=OFF"
    "-DBUILD_PYTHON_WRAPPER:BOOL=OFF"
    "-DBUILD_TESTING:BOOL=OFF"
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel
  INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR>
  STEP_TARGETS configure
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE
  USES_TERMINAL_INSTALL TRUE)

ExternalProject_Add(plascan_gdal_source
  SOURCE_DIR "${_plascan_gdal_source}"
  BINARY_DIR "${CMAKE_BINARY_DIR}/gdal-build"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS
    ${_plascan_external_cmake_args}
    "-DBUILD_SHARED_LIBS:BOOL=ON"
    "-DBUILD_APPS:BOOL=OFF"
    "-DBUILD_PYTHON_BINDINGS:BOOL=OFF"
    "-DBUILD_TESTING:BOOL=OFF"
    "-DGDAL_BUILD_OPTIONAL_DRIVERS:BOOL=ON"
    "-DOGR_BUILD_OPTIONAL_DRIVERS:BOOL=ON"
    "-DGDAL_ENABLE_PLUGINS:BOOL=OFF"
    "-DGDAL_ENABLE_PLUGINS_NO_DEPS:BOOL=OFF"
    "-DGDAL_USE_INTERNAL_LIBS:STRING=OFF"
    "-DGDAL_USE_GEOTIFF:BOOL=ON"
    "-DGDAL_USE_JPEG:BOOL=ON"
    "-DGDAL_USE_JSONC:BOOL=ON"
    "-DGDAL_USE_OPENJPEG:BOOL=ON"
    "-DGDAL_USE_PNG:BOOL=ON"
    "-DGDAL_USE_TIFF:BOOL=ON"
    "-DGDAL_USE_ZLIB:BOOL=ON"
    "-DGDAL_USE_ZSTD:BOOL=ON"
    "-DGDAL_ENABLE_DRIVER_JP2OPENJPEG:BOOL=ON"
    "-DGDAL_ENABLE_DRIVER_MBTILES:BOOL=OFF"
    "-DGDAL_ENABLE_DRIVER_PDS:BOOL=ON"
    "-DOGR_ENABLE_DRIVER_GPKG:BOOL=OFF"
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel
  INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR>
  BUILD_BYPRODUCTS
    "${PLASCAN_SOURCE_DEPENDENCY_PREFIX}/lib/cmake/gdal/GDALConfig.cmake"
  STEP_TARGETS configure
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE
  USES_TERMINAL_INSTALL TRUE)

add_custom_target(plascan_source_dependencies ALL
  DEPENDS plascan_qt_source plascan_opencv_source plascan_apriltag_source plascan_gdal_source)

message(STATUS "PlaScan source dependency superbuild")
message(STATUS "  Qt: ${PLASCAN_SOURCE_QT_VERSION} (${_plascan_qt_source})")
message(STATUS "  OpenCV: ${PLASCAN_SOURCE_OPENCV_VERSION} (${_plascan_opencv_source})")
message(STATUS "  AprilTag: ${PLASCAN_SOURCE_APRILTAG_VERSION} (${_plascan_apriltag_source})")
message(STATUS "  GDAL: ${PLASCAN_SOURCE_GDAL_VERSION} (${_plascan_gdal_source})")
message(STATUS "  PoissonRecon: ${PLASCAN_SOURCE_POISSON_RECON_COMMIT} (${_plascan_poisson_recon_source})")
message(STATUS "  Install prefix: ${PLASCAN_SOURCE_DEPENDENCY_PREFIX}")
