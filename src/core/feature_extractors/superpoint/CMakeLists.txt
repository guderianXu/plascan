cmake_minimum_required(VERSION 3.14)
project(SuperPointP LANGUAGES CXX)

option(SUPERPOINT_BUILD_EXAMPLES "Build demo and feature_matching executables" ON)
option(SUPERPOINT_INSTALL "Install targets and export CMake package files" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 查找OpenCV
find_package(OpenCV REQUIRED)

# Try to find Qt6. We only need QtCore for QFile/QDataStream.

find_package(Qt6 COMPONENTS Core QUIET)
if (Qt6_FOUND)
    set(QT_LIBS Qt6::Core)
else()
    message(STATUS "Qt6 not found; QFileBinaryIO will not be compiled. If you want Qt I/O, install Qt6 and re-run CMake.")
    set(QT_LIBS "")
endif()

# 公共包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# SuperPoint 库 (拆分为 4 个文件, 每个 ≤400 行)
set(SUPERPOINT_SOURCES
    SuperPoint.cpp
    SuperPointPostprocess.cpp
    SuperPointDetect.cpp
    SuperPointBatch.cpp
    SuperPoint.h
)
if (QT_LIBS)
    list(APPEND SUPERPOINT_SOURCES ../FeatureFileIO.cpp ../FeatureFileIO.h)
endif()

add_library(superpoint STATIC ${SUPERPOINT_SOURCES})

if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(superpoint PRIVATE USE_CUDA=1)
endif()

target_include_directories(superpoint
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
           $<INSTALL_INTERFACE:include>
)

target_link_libraries(superpoint
    PUBLIC ${TORCH_LIBRARIES}
           ${OpenCV_LIBS}
)
if (QT_LIBS)
    target_link_libraries(superpoint PUBLIC ${QT_LIBS})
endif()

set_target_properties(superpoint PROPERTIES
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(PLASCAN_EXTRA_NVRTC_RPATHS)
    set_property(TARGET superpoint APPEND PROPERTY BUILD_RPATH "${PLASCAN_EXTRA_NVRTC_RPATHS}")
endif()

# ── 单元测试 ──────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_executable(test_superpoint tests/test_superpoint.cpp)
        target_link_libraries(test_superpoint PRIVATE
            superpoint
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_compile_definitions(test_superpoint PRIVATE
            TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
        )
        set_target_properties(test_superpoint PROPERTIES
            BUILD_RPATH_USE_ORIGIN TRUE
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
        if(PLASCAN_EXTRA_NVRTC_RPATHS)
            set_property(TARGET test_superpoint APPEND PROPERTY BUILD_RPATH "${PLASCAN_EXTRA_NVRTC_RPATHS}")
        endif()
        include(GoogleTest)
        gtest_discover_tests(test_superpoint)
    endif()
endif()

if(SUPERPOINT_INSTALL)
    install(TARGETS superpoint
        EXPORT SuperPointTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )

    install(FILES SuperPoint.h DESTINATION include)

    # 导出 CMake 包文件
    include(CMakePackageConfigHelpers)

    set(config_install_dir lib/cmake/SuperPoint)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SuperPointConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/SuperPointConfig.cmake
        INSTALL_DESTINATION ${config_install_dir}
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/SuperPointConfigVersion.cmake
        VERSION 1.0.0
        COMPATIBILITY AnyNewerVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/SuperPointConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/SuperPointConfigVersion.cmake
        DESTINATION ${config_install_dir}
    )

    install(EXPORT SuperPointTargets
        FILE SuperPointTargets.cmake
        NAMESPACE SuperPoint::
        DESTINATION ${config_install_dir}
    )
endif()
