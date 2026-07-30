if(TARGET sfm AND TARGET bundle_adjust AND TARGET lidar)
    add_plascan_cli(bundle_adjust_cli
        SOURCE cli_bundle_adjust.cpp
        SOURCES
            ${CMAKE_SOURCE_DIR}/src/gui/project/services/BundleAdjustService.cpp
        LIBRARIES
            sfm
            bundle_adjust
            lidar
            camera
            intersection
            plascan_common_log
            plascan_common_project
            Qt6::Core
            Qt6::Gui
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
            ${CMAKE_SOURCE_DIR}/src/core/camera
            ${CMAKE_SOURCE_DIR}/src/core/intersection
            ${CMAKE_SOURCE_DIR}/src/core/lidar
            ${CMAKE_SOURCE_DIR}/src/core/sfm
            ${CMAKE_SOURCE_DIR}/src/gui/project/archive
            ${CMAKE_SOURCE_DIR}/src/gui/project/data
            ${CMAKE_SOURCE_DIR}/src/gui/project/services
            ${CMAKE_SOURCE_DIR}/src/gui/project/support
            ${OpenCV_INCLUDE_DIRS}
        WITH_NVRTC_RPATH
    )
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
