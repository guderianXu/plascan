if(TARGET matchphototask)
    add_plascan_cli(feature_match_cli
        SOURCE cli_feature_match.cpp
        LIBRARIES
            matchphototask
            Qt6::Core
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/core/image_matching
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/pair_selection
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/task
        COMPILE_DEFINITIONS PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        WITH_NVRTC_RPATH
    )
    plascan_deploy_tensorrt_runtime(feature_match_cli)
endif()

if(TARGET matchphototask AND TARGET plascan_cli_photogrammetry_common)
    add_plascan_cli(match_photos_cli
        SOURCE cli_match_photos.cpp
        LIBRARIES
            matchphototask
            plascan_cli_photogrammetry_common
            plascan_common_project
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/algorithm
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/pair_selection
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/runtime
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/task
            ${OpenCV_INCLUDE_DIRS}
        COMPILE_DEFINITIONS PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        WITH_NVRTC_RPATH
    )
    plascan_deploy_tensorrt_runtime(match_photos_cli)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
