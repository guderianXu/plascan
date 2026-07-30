if(TARGET superpoint)
    add_plascan_cli(feature_extract_cli
        SOURCE cli_feature_extract.cpp
        LIBRARIES
            superpoint
            feature_extractors_disk
            feature_extractors_aliked
            feature_extractors_traditional
            plascan_common_io
            plascan_common_string_utils
            Qt6::Core
            ${TORCH_LIBRARIES}
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
            ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/superpoint
            ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/tradition
            ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/disk
            ${CMAKE_SOURCE_DIR}/src/core/feature_extractors/aliked
        COMPILE_DEFINITIONS PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )
endif()

if(TARGET superglue_matcher)
    add_plascan_cli(feature_match_cli
        SOURCE cli_feature_match.cpp
        LIBRARIES
            feature_match_factory
            superglue_matcher
            lightglue_matcher
            feature_match_traditional
            loftr_matcher
            plascan_common_io
            Qt6::Core
            ${TORCH_LIBRARIES}
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/core/feature_match
            ${CMAKE_SOURCE_DIR}/src/core/feature_match/superglue
            ${CMAKE_SOURCE_DIR}/src/core/feature_match/lightglue
            ${CMAKE_SOURCE_DIR}/src/core/feature_match/tradition
            ${CMAKE_SOURCE_DIR}/src/core/feature_match/loftr
            ${CMAKE_SOURCE_DIR}/src/core/feature_extractors
    )
endif()

if(TARGET matchphototask AND TARGET plascan_cli_photogrammetry_common)
    add_plascan_cli(match_photos_cli
        SOURCE cli_match_photos.cpp
        LIBRARIES
            matchphototask
            plascan_cli_photogrammetry_common
            plascan_common_project
            ${TORCH_LIBRARIES}
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/algorithm
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/pair_selection
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/runtime
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask/task
            ${TORCH_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
        COMPILE_DEFINITIONS PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        WITH_NVRTC_RPATH
    )
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
