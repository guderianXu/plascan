if(TARGET qc AND TARGET meshing AND TARGET camera AND TARGET plascan_cli_photogrammetry_common)
    add_plascan_cli(model_quality_cli
        SOURCE cli_model_quality.cpp
        LIBRARIES
            qc
            meshing
            plascan_cli_photogrammetry_common
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/core/camera
            ${CMAKE_SOURCE_DIR}/src/core/mesh
            ${CMAKE_SOURCE_DIR}/src/core/qc
            ${OpenCV_INCLUDE_DIRS}
    )
endif()

if(TARGET aerial_triangulation AND TARGET matchphototask)
    add_plascan_cli(mvs_pair_audit_cli
        SOURCE cli_mvs_pair_audit.cpp
        LIBRARIES
            aerial_triangulation
            matchphototask
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask
    )
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
