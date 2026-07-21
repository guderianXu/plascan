if(TARGET dense_match)
    add_plascan_cli(dense_match_cli
        SOURCE cli_dense_match.cpp
        LIBRARIES dense_match
        INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/src/core/dense_match
    )
endif()

if(TARGET mvs)
    set(mvs_cli_include_dirs
        ${CMAKE_SOURCE_DIR}/src/core/mvs
        ${CMAKE_SOURCE_DIR}/src/core/camera
    )
    set(mvs_cli_libraries mvs camera plascan_common_io Qt6::Core)

    add_plascan_cli(rectify_cli
        SOURCE cli_epipolar_rectify.cpp
        LIBRARIES ${mvs_cli_libraries}
        INCLUDE_DIRECTORIES ${mvs_cli_include_dirs}
    )
    add_plascan_cli(triangulate_cli
        SOURCE cli_disparity_triangulate.cpp
        LIBRARIES ${mvs_cli_libraries}
        INCLUDE_DIRECTORIES ${mvs_cli_include_dirs}
    )
    add_plascan_cli(dense_cloud_refine_cli
        SOURCE cli_dense_cloud_refine.cpp
        LIBRARIES mvs
        INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/src/core/mvs
    )
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
