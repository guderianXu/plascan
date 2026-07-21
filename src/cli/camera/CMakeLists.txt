if(TARGET camera)
    add_plascan_cli(camera_convert_cli
        SOURCE cli_camera_convert.cpp
        LIBRARIES camera
        INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/src/core/camera
    )
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
