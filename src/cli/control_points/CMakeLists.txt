if(TARGET control_points)
    add_plascan_cli(marker_detect_cli
        SOURCE marker_detect_cli.cpp
        LIBRARIES control_points Qt6::Core Qt6::Gui
        INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/src/core/control_points
    )
    add_plascan_cli(marker_print_cli
        SOURCE marker_print_cli.cpp
        LIBRARIES control_points Qt6::Core Qt6::Gui
        INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/src/core/control_points
    )
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
