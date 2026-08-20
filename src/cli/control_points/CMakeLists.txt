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
    if(APPLE AND TARGET Qt6::QOffscreenIntegrationPlugin)
        qt_import_plugins(marker_print_cli INCLUDE Qt6::QOffscreenIntegrationPlugin)
    endif()
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
