add_library(plascan_cli_support STATIC
    CliConsole.cpp
    CliConsole.h
    CliJsonIO.cpp
    CliJsonIO.h
    CliOutputPolicy.cpp
    CliOutputPolicy.h
    CliPathUtils.cpp
    CliPathUtils.h
    CliTokenUtils.cpp
    CliTokenUtils.h
)
target_link_libraries(plascan_cli_support PUBLIC
    plascan_common_io
    plascan_common_log
    plascan_common_string_utils
    Qt6::Core
)
target_include_directories(plascan_cli_support PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/common/log
)

if(TARGET camera)
    add_library(plascan_cli_photogrammetry_common STATIC
        cli_photogrammetry_common.cpp
        cli_photogrammetry_common.h
    )
    target_link_libraries(plascan_cli_photogrammetry_common PUBLIC
        camera
        plascan_cli_support
    )
    target_include_directories(plascan_cli_photogrammetry_common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_SOURCE_DIR}/src/core/camera
    )
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
