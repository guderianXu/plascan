cmake_minimum_required(VERSION 3.16)
project(plascan_cli LANGUAGES CXX)

include(CMakeParseArguments)

set(PLASCAN_CLI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    include(GoogleTest)

    function(add_plascan_cli_contract_test name source_file)
        add_executable(${name} ${source_file})
        target_link_libraries(${name} PRIVATE
            Qt6::Core
            ${PLASCAN_LIBZIP_TARGET}
            GTest::gtest_main
        )
        target_include_directories(${name} PRIVATE
            ${PLASCAN_CLI_ROOT}/common/tests
        )
        target_compile_definitions(${name} PRIVATE
            PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )
        gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST)
    endfunction()
endif()

# All CLI targets share the same output location and the small argument-parsing
# foundation. Domain-specific dependencies stay in each module CMakeLists.
function(add_plascan_cli name)
    set(options WITH_NVRTC_RPATH)
    set(one_value_args SOURCE)
    set(multi_value_args
        SOURCES
        LIBRARIES
        INCLUDE_DIRECTORIES
        COMPILE_DEFINITIONS
    )
    cmake_parse_arguments(CLI "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT CLI_SOURCE)
        message(FATAL_ERROR "add_plascan_cli(${name}) requires SOURCE")
    endif()

    add_executable(${name} ${CLI_SOURCE} ${CLI_SOURCES})
    target_link_libraries(${name} PRIVATE plascan_cli_support ${CLI_LIBRARIES} ${OpenCV_LIBS})
    target_include_directories(${name} PRIVATE
        ${PLASCAN_CLI_ROOT}/common
        ${CLI_INCLUDE_DIRECTORIES}
    )
    target_include_directories(${name} SYSTEM PRIVATE
        ${PLASCAN_CLI_ROOT}/third_party
    )
    if(CLI_COMPILE_DEFINITIONS)
        target_compile_definitions(${name} PRIVATE ${CLI_COMPILE_DEFINITIONS})
    endif()

    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(CLI_WITH_NVRTC_RPATH)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH_USE_ORIGIN TRUE
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()
endfunction()

add_subdirectory(common)
add_subdirectory(camera)
add_subdirectory(control_points)
add_subdirectory(features)
add_subdirectory(dense)
add_subdirectory(reconstruction)
add_subdirectory(workflows)
add_subdirectory(quality)
