add_executable(test_marker_detect_cli test_marker_detect_cli.cpp)
target_link_libraries(test_marker_detect_cli PRIVATE
    Qt6::Core
    Qt6::Gui
    GTest::gtest_main
)
if(TARGET marker_detect_cli)
    target_compile_definitions(test_marker_detect_cli PRIVATE
        PLASCAN_MARKER_DETECT_CLI_PATH="$<TARGET_FILE:marker_detect_cli>"
    )
    add_dependencies(test_marker_detect_cli marker_detect_cli)
endif()
gtest_discover_tests(test_marker_detect_cli DISCOVERY_MODE PRE_TEST)

add_executable(test_marker_print_cli test_marker_print_cli.cpp)
target_link_libraries(test_marker_print_cli PRIVATE
    Qt6::Core
    GTest::gtest_main
)
if(TARGET marker_print_cli)
    target_compile_definitions(test_marker_print_cli PRIVATE
        PLASCAN_MARKER_PRINT_CLI_PATH="$<TARGET_FILE:marker_print_cli>"
    )
    add_dependencies(test_marker_print_cli marker_print_cli)
endif()
gtest_discover_tests(test_marker_print_cli DISCOVERY_MODE PRE_TEST)
