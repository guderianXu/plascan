add_plascan_cli_contract_test(test_reconstruction_cli
    test_reconstruction_cli.cpp
)

if(TARGET bundle_adjust_cli)
    target_compile_definitions(test_reconstruction_cli PRIVATE
        PLASCAN_BUNDLE_ADJUST_CLI_PATH="$<TARGET_FILE:bundle_adjust_cli>"
    )
    add_dependencies(test_reconstruction_cli bundle_adjust_cli)
endif()

if(TARGET planetary_linescan_ba_cli)
    target_compile_definitions(test_reconstruction_cli PRIVATE
        PLASCAN_PLANETARY_LINE_SCAN_BA_CLI_PATH="$<TARGET_FILE:planetary_linescan_ba_cli>"
    )
    add_dependencies(test_reconstruction_cli planetary_linescan_ba_cli)
endif()

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME BundleAdjustCliPythonContractsTest
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_bundle_adjust_cli_contracts.py
    )
    set_tests_properties(BundleAdjustCliPythonContractsTest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
