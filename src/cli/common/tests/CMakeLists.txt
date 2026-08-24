add_plascan_cli_contract_test(test_cli_layout_contracts
    test_cli_layout_contracts.cpp
)

add_executable(test_cli_support
    test_cli_support.cpp
)
target_link_libraries(test_cli_support PRIVATE
    plascan_cli_support
    GTest::gtest_main
)
target_include_directories(test_cli_support SYSTEM PRIVATE
    ${PLASCAN_CLI_ROOT}/third_party
)
gtest_discover_tests(test_cli_support DISCOVERY_MODE PRE_TEST)

if(TARGET plascan_cli_photogrammetry_common)
    add_executable(test_cli_photogrammetry_common
        test_cli_photogrammetry_common.cpp
    )
    target_link_libraries(test_cli_photogrammetry_common PRIVATE
        plascan_cli_photogrammetry_common
        GTest::gtest_main
    )
    gtest_discover_tests(test_cli_photogrammetry_common DISCOVERY_MODE PRE_TEST)
endif()
