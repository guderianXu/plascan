add_plascan_cli_contract_test(test_dense_cli
    test_dense_cli.cpp
)
target_link_libraries(test_dense_cli PRIVATE plascan_common_io)
target_include_directories(test_dense_cli PRIVATE ${CMAKE_SOURCE_DIR}/src/common)

if(TARGET dense_match_cli)
    target_compile_definitions(test_dense_cli PRIVATE
        PLASCAN_DENSE_MATCH_CLI_PATH="$<TARGET_FILE:dense_match_cli>"
    )
    add_dependencies(test_dense_cli dense_match_cli)
endif()

if(TARGET dense_cloud_refine_cli)
    target_compile_definitions(test_dense_cli PRIVATE
        PLASCAN_DENSE_CLOUD_REFINE_CLI_PATH="$<TARGET_FILE:dense_cloud_refine_cli>"
    )
    add_dependencies(test_dense_cli dense_cloud_refine_cli)
endif()

if(TARGET triangulate_cli)
    add_executable(test_triangulate_cli test_triangulate_cli.cpp)
    target_link_libraries(test_triangulate_cli PRIVATE
        Qt6::Core
        ${OpenCV_LIBS}
        GTest::gtest_main
    )
    target_compile_definitions(test_triangulate_cli PRIVATE
        PLASCAN_TRIANGULATE_CLI_PATH="$<TARGET_FILE:triangulate_cli>"
    )
    add_dependencies(test_triangulate_cli triangulate_cli)
    gtest_discover_tests(test_triangulate_cli DISCOVERY_MODE PRE_TEST)
endif()
