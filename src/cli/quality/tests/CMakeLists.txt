add_plascan_cli_contract_test(test_quality_cli
    test_quality_cli.cpp
)
if(TARGET model_quality_cli)
    target_compile_definitions(test_quality_cli PRIVATE
        PLASCAN_MODEL_QUALITY_CLI_PATH="$<TARGET_FILE:model_quality_cli>"
    )
    add_dependencies(test_quality_cli model_quality_cli)
endif()
