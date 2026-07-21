add_plascan_cli_contract_test(test_features_cli
    test_features_cli.cpp
)

if(TARGET feature_match_cli)
    target_compile_definitions(test_features_cli PRIVATE
        PLASCAN_FEATURE_MATCH_CLI_PATH="$<TARGET_FILE:feature_match_cli>"
    )
    add_dependencies(test_features_cli feature_match_cli)
endif()
if(TARGET match_photos_cli)
    target_compile_definitions(test_features_cli PRIVATE
        PLASCAN_MATCH_PHOTOS_CLI_PATH="$<TARGET_FILE:match_photos_cli>"
    )
    add_dependencies(test_features_cli match_photos_cli)
endif()
