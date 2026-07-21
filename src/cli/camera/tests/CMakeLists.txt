add_plascan_cli_contract_test(test_camera_cli
    test_camera_cli.cpp
)

if(TARGET camera_convert_cli)
    target_compile_definitions(test_camera_cli PRIVATE
        PLASCAN_CAMERA_CONVERT_CLI_PATH="$<TARGET_FILE:camera_convert_cli>"
    )
    add_dependencies(test_camera_cli camera_convert_cli)
endif()
