add_plascan_cli_contract_test(test_workflow_cli
    test_workflow_cli.cpp
)
target_link_libraries(test_workflow_cli PRIVATE project_workflows)

if(TARGET reconstruct_pipeline_cli)
    target_compile_definitions(test_workflow_cli PRIVATE
        PLASCAN_RECONSTRUCT_PIPELINE_CLI_PATH="$<TARGET_FILE:reconstruct_pipeline_cli>"
    )
    add_dependencies(test_workflow_cli reconstruct_pipeline_cli)
endif()
if(TARGET mesh_reconstruct_cli)
    target_compile_definitions(test_workflow_cli PRIVATE
        PLASCAN_MESH_RECONSTRUCT_CLI_PATH="$<TARGET_FILE:mesh_reconstruct_cli>"
    )
    add_dependencies(test_workflow_cli mesh_reconstruct_cli)
endif()
if(TARGET aerial_triangulation_cli)
    target_compile_definitions(test_workflow_cli PRIVATE
        PLASCAN_AERIAL_TRIANGULATION_CLI_PATH="$<TARGET_FILE:aerial_triangulation_cli>"
    )
    add_dependencies(test_workflow_cli aerial_triangulation_cli)
endif()

if(TARGET mvs_depth_reprocess_cli)
    target_compile_definitions(test_workflow_cli PRIVATE
        PLASCAN_MVS_DEPTH_REPROCESS_CLI_PATH="$<TARGET_FILE:mvs_depth_reprocess_cli>"
    )
    add_dependencies(test_workflow_cli mvs_depth_reprocess_cli)
endif()
