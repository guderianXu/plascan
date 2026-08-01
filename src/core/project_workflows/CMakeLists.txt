add_library(project_workflows STATIC
    ProjectWorkflowOperations.cpp
)

target_include_directories(project_workflows PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(project_workflows PUBLIC cxx_std_20)

target_link_libraries(project_workflows
    PUBLIC
        Qt6::Core
        plascan_common_io
        sfm_postprocess
        terrain
        plapoint::plapoint
)
