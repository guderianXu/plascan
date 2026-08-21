# 与 GUI“工作流程”菜单对应的无界面入口集中在此模块。
if(TARGET aerial_triangulation AND TARGET plascan_cli_photogrammetry_common)
    add_plascan_cli(aerial_triangulation_cli
        SOURCE cli_aerial_triangulation.cpp
        LIBRARIES
            aerial_triangulation
            plascan_cli_photogrammetry_common
            plascan_common_project
        COMPILE_DEFINITIONS PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        WITH_NVRTC_RPATH
    )
endif()

if(TARGET meshing)
    add_plascan_cli(mesh_reconstruct_cli
        SOURCE cli_mesh_reconstruct.cpp
        LIBRARIES meshing plapoint::plapoint plascan_common_io Qt6::Core Qt6::Gui
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/core/mesh
    )

    add_plascan_cli(texture_map_cli
        SOURCE cli_texture_map.cpp
        LIBRARIES meshing plapoint::plapoint plascan_common_io Qt6::Core Qt6::Gui
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/core/mesh
    )
endif()

if(TARGET terrain)
    add_plascan_cli(small_body_terrain_cli
        SOURCE cli_small_body_terrain.cpp
        LIBRARIES terrain Qt6::Core Qt6::Gui
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/core/terrain
            ${OpenCV_INCLUDE_DIRS}
    )
endif()

set(workflow_cli_base_targets
    image_matching
    matchphototask
    sfm
    mvs
    project_workflows
    meshing
    aerial_triangulation
    plascan_cli_photogrammetry_common
)
set(workflow_cli_available TRUE)
foreach(required_target IN LISTS workflow_cli_base_targets)
    if(NOT TARGET ${required_target})
        set(workflow_cli_available FALSE)
    endif()
endforeach()

function(add_reconstruction_workflow_cli name)
    add_plascan_cli(${name}
        SOURCE cli_reconstruct_pipeline.cpp
        SOURCES
            ReconstructionCliOptions.cpp
            ReconstructionCliOptions.h
            ReconstructionCliProgress.cpp
            ReconstructionCliProgress.h
            ReconstructionCliReport.cpp
            ReconstructionCliReport.h
            ReconstructionPipelineRunner.cpp
            ReconstructionPipelineRunner.h
        LIBRARIES
            aerial_triangulation
            plascan_cli_photogrammetry_common
            plascan_common_project
            sfm
            mvs
            project_workflows
            meshing
            ${ARGN}
            image_matching
            matchphototask
            camera
            control_points
            overlap
            Qt6::Core
            Qt6::Concurrent
            Qt6::Gui
            plascan_common_log
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/core/camera
            ${CMAKE_SOURCE_DIR}/src/core/control_points
            ${CMAKE_SOURCE_DIR}/src/core/image_matching
            ${CMAKE_SOURCE_DIR}/src/core/matchphototask
            ${CMAKE_SOURCE_DIR}/src/core/mesh
            ${CMAKE_SOURCE_DIR}/src/core/mvs
            ${CMAKE_SOURCE_DIR}/src/core/overlap
            ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
            ${CMAKE_SOURCE_DIR}/src/core/sfm
            ${CMAKE_SOURCE_DIR}/src/core/terrain
            ${OpenCV_INCLUDE_DIRS}
        COMPILE_DEFINITIONS PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        WITH_NVRTC_RPATH
    )
endfunction()

if(workflow_cli_available)
    add_reconstruction_workflow_cli(three_d_reconstruction_cli)
    target_compile_definitions(three_d_reconstruction_cli PRIVATE PLASCAN_THREE_D_ONLY)

    if(TARGET terrain)
        add_reconstruction_workflow_cli(reconstruct_pipeline_cli terrain)
    endif()
endif()

if(TARGET mvs)
    add_plascan_cli(mvs_depth_reprocess_cli
        SOURCE cli_mvs_depth_reprocess.cpp
        LIBRARIES
            mvs
            project_workflows
            camera
            Qt6::Core
            Qt6::Concurrent
            plascan_common_io
            plapoint::plapoint
            ${OpenCV_LIBS}
        INCLUDE_DIRECTORIES
            ${CMAKE_SOURCE_DIR}/src/common
            ${CMAKE_SOURCE_DIR}/src/core/camera
            ${CMAKE_SOURCE_DIR}/src/core/mvs
            ${OpenCV_INCLUDE_DIRS}
        WITH_NVRTC_RPATH
    )
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
