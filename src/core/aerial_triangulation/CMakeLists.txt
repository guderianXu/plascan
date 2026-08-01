cmake_minimum_required(VERSION 3.16)
project(plascan_aerial_triangulation LANGUAGES CXX)

add_library(aerial_triangulation STATIC
    model/AerialTriangulationOptions.h
    model/AerialTriangulationResolvedConfig.h
    model/AerialTriangulationResult.h
    workflow/AerialTriangulationWorkflow.cpp
    workflow/AerialTriangulationWorkflow.h
    workflow/AerialTriangulationPipeline.cpp
    workflow/AerialTriangulationPipeline.h
    preparation/MatchResultCatalog.cpp
    preparation/MatchResultCatalog.h
    preparation/TiePointPreparation.cpp
    preparation/TiePointPreparation.h
    reconstruction/MarkerPriorLoader.cpp
    reconstruction/MarkerPriorLoader.h
    reconstruction/CameraIntrinsicPriorSanitizer.cpp
    reconstruction/CameraIntrinsicPriorSanitizer.h
    reconstruction/SfmAttemptRunner.cpp
    reconstruction/SfmAttemptRunner.h
    reporting/AerialTriangulationResultWriter.cpp
    reporting/AerialTriangulationResultWriter.h
    reporting/QualityReportWriter.cpp
    reporting/QualityReportWriter.h
    search/AdaptiveFocalSearch.cpp
    search/AdaptiveFocalSearch.h
    preparation/ReconstructionPrerequisiteReport.cpp
    preparation/ReconstructionPrerequisiteReport.h
    reconstruction/SfmMatchDiagnostics.h
    reconstruction/SfmPairPlanner.h
    search/SfmSearchPolicy.cpp
    search/SfmSearchPolicy.h
)

target_include_directories(aerial_triangulation
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(aerial_triangulation
    PUBLIC
        Qt6::Core
        camera
        sfm
        matchphototask
    PRIVATE
        Qt6::Gui
        control_points
        plascan_common_io
        plascan_common_project
)

target_compile_definitions(aerial_triangulation PRIVATE
    PLASCAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_compile_features(aerial_triangulation PUBLIC cxx_std_20)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
