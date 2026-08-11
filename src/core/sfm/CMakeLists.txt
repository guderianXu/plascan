cmake_minimum_required(VERSION 3.16)
project(plascan_sfm LANGUAGES CXX)

set(SFM_CORE_SOURCES
    geometry/ProjectionGeometry.cpp
    geometry/TriangulationQuality.cpp
    geometry/OpenCvCameraAdapter.cpp
    geometry/SimilarityGaugeNormalizer.cpp
    graph/CorrespondenceGraph.cpp
    graph/CovisibilityPartitioner.cpp
    graph/ObservationNetworkBuilder.cpp
    tracks/CorrespondenceTrackThinner.cpp
    tracks/MultiViewTrackBuilder.cpp
    reconstruction/SfmReconstruction.cpp
    pose/PnpSolver.cpp
    triangulation/Triangulator.cpp
    triangulation/InitialSparsePointFilter.cpp
    pipeline/IncrementalSfm.cpp
    pipeline/IncrementalSfmDetail.cpp
    pipeline/InitialPairInitializer.cpp
    pipeline/ImageRegistrationEngine.cpp
    pipeline/HierarchicalBaBlockSolver.cpp
    pipeline/HierarchicalBundleAdjuster.cpp
    pipeline/KnownPoseReconstructor.cpp
    pipeline/SfmCalibrationPreviewSampler.cpp
    pipeline/SfmBundleAdjustCoordinator.cpp
    ReferenceTerrainPrior.cpp
)

set(SFM_CORE_HEADERS
    common/DisjointSet.h
    common/SfmTypes.h
    geometry/ProjectionGeometry.h
    geometry/TriangulationQuality.h
    geometry/OpenCvCameraAdapter.h
    geometry/SimilarityGaugeNormalizer.h
    graph/CorrespondenceGraph.h
    graph/CovisibilityPartitioner.h
    graph/ObservationNetworkBuilder.h
    tracks/CorrespondenceTrackThinner.h
    tracks/MultiViewTrackBuilder.h
    reconstruction/SfmReconstruction.h
    pose/PnpSolver.h
    triangulation/Triangulator.h
    triangulation/InitialSparsePointFilter.h
    pipeline/IncrementalSfm.h
    pipeline/IncrementalSfmDetail.h
    pipeline/InitialPairInitializer.h
    pipeline/ImageRegistrationEngine.h
    pipeline/HierarchicalBaBlockSolver.h
    pipeline/HierarchicalBundleAdjuster.h
    pipeline/KnownPoseReconstructor.h
    pipeline/SfmCalibrationPreviewSampler.h
    pipeline/SfmBundleAdjustCoordinator.h
    ReferenceTerrainPrior.h
)

add_library(sfm_core STATIC ${SFM_CORE_SOURCES} ${SFM_CORE_HEADERS})

target_include_directories(sfm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
    ${CMAKE_CURRENT_SOURCE_DIR}/../intersection
    ${CMAKE_CURRENT_SOURCE_DIR}/../bundle_adjust
    ${CMAKE_CURRENT_SOURCE_DIR}/../control_points
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(sfm_core
    PUBLIC
        camera
        intersection
        bundle_adjust
        control_network
    PRIVATE
        plapoint::plapoint
        plamatrix::plamatrix
        plascan_common_log
        ${OpenCV_LIBS}
)

target_compile_features(sfm_core PUBLIC cxx_std_20)

set(SFM_POSTPROCESS_SOURCES
    quality/SfmQualityMetrics.cpp
    filtering/SparsePointCloudWorkspace.cpp
    filtering/SparsePointCloudProcessor.cpp
)

set(SFM_POSTPROCESS_HEADERS
    quality/SfmQualityMetrics.h
    filtering/SparsePointCloudWorkspace.h
    filtering/SparsePointCloudProcessor.h
)

add_library(sfm_postprocess STATIC ${SFM_POSTPROCESS_SOURCES} ${SFM_POSTPROCESS_HEADERS})

target_include_directories(sfm_postprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sfm_postprocess
    PUBLIC
        sfm_core
        plapoint::plapoint
)

target_compile_features(sfm_postprocess PUBLIC cxx_std_20)

set(SFM_PROJECT_SOURCES
    project/BaInputBuilder.cpp
    project/ProjectMatchInputReader.cpp
    project/BaTrackBuilder.cpp
    project/SurveyControlBaAdapter.cpp
    project/MarkerBaAdapter.cpp
    project/SfmQualityJsonSerializer.cpp
    TriangulationService.cpp
)

set(SFM_PROJECT_HEADERS
    project/BaInputBuilder.h
    project/ProjectMatchInputReader.h
    project/BaTrackBuilder.h
    project/SurveyControlBaAdapter.h
    project/MarkerBaAdapter.h
    project/SfmQualityJsonSerializer.h
    TriangulationService.h
)

add_library(sfm_project STATIC ${SFM_PROJECT_SOURCES} ${SFM_PROJECT_HEADERS})

target_include_directories(sfm_project PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(sfm_project
    PUBLIC
        sfm_core
        sfm_postprocess
        control_points
        Qt6::Core
        Qt6::Gui
    PRIVATE
        image_matching
        plascan_common_log
        plascan_common_io
        plascan_common_project
        ${OpenCV_LIBS}
)

target_compile_features(sfm_project PUBLIC cxx_std_20)

add_library(sfm INTERFACE)
target_link_libraries(sfm INTERFACE sfm_core sfm_postprocess sfm_project)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
