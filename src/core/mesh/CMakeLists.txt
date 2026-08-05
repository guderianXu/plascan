cmake_minimum_required(VERSION 3.16)

add_library(meshing STATIC
    SurfaceReconstructor.cpp
    PointCloudPreprocess.cpp
    SurfaceReconstructorPostprocess.cpp
    BoundaryAwareVoxelSimplifier.cpp
    SurfaceReconstructorHeightGrid.cpp
    DepthMapMeshBuilder.cpp
    DepthFusionFramePolicy.cpp
    DepthMeshCompleteness.cpp
    TriangleDistanceIndex.cpp
    DepthRayMetric.cpp
    DepthTsdfSurfaceBuilder.cpp
    DepthMeasuredSupportConnectivity.cpp
    DepthTsdfNarrowBandActivation.cpp
    BinaryGridMinCutSolver.cpp
    BinaryGridMinCutSolverFlow.cpp
    VisibilityOccupancyCleanup.cpp
    VisibilityOccupancyBoundaryExtractor.cpp
    VisibilityOccupancyCarrierSubdivider.cpp
    VisibilityOccupancyCarrierFairer.cpp
    VisibilityOccupancyCarrierFieldProjector.cpp
    VisibilityOccupancyHandleRepair.cpp
    VisibilityOccupancyWellComposedRepair.cpp
    VisibilityOccupancyDistanceField.cpp
    VisibilityOccupancySurfaceBuilder.cpp
    VisibilityOccupancyTsdfCompletion.cpp
    DepthTsdfCellSheetRecovery.cpp
    IsoSurfaceTopology.cpp
    ConsistentIsoSurfaceInternal.cpp
    ConsistentIsoSurfaceExtractor.cpp
    Mc33IsoSurfaceExtractor.cpp
    DepthImplicitFieldRegularizer.cpp
    DepthVisibilityHistogram.cpp
    AdaptiveTsdfOctree.cpp
    SparseTgvSolver.cpp
    VisualHullReconstructor.cpp
    VisualHullFieldEvaluator.cpp
    RobustSurfaceDisplacementSolver.cpp
    VisualHullDepthRefiner.cpp
    DepthConstrainedSurfaceRefiner.cpp
    MeshIO.cpp
    MeshColorizer.cpp
    MeshFaceColorOptimizer.cpp
    MeshQuadricSimplifier.cpp
    MeshFaceOrientation.cpp
    OpenMeshSimplifier.cpp
    MeshTopologyQuality.cpp
    MeshBoundaryAttribution.cpp
    MeshAcquisitionGapReport.cpp
    MeshIsotropicRemesher.cpp
    TextureMapper.cpp
    CameraTextureMapper.cpp
    CameraTextureMapperV4.cpp
    TextureAtlasPacker.cpp
    TextureSourcePreprocessor.cpp
    TextureVisibilityEvaluator.cpp
    TextureChartBuilder.cpp
    TextureAtlasBaker.cpp
    ModelWorkflowService.cpp
)

set(PLASCAN_MC33_INCLUDE_DIR "" CACHE PATH
    "Optional directory containing the MIT-licensed mc33cpp.h header")
if(PLASCAN_MC33_INCLUDE_DIR)
    if(NOT EXISTS "${PLASCAN_MC33_INCLUDE_DIR}/mc33cpp.h")
        message(FATAL_ERROR
            "PLASCAN_MC33_INCLUDE_DIR does not contain mc33cpp.h: "
            "${PLASCAN_MC33_INCLUDE_DIR}")
    endif()
    target_include_directories(meshing SYSTEM PRIVATE
        "${PLASCAN_MC33_INCLUDE_DIR}")
    target_compile_definitions(meshing PRIVATE PLASCAN_HAS_MC33=1)
else()
    target_compile_definitions(meshing PRIVATE PLASCAN_HAS_MC33=0)
endif()

find_package(OpenMesh CONFIG QUIET)
if(OpenMesh_FOUND)
    target_link_libraries(meshing PRIVATE OpenMeshCore OpenMeshTools)
    target_compile_definitions(meshing PRIVATE PLASCAN_HAS_OPENMESH=1)
else()
    target_compile_definitions(meshing PRIVATE PLASCAN_HAS_OPENMESH=0)
endif()

target_include_directories(meshing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/mvs
)

target_link_libraries(meshing
    PUBLIC
        camera
        ${OpenCV_LIBS}
    PRIVATE
        mvs
        plapoint::plapoint
        plascan_common_io
        Qt6::Core
        Qt6::Gui
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(meshing PUBLIC MESHING_OPENMP)
endif()

target_compile_definitions(meshing PRIVATE RELEASE)

if(NOT MSVC)
    # Meshing explicitly rejects NaN/Inf inputs. -ffast-math permits the
    # compiler to assume those values do not exist and invalidates the guards.
    target_compile_options(meshing PRIVATE -funroll-loops)
endif()
