cmake_minimum_required(VERSION 3.16)

include(FetchContent)

set(PLASCAN_POISSON_RECON_COMMIT
    262b0f539d404057d1f36e1adc07fc9388678899)
FetchContent_Declare(plascan_poisson_recon
    GIT_REPOSITORY https://github.com/mkazhdan/PoissonRecon.git
    GIT_TAG ${PLASCAN_POISSON_RECON_COMMIT}
    GIT_SHALLOW FALSE
    GIT_SUBMODULES ""
    UPDATE_DISCONNECTED TRUE
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(plascan_poisson_recon)
if(NOT EXISTS "${plascan_poisson_recon_SOURCE_DIR}/Src/Reconstructors.h")
    message(FATAL_ERROR
        "Pinned PoissonRecon source is missing Reconstructors.h: "
        "${plascan_poisson_recon_SOURCE_DIR}")
endif()

add_library(plascan_poisson_recon_headers INTERFACE)
target_include_directories(plascan_poisson_recon_headers SYSTEM INTERFACE
    "${plascan_poisson_recon_SOURCE_DIR}/Src"
)

add_library(meshing STATIC
    SurfaceReconstructor.cpp
    PointCloudPreprocess.cpp
    SurfaceReconstructorPostprocess.cpp
    BoundaryAwareVoxelSimplifier.cpp
    SurfaceReconstructorHeightGrid.cpp
    ScreenedPoissonSurfaceBuilder.cpp
    SparseSfmPointQualityReader.cpp
    SparseOrbitalScaffoldBuilder.cpp
    MeshVoxelTopologyRepair.cpp
    OrbitalSparseScaffoldSurfaceBuilder.cpp
    StudioForegroundMask.cpp
    DepthMapMeshBuilder.cpp
    DepthAuxiliaryBridgeSelector.cpp
    DepthFusionFramePolicy.cpp
    DepthMeshCompleteness.cpp
    TriangleDistanceIndex.cpp
    DepthRayMetric.cpp
    DepthGeometrySourceEncoding.cpp
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
    NativeMeshSimplifier.cpp
    MeshTopologyQuality.cpp
    DepthTsdfFinalQualityGate.cpp
    DepthTsdfRecoveryTransaction.cpp
    MeshBoundaryAttribution.cpp
    MeshAcquisitionGapReport.cpp
    MeshIsotropicRemesher.cpp
    TextureMapper.cpp
    CameraTextureMapper.cpp
    CameraTextureMapperV4.cpp
    TextureAtlasPacker.cpp
    TextureAtlasMaxRects.cpp
    TextureSourcePreprocessor.cpp
    TextureOverlapExposure.cpp
    TextureOverlapExposureSolver.cpp
    TextureMeshVisibility.cpp
    TextureVisibilityEvaluator.cpp
    TextureChartBuilder.cpp
    TextureAtlasSampling.cpp
    TextureSeamLeveling.cpp
    TextureAtlasBaker.cpp
    ModelOutputPolicy.cpp
    ModelWorkflowService.cpp
)

# PoissonRecon is pinned third-party code. Its templates still use
# std::is_pod, which is deprecated in C++20 and emits diagnostics when they are
# instantiated by this wrapper. Keep the exception scoped to the only source
# file that instantiates those templates so project-owned meshing code retains
# the normal deprecation warnings.
if(MSVC)
    set_property(SOURCE ScreenedPoissonSurfaceBuilder.cpp APPEND PROPERTY
        COMPILE_OPTIONS /wd4996)
else()
    set_property(SOURCE ScreenedPoissonSurfaceBuilder.cpp APPEND PROPERTY
        COMPILE_OPTIONS -Wno-deprecated-declarations)
endif()

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
        plascan_poisson_recon_headers
        plapoint::plapoint
        plascan_common_concurrency
        plascan_common_io
        Qt6::Core
        Qt6::Gui
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

install(FILES
    ${CMAKE_SOURCE_DIR}/docs/third_party/PoissonRecon_NOTICE.md
    DESTINATION share/plascan/licenses
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
