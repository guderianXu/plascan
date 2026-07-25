cmake_minimum_required(VERSION 3.16)

add_library(meshing STATIC
    SurfaceReconstructor.cpp
    PointCloudPreprocess.cpp
    SurfaceReconstructorPostprocess.cpp
    BoundaryAwareVoxelSimplifier.cpp
    SurfaceReconstructorHeightGrid.cpp
    DepthMapMeshBuilder.cpp
    DepthTsdfSurfaceBuilder.cpp
    VisualHullReconstructor.cpp
    MeshIO.cpp
    MeshColorizer.cpp
    MeshFaceColorOptimizer.cpp
    MeshQuadricSimplifier.cpp
    MeshTopologyQuality.cpp
    MeshIsotropicRemesher.cpp
    TextureMapper.cpp
    CameraTextureMapper.cpp
    ModelWorkflowService.cpp
)

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
    target_compile_options(meshing PRIVATE -funroll-loops -ffast-math)
endif()
