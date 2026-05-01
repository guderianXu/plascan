cmake_minimum_required(VERSION 3.16)

add_library(meshing STATIC
    SurfaceReconstructor.cpp
    SurfaceReconstructorIO.cpp
    SurfaceReconstructorPostprocess.cpp
    SurfaceReconstructorHeightGrid.cpp
    MeshIO.cpp
    TextureMapper.cpp
    ModelWorkflowService.cpp
    MarchingCubesTable.cpp
    poisson/PoissonBranchPipeline.cpp
    poisson/PoissonLite.cpp
    poisson/PoissonPreprocess.cpp
    poisson/PoissonVoxel.cpp
)

target_include_directories(meshing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/core/mvs
    ${CMAKE_SOURCE_DIR}/src/core/pointcloud
)

target_link_libraries(meshing
    PRIVATE
        Qt6::Core
        Qt6::Gui
        ${OpenCV_LIBS}
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(meshing PUBLIC MESHING_OPENMP)
endif()

target_compile_definitions(meshing PRIVATE RELEASE)

if(NOT MSVC)
    target_compile_options(meshing PRIVATE -funroll-loops -ffast-math)
endif()
