add_library(bundle_adjust STATIC
    BundleAdjust.cpp
    BundleAdjustOptions.h
    BundleAdjustProblem.h
    BundleAdjustResult.h
    BundleAdjustSolver.h
    BundleAdjustTypes.h
    BundleAdjustAdaptiveCameraModel.cpp
    BundleAdjustProjection.cpp
    BundleAdjustQuality.cpp
    BundleAdjustValidation.cpp
    BundleAdjustPlaMatrix.cpp
    BundleAdjustPlaMatrixProblem.cpp
    BundleAdjustPlaMatrixModel.cpp
    BundleAdjustPlaMatrixConstraints.cpp
    BundleAdjustPlaMatrixAssembly.cpp
    BundleAdjustPlaMatrixAssemblySurvey.cpp
    BundleAdjustPlaMatrixReferenceSchur.cpp
    BundleAdjustPlaMatrixProjection.cpp
    BundleAdjustPlaMatrixRuntime.cpp
)

target_include_directories(bundle_adjust PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

target_include_directories(bundle_adjust PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(bundle_adjust
    PUBLIC
        camera
    PRIVATE
        plascan_common_log
        plamatrix::plamatrix
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

if(OpenMP_CXX_FOUND)
    message(STATUS "bundle_adjust: OpenMP 并行已启用 (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "bundle_adjust: OpenMP 未找到，将使用单线程模式")
endif()

add_executable(ba_backend_benchmark
    tools/BaBenchmarkRealDataset.cpp
    tools/ba_backend_benchmark.cpp
)

target_link_libraries(ba_backend_benchmark PRIVATE
    bundle_adjust
    camera
    plascan_common_io
    Qt6::Core
)

target_include_directories(ba_backend_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
    ${CMAKE_SOURCE_DIR}/src/common
)

set_target_properties(ba_backend_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
