if(DEFINED CMAKE_CUDA_COMPILER AND CMAKE_CUDA_COMPILER)
    set(BA_HAS_NATIVE_CUDA TRUE)
    set(BA_NATIVE_CUDA_SOURCES BundleAdjustNativeCuda.cu)
    message(STATUS "bundle_adjust: native CUDA 后端已启用 (${CMAKE_CUDA_COMPILER})")
else()
    set(BA_HAS_NATIVE_CUDA FALSE)
    set(BA_NATIVE_CUDA_SOURCES)
    message(STATUS "bundle_adjust: native CUDA 后端未启用")
endif()

add_library(bundle_adjust STATIC
    BundleAdjust.cpp
    BundleAdjustAdaptiveCameraModel.cpp
    BundleAdjustCeres.cpp
    BundleAdjustCeresPlanning.cpp
    BundleAdjustProjection.cpp
    BundleAdjustQuality.cpp
    BundleAdjustValidation.cpp
    BundleAdjustNativeCuda.cpp
    BundleAdjustNativeCudaMath.cpp
    BundleAdjustNativeCudaWorkset.cpp
    BundleAdjustPlaMatrix.cpp
    BundleAdjustPlaMatrixProblem.cpp
    BundleAdjustPlaMatrixModel.cpp
    BundleAdjustPlaMatrixConstraints.cpp
    BundleAdjustPlaMatrixAssembly.cpp
    BundleAdjustPlaMatrixAssemblySurvey.cpp
    BundleAdjustPlaMatrixProjection.cpp
    BundleAdjustPlaMatrixRuntime.cpp
    ${BA_NATIVE_CUDA_SOURCES}
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

if(PLASCAN_ENABLE_CERES_REFERENCE)
    target_link_libraries(bundle_adjust PRIVATE Ceres::ceres)
    target_compile_definitions(bundle_adjust PRIVATE PLASCAN_BA_HAS_CERES=1)
    message(STATUS "bundle_adjust: optional Ceres reference backend enabled (${CERES_VERSION})")
else()
    message(STATUS "bundle_adjust: production build uses PlaMatrix without Ceres")
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "bundle_adjust: OpenMP 并行已启用 (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "bundle_adjust: OpenMP 未找到，将使用单线程模式")
endif()

if(BA_HAS_NATIVE_CUDA)
    set_target_properties(bundle_adjust PROPERTIES
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_definitions(bundle_adjust PRIVATE PLASCAN_BA_HAS_NATIVE_CUDA=1)
    target_compile_options(bundle_adjust PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    )
endif()

add_executable(ba_backend_benchmark
    tools/BaBenchmarkRealDataset.cpp
    tools/ba_backend_benchmark.cpp
)

target_link_libraries(ba_backend_benchmark PRIVATE
    bundle_adjust
    camera
    Qt6::Core
)

target_include_directories(ba_backend_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

set_target_properties(ba_backend_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
