find_package(Ceres CONFIG QUIET)

add_library(bundle_adjust STATIC
    BundleAdjust.cpp
    BundleAdjustCeres.cpp
    BundleAdjustNativeCuda.cpp
)

target_include_directories(bundle_adjust PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

target_link_libraries(bundle_adjust
    PUBLIC
        camera
        plascan_common_log
    PRIVATE
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

if(Ceres_FOUND)
    target_link_libraries(bundle_adjust PRIVATE Ceres::ceres)
    target_compile_definitions(bundle_adjust PRIVATE PLASCAN_BA_HAS_CERES=1)
    message(STATUS "bundle_adjust: Ceres 后端已启用 (${CERES_VERSION})")
else()
    message(STATUS "bundle_adjust: Ceres 未找到，仅启用 legacy CPU 后端")
endif()

if(OpenMP_CXX_FOUND)
    target_compile_definitions(bundle_adjust PRIVATE HAS_OPENMP)
    message(STATUS "bundle_adjust: OpenMP 并行已启用 (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "bundle_adjust: OpenMP 未找到，将使用单线程模式")
endif()

add_executable(ba_backend_benchmark
    tools/ba_backend_benchmark.cpp
)

target_link_libraries(ba_backend_benchmark PRIVATE
    bundle_adjust
    camera
)

target_include_directories(ba_backend_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)

set_target_properties(ba_backend_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
