set(TERRAIN_GPU_SOURCES)
if(PLASCAN_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    list(APPEND TERRAIN_GPU_SOURCES TerrainCudaBackend.cu)
    set(TERRAIN_HAS_CUDA TRUE)
    message(STATUS "terrain: CUDA backend enabled")
else()
    list(APPEND TERRAIN_GPU_SOURCES TerrainNoCudaBackend.cpp)
    set(TERRAIN_HAS_CUDA FALSE)
    message(STATUS "terrain: CUDA backend unavailable")
endif()

if(PLASCAN_ENABLE_OPENCL AND TARGET OpenCL::OpenCL)
    list(APPEND TERRAIN_GPU_SOURCES TerrainOpenClBackend.cpp)
    set(TERRAIN_HAS_OPENCL TRUE)
    message(STATUS "terrain: OpenCL backend enabled")
else()
    list(APPEND TERRAIN_GPU_SOURCES TerrainNoOpenClBackend.cpp)
    set(TERRAIN_HAS_OPENCL FALSE)
    message(STATUS "terrain: OpenCL backend unavailable")
endif()

add_library(terrain STATIC
    TerrainComputeBackend.cpp
    OrthoProjectorGpu.cpp
    ${TERRAIN_GPU_SOURCES}
    TerrainPipeline.cpp
    SmallBodyGlobalProducts.cpp
    SmallBodyGlobalProductGenerator.cpp
    SmallBodyMeshRaycaster.cpp
    GlobalTerrainReportRenderer.cpp
    OrthoGenerationOptions.cpp
    OrthoProjector.cpp
    OrthoProjectorGrid.cpp
    OrthoProjectorSupport.cpp
    PointCloudDomGenerator.cpp
    PointCloudDomSupport.cpp
    DemGridAggregator.cpp
    DemMosaic.cpp
    TerrainProductManifest.cpp
    DemGenerator.cpp
    DemGeneratorFromDepth.cpp
    DomGenerator.cpp
    DemDomIO.cpp
    ObjMtlLoader.cpp
    projection/AsteroidProjection.cpp
)

if(TERRAIN_HAS_CUDA)
    set_target_properties(terrain PROPERTIES
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_options(terrain PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endif()

if(TERRAIN_HAS_OPENCL)
    target_link_libraries(terrain PRIVATE OpenCL::OpenCL)
endif()

target_include_directories(terrain PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(GDAL_INCLUDE_DIRS)
    target_include_directories(terrain PRIVATE ${GDAL_INCLUDE_DIRS})
endif()

target_link_libraries(terrain
    PUBLIC
        camera
        plapoint::plapoint
        plascan_common_io
    PRIVATE
        Qt6::Core
        Qt6::Gui
        ${PLASCAN_GDAL_TARGET}
        ${OpenCV_LIBS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(terrain PRIVATE OpenMP::OpenMP_CXX)
endif()

if(BUILD_TESTS)
    include(GoogleTest)

    add_executable(test_terrain_dem_dom
        tests/test_dem_dom.cpp
    )

    target_link_libraries(test_terrain_dem_dom PRIVATE
        terrain
        Qt6::Core
        Qt6::Gui
        ${OpenCV_LIBS}
        GTest::gtest_main
    )

    target_include_directories(test_terrain_dem_dom PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_executable(terrain_dem_dom_tool
        tests/terrain_dem_dom_tool.cpp
    )

    target_link_libraries(terrain_dem_dom_tool PRIVATE
        terrain
        Qt6::Core
        Qt6::Gui
    )

    target_include_directories(terrain_dem_dom_tool PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_terrain_dem_dom)

    add_executable(test_ortho_generation
        tests/test_ortho_generation.cpp
    )

    target_link_libraries(test_ortho_generation PRIVATE
        terrain
        Qt6::Core
        Qt6::Gui
        ${OpenCV_LIBS}
        GTest::gtest_main
    )

    target_include_directories(test_ortho_generation PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_ortho_generation)

    add_executable(test_dem_grid_aggregator
        tests/test_dem_grid_aggregator.cpp
    )

    target_link_libraries(test_dem_grid_aggregator PRIVATE
        terrain
        Qt6::Core
        ${OpenCV_LIBS}
        GTest::gtest_main
    )

    target_include_directories(test_dem_grid_aggregator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_dem_grid_aggregator)

    add_executable(test_dem_mosaic
        tests/test_dem_mosaic.cpp
    )

    target_link_libraries(test_dem_mosaic PRIVATE
        terrain
        Qt6::Core
        ${OpenCV_LIBS}
        GTest::gtest_main
    )

    target_include_directories(test_dem_mosaic PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_dem_mosaic)

    add_executable(test_terrain_product_manifest
        tests/test_terrain_product_manifest.cpp
    )

    target_link_libraries(test_terrain_product_manifest PRIVATE
        terrain
        Qt6::Core
        GTest::gtest_main
    )

    target_include_directories(test_terrain_product_manifest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_terrain_product_manifest)

    add_executable(test_dem_quality_rasters
        tests/test_dem_quality_rasters.cpp
    )

    target_link_libraries(test_dem_quality_rasters PRIVATE
        terrain
        Qt6::Core
        ${OpenCV_LIBS}
        GTest::gtest_main
    )

    target_include_directories(test_dem_quality_rasters PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_dem_quality_rasters)

    add_executable(test_small_body_global_products
        tests/test_small_body_global_products.cpp
    )

    target_link_libraries(test_small_body_global_products PRIVATE
        terrain
        Qt6::Core
        Qt6::Gui
        ${OpenCV_LIBS}
        GTest::gtest_main
    )

    target_include_directories(test_small_body_global_products PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_small_body_global_products)

endif()
