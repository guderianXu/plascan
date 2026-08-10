add_library(terrain STATIC
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

    # terrain_stereo_test / stereo_pipeline_test / stereo_pipeline_benchmark
    # require the old pointcloud module which has been removed.
    # Re-enable after migrating these tests to plapoint.
endif()
