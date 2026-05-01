add_library(terrain STATIC
    TerrainPipeline.cpp
    DemGenerator.cpp
    DemGeneratorFromDepth.cpp
    DomGenerator.cpp
    DemDomIO.cpp
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
        pointcloud
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

    add_executable(terrain_stereo_test
        tests/terrain_stereo_test.cpp
    )

    target_link_libraries(terrain_stereo_test PRIVATE
        terrain
        mvs
        pointcloud
        camera
        Qt6::Core
        Qt6::Concurrent
        ${OpenCV_LIBS}
        ${PLASCAN_GDAL_TARGET}
    )

    target_include_directories(terrain_stereo_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/core/mvs
        ${CMAKE_SOURCE_DIR}/src/core/camera
        ${CMAKE_SOURCE_DIR}/src/core/pointcloud
    )

    add_executable(stereo_pipeline_test
        tests/stereo_pipeline_test.cpp
    )

    target_link_libraries(stereo_pipeline_test PRIVATE
        terrain
        mvs
        pointcloud
        camera
        Qt6::Core
        Qt6::Concurrent
        ${OpenCV_LIBS}
        ${PLASCAN_GDAL_TARGET}
    )

    target_include_directories(stereo_pipeline_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/core/mvs
        ${CMAKE_SOURCE_DIR}/src/core/camera
        ${CMAKE_SOURCE_DIR}/src/core/pointcloud
    )

    add_executable(stereo_pipeline_benchmark
        tests/stereo_pipeline_benchmark.cpp
    )

    target_link_libraries(stereo_pipeline_benchmark PRIVATE
        terrain
        mvs
        pointcloud
        camera
        Qt6::Core
        Qt6::Concurrent
        ${OpenCV_LIBS}
        ${PLASCAN_GDAL_TARGET}
    )

    target_include_directories(stereo_pipeline_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/core/mvs
        ${CMAKE_SOURCE_DIR}/src/core/camera
        ${CMAKE_SOURCE_DIR}/src/core/pointcloud
    )
endif()
