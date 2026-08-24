add_library(stereo_dem STATIC
    RpcGeospatialSupport.cpp
    RpcStereoDemGenerator.cpp
    RpcDomGenerator.cpp
)

target_include_directories(stereo_dem PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(GDAL_INCLUDE_DIRS)
    target_include_directories(stereo_dem PRIVATE ${GDAL_INCLUDE_DIRS})
endif()

target_compile_features(stereo_dem PUBLIC cxx_std_20)
target_link_libraries(stereo_dem
    PUBLIC
        camera
        terrain
    PRIVATE
        Qt6::Core
        plascan_common_io
        ${PLASCAN_GDAL_TARGET}
        ${OpenCV_LIBS}
)

if(BUILD_TESTS)
    include(GoogleTest)
    add_executable(test_rpc_stereo_products
        tests/test_rpc_stereo_products.cpp
    )
    target_link_libraries(test_rpc_stereo_products PRIVATE
        stereo_dem
        Qt6::Core
        ${OpenCV_LIBS}
        GTest::gtest_main
    )
    target_compile_definitions(test_rpc_stereo_products PRIVATE
        TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
    )
    gtest_discover_tests(test_rpc_stereo_products DISCOVERY_MODE PRE_TEST)
endif()
