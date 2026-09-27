include(GoogleTest)

add_executable(test_placamera
    public_header_calibration.cpp
    public_header_capabilities.cpp
    public_header_camera_baseline.cpp
    public_header_camera_topology.cpp
    public_header_colmap.cpp
    public_header_dataset_formats.cpp
    public_header_errors.cpp
    public_header_frame_camera.cpp
    public_header_formats.cpp
    public_header_frame_numeric_state.cpp
    public_header_instance_set.cpp
    public_header_isd.cpp
    public_header_linescan_camera.cpp
    public_header_linescan_numeric_state.cpp
    public_header_model.cpp
    public_header_optimization.cpp
    public_header_project_import.cpp
    public_header_placamera.cpp
    public_header_registry.cpp
    public_header_result.cpp
    public_header_rpc_adjustment.cpp
    public_header_rpc_camera.cpp
    public_header_rpc_metadata.cpp
    public_header_rpc_raster.cpp
    public_header_state_codec.cpp
    public_header_types.cpp
    public_header_tsai.cpp
    public_header_version.cpp
    test_frame_camera.cpp
    test_camera_baseline.cpp
    test_camera_topology.cpp
    test_colmap.cpp
    test_dataset_formats.cpp
    test_frame_numeric_state.cpp
    test_instance_set.cpp
    test_linescan_camera.cpp
    test_linescan_numeric_state.cpp
    test_optimization.cpp
    test_project_import.cpp
    test_registry.cpp
    test_rpc_adjustment.cpp
    test_rpc_camera.cpp
    test_rpc_metadata.cpp
    test_result.cpp
    test_tsai_file.cpp
    test_types.cpp
    test_tsai.cpp
)
target_link_libraries(test_placamera PRIVATE
    placamera::placamera
    GTest::gtest_main
)
if(PROJECT_IS_TOP_LEVEL)
    set(_placamera_test_tmp_root "${CMAKE_BINARY_DIR}/tmp/placamera-project-import")
else()
    set(_placamera_test_tmp_root "${CMAKE_SOURCE_DIR}/build/tmp/placamera-project-import")
endif()
target_compile_definitions(test_placamera PRIVATE PLACAMERA_TEST_TMP_ROOT="${_placamera_test_tmp_root}")
target_compile_definitions(test_placamera PRIVATE
    PLACAMERA_TSAI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tsai"
)
set_target_properties(test_placamera PROPERTIES CXX_EXTENSIONS OFF)
gtest_discover_tests(test_placamera TEST_PREFIX "PlaCamera.")

add_executable(test_linescan_hot_path_allocations test_linescan_hot_path_allocations.cpp)
target_link_libraries(test_linescan_hot_path_allocations PRIVATE placamera::placamera)
set_target_properties(test_linescan_hot_path_allocations PROPERTIES CXX_EXTENSIONS OFF)
add_test(NAME PlaCamera.LineScanHotPathZeroAllocations COMMAND test_linescan_hot_path_allocations)

if(TARGET placamera::state)
    add_executable(test_placamera_state test_isd.cpp test_isd_file.cpp test_state_codec.cpp)
    target_link_libraries(test_placamera_state PRIVATE
        placamera::state
        GTest::gtest_main
    )
    target_compile_definitions(test_placamera_state PRIVATE PLACAMERA_TEST_TMP_ROOT="${_placamera_test_tmp_root}")
    if(NOT PROJECT_IS_TOP_LEVEL AND EXISTS "${CMAKE_SOURCE_DIR}/testData")
        target_compile_definitions(test_placamera_state PRIVATE
            PLANETARY_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
        )
    endif()
    set_target_properties(test_placamera_state PROPERTIES CXX_EXTENSIONS OFF)
    gtest_discover_tests(test_placamera_state TEST_PREFIX "PlaCameraState.")
endif()

if(TARGET placamera::gdal)
    add_executable(test_placamera_gdal test_rpc_raster.cpp)
    target_link_libraries(test_placamera_gdal PRIVATE
        placamera::gdal
        GDAL::GDAL
        GTest::gtest_main
    )
    set_target_properties(test_placamera_gdal PROPERTIES CXX_EXTENSIONS OFF)
    gtest_discover_tests(test_placamera_gdal TEST_PREFIX "PlaCameraGdal.")
endif()

if(TARGET placamera::reference)
    find_package(nlohmann_json CONFIG REQUIRED)
    add_executable(test_placamera_reference
        public_header_metashape_camera_reference_adapter.cpp
        test_metalign_differential.cpp
        test_reference_types.cpp
        test_reference_resolver.cpp
        test_reference_comparator.cpp
        test_metashape_camera_reference_adapter.cpp
    )
    target_link_libraries(test_placamera_reference PRIVATE
        placamera::reference
        GTest::gtest_main
        nlohmann_json::nlohmann_json
    )
    target_compile_definitions(test_placamera_reference PRIVATE
        PLACAMERA_METALIGN_CAMERA_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/metalign/camera_contract_v1.json"
    )
    set_target_properties(test_placamera_reference PROPERTIES CXX_EXTENSIONS OFF)
    gtest_discover_tests(test_placamera_reference TEST_PREFIX "PlaCameraReference.")
endif()

add_test(
    NAME PlaCamera.PublicDependencyBoundary
    COMMAND ${CMAKE_COMMAND}
        -DPLACAMERA_PUBLIC_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../include/placamera
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_public_dependencies.cmake
)
