find_package(GTest REQUIRED)
include(GoogleTest)

function(add_aerial_triangulation_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
        aerial_triangulation
        GTest::gtest_main
    )
    target_compile_definitions(${target} PRIVATE
        PLASCAN_AERIAL_IO_TEST_TMP_DIR="${CMAKE_SOURCE_DIR}/build/tmp/aerial-image-io-tests")
    gtest_discover_tests(${target} DISCOVERY_MODE PRE_TEST)
endfunction()

add_aerial_triangulation_test(test_sfm_pair_planner test_sfm_pair_planner.cpp)
add_aerial_triangulation_test(test_sfm_search_policy test_sfm_search_policy.cpp)
add_aerial_triangulation_test(test_reconstruction_prerequisites
                              test_reconstruction_prerequisites.cpp)
add_aerial_triangulation_test(test_match_result_catalog test_match_result_catalog.cpp)
add_aerial_triangulation_test(test_aerial_triangulation_workflow
                              test_aerial_triangulation_workflow.cpp)
add_aerial_triangulation_test(test_tie_point_preparation test_tie_point_preparation.cpp)
add_aerial_triangulation_test(test_sfm_attempt_runner test_sfm_attempt_runner.cpp)
add_aerial_triangulation_test(test_camera_intrinsic_prior_sanitizer
                              test_camera_intrinsic_prior_sanitizer.cpp)
add_executable(test_camera_intrinsic_prior
    test_camera_intrinsic_prior.cpp
    ../preparation/CameraIntrinsicPrior.cpp
)
target_include_directories(test_camera_intrinsic_prior PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/core/aerial_triangulation
)
target_link_libraries(test_camera_intrinsic_prior PRIVATE Qt6::Core GTest::gtest_main)
gtest_discover_tests(test_camera_intrinsic_prior DISCOVERY_MODE PRE_TEST)
add_aerial_triangulation_test(test_adaptive_focal_search test_adaptive_focal_search.cpp)
add_aerial_triangulation_test(test_aerial_triangulation_pipeline
                              test_aerial_triangulation_pipeline.cpp)
target_link_libraries(test_aerial_triangulation_pipeline PRIVATE plascan::placamera_runtime)
add_aerial_triangulation_test(test_aerial_triangulation_result_writer
                              test_aerial_triangulation_result_writer.cpp)
add_aerial_triangulation_test(test_rpc_aerial_triangulation_runner
                              test_rpc_aerial_triangulation_runner.cpp)

target_compile_definitions(test_rpc_aerial_triangulation_runner PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
)

target_link_libraries(test_sfm_attempt_runner PRIVATE control_points)

add_aerial_triangulation_test(test_aerial_image_io test_aerial_image_io.cpp)
foreach(target test_sfm_attempt_runner test_aerial_triangulation_pipeline
               test_aerial_triangulation_result_writer test_aerial_image_io test_rpc_aerial_triangulation_runner)
    target_link_libraries(${target} PRIVATE plascan_common_io)
endforeach()

add_executable(test_standard_aerial_engine test_standard_aerial_engine.cpp)
target_link_libraries(test_standard_aerial_engine PRIVATE aerial_triangulation_engine GTest::gtest_main)
target_compile_definitions(test_standard_aerial_engine PRIVATE
    PLASCAN_STANDARD_AERIAL_TMP_DIR="${CMAKE_SOURCE_DIR}/build/tmp/aerial-standard-engine-tests")
gtest_discover_tests(test_standard_aerial_engine DISCOVERY_MODE PRE_TEST)
