find_package(GTest REQUIRED)
include(GoogleTest)

function(add_aerial_triangulation_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
        aerial_triangulation
        GTest::gtest_main
    )
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
add_aerial_triangulation_test(test_adaptive_focal_search test_adaptive_focal_search.cpp)
add_aerial_triangulation_test(test_aerial_triangulation_pipeline
                              test_aerial_triangulation_pipeline.cpp)
add_aerial_triangulation_test(test_aerial_triangulation_result_writer
                              test_aerial_triangulation_result_writer.cpp)

target_link_libraries(test_sfm_attempt_runner PRIVATE control_points)
