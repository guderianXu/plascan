include(GoogleTest)

function(add_sfm_test target module_target)
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE ${module_target} GTest::gtest_main ${ARGN})
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/sfm
    )
    gtest_discover_tests(${target} DISCOVERY_MODE PRE_TEST)
endfunction()

add_sfm_test(test_sparse_point_cloud_processor sfm_postprocess)
add_sfm_test(test_sparse_point_cloud_workspace sfm_postprocess)

add_sfm_test(test_sfm_params sfm_core
    sfm_postprocess
    camera
    intersection
    bundle_adjust
    ${OpenCV_LIBS}
)
target_include_directories(test_sfm_params PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
)

add_sfm_test(test_sfm_quality_report sfm_project Qt6::Core)

add_sfm_test(test_sfm_pipeline sfm_core
    camera
    intersection
    bundle_adjust
)
target_include_directories(test_sfm_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
)
target_compile_definitions(test_sfm_pipeline PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
)

add_sfm_test(test_sfm_prior_tracks sfm_core
    camera
    intersection
    bundle_adjust
)
target_include_directories(test_sfm_prior_tracks PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/control_points
)

add_sfm_test(test_ba_input_builder sfm_project
    camera
    intersection
    bundle_adjust
    Qt6::Core
)
target_include_directories(test_ba_input_builder PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/common
)

add_sfm_test(test_ba_track_builder sfm_project)

add_sfm_test(test_initial_sparse_triangulator sfm_core
    camera
    intersection
    bundle_adjust
)
target_include_directories(test_initial_sparse_triangulator PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/camera
    ${CMAKE_SOURCE_DIR}/src/core/intersection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
)

add_sfm_test(test_reference_terrain_prior sfm_core
    bundle_adjust
    camera
)
target_include_directories(test_reference_terrain_prior PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)

add_sfm_test(test_multiview_track_builder sfm_core)
add_sfm_test(test_observation_network_builder sfm_core)
add_sfm_test(test_projection_geometry sfm_core)
add_sfm_test(test_similarity_gauge_normalizer sfm_core)
add_sfm_test(test_triangulation_quality sfm_core)
add_sfm_test(test_opencv_camera_adapter sfm_core ${OpenCV_LIBS})
