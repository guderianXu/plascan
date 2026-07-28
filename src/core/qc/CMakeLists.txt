add_library(qc STATIC
    DemDifference.cpp
    PointCloudAlignment.cpp
    ReconstructionQualityReport.cpp
    SurveyControlReport.cpp
    ModelMeshRenderer.cpp
    ModelImageMetrics.cpp
    ModelGeometryComparator.cpp
    ModelImageQualityEvaluator.cpp
    ProcessingBaselineManager.cpp
)

target_include_directories(qc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(qc PUBLIC
    plapoint::plapoint
    camera
    meshing
    Qt6::Core
    ${OpenCV_LIBS}
)

target_compile_features(qc PUBLIC cxx_std_17)

if(BUILD_TESTS)
    include(GoogleTest)

    add_executable(test_reconstruction_quality_report
        tests/test_reconstruction_quality_report.cpp
    )

    target_link_libraries(test_reconstruction_quality_report PRIVATE
        qc
        Qt6::Core
        GTest::gtest_main
    )

    target_include_directories(test_reconstruction_quality_report PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_reconstruction_quality_report)

    add_executable(test_survey_control_import
        tests/test_survey_control_import.cpp
    )

    target_link_libraries(test_survey_control_import PRIVATE
        control_points
        Qt6::Core
        GTest::gtest_main
    )

    target_include_directories(test_survey_control_import PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_survey_control_import DISCOVERY_MODE PRE_TEST)

    add_executable(test_point_cloud_alignment
        tests/test_point_cloud_alignment.cpp
    )

    target_link_libraries(test_point_cloud_alignment PRIVATE
        qc
        Qt6::Core
        GTest::gtest_main
    )

    target_include_directories(test_point_cloud_alignment PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_point_cloud_alignment DISCOVERY_MODE PRE_TEST)

    add_executable(test_dem_difference
        tests/test_dem_difference.cpp
    )

    target_link_libraries(test_dem_difference PRIVATE
        qc
        Qt6::Core
        GTest::gtest_main
    )

    target_include_directories(test_dem_difference PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_dem_difference)

    add_executable(test_model_image_quality
        tests/test_model_image_quality.cpp
    )

    target_link_libraries(test_model_image_quality PRIVATE
        qc
        meshing
        Qt6::Core
        GTest::gtest_main
    )

    target_include_directories(test_model_image_quality PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/core/mesh
    )

    gtest_discover_tests(test_model_image_quality DISCOVERY_MODE PRE_TEST)

    add_executable(test_processing_baseline_manager
        tests/test_processing_baseline_manager.cpp
    )

    target_link_libraries(test_processing_baseline_manager PRIVATE
        qc
        meshing
        Qt6::Core
        GTest::gtest_main
    )

    target_include_directories(test_processing_baseline_manager PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/core/mesh
    )

    gtest_discover_tests(test_processing_baseline_manager DISCOVERY_MODE PRE_TEST)
endif()
