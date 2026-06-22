add_library(qc STATIC
    DemDifference.cpp
    PointCloudAlignment.cpp
    SurveyControlImport.cpp
    ReconstructionQualityReport.cpp
    SurveyControlReport.cpp
)

target_include_directories(qc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(qc PUBLIC
    plapoint::plapoint
    Qt6::Core
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
        qc
        Qt6::Core
        GTest::gtest_main
    )

    target_include_directories(test_survey_control_import PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    gtest_discover_tests(test_survey_control_import)

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

    gtest_discover_tests(test_point_cloud_alignment)

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
endif()
