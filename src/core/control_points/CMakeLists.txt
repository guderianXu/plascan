add_library(control_network STATIC
    registration/ControlNetworkSolver.cpp
    registration/ControlNetworkSolver.h
    registration/PriorTrack.h
    model/MarkerRoles.h
)

target_include_directories(control_network PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/core/sfm
)

target_link_libraries(control_network PUBLIC
    ${OpenCV_LIBS}
)

target_compile_features(control_network PUBLIC cxx_std_20)

add_library(control_points STATIC
    model/MarkerSet.cpp
    model/MarkerSetValidator.cpp
    io/MarkerSetJson.cpp
    io/MarkerSetStore.cpp
    io/MarkerCsv.cpp
    io/MarkerCsvExport.cpp
    commands/MarkerChangeSet.cpp
    geometry/MarkerGeometry.cpp
    geometry/MarkerProjectionPredictor.cpp
    reference/CoordinateReference.cpp
    quality/MarkerQualityReport.cpp
    detection/AprilTagDetector.cpp
    detection/DetectionMerger.cpp
    detection/DetectionIntegrator.cpp
    detection/DetectionReviewStore.cpp
    detection/NonCodedTargetDetector.cpp
    detection/MarkerDetectorFactory.cpp
    print/MarkerSheetRenderer.cpp
    print/MarkerPdfWriter.cpp
)

target_include_directories(control_points PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(control_points PUBLIC
    control_network
    Qt6::Core
    Qt6::Gui
    ${OpenCV_LIBS}
    ${PLASCAN_GDAL_TARGET}
    apriltag::apriltag
)

target_include_directories(control_points PUBLIC ${OpenCV_INCLUDE_DIRS})
if(GDAL_INCLUDE_DIRS)
    target_include_directories(control_points PRIVATE ${GDAL_INCLUDE_DIRS})
endif()

target_compile_features(control_points PUBLIC cxx_std_20)

if(BUILD_TESTS)
    include(GoogleTest)

    add_executable(test_marker_set
        tests/test_marker_set.cpp
    )

    target_link_libraries(test_marker_set PRIVATE
        control_points
        GTest::gtest_main
    )

    gtest_discover_tests(test_marker_set DISCOVERY_MODE PRE_TEST)

    add_executable(test_marker_set_store
        tests/test_marker_set_store.cpp
    )

    target_link_libraries(test_marker_set_store PRIVATE
        control_points
        GTest::gtest_main
    )

    gtest_discover_tests(test_marker_set_store DISCOVERY_MODE PRE_TEST)

    add_executable(test_marker_change_set
        tests/test_marker_change_set.cpp
    )

    target_link_libraries(test_marker_change_set PRIVATE
        control_points
        GTest::gtest_main
    )

    gtest_discover_tests(test_marker_change_set DISCOVERY_MODE PRE_TEST)

    add_executable(test_marker_geometry
        tests/test_marker_geometry.cpp
    )
    target_link_libraries(test_marker_geometry PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_marker_geometry DISCOVERY_MODE PRE_TEST)

    add_executable(test_coordinate_reference
        tests/test_coordinate_reference.cpp
    )
    target_link_libraries(test_coordinate_reference PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_coordinate_reference DISCOVERY_MODE PRE_TEST)

    add_executable(test_marker_csv
        tests/test_marker_csv.cpp
    )
    target_link_libraries(test_marker_csv PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_marker_csv DISCOVERY_MODE PRE_TEST)

    add_executable(test_control_network_solver
        tests/test_control_network_solver.cpp
    )
    target_link_libraries(test_control_network_solver PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_control_network_solver DISCOVERY_MODE PRE_TEST)

    add_executable(test_marker_quality_report
        tests/test_marker_quality_report.cpp
    )
    target_link_libraries(test_marker_quality_report PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_marker_quality_report DISCOVERY_MODE PRE_TEST)

    add_executable(test_apriltag_detector
        tests/test_apriltag_detector.cpp
    )
    target_link_libraries(test_apriltag_detector PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_apriltag_detector DISCOVERY_MODE PRE_TEST)

    add_executable(test_non_coded_target_detector
        tests/test_non_coded_target_detector.cpp
    )
    target_link_libraries(test_non_coded_target_detector PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_non_coded_target_detector DISCOVERY_MODE PRE_TEST)

    add_executable(test_detection_integrator
        tests/test_detection_integrator.cpp
    )
    target_link_libraries(test_detection_integrator PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_detection_integrator DISCOVERY_MODE PRE_TEST)

    add_executable(test_detection_review_store
        tests/test_detection_review_store.cpp
    )
    target_link_libraries(test_detection_review_store PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_detection_review_store DISCOVERY_MODE PRE_TEST)

    add_executable(test_marker_detector_factory
        tests/test_marker_detector_factory.cpp
    )
    target_link_libraries(test_marker_detector_factory PRIVATE
        control_points
        GTest::gtest_main
    )
    gtest_discover_tests(test_marker_detector_factory DISCOVERY_MODE PRE_TEST)

    add_executable(test_marker_pdf_writer
        tests/test_marker_pdf_writer.cpp
    )
    target_link_libraries(test_marker_pdf_writer PRIVATE
        control_points
        GTest::gtest
    )
    if(APPLE AND TARGET Qt6::QOffscreenIntegrationPlugin)
        qt_import_plugins(test_marker_pdf_writer INCLUDE Qt6::QOffscreenIntegrationPlugin)
    endif()
    set(_marker_pdf_qt_platform_plugin_path
        "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/Qt6/plugins/platforms")
    gtest_discover_tests(test_marker_pdf_writer
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            ENVIRONMENT_MODIFICATION
                "QT_QPA_PLATFORM_PLUGIN_PATH=set:${_marker_pdf_qt_platform_plugin_path}"
    )
endif()
