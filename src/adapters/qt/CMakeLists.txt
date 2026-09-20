# Qt presentation and application lifecycle adapters. Core never links these targets.
if(TARGET control_points)
    add_library(marker_detection_qt STATIC markers/detection/MarkerImageAdapter.cpp)
    target_include_directories(marker_detection_qt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/markers)
    target_link_libraries(marker_detection_qt PUBLIC control_points Qt6::Gui)
    target_compile_features(marker_detection_qt PUBLIC cxx_std_20)
endif()

if(PLASCAN_BUILD_QT_PRESENTATION AND TARGET control_points)
    add_library(marker_print_qt STATIC
        markers/print/MarkerSheetRenderer.cpp
        markers/print/MarkerPdfWriter.cpp
    )
    target_include_directories(marker_print_qt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/markers)
    target_link_libraries(marker_print_qt PUBLIC control_points Qt6::Core Qt6::Gui
        PRIVATE apriltag::apriltag)
    target_compile_features(marker_print_qt PUBLIC cxx_std_20)
endif()

if(PLASCAN_BUILD_QT_PRESENTATION AND TARGET terrain)
    add_library(terrain_report_qt STATIC terrain/GlobalTerrainReportRenderer.cpp)
    target_include_directories(terrain_report_qt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/terrain)
    target_link_libraries(terrain_report_qt PUBLIC terrain Qt6::Core Qt6::Gui
        PRIVATE plascan_common_io ${OpenCV_LIBS})
    target_compile_features(terrain_report_qt PUBLIC cxx_std_20)
endif()

if(TARGET project_workflows)
    add_library(project_recovery_qt STATIC project/ProjectResourceRecoveryBinding.cpp)
    target_include_directories(project_recovery_qt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/project)
    target_link_libraries(project_recovery_qt PUBLIC project_workflows
        PRIVATE plascan_common_project plascan_common_log Qt6::Core)
    target_compile_features(project_recovery_qt PUBLIC cxx_std_20)
endif()
