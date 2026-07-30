# Dialog sources are grouped by user-facing domain. Keep non-dialog helpers in
# their owning modules (for example, camera rendering helpers belong in views).

set(GUI_APPLICATION_DIALOG_SOURCES
  dialogs/application/AboutDialog.cpp
  dialogs/application/AboutDialog.h
  dialogs/application/WorkflowReportDialog.cpp
  dialogs/application/WorkflowReportDialog.h
  dialogs/application/WorkflowReportDialog.ui
)

set(GUI_CAMERA_DIALOG_SOURCES
  dialogs/camera/CameraConvertDialog.cpp
  dialogs/camera/CameraConvertDialog.h
  dialogs/camera/CameraModel3DDialog.cpp
  dialogs/camera/CameraModel3DDialog.h
  dialogs/camera/CameraModel3DDialog.ui
  dialogs/camera/ForwardIntersectionCheckDialog.cpp
  dialogs/camera/ForwardIntersectionCheckDialog.h
  dialogs/camera/ForwardIntersectionCheckDialog.ui
  dialogs/camera/ForwardIntersectionResultsDialog.cpp
  dialogs/camera/ForwardIntersectionResultsDialog.h
  dialogs/camera/ForwardIntersectionResultsDialog.ui
  dialogs/camera/SurveyControlDialog.cpp
  dialogs/camera/SurveyControlDialog.h
)

set(GUI_IMAGE_DIALOG_SOURCES
  dialogs/image/GenerateMaskDialog.cpp
  dialogs/image/GenerateMaskDialog.h
)

set(GUI_RECONSTRUCTION_DIALOG_SOURCES
  dialogs/reconstruction/AerialTriangulationDialog.cpp
  dialogs/reconstruction/AerialTriangulationDialog.h
  dialogs/reconstruction/AerialTriangulationDialog.ui
  dialogs/reconstruction/CreateDemDialog.cpp
  dialogs/reconstruction/CreateDemDialog.h
  dialogs/reconstruction/CreateDemDialog.ui
  dialogs/reconstruction/GenerateModelDialog.cpp
  dialogs/reconstruction/GenerateModelDialog.h
  dialogs/reconstruction/MapProjectDialog.cpp
  dialogs/reconstruction/MapProjectDialog.h
  dialogs/reconstruction/MapProjectDialog.ui
  dialogs/reconstruction/TextureMappingDialog.cpp
  dialogs/reconstruction/TextureMappingDialog.h
  dialogs/reconstruction/TextureMappingDialog.ui
)

set(GUI_TIE_POINT_DIALOG_SOURCES
  dialogs/tie_points/CleanTiePointsDialog.cpp
  dialogs/tie_points/CleanTiePointsDialog.h
  dialogs/tie_points/CreateTiePointsDialog.cpp
  dialogs/tie_points/CreateTiePointsDialog.h
  dialogs/tie_points/FeaturePointVisualizationDialog.cpp
  dialogs/tie_points/FeaturePointVisualizationDialog.h
  dialogs/tie_points/FeaturePointVisualizationDialog.ui
  dialogs/tie_points/MatchPairSelectorDialog.cpp
  dialogs/tie_points/MatchPairSelectorDialog.h
  dialogs/tie_points/MatchPairSelectorDialog.ui
  dialogs/tie_points/MatchViewerDialog.cpp
  dialogs/tie_points/MatchViewerDialog.h
  dialogs/tie_points/MatchViewerDialog.ui
  dialogs/tie_points/OverlapAnalysisDialog.cpp
  dialogs/tie_points/OverlapAnalysisDialog.h
  dialogs/tie_points/OverlapAnalysisDialog.ui
  dialogs/tie_points/ThinTiePointsDialog.cpp
  dialogs/tie_points/ThinTiePointsDialog.h
)

set(GUI_SHARED_DIALOG_SOURCES
  dialogs/shared/WorkflowParameterDialogStyle.cpp
  dialogs/shared/WorkflowParameterDialogStyle.h
)

set(GUI_DIALOG_SOURCES
  ${GUI_APPLICATION_DIALOG_SOURCES}
  ${GUI_CAMERA_DIALOG_SOURCES}
  ${GUI_IMAGE_DIALOG_SOURCES}
  ${GUI_RECONSTRUCTION_DIALOG_SOURCES}
  ${GUI_TIE_POINT_DIALOG_SOURCES}
  ${GUI_SHARED_DIALOG_SOURCES}
)
