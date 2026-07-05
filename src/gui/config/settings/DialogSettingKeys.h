#pragma once

#include <QString>

namespace DialogSettingKeys {

inline const QString FeatureExtraction = QStringLiteral("feature_extraction");
inline const QString FeaturePointVisualization = QStringLiteral("feature_point_visualization");
inline const QString BundleAdjust = QStringLiteral("bundle_adjust");
inline const QString AerialTriangulation = QStringLiteral("aerial_triangulation");
inline const QString ThreeDReconstruction = QStringLiteral("three_d_reconstruction");
inline const QString CreateDem = QStringLiteral("create_dem");
inline const QString MapProject = QStringLiteral("mapproject");
inline const QString DenseCloud = QStringLiteral("dense_cloud");
inline const QString MatchViewer = QStringLiteral("match_viewer");
inline const QString FeatureMatching = QStringLiteral("feature_matching");
inline const QString VocabularyOverlap = QStringLiteral("vocabulary_overlap");
inline const QString MainWindowUi = QStringLiteral("main_window_ui");

// ==== 重建菜单对话框 ====
inline const QString ObservationNetwork   = QStringLiteral("observation_network");
inline const QString InitCameraPose       = QStringLiteral("init_camera_pose");
inline const QString Triangulation        = QStringLiteral("triangulation");
inline const QString ReconBundleAdjust    = QStringLiteral("recon_bundle_adjust");
inline const QString SparseCloudPostProcess = QStringLiteral("sparse_cloud_post_process");
inline const QString DepthMapEstimate     = QStringLiteral("depth_map_estimate");
inline const QString DenseMatch           = QStringLiteral("dense_match");
inline const QString DepthFusion          = QStringLiteral("depth_fusion");
inline const QString DenseCloudRefine     = QStringLiteral("dense_cloud_refine");
inline const QString GenerateModel        = QStringLiteral("generate_model");
inline const QString MeshReconstruction   = QStringLiteral("mesh_reconstruction");
inline const QString TextureMapping       = QStringLiteral("texture_mapping");
inline const QString ModelExport          = QStringLiteral("model_export");

} // namespace DialogSettingKeys
