#pragma once

#include <QString>

namespace DialogSettingKeys {

inline const QString FeaturePointVisualization = QStringLiteral("feature_point_visualization");
inline const QString AerialTriangulation = QStringLiteral("aerial_triangulation");
inline const QString MapProject = QStringLiteral("mapproject");
inline const QString MatchViewer = QStringLiteral("match_viewer");
// Retained for reading pair constraints written by projects created before
// the granular reconstruction menu was removed.
inline const QString FeatureMatching = QStringLiteral("feature_matching");
inline const QString GenerateModel        = QStringLiteral("generate_model");
inline const QString TextureMapping       = QStringLiteral("texture_mapping");

} // namespace DialogSettingKeys
