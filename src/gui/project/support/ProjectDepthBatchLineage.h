#pragma once

#include "DepthFrameUtils.h"

#include <QJsonObject>
#include <QString>

namespace xjw::gui::project
{

QString canonicalProjectDepthInputSignature(
    const QJsonObject &projectMetadata,
    int aerialTriangulationResultIndex,
    int signatureVersion);

bool legacyDepthCamerasMatchCurrentProject(
    const xjw::core::project::StoredDepthFramesResult &storedFrames,
    const QJsonObject &projectMetadata);

} // namespace xjw::gui::project
