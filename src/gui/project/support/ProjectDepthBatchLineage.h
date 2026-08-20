#pragma once

#include "DepthFrameUtils.h"

#include <QJsonObject>
#include <QString>

namespace xjw::gui::project
{

QString canonicalProjectDepthInputSignature(
    const QJsonObject &projectMetadata,
    int aerialTriangulationResultIndex);

} // namespace xjw::gui::project
