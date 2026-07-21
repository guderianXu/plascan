#pragma once

#include "project/BaInputBuilder.h"

#include <QMap>
#include <QString>

namespace xjw::core::project
{

void appendMarkerBaInput(const MarkerBaInput *input,
                         const QMap<QString, int> &cameraIndexByPath,
                         BaInputBuildResult *result);

} // namespace xjw::core::project
