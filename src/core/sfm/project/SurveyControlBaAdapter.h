#pragma once

#include "project/BaInputBuilder.h"

#include <QJsonObject>
#include <QMap>
#include <QString>

namespace xjw::core::project
{

void appendSurveyControlBaInput(const QJsonObject &meta,
                                const QMap<QString, int> &cameraIndexByPath,
                                BaInputBuildResult *result);

} // namespace xjw::core::project
