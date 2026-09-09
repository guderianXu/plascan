#pragma once

#include <functional>
#include <QJsonObject>
#include <QString>

#include "metmodel/types.hpp"

namespace xjw::mesh
{
    QJsonObject colorizeRecoveredModel(metmodel::Mesh& mesh,
                                       metmodel::Scene& scene,
                                       const QJsonObject& settings,
                                       int device,
                                       const std::function<bool()>& is_cancelled,
                                       const std::function<void(const QString&, int)>& progress);
}
