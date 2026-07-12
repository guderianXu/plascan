#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::gui::config
{

int normalizeImageViewRotationDegrees(int degrees);
QString imageViewRotationPathKey(const QString &imagePath);
int imageViewRotationForPath(const QJsonObject &rotations, const QString &imagePath);
QJsonObject withImageViewRotation(const QJsonObject &rotations,
                                  const QString &imagePath,
                                  int degrees);

} // namespace xjw::gui::config
