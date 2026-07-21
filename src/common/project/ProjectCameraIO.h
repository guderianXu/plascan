#pragma once

#include "Camera.h"

#include <QJsonObject>
#include <QString>

namespace xjw::common::project
{

QJsonObject cameraToJson(const xjw::Camera &camera);
bool parseTsaiCamera(const QString &tsai_path,
                     QJsonObject *camera_metadata,
                     QString *error_message = nullptr);
bool cameraFromJson(const QJsonObject &camera_object, xjw::Camera *camera);
bool imageCameraFromEntry(const QJsonObject &image_object, xjw::Camera *camera);

} // namespace xjw::common::project
