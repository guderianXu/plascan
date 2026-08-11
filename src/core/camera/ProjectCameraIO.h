#pragma once

#include "FramePinholeCamera.h"

#include <QJsonObject>
#include <QString>

namespace xjw::common::project
{

QJsonObject cameraToJson(const xjw::FramePinholeCamera &camera);
bool parseTsaiCamera(const QString &tsai_path,
                     QJsonObject *camera_metadata,
                     QString *error_message = nullptr);
bool cameraFromJson(const QJsonObject &camera_object, xjw::FramePinholeCamera *camera);
bool imageCameraFromEntry(const QJsonObject &image_object, xjw::FramePinholeCamera *camera);

} // namespace xjw::common::project
