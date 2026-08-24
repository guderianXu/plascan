#pragma once

#include "CameraModel.h"
#include "FramePinholeCamera.h"
#include "RpcCameraModel.h"

#include <QJsonObject>
#include <QString>

#include <memory>

namespace xjw::common::project
{

QJsonObject cameraToJson(const xjw::FramePinholeCamera &camera);
QJsonObject cameraToJson(const xjw::RpcCameraModel &camera);
bool parseTsaiCamera(const QString &tsai_path,
                     QJsonObject *camera_metadata,
                     QString *error_message = nullptr);
bool parseRpcCameraRaster(const QString &raster_path,
                          QJsonObject *camera_metadata,
                          QString *error_message = nullptr);
bool cameraFromJson(const QJsonObject &camera_object, xjw::FramePinholeCamera *camera);
bool cameraFromJson(const QJsonObject &camera_object, xjw::RpcCameraModel *camera);
bool imageCameraFromEntry(const QJsonObject &image_object, xjw::FramePinholeCamera *camera);
bool imageCameraFromEntry(const QJsonObject &image_object, xjw::RpcCameraModel *camera);
std::unique_ptr<xjw::CameraModel> cameraModelFromJson(const QJsonObject &camera_object);
std::unique_ptr<xjw::CameraModel> imageCameraModelFromEntry(const QJsonObject &image_object);

} // namespace xjw::common::project
