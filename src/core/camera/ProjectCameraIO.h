#pragma once

#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/models/linescan/LineScanInstance.h"
#include "camera/models/rpc/RpcInstance.h"

#include <QJsonObject>
#include <QString>

namespace xjw::common::project
{

    QJsonObject
    serializeFramePinholeNumericState(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera);
    bool saveFramePinholeNumericState(const xjw::camera_models::frame_pinhole::FramePinholeNumericState& camera,
                                      const std::string& path);
    bool loadFramePinholeNumericStateFromFile(const QString& path,
                                              xjw::camera_models::frame_pinhole::FramePinholeNumericState* camera,
                                              QString* error_message = nullptr);
    QJsonObject serializeRpcInstance(const xjw::camera_models::rpc::RpcInstance& camera);
    QJsonObject serializeLineScanInstance(const xjw::camera_models::linescan::LineScanInstance& camera);
    bool parseTsaiCamera(const QString& tsai_path, QJsonObject* camera_metadata, QString* error_message = nullptr);
    bool
    parseRpcCameraRaster(const QString& raster_path, QJsonObject* camera_metadata, QString* error_message = nullptr);
    bool decodeFramePinholeNumericState(const QJsonObject& camera_object,
                                        xjw::camera_models::frame_pinhole::FramePinholeNumericState* camera);

} // namespace xjw::common::project
