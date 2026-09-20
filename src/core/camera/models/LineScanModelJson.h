#pragma once

#include "camera/models/linescan/LineScanInstance.h"

#include <QJsonObject>

#include <memory>
#include <string_view>

namespace xjw::camera_models
{

    std::unique_ptr<camera_core::CameraDefinition>
    createLineScanDefinitionFromJson(const camera_core::CameraDefinitionId& id,
                                     const xjw::coordinate_system::CoordinateFrameId& frame,
                                     int parameterSchemaVersion,
                                     std::string_view serializedParameters);

    std::unique_ptr<camera_core::CameraInstance>
    createLineScanInstanceFromJson(const camera_core::CameraInstanceId& id,
                                   const camera_core::ImageId& image,
                                   std::shared_ptr<const camera_core::CameraDefinition> definition,
                                   std::string_view serializedState);

    QJsonObject lineScanDefinitionParametersToJson(const linescan::LineScanDefinition& definition);
    QJsonObject lineScanInstanceStateToJson(const linescan::LineScanInstance& instance);

} // namespace xjw::camera_models
