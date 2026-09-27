#pragma once

#include <placamera/frame_camera.h>

#include <QJsonObject>
#include <QString>

#include <memory>
#include <optional>

namespace xjw::common::project
{

    struct ProjectFramePinhole
    {
        std::shared_ptr<const placamera::FramePinholeDefinition> definition;
        placamera::Pose pose;
    };

    /** Decode project/report frame-pinhole metadata directly into PlaCamera geometry. */
    std::optional<ProjectFramePinhole> decodeFramePinholeMetadata(const QJsonObject& metadata,
                                                                  placamera::CameraDefinitionId definition_id,
                                                                  QString* error = nullptr);

} // namespace xjw::common::project
