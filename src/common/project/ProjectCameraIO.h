#pragma once

#include <placamera/frame_camera.h>

#include <QJsonObject>

namespace xjw::common::project
{

    /** Serialize a validated PlaCamera instance for project reports. */
    QJsonObject serializeFramePinholeModel(const placamera::FramePinholeModel& camera);

} // namespace xjw::common::project
