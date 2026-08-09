#pragma once

#include "model/CameraReferenceSet.h"

#include <QJsonObject>
#include <QString>

namespace xjw::gui::reference_import
{
struct MetashapeCameraReferenceImportResult;
}

namespace xjw::gui::reference
{

camera_reference::CameraReferenceSet buildMetashapeCameraReferenceSet(
    const reference_import::MetashapeCameraReferenceImportResult &imported,
    const QJsonObject &metadata,
    const QString &cameraPath,
    const QString &offsetPath,
    const QString &contentHash);

} // namespace xjw::gui::reference
