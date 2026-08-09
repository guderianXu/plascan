#pragma once

#include <QString>

namespace xjw::camera_reference
{
class CameraReferenceSet;
struct RawCameraReference;
}

namespace xjw::gui::reference
{

bool exportCameraReferenceCsv(const camera_reference::CameraReferenceSet &referenceSet,
                              const QString &path,
                              QString *error = nullptr);

} // namespace xjw::gui::reference
