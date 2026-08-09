#pragma once

#include "model/CameraReferenceSet.h"

#include <QJsonObject>

namespace xjw::camera_reference
{

class CameraReferenceSetJson
{
public:
    static constexpr const char *Format = "plascan_camera_reference_set";

    static QJsonObject encode(const CameraReferenceSet &referenceSet);
    static bool decode(const QJsonObject &object,
                       CameraReferenceSet *referenceSet,
                       QString *error = nullptr);
};

} // namespace xjw::camera_reference
