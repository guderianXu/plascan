#pragma once

#include "model/MarkerSet.h"

#include <QJsonObject>
#include <QString>

namespace xjw::control_points
{

class MarkerSetJson
{
public:
    static QJsonObject encode(const MarkerSet &markerSet);
    static bool decode(const QJsonObject &object, MarkerSet *markerSet, QString *error = nullptr);
};

} // namespace xjw::control_points
