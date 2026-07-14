#pragma once

#include "MarkerSet.h"

#include <QString>
#include <QVector>

namespace xjw::control_points
{

struct MarkerValidationIssue
{
    QString code;
    QString message;
    MarkerId markerId;
    QString imageId;
};

class MarkerSetValidator
{
public:
    static QVector<MarkerValidationIssue> validate(const MarkerSet &set);
};

} // namespace xjw::control_points
