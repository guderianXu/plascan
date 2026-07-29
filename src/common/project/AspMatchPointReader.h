#pragma once

#include <QPointF>
#include <QString>
#include <QVector>

namespace xjw::common::project
{

struct AspMatchPointResult
{
    bool success = false;
    QString errorMessage;
    QVector<QPointF> leftPoints;
    QVector<QPointF> rightPoints;
};

/// Reads the point coordinates from an ASP binary match file without loading
/// descriptor payloads into memory.
AspMatchPointResult readAspMatchPoints(const QString &path);

} // namespace xjw::common::project
