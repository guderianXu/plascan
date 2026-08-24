#pragma once

#include <QProcessEnvironment>
#include <QString>

namespace xjw::gui::runtime
{

struct GeospatialDataPaths
{
    QString projData;
    QString gdalData;
};

GeospatialDataPaths resolveGeospatialDataPaths(
    const QString &applicationDirectory,
    const QProcessEnvironment &environment = QProcessEnvironment::systemEnvironment());

GeospatialDataPaths configureGeospatialDataPaths(const QString &applicationDirectory);

} // namespace xjw::gui::runtime
