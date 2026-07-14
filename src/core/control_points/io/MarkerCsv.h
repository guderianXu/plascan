#pragma once

#include "model/MarkerSet.h"

#include <QHash>
#include <QString>

namespace xjw::control_points
{

struct MarkerCsvImportOptions
{
    QString defaultRole;
    QString sourceCrs;
    QString axisOrder = QStringLiteral("traditional_gis");
    QString verticalDatum;
    QString verticalUnit;
    QHash<QString, QString> imageIdentityByPath;
};

struct MarkerCsvImportResult
{
    bool ok = false;
    QString error;
    MarkerSet markerSet;
    int controlPointCount = 0;
    int checkPointCount = 0;
    int scaleBarCount = 0;
};

MarkerCsvImportResult parseMarkerCsv(const QString &csvText,
                                     const MarkerCsvImportOptions &options = {});
MarkerCsvImportResult readMarkerCsvFile(const QString &path,
                                        const MarkerCsvImportOptions &options = {});
QString markerSetToCsv(const MarkerSet &markerSet);

} // namespace xjw::control_points
