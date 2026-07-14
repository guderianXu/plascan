#pragma once

#include "model/MarkerSet.h"

#include <QHash>
#include <QJsonObject>
#include <QString>

namespace xjw::control_points
{

struct SurveyControlMigrationResult
{
    bool ok = false;
    MarkerSet markerSet;
    QString error;
    int migratedMarkers = 0;
    int migratedProjections = 0;
    int migratedScaleBars = 0;
};

SurveyControlMigrationResult migrateSurveyControl(
    const QJsonObject &legacy,
    const QHash<QString, QString> &imageIdentityByPath);

} // namespace xjw::control_points
