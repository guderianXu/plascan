#pragma once

#include "PlanetaryLineScanBundleAdjust.h"

#include <QJsonObject>
#include <QString>

namespace xjw
{
namespace cli
{

QJsonObject planetaryLineScanBaResultToJson(
    const lidar::PlanetaryLineScanBaResult &result);

bool writePlanetaryLineScanBaArtifacts(
    const QString &outputDirectory,
    const QString &prefix,
    const lidar::PlanetaryLineScanBaResult &result,
    QString *errorMessage = nullptr);

} // namespace cli
} // namespace xjw
