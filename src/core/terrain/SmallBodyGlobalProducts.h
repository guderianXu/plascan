#pragma once

#include "DemDomTypes.h"

#include <QJsonObject>
#include <QString>

#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <functional>

namespace xjw
{

/** Parameters for a planetocentric, positive-east, body-fixed global grid. */
struct SmallBodyGlobalOptions
{
    QString targetName = QStringLiteral("Small Body");
    QString bodyFixedFrame = QStringLiteral("MODEL_LOCAL_BODY_FIXED");
    QString surfaceCoordinateUnit = QStringLiteral("m");
    bool automaticCenter = true;
    cv::Vec3d bodyCenter = cv::Vec3d(0.0, 0.0, 0.0);
    double referenceRadiusM = 0.0;
    double angularResolutionDeg = 0.25;
    double centralMeridianDeg = 0.0;
    std::int64_t maximumPixelCount = 25000000;
    bool writeReportPreview = true;

    bool validate(QString *errorMessage = nullptr) const;
};

/** In-memory and on-disk products generated on one shared longitude/latitude grid. */
struct SmallBodyGlobalProducts
{
    DemGridData radialDem;
    DemGridData elevationDem;
    cv::Mat domBgr;
    cv::Mat validMask;
    cv::Mat reliability;
    cv::Mat ambiguousMask;

    cv::Vec3d bodyCenter = cv::Vec3d(0.0, 0.0, 0.0);
    double referenceRadiusM = 0.0;
    double coverageRatio = 0.0;
    double solidAngleWeightedCoverageRatio = 0.0;
    double ambiguousRatio = 0.0;
    QString sourceSurfacePath;
    QString domColorSource;

    QString radialDemPath;
    QString elevationDemPath;
    QString domPath;
    QString reliabilityPath;
    QString coveragePath;
    QString ambiguityPath;
    QString previewPath;
    QString reportPath;
    QJsonObject report;
};

using SmallBodyProgressCallback = std::function<void(const QString &, int)>;

} // namespace xjw
