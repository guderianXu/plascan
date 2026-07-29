#pragma once

#include <QString>

namespace xjw::matchphotos
{

struct StoredPairGeometryAuditResult
{
    bool statisticsAvailable = false;
    bool verified = false;
    int totalMatches = 0;
    int geometricInliers = 0;
    double inlierRatio = 0.0;
    double coverageScore = 0.0;
    QString reason;
};

StoredPairGeometryAuditResult auditStoredPairGeometry(
    const QString &matchPath,
    const QString &sidecarPath,
    int minimumInliers = 20,
    double reprojectionThreshold = 1.5);

} // namespace xjw::matchphotos
