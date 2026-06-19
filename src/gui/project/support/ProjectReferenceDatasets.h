#pragma once

#include <QString>
#include <QJsonObject>

class ProjectData;

namespace xjw::gui::project {

struct ReferenceDatasetQualityReportResult
{
    bool saved = false;
    QString jsonPath;
    QString csvPath;
    QJsonObject record;
    QString errorMessage;
};

QString referenceDatasetTypeForPath(const QString &path);

QString normalizeReferenceDatasetType(const QString &type, const QString &path);

bool registerReferenceDataset(ProjectData *projectData,
                              const QString &path,
                              const QString &type = QString(),
                              const QString &role = QStringLiteral("validation"),
                              QString *errorMsg = nullptr);

ReferenceDatasetQualityReportResult writeReferenceDatasetQualityReport(
    ProjectData *projectData,
    const QString &baseName = QStringLiteral("reference_quality_report"));

ReferenceDatasetQualityReportResult writeReferenceTerrainPriorPreflightReport(
    ProjectData *projectData,
    const QString &baseName = QStringLiteral("reference_terrain_prior_preflight"));

} // namespace xjw::gui::project
