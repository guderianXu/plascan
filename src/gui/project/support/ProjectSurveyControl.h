#pragma once

#include "model/MarkerSet.h"

#include <QByteArray>
#include <QHash>
#include <QJsonObject>
#include <QString>

class ProjectData;

namespace xjw::gui::project
{

struct SurveyControlProjectImportResult
{
    bool imported = false;
    QString errorMessage;
    int controlPointCount = 0;
    int checkPointCount = 0;
    int scaleBarCount = 0;
};

struct PreparedSurveyControlImport
{
    bool prepared = false;
    QString errorMessage;
    control_points::MarkerSet markerSet;
};

// A GUI-side transaction keeps the previous bytes until both the sidecar and
// ProjectData metadata have been accepted.  Workers must only prepare the
// MarkerSet; they must not modify this path.
struct SurveyControlSidecarSnapshot
{
    QString path;
    bool existed = false;
    QByteArray bytes;
};

PreparedSurveyControlImport prepareSurveyControlCsv(
    const QHash<QString, QString>& imageIdentityByPath,
    const QString& csvPath,
    const QString& defaultRole);

SurveyControlProjectImportResult writeSurveyControlMarkerSet(
    const QString& sidecarPath,
    const control_points::MarkerSet& markerSet);

SurveyControlProjectImportResult commitSurveyControlMarkerSet(
    ProjectData* projectData,
    const control_points::MarkerSet& markerSet);

SurveyControlProjectImportResult importSurveyControlCsv(ProjectData *projectData,
                                                        const QString &csvPath,
                                                        const QString &defaultRole);

bool captureSurveyControlSidecar(const QString& sidecarPath,
                                 SurveyControlSidecarSnapshot* snapshot,
                                 QString* errorMessage = nullptr);

bool restoreSurveyControlSidecar(const SurveyControlSidecarSnapshot& snapshot,
                                 QString* errorMessage = nullptr);

QJsonObject surveyControlDialogMetadata(ProjectData *projectData,
                                        QString *errorMessage = nullptr);

} // namespace xjw::gui::project
