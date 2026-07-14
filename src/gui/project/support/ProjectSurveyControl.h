#pragma once

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

SurveyControlProjectImportResult importSurveyControlCsv(ProjectData *projectData,
                                                        const QString &csvPath,
                                                        const QString &defaultRole);

SurveyControlProjectImportResult migrateLegacySurveyControl(ProjectData *projectData);

} // namespace xjw::gui::project
