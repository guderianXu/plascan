#include "ProjectSurveyControl.h"

#include "ProjectData.h"
#include "SurveyControlImport.h"

#include <QDateTime>
#include <QFileInfo>
#include <QJsonObject>

namespace xjw::gui::project
{

SurveyControlProjectImportResult importSurveyControlCsv(ProjectData *projectData,
                                                        const QString &csvPath,
                                                        const QString &defaultRole)
{
    SurveyControlProjectImportResult result;
    if (!projectData || !projectData->hasProject())
    {
        result.errorMessage = QStringLiteral("项目未打开，无法导入控制点数据");
        return result;
    }

    xjw::qc::SurveyControlImportOptions options;
    options.defaultRole = defaultRole;
    const auto importResult = xjw::qc::readSurveyControlCsvFile(csvPath, options);
    if (!importResult.ok)
    {
        result.errorMessage = importResult.error;
        return result;
    }

    QJsonObject surveyControl = importResult.surveyControl;
    surveyControl[QStringLiteral("source_path")] = QFileInfo(csvPath).absoluteFilePath();
    surveyControl[QStringLiteral("imported_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    surveyControl[QStringLiteral("format")] = QStringLiteral("csv");

    QJsonObject meta = projectData->metadata();
    meta[QStringLiteral("survey_control")] = surveyControl;
    projectData->updateMetadata(meta, true);

    result.imported = true;
    result.controlPointCount = importResult.controlPointCount;
    result.checkPointCount = importResult.checkPointCount;
    result.scaleBarCount = importResult.scaleBarCount;
    return result;
}

} // namespace xjw::gui::project
