#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::qc
{

struct SurveyControlImportOptions
{
    QString defaultRole;
};

struct SurveyControlImportResult
{
    bool ok = false;
    QString error;
    QJsonObject surveyControl;
    int controlPointCount = 0;
    int checkPointCount = 0;
    int scaleBarCount = 0;
};

SurveyControlImportResult parseSurveyControlCsv(const QString &csvText,
                                                const SurveyControlImportOptions &options = {});

SurveyControlImportResult readSurveyControlCsvFile(const QString &path,
                                                   const SurveyControlImportOptions &options = {});

} // namespace xjw::qc
