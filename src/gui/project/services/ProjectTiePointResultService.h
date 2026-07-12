#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>

class ProjectData;

namespace xjw::gui::project
{

struct TiePointResultSelection
{
    int sourceIndex = -1;
    int pointCount = -1;
    QJsonObject record;
    QString sparseCloudPath;

    bool isValid() const;
};

struct TiePointMutationResult
{
    bool success = false;
    int removedRecordCount = 0;
    QString reconstructionGenerationId;
    QString errorMessage;
    QStringList cleanupWarnings;
};

class ProjectTiePointResultService
{
public:
    static TiePointResultSelection selectCurrent(const QJsonObject &meta,
                                                 const QString &projectPath);
    static QJsonObject metadataWithCurrentOnly(const QJsonObject &meta,
                                               const QString &projectPath);
    static TiePointMutationResult replaceCurrent(ProjectData *projectData,
                                                 const QJsonObject &newRecord);
    static TiePointMutationResult deleteAll(ProjectData *projectData);
};

} // namespace xjw::gui::project
