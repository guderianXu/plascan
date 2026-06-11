#include "ProjectMetadataOperations.h"

#include "ProjectData.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowUtils.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>

namespace xjw::gui::project {

QJsonObject projectFilesMeta(ProjectData *projectData)
{
    if (!projectData)
    {
        return QJsonObject();
    }

    const QJsonObject meta = projectData->coreFilesMeta();
    if (meta.value(QStringLiteral("project_files")).isObject())
    {
        return meta.value(QStringLiteral("project_files")).toObject();
    }
    return meta;
}

void persistProjectMeta(ProjectData *projectData,
                        const QJsonObject &meta,
                        bool markDirty)
{
    if (!projectData)
    {
        return;
    }

    projectData->updateMetadata(meta, markDirty);
    projectData->saveTemporaryMetadata();
}

QString resolveProjectOutputDir(const QString &projectPath,
                                const QString &requestedDir,
                                const QString &fallbackRelativeDir)
{
    QString outputDir = requestedDir.trimmed();
    if (outputDir.isEmpty())
    {
        const QString projectRoot = QFileInfo(projectPath).absolutePath();
        outputDir = QDir(projectRoot).filePath(fallbackRelativeDir);
    }
    outputDir = QDir::cleanPath(outputDir);
    QDir().mkpath(outputDir);
    return outputDir;
}

bool resolveLatestDenseCloudPath(ProjectData *projectData,
                                 QString *denseCloudPath,
                                 QString *errorMessage)
{
    if (!denseCloudPath)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("内部错误：缺少密集点云输出参数");
        }
        return false;
    }

    *denseCloudPath = QString();
    if (!projectData)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("项目未就绪");
        }
        return false;
    }

    const QJsonArray denseResults = projectData->metadata().value(QStringLiteral("dense_cloud_results")).toArray();
    for (int index = denseResults.size() - 1; index >= 0; --index)
    {
        const QString candidate = denseResults.at(index).toObject().value(QStringLiteral("dense_cloud_xyz")).toString();
        if (!candidate.isEmpty() && QFileInfo::exists(candidate))
        {
            *denseCloudPath = candidate;
            return true;
        }
    }

    if (errorMessage)
    {
        *errorMessage = QStringLiteral("未找到可用的密集点云结果");
    }
    return false;
}

void upsertProjectRecordByPath(ProjectData *projectData,
                               const QString &arrayKey,
                               const QString &pathKey,
                               const QJsonObject &record,
                               bool markDirty)
{
    if (!projectData)
    {
        return;
    }

    projectData->upsertResultRecordByPath(arrayKey, pathKey, record, markDirty);
}

void replaceProjectRecordWithLatest(ProjectData *projectData,
                                    const QString &arrayKey,
                                    const QJsonObject &record,
                                    bool markDirty)
{
    if (!projectData)
    {
        return;
    }

    projectData->replaceResultRecordWithLatest(arrayKey, record, markDirty);
}

void appendAtResult(ProjectData *projectData,
                    const QString &sparseCloudPath,
                    int sparsePointCount,
                    const QStringList &selectedImages,
                    const QString &outputDir,
                    const QJsonObject &extraRecord,
                    int replaceIndex)
{
    if (!projectData || !projectData->hasProject())
    {
        return;
    }

    QJsonObject entry = makeAtResultRecord(QDateTime::currentDateTimeUtc().toString(Qt::ISODate),
                                           sparseCloudPath,
                                           sparsePointCount,
                                           selectedImages,
                                           outputDir,
                                           extraRecord);
    entry[QStringLiteral("operation_display_name")] =
        sparseOperationDisplayName(entry.value(QStringLiteral("operation")).toString());

    projectData->upsertResultRecordByIndex(QStringLiteral("aerial_triangulation_results"),
                                           entry,
                                           replaceIndex,
                                           true);
}

void appendObsNetResult(ProjectData *projectData,
                        int nodeCount,
                        int edgeCount,
                        const QString &algorithmName,
                        const QJsonObject &extraInfo)
{
    if (!projectData || !projectData->hasProject())
    {
        return;
    }

    QJsonObject entry;
    entry[QStringLiteral("algorithm")] = algorithmName;
    entry[QStringLiteral("node_count")] = nodeCount;
    entry[QStringLiteral("edge_count")] = edgeCount;
    entry[QStringLiteral("timestamp")] = QDateTime::currentDateTime().toString(Qt::ISODate);
    for (auto it = extraInfo.begin(); it != extraInfo.end(); ++it)
    {
        entry[it.key()] = it.value();
    }

    projectData->replaceResultRecordWithLatest(QStringLiteral("observation_network_results"),
                                               entry,
                                               true);
}

} // namespace xjw::gui::project
