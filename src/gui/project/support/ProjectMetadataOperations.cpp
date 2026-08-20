#include "ProjectMetadataOperations.h"

#include "project/ProjectSessionModel.h"
#include "ProjectResultRecords.h"
#include "ProjectWorkflowOperations.h"
#include "project/ProjectIO.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>

#include <limits>

namespace xjw::gui::project {
namespace {

QString denseCloudPathFromRecord(const QJsonObject &record)
{
    QString path = record.value(QStringLiteral("dense_cloud_xyz")).toString().trimmed();
    if (path.isEmpty())
    {
        path = record.value(QStringLiteral("path")).toString().trimmed();
    }
    return QDir::cleanPath(path);
}

bool containsAnyToken(const QString &text, const QStringList &tokens)
{
    for (const QString &token : tokens)
    {
        if (text.contains(token))
        {
            return true;
        }
    }
    return false;
}

int denseCloudMeshingPreferenceScore(const QJsonObject &record)
{
    const QString path = denseCloudPathFromRecord(record);
    const QString haystack = QStringList{
        record.value(QStringLiteral("stage")).toString(),
        record.value(QStringLiteral("quality_stage")).toString(),
        record.value(QStringLiteral("operation")).toString(),
        record.value(QStringLiteral("source")).toString(),
        record.value(QStringLiteral("result_type")).toString(),
        record.value(QStringLiteral("kind")).toString(),
        QFileInfo(path).completeBaseName()
    }.join(QLatin1Char(' ')).toLower();

    int score = 0;
    if (containsAnyToken(haystack, {QStringLiteral("production"),
                                   QStringLiteral("final"),
                                   QStringLiteral("deliverable")}))
    {
        score += 3000;
    }
    if (containsAnyToken(haystack, {QStringLiteral("terrain"),
                                   QStringLiteral("surface"),
                                   QStringLiteral("dem_surface"),
                                   QStringLiteral("height_grid")}))
    {
        score += 1200;
    }
    if (containsAnyToken(haystack, {QStringLiteral("refined"),
                                   QStringLiteral("refine"),
                                   QStringLiteral("cleaned"),
                                   QStringLiteral("clean"),
                                   QStringLiteral("filtered"),
                                   QStringLiteral("filter"),
                                   QStringLiteral("postprocess"),
                                   QStringLiteral("denoise")}))
    {
        score += 1000;
    }
    if (containsAnyToken(haystack, {QStringLiteral("raw"),
                                   QStringLiteral("mvs_fusion"),
                                   QStringLiteral("depth_fusion")}))
    {
        score -= 100;
    }
    if (containsAnyToken(haystack, {QStringLiteral("debug"),
                                   QStringLiteral("preview"),
                                   QStringLiteral("sample"),
                                   QStringLiteral("temporary")}))
    {
        score -= 800;
    }
    return score;
}

} // namespace

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
    projectData->scheduleTemporaryMetadataSave();
}

QString resolveProjectOutputDir(const QString &projectPath,
                                const QString &requestedDir,
                                const QString &fallbackRelativeDir)
{
    QString outputDir = requestedDir.trimmed();
    if (outputDir.isEmpty())
    {
        const QString projectRoot =
            xjw::common::project::ProjectIO::projectRootFromPlascan(
                projectPath);
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
    int bestScore = std::numeric_limits<int>::min();
    int bestIndex = -1;
    QString bestPath;
    for (int index = 0; index < denseResults.size(); ++index)
    {
        const QJsonObject record = denseResults.at(index).toObject();
        const QString candidate = denseCloudPathFromRecord(record);
        if (!candidate.isEmpty() && QFileInfo::exists(candidate))
        {
            const int score = denseCloudMeshingPreferenceScore(record);
            if (score > bestScore || (score == bestScore && index > bestIndex))
            {
                bestScore = score;
                bestIndex = index;
                bestPath = candidate;
            }
        }
    }
    if (!bestPath.isEmpty())
    {
        *denseCloudPath = bestPath;
        return true;
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

TiePointMutationResult replaceTiePointResult(ProjectData *projectData,
                                             const QString &sparseCloudPath,
                                             int sparsePointCount,
                                             const QStringList &selectedImages,
                                             const QString &outputDir,
                                             const QJsonObject &extraRecord)
{
    if (!projectData || !projectData->hasProject())
    {
        TiePointMutationResult result;
        result.errorMessage = QStringLiteral("项目数据未初始化");
        return result;
    }

    QJsonObject entry = makeAtResultRecord(QDateTime::currentDateTimeUtc().toString(Qt::ISODate),
                                           sparseCloudPath,
                                           sparsePointCount,
                                           selectedImages,
                                           outputDir,
                                           extraRecord);
    entry[QStringLiteral("operation_display_name")] =
        xjw::core::project::sparseOperationDisplayName(
            entry.value(QStringLiteral("operation")).toString());

    return ProjectTiePointResultService::replaceCurrent(projectData, entry);
}

} // namespace xjw::gui::project
