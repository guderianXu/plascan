#include "ProjectTiePointResultService.h"

#include "ProjectData.h"
#include "project/ProjectIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonValue>
#include <QSet>
#include <QUuid>

#include <algorithm>

namespace xjw::gui::project
{
namespace
{

QString resolveProjectPath(const QString &projectPath, const QString &path)
{
    const QString trimmedPath = path.trimmed();
    if (trimmedPath.isEmpty())
    {
        return QString();
    }

    const QFileInfo pathInfo(trimmedPath);
    if (pathInfo.isAbsolute())
    {
        return QDir::cleanPath(pathInfo.absoluteFilePath());
    }

    return xjw::common::project::ProjectIO::resolveProjectResourcePath(
        projectPath, trimmedPath);
}

QString pathKey(const QString &path)
{
    QString key = QDir::cleanPath(path);
#ifdef Q_OS_WIN
    key = key.toLower();
#endif
    return key;
}

void appendUniquePath(QStringList *paths, const QString &path)
{
    if (!paths || path.isEmpty())
    {
        return;
    }

    const QString cleanPath = QDir::cleanPath(path);
    const QString cleanKey = pathKey(cleanPath);
    for (const QString &existing : *paths)
    {
        if (pathKey(existing) == cleanKey)
        {
            return;
        }
    }
    paths->append(cleanPath);
}

void collectRecordArtifacts(const QJsonObject &record,
                            const QString &projectPath,
                            QStringList *filePaths,
                            QStringList *directoryPaths)
{
    const QJsonObject files = record.value(QStringLiteral("files")).toObject();
    for (auto it = files.begin(); it != files.end(); ++it)
    {
        if (it.value().isString())
        {
            appendUniquePath(filePaths, resolveProjectPath(projectPath, it.value().toString()));
        }
    }

    appendUniquePath(directoryPaths,
                     resolveProjectPath(projectPath,
                                        record.value(QStringLiteral("output_dir")).toString()));
}

void collectJsonPaths(const QJsonValue &value,
                      const QString &projectPath,
                      QSet<QString> *protectedPaths)
{
    if (!protectedPaths)
    {
        return;
    }

    if (value.isString())
    {
        const QString resolved = resolveProjectPath(projectPath, value.toString());
        if (!resolved.isEmpty())
        {
            protectedPaths->insert(pathKey(resolved));
        }
        return;
    }

    if (value.isArray())
    {
        for (const QJsonValue &item : value.toArray())
        {
            collectJsonPaths(item, projectPath, protectedPaths);
        }
        return;
    }

    if (value.isObject())
    {
        const QJsonObject object = value.toObject();
        for (auto it = object.begin(); it != object.end(); ++it)
        {
            collectJsonPaths(it.value(), projectPath, protectedPaths);
        }
    }
}

QSet<QString> protectedPathsWithoutTiePoints(const QJsonObject &meta,
                                             const QString &projectPath)
{
    QJsonObject protectedMeta = meta;
    protectedMeta.remove(QStringLiteral("aerial_triangulation_results"));
    QSet<QString> protectedPaths;
    collectJsonPaths(protectedMeta, projectPath, &protectedPaths);
    return protectedPaths;
}

QStringList unprotectedPaths(const QStringList &paths, const QSet<QString> &protectedPaths)
{
    QStringList result;
    for (const QString &path : paths)
    {
        if (!protectedPaths.contains(pathKey(path)))
        {
            result.append(path);
        }
    }
    return result;
}

QString invalidFileCandidate(const QStringList &paths)
{
    for (const QString &path : paths)
    {
        const QFileInfo info(path);
        if (info.exists() && !info.isFile())
        {
            return path;
        }
    }
    return QString();
}

QString missingFileCandidate(const QStringList &paths)
{
    for (const QString &path : paths)
    {
        if (!QFileInfo(path).isFile())
        {
            return path;
        }
    }
    return QString();
}

QStringList removeFiles(const QStringList &paths)
{
    QStringList failedPaths;
    for (const QString &path : paths)
    {
        const QFileInfo info(path);
        if (info.exists() && !QFile::remove(path))
        {
            failedPaths.append(path);
        }
    }
    return failedPaths;
}

void removeEmptyDirectories(QStringList paths, const QSet<QString> &protectedPaths)
{
    std::sort(paths.begin(), paths.end(), [](const QString &left, const QString &right) {
        return left.size() > right.size();
    });

    for (const QString &path : paths)
    {
        if (protectedPaths.contains(pathKey(path)))
        {
            continue;
        }

        QDir dir(path);
        if (dir.exists() && dir.entryList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty())
        {
            QDir().rmdir(path);
        }
    }
}

QStringList collectFiles(const QJsonArray &records,
                         const QString &projectPath,
                         QStringList *directories)
{
    QStringList files;
    for (const QJsonValue &value : records)
    {
        collectRecordArtifacts(value.toObject(), projectPath, &files, directories);
    }
    return files;
}

QStringList collectDepthArtifactFiles(const QJsonArray &records,
                                      const QString &projectPath)
{
    static const QStringList artifactKeys{
        QStringLiteral("depth_png"),
        QStringLiteral("raw_depth_path"),
        QStringLiteral("raw_confidence_path"),
        QStringLiteral("valid_mask_path"),
        QStringLiteral("missing_reason_path"),
        QStringLiteral("missing_reason_preview_path"),
        QStringLiteral("targeted_gap_recovered_mask_path")
    };

    QStringList files;
    for (const QJsonValue &value : records)
    {
        const QJsonObject record = value.toObject();
        for (const QString &key : artifactKeys)
        {
            appendUniquePath(&files, resolveProjectPath(projectPath, record.value(key).toString()));
        }
    }
    return files;
}

} // namespace

bool TiePointResultSelection::isValid() const
{
    return sourceIndex >= 0 && !record.isEmpty() && !sparseCloudPath.isEmpty();
}

TiePointResultSelection ProjectTiePointResultService::selectCurrent(const QJsonObject &meta,
                                                                    const QString &projectPath)
{
    const QJsonArray records = meta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    for (int index = records.size() - 1; index >= 0; --index)
    {
        const QJsonObject record = records.at(index).toObject();
        const QString storedPath = record.value(QStringLiteral("files"))
                                       .toObject()
                                       .value(QStringLiteral("sparse_cloud_xyz"))
                                       .toString();
        const QString resolvedPath = resolveProjectPath(projectPath, storedPath);
        const QFileInfo sparseInfo(resolvedPath);
        if (resolvedPath.isEmpty() || !sparseInfo.isFile())
        {
            continue;
        }

        TiePointResultSelection selection;
        selection.sourceIndex = index;
        selection.pointCount = record.value(QStringLiteral("sparse_point_count")).toInt(-1);
        if (selection.pointCount < 0)
        {
            selection.pointCount = record.value(QStringLiteral("point_count")).toInt(-1);
        }
        if (selection.pointCount < 0)
        {
            selection.pointCount = record.value(QStringLiteral("quality"))
                                       .toObject()
                                       .value(QStringLiteral("point_count"))
                                       .toInt(-1);
        }
        selection.record = record;
        selection.sparseCloudPath = resolvedPath;
        return selection;
    }

    return {};
}

QJsonObject ProjectTiePointResultService::metadataWithCurrentOnly(const QJsonObject &meta,
                                                                  const QString &projectPath)
{
    QJsonObject normalizedMeta = meta;
    const TiePointResultSelection selection = selectCurrent(meta, projectPath);
    normalizedMeta[QStringLiteral("aerial_triangulation_results")] =
        selection.isValid() ? QJsonArray{selection.record} : QJsonArray{};
    return normalizedMeta;
}

TiePointMutationResult ProjectTiePointResultService::replaceCurrent(ProjectData *projectData,
                                                                     const QJsonObject &newRecord)
{
    TiePointMutationResult result;
    if (!projectData || !projectData->hasProject())
    {
        result.errorMessage = QStringLiteral("项目数据未初始化");
        return result;
    }

    QJsonObject recordWithGeneration = newRecord;
    recordWithGeneration[QStringLiteral("reconstruction_generation_id")] =
        QUuid::createUuid().toString(QUuid::WithoutBraces);

    const QString projectPath = projectData->currentProjectPath();
    const QJsonObject validationMeta{
        {QStringLiteral("aerial_triangulation_results"), QJsonArray{recordWithGeneration}}
    };
    if (!selectCurrent(validationMeta, projectPath).isValid())
    {
        result.errorMessage = QStringLiteral("新的连接点文件不存在或不是普通文件");
        return result;
    }

    QStringList newFiles;
    collectRecordArtifacts(recordWithGeneration, projectPath, &newFiles, nullptr);
    const QString missingArtifact = missingFileCandidate(newFiles);
    if (!missingArtifact.isEmpty())
    {
        result.errorMessage = QStringLiteral("新的连接点关联文件不存在：%1").arg(missingArtifact);
        return result;
    }

    const QJsonObject oldMeta = projectData->metadata();
    const QJsonArray oldRecords = oldMeta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    const QStringList oldDepthArtifacts = collectDepthArtifactFiles(
        oldMeta.value(QStringLiteral("depth_map_results")).toArray(),
        projectPath);
    QStringList oldDirectories;
    const QStringList oldFiles = collectFiles(oldRecords, projectPath, &oldDirectories);

    QSet<QString> protectedPaths = protectedPathsWithoutTiePoints(oldMeta, projectPath);
    collectJsonPaths(recordWithGeneration, projectPath, &protectedPaths);
    const QStringList filesToDelete = unprotectedPaths(oldFiles, protectedPaths);

    QJsonObject retainedMeta = oldMeta;
    for (const QString &key : {QStringLiteral("aerial_triangulation_results"),
                               QStringLiteral("depth_map_results"),
                               QStringLiteral("dense_cloud_results"),
                               QStringLiteral("model_results"),
                               QStringLiteral("dem_results"),
                               QStringLiteral("ortho_results")})
    {
        retainedMeta.remove(key);
    }
    QSet<QString> retainedPaths;
    collectJsonPaths(retainedMeta, projectPath, &retainedPaths);
    collectJsonPaths(recordWithGeneration, projectPath, &retainedPaths);
    const QStringList depthArtifactsToDelete = unprotectedPaths(oldDepthArtifacts, retainedPaths);

    QJsonObject updatedMeta = oldMeta;
    updatedMeta[QStringLiteral("aerial_triangulation_results")] =
        QJsonArray{recordWithGeneration};
    for (const QString &key : {QStringLiteral("depth_map_results"),
                               QStringLiteral("dense_cloud_results"),
                               QStringLiteral("model_results"),
                               QStringLiteral("dem_results"),
                               QStringLiteral("ortho_results")})
    {
        updatedMeta[key] = QJsonArray{};
    }
    projectData->updateMetadata(updatedMeta, true);
    projectData->scheduleTemporaryMetadataSave();

    result.removedRecordCount = oldRecords.size();
    result.reconstructionGenerationId =
        recordWithGeneration.value(QStringLiteral("reconstruction_generation_id")).toString();
    result.cleanupWarnings = removeFiles(filesToDelete);
    const QStringList depthCleanupWarnings = removeFiles(depthArtifactsToDelete);
    for (const QString &warning : depthCleanupWarnings)
    {
        appendUniquePath(&result.cleanupWarnings, warning);
    }
    removeEmptyDirectories(oldDirectories, protectedPaths);
    result.success = true;
    return result;
}

TiePointMutationResult ProjectTiePointResultService::deleteAll(ProjectData *projectData)
{
    TiePointMutationResult result;
    if (!projectData || !projectData->hasProject())
    {
        result.errorMessage = QStringLiteral("项目数据未初始化");
        return result;
    }

    const QJsonObject oldMeta = projectData->metadata();
    const QJsonArray oldRecords = oldMeta.value(QStringLiteral("aerial_triangulation_results")).toArray();
    if (oldRecords.isEmpty())
    {
        result.success = true;
        return result;
    }

    const QString projectPath = projectData->currentProjectPath();
    QStringList oldDirectories;
    const QStringList oldFiles = collectFiles(oldRecords, projectPath, &oldDirectories);
    const QSet<QString> protectedPaths = protectedPathsWithoutTiePoints(oldMeta, projectPath);
    const QStringList filesToDelete = unprotectedPaths(oldFiles, protectedPaths);
    const QString invalidPath = invalidFileCandidate(filesToDelete);
    if (!invalidPath.isEmpty())
    {
        result.errorMessage = QStringLiteral("连接点文件路径不是普通文件：%1").arg(invalidPath);
        return result;
    }

    const QStringList failedPaths = removeFiles(filesToDelete);
    if (!failedPaths.isEmpty())
    {
        result.errorMessage = QStringLiteral("无法删除连接点文件：%1").arg(failedPaths.join(QStringLiteral("；")));
        return result;
    }

    QJsonObject newMeta = oldMeta;
    newMeta[QStringLiteral("aerial_triangulation_results")] = QJsonArray{};
    projectData->updateMetadata(newMeta, true);
    projectData->scheduleTemporaryMetadataSave();
    removeEmptyDirectories(oldDirectories, protectedPaths);

    result.success = true;
    result.removedRecordCount = oldRecords.size();
    return result;
}

} // namespace xjw::gui::project
