#include "ProjectResourceCleanup.h"

#include "project/ProjectSessionModel.h"
#include "project/ProjectIO.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QSet>

#include <algorithm>

namespace xjw::core::project
{
namespace
{

QString normalizedProjectPath(const QString &projectRoot, const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return QString();
    }

    QFileInfo info(path);
    if (info.isAbsolute())
    {
        return QDir::cleanPath(info.absoluteFilePath());
    }
    return QDir::cleanPath(QDir(projectRoot).filePath(path));
}

QString dataSectionToArrayKey(const QString &section)
{
    if (section == QStringLiteral("观测网络")) return QStringLiteral("observation_network_results");
    if (section == QStringLiteral("连接点")) return QStringLiteral("aerial_triangulation_results");
    if (section == QStringLiteral("深度图")) return QStringLiteral("depth_map_results");
    if (section == QStringLiteral("稠密点云")) return QStringLiteral("dense_cloud_results");
    if (section == QStringLiteral("3D模型")) return QStringLiteral("model_results");
    if (section == QStringLiteral("DEM")) return QStringLiteral("dem_results");
    if (section == QStringLiteral("正射影像")) return QStringLiteral("ortho_results");
    return QString();
}

QString primaryPathForSectionRecord(const QString &section,
                                    const QJsonObject &record,
                                    const QString &projectRoot)
{
    QString path;
    if (section == QStringLiteral("连接点"))
    {
        path = record.value(QStringLiteral("files")).toObject().value(QStringLiteral("sparse_cloud_xyz")).toString();
    }
    else if (section == QStringLiteral("深度图"))
    {
        path = record.value(QStringLiteral("depth_png")).toString();
        if (path.isEmpty()) path = record.value(QStringLiteral("raw_depth_path")).toString();
        if (path.isEmpty()) path = record.value(QStringLiteral("valid_mask_path")).toString();
        if (path.isEmpty()) path = record.value(QStringLiteral("preview_path")).toString();
    }
    else if (section == QStringLiteral("稠密点云"))
    {
        path = record.value(QStringLiteral("dense_cloud_xyz")).toString();
    }
    else if (section == QStringLiteral("3D模型"))
    {
        path = record.value(QStringLiteral("final_model_path")).toString();
        if (path.isEmpty()) path = record.value(QStringLiteral("model_obj")).toString();
        if (path.isEmpty()) path = record.value(QStringLiteral("model_ply")).toString();
    }
    else if (section == QStringLiteral("DEM"))
    {
        path = record.value(QStringLiteral("dem_tif")).toString();
    }
    else if (section == QStringLiteral("正射影像"))
    {
        path = record.value(QStringLiteral("output_path")).toString();
    }
    return normalizedProjectPath(projectRoot, path);
}

void appendUniquePath(QStringList *paths, const QString &path)
{
    if (!paths) return;

    const QString clean = QDir::cleanPath(path);
    if (!clean.isEmpty() && !paths->contains(clean))
    {
        paths->append(clean);
    }
}

void collectSectionRecordArtifacts(const QString &section,
                                   const QJsonObject &record,
                                   const QString &projectRoot,
                                   QStringList *filePaths,
                                   QStringList *dirPaths)
{
    if (section == QStringLiteral("连接点"))
    {
        const QJsonObject files = record.value(QStringLiteral("files")).toObject();
        for (auto it = files.begin(); it != files.end(); ++it)
        {
            if (it.value().isString())
            {
                appendUniquePath(filePaths, normalizedProjectPath(projectRoot, it.value().toString()));
            }
        }
        appendUniquePath(dirPaths,
                         normalizedProjectPath(projectRoot,
                                               record.value(QStringLiteral("output_dir")).toString()));
        return;
    }

    if (section == QStringLiteral("深度图"))
    {
        const auto collectDepthPaths = [&](const QJsonObject &depthArtifact)
        {
            for (const QString &key : {
                     QStringLiteral("depth_png"),
                     QStringLiteral("raw_depth_path"),
                     QStringLiteral("raw_confidence_path"),
                     QStringLiteral("raw_support_count_path"),
                     QStringLiteral("raw_uncertainty_path"),
                     QStringLiteral("raw_geometry_support_path"),
                     QStringLiteral("raw_geometry_source_mask_path"),
                     QStringLiteral("raw_inverse_depth_mean_path"),
                     QStringLiteral("raw_inverse_depth_spread_path"),
                     QStringLiteral("raw_adaptive_geometry_support_weight_path"),
                     QStringLiteral("raw_adaptive_geometry_effective_view_count_path"),
                     QStringLiteral("raw_adaptive_geometry_conflict_ratio_path"),
                     QStringLiteral("raw_adaptive_geometry_conflict_weight_path"),
                     QStringLiteral("valid_mask_path"),
                     QStringLiteral("support_mask_path"),
                     QStringLiteral("missing_reason_path"),
                     QStringLiteral("missing_reason_preview_path"),
                     QStringLiteral("normal_map_path"),
                     QStringLiteral("raw_normal_path"),
                     QStringLiteral("preview_path"),
                     QStringLiteral("confidence_preview_path")})
            {
                appendUniquePath(filePaths,
                                 normalizedProjectPath(projectRoot,
                                                       depthArtifact.value(key).toString()));
            }
        };
        collectDepthPaths(record);
        for (const QJsonValue &levelValue : record.value(QStringLiteral("pyramid_levels")).toArray())
        {
            collectDepthPaths(levelValue.toObject());
        }
        return;
    }

    if (section == QStringLiteral("稠密点云"))
    {
        appendUniquePath(filePaths,
                         normalizedProjectPath(projectRoot,
                                               record.value(QStringLiteral("dense_cloud_xyz")).toString()));
        for (const QJsonValue &value : record.value(QStringLiteral("imported_dependencies")).toArray())
        {
            const QString dependency = normalizedProjectPath(projectRoot, value.toString());
            appendUniquePath(filePaths, dependency);
            appendUniquePath(dirPaths, QFileInfo(dependency).absolutePath());
        }
        appendUniquePath(dirPaths,
                         normalizedProjectPath(projectRoot,
                                               record.value(QStringLiteral("import_directory")).toString()));
        return;
    }

    if (section == QStringLiteral("3D模型"))
    {
        const QStringList modelKeys = {
            QStringLiteral("final_model_path"),
            QStringLiteral("model_obj"),
            QStringLiteral("model_mtl"),
            QStringLiteral("model_ply"),
            QStringLiteral("texture_png"),
            QStringLiteral("texture_image")
        };
        for (const QString &key : modelKeys)
        {
            appendUniquePath(filePaths, normalizedProjectPath(projectRoot, record.value(key).toString()));
        }
        for (const QJsonValue &value : record.value(QStringLiteral("imported_dependencies")).toArray())
        {
            const QString dependency = normalizedProjectPath(projectRoot, value.toString());
            appendUniquePath(filePaths, dependency);
            appendUniquePath(dirPaths, QFileInfo(dependency).absolutePath());
        }
        appendUniquePath(dirPaths,
                         normalizedProjectPath(projectRoot,
                                               record.value(QStringLiteral("import_directory")).toString()));

        const QString texturePath =
            normalizedProjectPath(projectRoot, record.value(QStringLiteral("texture_png")).toString());
        if (!texturePath.isEmpty())
        {
            appendUniquePath(dirPaths, QFileInfo(texturePath).absolutePath());
        }
        return;
    }

    if (section == QStringLiteral("DEM"))
    {
        appendUniquePath(filePaths,
                         normalizedProjectPath(projectRoot,
                                               record.value(QStringLiteral("dem_tif")).toString()));
        appendUniquePath(dirPaths,
                         normalizedProjectPath(projectRoot,
                                               record.value(QStringLiteral("output_dir")).toString()));
        return;
    }

    if (section == QStringLiteral("正射影像"))
    {
        appendUniquePath(filePaths,
                         normalizedProjectPath(projectRoot,
                                               record.value(QStringLiteral("output_path")).toString()));
    }
}

bool removeFileIfPresent(const QString &path)
{
    if (path.isEmpty())
    {
        return true;
    }

    const QFileInfo info(path);
    if (!info.exists())
    {
        return true;
    }

    if (info.isDir())
    {
        return QDir(path).removeRecursively();
    }

    return QFile::remove(path);
}

void removeDirectoryIfEmpty(const QString &path)
{
    if (path.isEmpty())
    {
        return;
    }

    QDir dir(path);
    if (!dir.exists())
    {
        return;
    }

    if (dir.entryList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty())
    {
        dir.rmdir(path);
    }
}

} // namespace

ResourceCleanupResult ProjectResourceCleanupService::cleanupGeneratedData(ProjectData *projectData,
                                                                          const QString &section,
                                                                          const QStringList &resourcePaths)
{
    ResourceCleanupResult result;

    if (!projectData)
    {
        result.errorMessage = QStringLiteral("ProjectData 未初始化");
        return result;
    }

    if (resourcePaths.isEmpty())
    {
        result.errorMessage = QStringLiteral("待删除资源为空");
        return result;
    }

    result.sectionArrayKey = dataSectionToArrayKey(section);
    if (result.sectionArrayKey.isEmpty())
    {
        result.unsupportedSection = true;
        return result;
    }

    const QString projectRoot =
        xjw::common::project::ProjectIO::projectRootFromPlascan(
            projectData->currentProjectPath());
    QJsonObject meta = projectData->metadata();
    const QJsonArray sourceArray = meta.value(result.sectionArrayKey).toArray();
    QJsonArray keptArray;
    QStringList filePathsToDelete;
    QStringList dirPathsToTrim;

    QSet<QString> normalizedTargets;
    QSet<int> indexTargets;
    for (const QString &path : resourcePaths)
    {
        if (section == QStringLiteral("观测网络"))
        {
            bool ok = false;
            const int index = path.toInt(&ok);
            if (ok)
            {
                indexTargets.insert(index);
            }
        }
        else
        {
            normalizedTargets.insert(normalizedProjectPath(projectRoot, path));
        }
    }

    for (int index = 0; index < sourceArray.size(); ++index)
    {
        const QJsonObject record = sourceArray.at(index).toObject();
        bool shouldDelete = false;
        if (section == QStringLiteral("观测网络"))
        {
            shouldDelete = indexTargets.contains(index);
        }
        else
        {
            const QString primaryPath = primaryPathForSectionRecord(section, record, projectRoot);
            shouldDelete = normalizedTargets.contains(primaryPath);
        }

        if (!shouldDelete)
        {
            keptArray.append(record);
            continue;
        }

        ++result.removedCount;
        collectSectionRecordArtifacts(section, record, projectRoot, &filePathsToDelete, &dirPathsToTrim);
    }

    if (result.removedCount <= 0)
    {
        result.noMatchedRecords = true;
        return result;
    }

    meta[result.sectionArrayKey] = keptArray;
    projectData->updateMetadata(meta, true);
    projectData->scheduleTemporaryMetadataSave();

    for (const QString &filePath : filePathsToDelete)
    {
        if (!removeFileIfPresent(filePath))
        {
            result.failedPaths.append(filePath);
        }
    }

    std::sort(dirPathsToTrim.begin(), dirPathsToTrim.end(), [](const QString &lhs, const QString &rhs) {
        return lhs.size() > rhs.size();
    });
    dirPathsToTrim.removeDuplicates();

    for (const QString &dirPath : dirPathsToTrim)
    {
        const QFileInfo dirInfo(dirPath);
        if (!dirInfo.exists())
        {
            continue;
        }

        if (section == QStringLiteral("连接点") || section == QStringLiteral("DEM"))
        {
            if (!removeFileIfPresent(dirPath))
            {
                result.failedPaths.append(dirPath);
            }
        }
        else
        {
            removeDirectoryIfEmpty(dirPath);
        }
    }

    result.success = true;
    return result;
}

} // namespace xjw::core::project
