#include "ProjectResourceCleanupArtifacts.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSet>

namespace xjw::core::project::detail
{
namespace
{

QString comparablePath(QString path)
{
    path = QDir::cleanPath(QDir::fromNativeSeparators(path));
#ifdef Q_OS_WIN
    path = path.toCaseFolded();
#endif
    return path;
}

void collectDepthPaths(const QJsonObject &artifact,
                       const QString &projectRoot,
                       QStringList *filePaths)
{
    static const QStringList keys{
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
        QStringLiteral("prepared_image"),
        QStringLiteral("prepared_valid_mask_path"),
        QStringLiteral("valid_mask_path"),
        QStringLiteral("support_mask_path"),
        QStringLiteral("missing_reason_path"),
        QStringLiteral("missing_reason_preview_path"),
        QStringLiteral("targeted_gap_recovered_mask_path"),
        QStringLiteral("depth_provenance_path"),
        QStringLiteral("normal_map_path"),
        QStringLiteral("raw_normal_path"),
        QStringLiteral("preview_path"),
        QStringLiteral("confidence_preview_path")
    };
    for (const QString &key : keys)
    {
        appendUniqueCleanupPath(
            filePaths,
            normalizedCleanupPath(projectRoot, artifact.value(key).toString()));
    }
}

void collectJsonStringPaths(const QJsonValue &value,
                            const QString &projectRoot,
                            QStringList *paths)
{
    if (value.isString())
    {
        const QString text = value.toString().trimmed();
        if (!text.isEmpty())
        {
            appendUniqueCleanupPath(paths,
                                    normalizedCleanupPath(projectRoot, text));
        }
        return;
    }
    if (value.isArray())
    {
        for (const QJsonValue &child : value.toArray())
        {
            collectJsonStringPaths(child, projectRoot, paths);
        }
        return;
    }
    if (value.isObject())
    {
        const QJsonObject object = value.toObject();
        for (auto it = object.constBegin(); it != object.constEnd(); ++it)
        {
            collectJsonStringPaths(it.value(), projectRoot, paths);
        }
    }
}

bool isSafeRunId(const QString &runId)
{
    static const QRegularExpression expression(
        QStringLiteral("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"));
    return expression.match(runId).hasMatch();
}

bool pathContainsParentTraversal(const QString &path)
{
    const QStringList components = QDir::fromNativeSeparators(path).split(
        QLatin1Char('/'), Qt::SkipEmptyParts);
    return components.contains(QLatin1String(".."));
}

bool directoryMatchesRunId(const QString &directory,
                           const QString &runId,
                           const QString &expectedParentName)
{
    const QString trimmedRunId = runId.trimmed();
    if (!isSafeRunId(trimmedRunId)
        || QFileInfo(directory).fileName() != trimmedRunId)
    {
        return false;
    }
    return expectedParentName.isEmpty()
        || QFileInfo(QFileInfo(directory).absolutePath()).fileName()
            == expectedParentName;
}

bool ownershipManifestMatchesDirectory(const QJsonObject &record,
                                       const QString &directory,
                                       const QString &projectRoot,
                                       const QString &runIdKey)
{
    const QString storedManifestPath = record.value(
        QStringLiteral("ownership_manifest_path")).toString();
    if (pathContainsParentTraversal(storedManifestPath))
    {
        return false;
    }
    const QString manifestPath = normalizedCleanupPath(
        projectRoot,
        storedManifestPath);
    const QFileInfo manifestInfo(manifestPath);
    if (manifestPath.isEmpty()
        || !manifestInfo.isFile()
        || manifestInfo.isSymLink()
        || !cleanupPathIsInside(directory, manifestPath, false))
    {
        return false;
    }

    QFile manifestFile(manifestPath);
    constexpr qint64 kMaximumManifestBytes = 1024 * 1024;
    if (manifestInfo.size() <= 0
        || manifestInfo.size() > kMaximumManifestBytes
        || !manifestFile.open(QIODevice::ReadOnly))
    {
        return false;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        manifestFile.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError
        || !document.isObject())
    {
        return false;
    }

    const QJsonObject manifest = document.object();
    if (manifest.value(QStringLiteral("schema_version")).toInt() != 1
        || manifest.value(QStringLiteral("type")).toString()
            != QStringLiteral("plascan_owned_directory"))
    {
        return false;
    }

    const QString storedDeclaredDirectory = manifest.value(
        QStringLiteral("owned_directory")).toString();
    if (pathContainsParentTraversal(storedDeclaredDirectory))
    {
        return false;
    }
    const QString declaredDirectory = normalizedCleanupPath(
        projectRoot, storedDeclaredDirectory);
    const QString recordRunId = record.value(runIdKey).toString().trimmed();
    const QString manifestRunId = manifest.value(
        QStringLiteral("run_id")).toString().trimmed();
    return isSafeRunId(recordRunId)
        && manifestRunId == recordRunId
        && !declaredDirectory.isEmpty()
        && cleanupPathIdentity(declaredDirectory)
            == cleanupPathIdentity(directory);
}

void collectDirectoryCandidate(CleanupRecordArtifacts *artifacts,
                               const QJsonObject &record,
                               const QString &projectRoot,
                               const QString &directoryKey,
                               const QString &runIdKey,
                               const QString &expectedParentName = {})
{
    if (!artifacts)
    {
        return;
    }

    const QString storedDirectory = record.value(directoryKey).toString();
    const QString directory = normalizedCleanupPath(projectRoot,
                                                    storedDirectory);
    if (directory.isEmpty())
    {
        return;
    }

    const bool verified =
        !pathContainsParentTraversal(storedDirectory)
        && ((!expectedParentName.isEmpty()
             && directoryMatchesRunId(directory,
                                      record.value(runIdKey).toString(),
                                      expectedParentName))
            || ownershipManifestMatchesDirectory(record,
                                                 directory,
                                                 projectRoot,
                                                 runIdKey));
    appendUniqueCleanupPath(
        verified ? &artifacts->ownedDirectories
                 : &artifacts->unverifiedDirectories,
        directory);
}

} // namespace

QString normalizedCleanupPath(const QString &projectRoot,
                              const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return {};
    }

    const QFileInfo info(trimmed);
    return QDir::cleanPath(info.isAbsolute()
        ? info.absoluteFilePath()
        : QDir(projectRoot).absoluteFilePath(trimmed));
}

QString cleanupPathIdentity(const QString &path)
{
    if (path.trimmed().isEmpty())
    {
        return {};
    }
    QString current = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    if (current.isEmpty())
    {
        return {};
    }

    QStringList missingComponents;
    while (!QFileInfo::exists(current)
           && !QFileInfo(current).isSymLink())
    {
        const QFileInfo info(current);
        const QString parent = info.absolutePath();
        if (parent == current || info.fileName().isEmpty())
        {
            break;
        }
        missingComponents.prepend(info.fileName());
        current = parent;
    }

    const QString canonical = QFileInfo(current).canonicalFilePath();
    QString identity = canonical.isEmpty()
        ? QDir::cleanPath(QFileInfo(current).absoluteFilePath())
        : QDir::cleanPath(canonical);
    for (const QString &component : missingComponents)
    {
        identity = QDir(identity).filePath(component);
    }
    return comparablePath(identity);
}

bool cleanupPathIsInside(const QString &directoryPath,
                         const QString &path,
                         bool allowEqual)
{
    const QString directoryIdentity = cleanupPathIdentity(directoryPath);
    const QString pathIdentity = cleanupPathIdentity(path);
    if (directoryIdentity.isEmpty() || pathIdentity.isEmpty())
    {
        return false;
    }
    if (directoryIdentity == pathIdentity)
    {
        return allowEqual;
    }

    const QString relative = QDir::fromNativeSeparators(
        QDir(directoryIdentity).relativeFilePath(pathIdentity));
    return relative != QLatin1String("..")
        && !relative.startsWith(QLatin1String("../"))
        && !QFileInfo(relative).isAbsolute();
}

bool cleanupPathTraversesLink(const QString &managedRoot,
                              const QString &path)
{
    const QString root = QDir::cleanPath(
        QFileInfo(managedRoot).absoluteFilePath());
    const QString absolutePath = QDir::cleanPath(
        QFileInfo(path).absoluteFilePath());
    const QString relative = QDir::fromNativeSeparators(
        QDir(root).relativeFilePath(absolutePath));
    if (relative == QLatin1String("..")
        || relative.startsWith(QLatin1String("../"))
        || QFileInfo(relative).isAbsolute())
    {
        return false;
    }

    QString current = root;
    const QStringList components = relative.split(
        QLatin1Char('/'), Qt::SkipEmptyParts);
    for (const QString &component : components)
    {
        if (component == QLatin1String("."))
        {
            continue;
        }
        current = QDir(current).filePath(component);
        const QFileInfo info(current);
        bool isLink = info.isSymLink();
#ifdef Q_OS_WIN
        isLink = isLink || info.isJunction();
#endif
        if (isLink)
        {
            return true;
        }
    }
    return false;
}

bool cleanupDirectoryContainsLink(const QString &directoryPath)
{
    const QFileInfo rootInfo(directoryPath);
    if (!rootInfo.exists() || !rootInfo.isDir())
    {
        return false;
    }

    QStringList pending{rootInfo.absoluteFilePath()};
    QSet<QString> visited;
    while (!pending.isEmpty())
    {
        const QString current = pending.takeLast();
        const QString identity = cleanupPathIdentity(current);
        if (identity.isEmpty() || visited.contains(identity))
        {
            return true;
        }
        visited.insert(identity);

        const QDir directory(current);
        if (!directory.isReadable())
        {
            return true;
        }
        const QFileInfoList entries = directory.entryInfoList(
            QDir::AllEntries | QDir::NoDotAndDotDot
                | QDir::Hidden | QDir::System);
        for (const QFileInfo &entry : entries)
        {
            bool isLink = entry.isSymLink();
#ifdef Q_OS_WIN
            isLink = isLink || entry.isJunction();
#endif
            if (isLink)
            {
                return true;
            }
            if (entry.isDir())
            {
                pending.append(entry.absoluteFilePath());
            }
        }
    }
    return false;
}

QString cleanupPrimaryPath(const QString &section,
                           const QJsonObject &record,
                           const QString &projectRoot)
{
    QString path;
    if (section == QStringLiteral("连接点"))
    {
        path = record.value(QStringLiteral("files")).toObject()
            .value(QStringLiteral("sparse_cloud_xyz")).toString();
    }
    else if (section == QStringLiteral("深度图"))
    {
        path = record.value(QStringLiteral("depth_png")).toString();
        if (path.isEmpty())
        {
            path = record.value(QStringLiteral("raw_depth_path")).toString();
        }
        if (path.isEmpty())
        {
            path = record.value(QStringLiteral("valid_mask_path")).toString();
        }
        if (path.isEmpty())
        {
            path = record.value(QStringLiteral("preview_path")).toString();
        }
    }
    else if (section == QStringLiteral("稠密点云"))
    {
        path = record.value(QStringLiteral("dense_cloud_xyz")).toString();
    }
    else if (section == QStringLiteral("3D模型"))
    {
        path = record.value(QStringLiteral("final_model_path")).toString();
        if (path.isEmpty())
        {
            path = record.value(QStringLiteral("model_obj")).toString();
        }
        if (path.isEmpty())
        {
            path = record.value(QStringLiteral("model_ply")).toString();
        }
    }
    else if (section == QStringLiteral("DEM"))
    {
        path = record.value(QStringLiteral("dem_tif")).toString();
    }
    else if (section == QStringLiteral("正射影像"))
    {
        path = record.value(QStringLiteral("output_path")).toString();
    }
    return normalizedCleanupPath(projectRoot, path);
}

CleanupRecordArtifacts collectCleanupRecordArtifacts(
    const QString &section,
    const QJsonObject &record,
    const QString &projectRoot)
{
    CleanupRecordArtifacts artifacts;
    if (section == QStringLiteral("连接点"))
    {
        const QJsonObject files = record.value(QStringLiteral("files")).toObject();
        for (auto it = files.constBegin(); it != files.constEnd(); ++it)
        {
            if (it.value().isString())
            {
                appendUniqueCleanupPath(
                    &artifacts.files,
                    normalizedCleanupPath(projectRoot, it.value().toString()));
            }
        }
        collectDirectoryCandidate(&artifacts,
                                  record,
                                  projectRoot,
                                  QStringLiteral("output_dir"),
                                  QStringLiteral("run_id"));
    }
    else if (section == QStringLiteral("深度图"))
    {
        collectDepthPaths(record, projectRoot, &artifacts.files);
        for (const QJsonValue &level : record.value(
                 QStringLiteral("pyramid_levels")).toArray())
        {
            collectDepthPaths(level.toObject(), projectRoot, &artifacts.files);
        }
    }
    else if (section == QStringLiteral("稠密点云"))
    {
        appendUniqueCleanupPath(
            &artifacts.files,
            normalizedCleanupPath(
                projectRoot,
                record.value(QStringLiteral("dense_cloud_xyz")).toString()));
        for (const QJsonValue &value : record.value(
                 QStringLiteral("imported_dependencies")).toArray())
        {
            appendUniqueCleanupPath(
                &artifacts.files,
                normalizedCleanupPath(projectRoot, value.toString()));
        }
        collectDirectoryCandidate(&artifacts,
                                  record,
                                  projectRoot,
                                  QStringLiteral("import_directory"),
                                  QStringLiteral("run_id"));
    }
    else if (section == QStringLiteral("3D模型"))
    {
        static const QStringList modelKeys{
            QStringLiteral("final_model_path"),
            QStringLiteral("model_obj"),
            QStringLiteral("model_mtl"),
            QStringLiteral("model_ply"),
            QStringLiteral("texture_png"),
            QStringLiteral("texture_image"),
            QStringLiteral("model_diagnostics_path"),
            QStringLiteral("texture_diagnostics_path")
        };
        for (const QString &key : modelKeys)
        {
            appendUniqueCleanupPath(
                &artifacts.files,
                normalizedCleanupPath(projectRoot, record.value(key).toString()));
        }
        for (const QJsonValue &value : record.value(
                 QStringLiteral("imported_dependencies")).toArray())
        {
            appendUniqueCleanupPath(
                &artifacts.files,
                normalizedCleanupPath(projectRoot, value.toString()));
        }
        collectDirectoryCandidate(&artifacts,
                                  record,
                                  projectRoot,
                                  QStringLiteral("import_directory"),
                                  QStringLiteral("run_id"));
        collectDirectoryCandidate(&artifacts,
                                  record,
                                  projectRoot,
                                  QStringLiteral("model_run_directory"),
                                  QStringLiteral("model_run_id"),
                                  QStringLiteral("model_runs"));
        collectDirectoryCandidate(&artifacts,
                                  record,
                                  projectRoot,
                                  QStringLiteral("texture_run_directory"),
                                  QStringLiteral("texture_run_id"),
                                  QStringLiteral("texture_runs"));
    }
    else if (section == QStringLiteral("DEM"))
    {
        appendUniqueCleanupPath(
            &artifacts.files,
            normalizedCleanupPath(
                projectRoot,
                record.value(QStringLiteral("dem_tif")).toString()));
        collectDirectoryCandidate(&artifacts,
                                  record,
                                  projectRoot,
                                  QStringLiteral("output_dir"),
                                  QStringLiteral("run_id"));
    }
    else if (section == QStringLiteral("正射影像"))
    {
        appendUniqueCleanupPath(
            &artifacts.files,
            normalizedCleanupPath(
                projectRoot,
                record.value(QStringLiteral("output_path")).toString()));
    }
    return artifacts;
}

void collectCleanupStringPaths(const QJsonValue &value,
                               const QString &projectRoot,
                               QStringList *paths)
{
    collectJsonStringPaths(value, projectRoot, paths);
}

bool cleanupPathIsProtected(const QString &managedRoot,
                            const QString &path)
{
    const QString identity = cleanupPathIdentity(path);
    if (identity.isEmpty()
        || identity == cleanupPathIdentity(managedRoot))
    {
        return true;
    }

    const QString chunkArchive = QDir(managedRoot).filePath(
        QStringLiteral("chunk.zip"));
    if (identity == cleanupPathIdentity(chunkArchive))
    {
        return true;
    }

    for (const QString &relative : {
             QStringLiteral(".plascan_tmp"),
             QStringLiteral(".plascan_cleanup_trash"),
             QStringLiteral(".plascan_cleanup_purging")})
    {
        if (cleanupPathIsInside(QDir(managedRoot).filePath(relative),
                                path,
                                true))
        {
            return true;
        }
    }

    for (const QString &relative : {
             QStringLiteral("assets"),
             QStringLiteral("bundle_adjust"),
             QStringLiteral("reconstruction"),
             QStringLiteral("reports")})
    {
        if (identity == cleanupPathIdentity(
                QDir(managedRoot).filePath(relative)))
        {
            return true;
        }
    }
    return false;
}

void appendUniqueCleanupPath(QStringList *paths, const QString &path)
{
    if (!paths)
    {
        return;
    }
    const QString clean = QDir::cleanPath(path);
    if (!clean.isEmpty() && clean != QLatin1String(".")
        && !paths->contains(clean))
    {
        paths->append(clean);
    }
}

} // namespace xjw::core::project::detail
