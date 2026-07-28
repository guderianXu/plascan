#include "ProjectMetadata.h"

#include "ProjectIO.h"

#include <QDir>
#include <QFileInfo>
#include <QSet>

#include <algorithm>

namespace xjw::common::project
{

namespace
{

QString normalizedFeatureSuffix(QString suffix)
{
    suffix = suffix.trimmed().toLower();
    if (suffix.isEmpty())
    {
        return QString();
    }
    if (!suffix.startsWith(QLatin1Char('.')))
    {
        suffix.prepend(QLatin1Char('.'));
    }
    return suffix;
}

QSet<QString> collectProjectFeatureSuffixes(const QString &project_path,
                                            const QJsonObject &metadata)
{
    QSet<QString> suffixes;
    for (const QString &image_path : projectImagePaths(metadata))
    {
        for (const QString &suffix : ProjectIO::availableFeatureSuffixes(project_path,
                                                                          image_path))
        {
            const QString normalized = normalizedFeatureSuffix(suffix);
            if (!normalized.isEmpty())
            {
                suffixes.insert(normalized);
            }
        }
    }

    const QDir feature_dir(ProjectIO::ipfindOutputDir(project_path));
    if (!feature_dir.exists())
    {
        return suffixes;
    }

    static const QStringList known_suffixes = {
        QStringLiteral(".sp"),
        QStringLiteral(".dsk"),
        QStringLiteral(".alk"),
        QStringLiteral(".sift"),
        QStringLiteral(".orb"),
        QStringLiteral(".akz"),
        QStringLiteral(".dedode")
    };
    const QFileInfoList files = feature_dir.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);
    for (const QFileInfo &file_info : files)
    {
        const QString file_name = file_info.fileName().toLower();
        for (const QString &suffix : known_suffixes)
        {
            if (file_name.endsWith(suffix))
            {
                suffixes.insert(suffix);
            }
        }
    }
    return suffixes;
}

} // namespace

QString normalizePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

QString normalizedImageToken(const QString &token)
{
    const QString trimmed = token.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }

    QString normalized = QDir::cleanPath(QDir::fromNativeSeparators(trimmed));
    normalized.replace(QLatin1Char('\\'), QLatin1Char('/'));
    return normalized.toCaseFolded();
}

QString imageBaseToken(const QString &token)
{
    const QString trimmed = QDir::fromNativeSeparators(token.trimmed());
    if (trimmed.isEmpty())
    {
        return QString();
    }

    const QFileInfo info(trimmed);
    const QString base = info.completeBaseName();
    return (base.isEmpty() ? info.fileName() : base).toCaseFolded();
}

bool imageTokensReferToSameImage(const QString &lhs, const QString &rhs)
{
    const QString normalized_lhs = normalizedImageToken(lhs);
    const QString normalized_rhs = normalizedImageToken(rhs);
    if (normalized_lhs.isEmpty() || normalized_rhs.isEmpty())
    {
        return false;
    }
    if (normalized_lhs == normalized_rhs)
    {
        return true;
    }

    const QFileInfo lhs_info(QDir::fromNativeSeparators(lhs.trimmed()));
    const QFileInfo rhs_info(QDir::fromNativeSeparators(rhs.trimmed()));
    if (lhs_info.fileName().toCaseFolded() == rhs_info.fileName().toCaseFolded())
    {
        return true;
    }
    return imageBaseToken(lhs) == imageBaseToken(rhs);
}

bool imageReferenceMatchesToken(const QString &path_token,
                                const QString &name_token,
                                const QString &candidate)
{
    if (candidate.trimmed().isEmpty())
    {
        return false;
    }
    return imageTokensReferToSameImage(path_token, candidate) ||
           imageTokensReferToSameImage(name_token, candidate);
}

bool pathTokenMatchesImage(const QString &token, const QString &image_path)
{
    return imageTokensReferToSameImage(token, image_path);
}

QJsonObject projectFilesRootObject(const QJsonObject &metadata)
{
    if (metadata.value(QStringLiteral("project_files")).isObject())
    {
        return metadata.value(QStringLiteral("project_files")).toObject();
    }
    return metadata;
}

QJsonArray projectImageEntries(const QJsonObject &metadata)
{
    return projectFilesRootObject(metadata).value(QStringLiteral("images")).toArray();
}

QStringList projectImagePaths(const QJsonObject &metadata)
{
    QStringList image_paths;
    const QJsonArray entries = projectImageEntries(metadata);
    image_paths.reserve(entries.size());
    for (const QJsonValue &value : entries)
    {
        const QString image_path = value.toObject().value(QStringLiteral("path")).toString();
        if (!image_path.isEmpty())
        {
            image_paths.push_back(image_path);
        }
    }
    return image_paths;
}

QMap<QString, QJsonObject> projectImageMetaByPath(const QJsonObject &metadata,
                                                  bool normalize_paths)
{
    QMap<QString, QJsonObject> result;
    for (const QJsonValue &value : projectImageEntries(metadata))
    {
        const QJsonObject image = value.toObject();
        QString image_path = image.value(QStringLiteral("path")).toString();
        if (image_path.isEmpty())
        {
            continue;
        }
        if (normalize_paths)
        {
            image_path = normalizePath(image_path);
        }
        result.insert(image_path, image);
    }
    return result;
}

QString resolveProjectImagePathFromToken(const QString &token,
                                         const QJsonObject &metadata)
{
    return resolveProjectImagePathFromToken(token, projectImagePaths(metadata));
}

QString resolveProjectImagePathFromToken(const QString &token,
                                         const QStringList &project_image_paths)
{
    if (token.isEmpty())
    {
        return QString();
    }
    for (const QString &image_path : project_image_paths)
    {
        if (pathTokenMatchesImage(token, image_path))
        {
            return image_path;
        }
    }
    return QString();
}

QString resolveProjectFeaturePathFromToken(const QString &project_path,
                                           const QJsonObject &metadata,
                                           const QString &token)
{
    const QString image_path = resolveProjectImagePathFromToken(token, metadata);
    if (!image_path.isEmpty())
    {
        const QString feature_path = ProjectIO::findFeatureForImage(project_path, image_path);
        if (!feature_path.isEmpty())
        {
            return feature_path;
        }
    }
    return ProjectIO::findFeatureForImage(project_path, token);
}

QString resolveFeaturePathBySuffix(const QString &project_path,
                                   const QJsonObject &metadata,
                                   const QString &token,
                                   const QString &suffix)
{
    const QString image_path = resolveProjectImagePathFromToken(token, metadata);
    if (image_path.isEmpty())
    {
        return QString();
    }
    return ProjectIO::featureFileForSuffix(project_path, image_path, suffix);
}

QStringList projectFeatureSuffixes(const QString &project_path,
                                   const QJsonObject &metadata)
{
    QSet<QString> available = collectProjectFeatureSuffixes(project_path, metadata);
    static const QStringList preferred_order = {
        QStringLiteral(".sift"),
        QStringLiteral(".dsk"),
        QStringLiteral(".alk"),
        QStringLiteral(".sp"),
        QStringLiteral(".orb"),
        QStringLiteral(".akz"),
        QStringLiteral(".dedode")
    };

    QStringList ordered;
    for (const QString &suffix : preferred_order)
    {
        if (available.remove(suffix))
        {
            ordered.append(suffix);
        }
    }
    QStringList extras = available.values();
    std::sort(extras.begin(), extras.end());
    ordered.append(extras);
    return ordered;
}

QString inferPreferredFeatureSuffix(const QString &project_path,
                                    const QJsonObject &metadata)
{
    const QStringList suffixes = projectFeatureSuffixes(project_path, metadata);
    return suffixes.isEmpty() ? QString() : suffixes.first();
}

QString resolvePreferredFeatureSuffix(const QString &project_path,
                                      const QJsonObject &metadata,
                                      const QString &requested_suffix,
                                      const QString &fallback_suffix)
{
    const QString requested = normalizedFeatureSuffix(requested_suffix);
    const QStringList available = projectFeatureSuffixes(project_path, metadata);
    if (!requested.isEmpty() && available.contains(requested))
    {
        return requested;
    }
    if (!available.isEmpty())
    {
        return available.first();
    }
    const QString fallback = normalizedFeatureSuffix(fallback_suffix);
    return fallback.isEmpty() ? QStringLiteral(".sift") : fallback;
}

bool projectHasFeatureSuffix(const QString &project_path,
                             const QJsonObject &metadata,
                             const QString &suffix)
{
    const QString normalized = normalizedFeatureSuffix(suffix);
    return !normalized.isEmpty()
        && collectProjectFeatureSuffixes(project_path, metadata).contains(normalized);
}

} // namespace xjw::common::project
