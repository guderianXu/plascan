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
        return {};
    }
    if (!suffix.startsWith(QLatin1Char('.')))
    {
        suffix.prepend(QLatin1Char('.'));
    }
    return suffix;
}

QSet<QString> collectProjectFeatureSuffixes(const QString &projectPath,
                                            const QJsonObject &metadata)
{
    QSet<QString> suffixes;
    for (const QString &image_path : projectImagePaths(metadata))
    {
        for (const QString &suffix : ProjectIO::availableFeatureSuffixes(
                 projectPath, image_path))
        {
            const QString normalized = normalizedFeatureSuffix(suffix);
            if (!normalized.isEmpty())
            {
                suffixes.insert(normalized);
            }
        }
    }

    const QDir feature_dir(ProjectIO::ipfindOutputDir(projectPath));
    if (!feature_dir.exists())
    {
        return suffixes;
    }

    static const QStringList known_suffixes{
        QStringLiteral(".sp"),
        QStringLiteral(".dsk"),
        QStringLiteral(".alk"),
        QStringLiteral(".sift"),
        QStringLiteral(".orb"),
        QStringLiteral(".akz"),
        QStringLiteral(".dedode")
    };
    const QFileInfoList files = feature_dir.entryInfoList(
        QDir::Files | QDir::NoDotAndDotDot);
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

QString resolveProjectFeaturePathFromToken(const QString &projectPath,
                                           const QJsonObject &metadata,
                                           const QString &token)
{
    const QString image_path =
        resolveProjectImagePathFromToken(token, metadata);
    if (!image_path.isEmpty())
    {
        const QString feature_path =
            ProjectIO::findFeatureForImage(projectPath, image_path);
        if (!feature_path.isEmpty())
        {
            return feature_path;
        }
    }
    return {};
}

QString resolveFeaturePathBySuffix(const QString &projectPath,
                                   const QJsonObject &metadata,
                                   const QString &token,
                                   const QString &suffix)
{
    const QString image_path =
        resolveProjectImagePathFromToken(token, metadata);
    return image_path.isEmpty()
        ? QString()
        : ProjectIO::featureFileForSuffix(
              projectPath, image_path, suffix);
}

QStringList projectFeatureSuffixes(const QString &projectPath,
                                   const QJsonObject &metadata)
{
    QSet<QString> available =
        collectProjectFeatureSuffixes(projectPath, metadata);
    static const QStringList preferred_order{
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

QString inferPreferredFeatureSuffix(const QString &projectPath,
                                    const QJsonObject &metadata)
{
    const QStringList suffixes =
        projectFeatureSuffixes(projectPath, metadata);
    return suffixes.isEmpty() ? QString() : suffixes.first();
}

QString resolvePreferredFeatureSuffix(const QString &projectPath,
                                      const QJsonObject &metadata,
                                      const QString &requestedSuffix,
                                      const QString &fallbackSuffix)
{
    const QString requested = normalizedFeatureSuffix(requestedSuffix);
    const QStringList available =
        projectFeatureSuffixes(projectPath, metadata);
    if (!requested.isEmpty() && available.contains(requested))
    {
        return requested;
    }
    if (!available.isEmpty())
    {
        return available.first();
    }
    const QString fallback = normalizedFeatureSuffix(fallbackSuffix);
    return fallback.isEmpty() ? QStringLiteral(".sift") : fallback;
}

bool projectHasFeatureSuffix(const QString &projectPath,
                             const QJsonObject &metadata,
                             const QString &suffix)
{
    const QString normalized = normalizedFeatureSuffix(suffix);
    return !normalized.isEmpty()
        && collectProjectFeatureSuffixes(projectPath, metadata)
               .contains(normalized);
}

} // namespace xjw::common::project
