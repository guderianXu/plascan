#include "ProjectMetadata.h"

#include <QDir>
#include <QFileInfo>
#include <QSet>

namespace xjw::common::project
{

namespace
{

QStringList uniqueCandidates(const QStringList &candidates)
{
    QStringList unique;
    QSet<QString> keys;
    for (const QString &candidate : candidates)
    {
        const QString key = normalizedImageToken(candidate);
        if (!key.isEmpty() && !keys.contains(key))
        {
            keys.insert(key);
            unique.append(candidate);
        }
    }
    return unique;
}

ImageResolveResult resultForCandidates(const QStringList &candidates)
{
    const QStringList unique = uniqueCandidates(candidates);
    if (unique.size() == 1)
    {
        return {ImageResolveStatus::Found, unique.first(), unique};
    }
    if (unique.size() > 1)
    {
        return {ImageResolveStatus::Ambiguous, {}, unique};
    }
    return {ImageResolveStatus::NotFound, {}, {}};
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
    return normalized_lhs == normalized_rhs;
}

bool imageReferenceMatchesToken(const QString &path_token,
                                const QString &name_token,
                                const QString &candidate)
{
    if (candidate.trimmed().isEmpty())
    {
        return false;
    }
    if (!path_token.trimmed().isEmpty())
    {
        return imageTokensReferToSameImage(path_token, candidate);
    }
    return imageTokensReferToSameImage(name_token, candidate);
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
    return resolveProjectImageToken(token, metadata).path;
}

ImageResolveResult resolveProjectImageToken(const QString &token,
                                            const QJsonObject &metadata)
{
    const QString trimmed_token = token.trimmed();
    if (trimmed_token.isEmpty())
    {
        return {ImageResolveStatus::InvalidToken, {}, {}};
    }

    QStringList uuid_matches;
    for (const QJsonValue &value : projectImageEntries(metadata))
    {
        const QJsonObject image = value.toObject();
        if (image.value(QStringLiteral("image_uuid")).toString().trimmed()
            == trimmed_token)
        {
            const QString path = image.value(QStringLiteral("path")).toString();
            if (!path.isEmpty())
            {
                uuid_matches.append(path);
            }
        }
    }
    if (!uuid_matches.isEmpty())
    {
        return resultForCandidates(uuid_matches);
    }
    return resolveProjectImageToken(token, projectImagePaths(metadata));
}

ImageResolveResult resolveProjectImageToken(
    const QString &token,
    const QStringList &project_image_paths)
{
    if (token.trimmed().isEmpty())
    {
        return {ImageResolveStatus::InvalidToken, {}, {}};
    }

    QStringList exact_matches;
    const QString normalized_token = normalizedImageToken(token);
    for (const QString &image_path : project_image_paths)
    {
        if (normalizedImageToken(image_path) == normalized_token)
        {
            exact_matches.append(image_path);
        }
    }
    if (!exact_matches.isEmpty())
    {
        return resultForCandidates(exact_matches);
    }
    if (token.contains(QLatin1Char('/')) || token.contains(QLatin1Char('\\')))
    {
        return {ImageResolveStatus::NotFound, {}, {}};
    }

    QStringList file_name_matches;
    const QString token_file_name =
        QFileInfo(QDir::fromNativeSeparators(token.trimmed()))
            .fileName().toCaseFolded();
    for (const QString &image_path : project_image_paths)
    {
        if (QFileInfo(QDir::fromNativeSeparators(image_path))
                .fileName().toCaseFolded() == token_file_name)
        {
            file_name_matches.append(image_path);
        }
    }
    if (!file_name_matches.isEmpty())
    {
        return resultForCandidates(file_name_matches);
    }

    QStringList stem_matches;
    const QString token_base = imageBaseToken(token);
    for (const QString &image_path : project_image_paths)
    {
        if (!token_base.isEmpty() && imageBaseToken(image_path) == token_base)
        {
            stem_matches.append(image_path);
        }
    }
    return resultForCandidates(stem_matches);
}

QString resolveProjectImagePathFromToken(
    const QString &token,
    const QStringList &project_image_paths)
{
    return resolveProjectImageToken(token, project_image_paths).path;
}

} // namespace xjw::common::project
