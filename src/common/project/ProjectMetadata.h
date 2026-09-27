#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

namespace xjw::common::project
{

    enum class ImageResolveStatus
    {
        Found,
        NotFound,
        Ambiguous,
        InvalidToken
    };

    struct ImageResolveResult
    {
        ImageResolveStatus status = ImageResolveStatus::NotFound;
        QString path;
        QStringList candidates;
    };

    QString normalizePath(const QString& path);
    QString normalizedImageToken(const QString& token);
    QString imageBaseToken(const QString& token);
    bool imageTokensReferToSameImage(const QString& lhs, const QString& rhs);
    bool imageReferenceMatchesToken(const QString& path_token, const QString& name_token, const QString& candidate);
    bool pathTokenMatchesImage(const QString& token, const QString& image_path);

    QJsonObject projectFilesRootObject(const QJsonObject& metadata);
    QJsonArray projectImageEntries(const QJsonObject& metadata);
    QStringList projectImagePaths(const QJsonObject& metadata);
    QMap<QString, QJsonObject> projectImageMetaByPath(const QJsonObject& metadata, bool normalize_paths = false);

    // Returns the normalized model adapter metadata for an image. Camera model
    // parameters live in camera_definitions/camera_instances; callers should use
    // this accessor instead of reading a camera object embedded in images[].

    ImageResolveResult resolveProjectImageToken(const QString& token, const QJsonObject& metadata);
    ImageResolveResult resolveProjectImageToken(const QString& token, const QStringList& project_image_paths);
    QString resolveProjectImagePathFromToken(const QString& token, const QJsonObject& metadata);
    QString resolveProjectImagePathFromToken(const QString& token, const QStringList& project_image_paths);
} // namespace xjw::common::project
