#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

namespace xjw::common::project
{

QString normalizePath(const QString &path);
QString normalizedImageToken(const QString &token);
QString imageBaseToken(const QString &token);
bool imageTokensReferToSameImage(const QString &lhs, const QString &rhs);
bool imageReferenceMatchesToken(const QString &path_token,
                                const QString &name_token,
                                const QString &candidate);
bool pathTokenMatchesImage(const QString &token, const QString &image_path);

QJsonObject projectFilesRootObject(const QJsonObject &metadata);
QJsonArray projectImageEntries(const QJsonObject &metadata);
QStringList projectImagePaths(const QJsonObject &metadata);
QMap<QString, QJsonObject> projectImageMetaByPath(const QJsonObject &metadata,
                                                  bool normalize_paths = false);

QString resolveProjectImagePathFromToken(const QString &token,
                                         const QJsonObject &metadata);
QString resolveProjectFeaturePathFromToken(const QString &project_path,
                                           const QJsonObject &metadata,
                                           const QString &token);
QString resolveFeaturePathBySuffix(const QString &project_path,
                                   const QJsonObject &metadata,
                                   const QString &token,
                                   const QString &suffix);
QStringList projectFeatureSuffixes(const QString &project_path,
                                   const QJsonObject &metadata);
QString inferPreferredFeatureSuffix(const QString &project_path,
                                    const QJsonObject &metadata);
QString resolvePreferredFeatureSuffix(const QString &project_path,
                                      const QJsonObject &metadata,
                                      const QString &requested_suffix,
                                      const QString &fallback_suffix = QString());
bool projectHasFeatureSuffix(const QString &project_path,
                             const QJsonObject &metadata,
                             const QString &suffix);

} // namespace xjw::common::project
