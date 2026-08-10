#pragma once

#include <QJsonObject>
#include <QJsonValue>
#include <QString>
#include <QStringList>

namespace xjw::core::project::detail
{

struct CleanupRecordArtifacts
{
    QStringList files;
    QStringList ownedDirectories;
    // Directory fields without a matching run ID or validated ownership
    // manifest are references only and must never be recursively removed.
    QStringList unverifiedDirectories;
};

QString normalizedCleanupPath(const QString &projectRoot,
                              const QString &path);
QString cleanupPathIdentity(const QString &path);
bool cleanupPathIsInside(const QString &directoryPath,
                         const QString &path,
                         bool allowEqual = false);
bool cleanupPathTraversesLink(const QString &managedRoot,
                              const QString &path);
bool cleanupDirectoryContainsLink(const QString &directoryPath);

QString cleanupPrimaryPath(const QString &section,
                           const QJsonObject &record,
                           const QString &projectRoot);
CleanupRecordArtifacts collectCleanupRecordArtifacts(
    const QString &section,
    const QJsonObject &record,
    const QString &projectRoot);

void collectCleanupStringPaths(const QJsonValue &value,
                               const QString &projectRoot,
                               QStringList *paths);

bool cleanupPathIsProtected(const QString &managedRoot,
                            const QString &path);
void appendUniqueCleanupPath(QStringList *paths, const QString &path);

} // namespace xjw::core::project::detail
