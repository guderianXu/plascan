#pragma once

#include "ProjectResourceCleanup.h"

#include <QJsonObject>
#include <QString>
#include <QStringList>

class ProjectData;
class ProjectResourceCleanupPersistence;

namespace xjw::core::project::detail
{

struct ResourceCleanupPlan
{
    QJsonObject originalMetadata;
    QJsonObject updatedMetadata;
    QString projectPath;
    QString chunkId;
    int chunkDirectory = 0;
    QString projectRoot;
    QString managedRoot;
    QStringList managedFiles;
    QStringList managedDirectories;
    QStringList preservedExternalPaths;
    QStringList preservedSharedPaths;
    QStringList preservedUnsafePaths;
    int removedCount = 0;
};

QString cleanupSectionArrayKey(const QString &section);

bool buildResourceCleanupPlan(ProjectData *projectData,
                              const QString &section,
                              const QStringList &resourcePaths,
                              ResourceCleanupPlan *plan,
                              QString *errorMessage);

bool executeResourceCleanupPlan(ProjectData *projectData,
                                const ResourceCleanupPlan &plan,
                                ResourceCleanupResult *result);

bool executePreparedResourceCleanupPlan(
    const ProjectResourceCleanupPersistence &persistence,
    const ResourceCleanupPlan &plan,
    ResourceCleanupResult *result);

bool recoverPendingResourceCleanupTransactions(
    ProjectData *projectData,
    ResourceCleanupResult *result);

bool recoverPendingResourceCleanupTransactionsBeforeOpen(
    const QString &projectPath,
    QString *errorMessage);

} // namespace xjw::core::project::detail
