#include "ProjectResourceCleanupTransaction.h"

#include "ProjectResourceCleanupArtifacts.h"

#include "project/ProjectIO.h"
#include "project/ProjectSessionModel.h"

#include <QDir>
#include <QFileInfo>
#include <QList>
#include <QPair>

#include <functional>

namespace xjw::core::project::detail
{
namespace
{

bool renamePath(const QString &source, const QString &destination)
{
    return QDir().rename(source, destination);
}

bool validateStagingSource(const ResourceCleanupPlan &plan,
                           const QString &path,
                           bool expectedDirectory)
{
    const QFileInfo info(path);
    if (!cleanupPathIsInside(plan.managedRoot, path, false)
        || cleanupPathIsProtected(plan.managedRoot, path)
        || cleanupPathTraversesLink(plan.managedRoot, path))
    {
        return false;
    }
    if (expectedDirectory)
    {
        return info.isDir()
            && !info.isSymLink()
            && !cleanupDirectoryContainsLink(path);
    }
    return !info.isDir() && !info.isSymLink();
}

bool stagePaths(const ResourceCleanupPlan &plan,
                const QString &transactionRoot,
                CleanupTransactionManifest *manifest,
                QString *errorMessage,
                QStringList *failedPaths)
{
    QList<QPair<QString, bool>> sources;
    for (const QString &directory : plan.managedDirectories)
    {
        sources.append({directory, true});
    }
    for (const QString &file : plan.managedFiles)
    {
        sources.append({file, false});
    }

    int sequence = 0;
    for (const auto &source : sources)
    {
        const QFileInfo info(source.first);
        if (!info.exists() && !info.isSymLink())
        {
            continue;
        }
        if (!validateStagingSource(plan, source.first, source.second))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "清理路径在执行前发生变化，已拒绝操作：%1")
                                    .arg(source.first);
            }
            appendUniqueCleanupPath(failedPaths, source.first);
            return false;
        }

        CleanupTransactionMove move;
        move.source = source.first;
        move.destination = QDir(transactionRoot).filePath(
            QStringLiteral("%1_%2")
                .arg(sequence++, 6, 10, QLatin1Char('0'))
                .arg(info.fileName()));
        move.directory = source.second;
        if (!appendPlannedCleanupMove(transactionRoot,
                                      manifest,
                                      move,
                                      errorMessage))
        {
            appendUniqueCleanupPath(failedPaths, source.first);
            return false;
        }

        const int moveIndex = manifest->moves.size() - 1;
        if (!validateStagingSource(plan, source.first, source.second)
            || QFileInfo::exists(move.destination)
            || QFileInfo(move.destination).isSymLink()
            || !renamePath(source.first, move.destination))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "无法将受管产物移入清理事务区：%1")
                                    .arg(source.first);
            }
            appendUniqueCleanupPath(failedPaths, source.first);
            return false;
        }
        if (!markCleanupMoveStaged(transactionRoot,
                                   manifest,
                                   moveIndex,
                                   errorMessage))
        {
            appendUniqueCleanupPath(failedPaths, source.first);
            return false;
        }
    }
    return true;
}

using MetadataCommit = std::function<bool(QString *, bool *)>;

bool rollbackCleanup(const MetadataCommit &commitOriginal,
                     const QString &transactionRoot,
                     const CleanupTransactionManifest &manifest,
                     ResourceCleanupResult *result,
                     bool *metadataFullyCommitted,
                     QString *metadataError)
{
    if (metadataFullyCommitted)
    {
        *metadataFullyCommitted = false;
    }
    QString artifactError;
    const bool artifactsRestored = transactionRoot.isEmpty()
        || restoreStagedCleanupTransaction(transactionRoot,
                                           manifest,
                                           &result->failedPaths,
                                           &artifactError);
    bool metadataRestored = false;
    bool archiveRestored = false;
    if (artifactsRestored && commitOriginal)
    {
        metadataRestored = commitOriginal(metadataError, &archiveRestored);
        if (metadataFullyCommitted)
        {
            *metadataFullyCommitted = metadataRestored;
        }
    }
    const bool metadataDurable = metadataRestored || archiveRestored;
    QString transactionError;
    const bool transactionRemoved = transactionRoot.isEmpty()
        || (artifactsRestored
            && metadataDurable
            && purgeCommittedCleanupTransaction(
                transactionRoot,
                manifest,
                &result->failedPaths,
                &transactionError));
    if (!artifactError.isEmpty())
    {
        if (metadataError && !metadataError->isEmpty())
        {
            *metadataError += QStringLiteral("; ");
        }
        if (metadataError)
        {
            *metadataError += artifactError;
        }
    }
    if (!transactionError.isEmpty())
    {
        if (metadataError && !metadataError->isEmpty())
        {
            *metadataError += QStringLiteral("; ");
        }
        if (metadataError)
        {
            *metadataError += transactionError;
        }
    }
    return artifactsRestored && metadataDurable && transactionRemoved;
}

QString rollbackStateMessage(bool rollbackSucceeded,
                             const QString &transactionRoot)
{
    if (rollbackSucceeded)
    {
        return QStringLiteral("产物和元数据已回滚");
    }
    if (transactionRoot.isEmpty())
    {
        return QStringLiteral("内存元数据已回滚，但持久化恢复失败");
    }
    return QStringLiteral(
        "恢复清单仍保留在项目事务区");
}

bool executeWithMetadataCommits(
    const ResourceCleanupPlan &plan,
    ResourceCleanupResult *result,
    const MetadataCommit &commitUpdated,
    const MetadataCommit &commitOriginal,
    bool metadataPrepared)
{
    if (!result || !commitUpdated || !commitOriginal)
    {
        return false;
    }

    QString transactionRoot;
    CleanupTransactionManifest manifest;
    const bool hasManagedArtifacts = !plan.managedFiles.isEmpty()
        || !plan.managedDirectories.isEmpty();
    if (hasManagedArtifacts
        && !createCleanupTransaction(plan,
                                     &transactionRoot,
                                     &manifest,
                                     &result->errorMessage))
    {
        if (metadataPrepared)
        {
            QString metadataError;
            result->metadataStateCommitted = commitOriginal(
                &metadataError, nullptr);
            if (!result->metadataStateCommitted)
            {
                result->errorMessage += QStringLiteral(
                    "; 原元数据恢复失败：%1").arg(metadataError);
            }
        }
        return false;
    }

    if (hasManagedArtifacts
        && !stagePaths(plan,
                       transactionRoot,
                       &manifest,
                       &result->errorMessage,
                       &result->failedPaths))
    {
        QString rollbackError;
        bool metadataFullyCommitted = false;
        const bool rollbackSucceeded = rollbackCleanup(
            commitOriginal,
            transactionRoot,
            manifest,
            result,
            &metadataFullyCommitted,
            &rollbackError);
        result->metadataStateCommitted = metadataFullyCommitted;
        if (!rollbackSucceeded)
        {
            if (!result->errorMessage.isEmpty())
            {
                result->errorMessage += QStringLiteral("; ");
            }
            result->errorMessage += QStringLiteral("回滚失败：%1")
                                        .arg(rollbackError);
        }
        return false;
    }

    QString persistenceError;
    if (!commitUpdated(&persistenceError, nullptr))
    {
        QString rollbackError;
        bool metadataFullyCommitted = false;
        const bool rollbackSucceeded = rollbackCleanup(
            commitOriginal,
            transactionRoot,
            manifest,
            result,
            &metadataFullyCommitted,
            &rollbackError);
        result->metadataStateCommitted = metadataFullyCommitted;
        result->errorMessage = QStringLiteral(
            "无法提交清理后的项目元数据，%1：%2")
                                   .arg(rollbackStateMessage(
                                            rollbackSucceeded,
                                            transactionRoot),
                                        persistenceError);
        if (!rollbackError.isEmpty())
        {
            result->errorMessage += QStringLiteral("; %1")
                                        .arg(rollbackError);
        }
        return false;
    }

    if (!transactionRoot.isEmpty()
        && !markCleanupMetadataCommitted(transactionRoot,
                                         &manifest,
                                         &result->errorMessage))
    {
        QString rollbackError;
        bool metadataFullyCommitted = false;
        const bool rollbackSucceeded = rollbackCleanup(
            commitOriginal,
            transactionRoot,
            manifest,
            result,
            &metadataFullyCommitted,
            &rollbackError);
        result->metadataStateCommitted = metadataFullyCommitted;
        result->errorMessage = QStringLiteral(
            "清理元数据已提交，但无法更新恢复清单，%1")
                                   .arg(rollbackStateMessage(
                                       rollbackSucceeded,
                                       transactionRoot));
        if (!rollbackError.isEmpty())
        {
            result->errorMessage += QStringLiteral("：%1")
                                        .arg(rollbackError);
        }
        return false;
    }

    if (!transactionRoot.isEmpty())
    {
        QString purgeError;
        if (!purgeCommittedCleanupTransaction(transactionRoot,
                                               manifest,
                                               &result->failedPaths,
                                               &purgeError))
        {
            // Metadata no longer references these artifacts. A transaction
            // that reached the purge-only namespace can only be deleted; the
            // next cleanup or project-open preflight will retry it.
            result->errorMessage = purgeError;
        }
    }

    result->metadataStateCommitted = true;
    result->success = true;
    return true;
}

} // namespace

bool executeResourceCleanupPlan(ProjectData *projectData,
                                const ResourceCleanupPlan &plan,
                                ResourceCleanupResult *result)
{
    if (!projectData)
    {
        return false;
    }
    const MetadataCommit commitUpdated =
        [projectData, &plan](QString *errorMessage,
                             bool *archiveCommitted)
        {
            return projectData->commitResourceCleanupMetadata(
                plan.updatedMetadata, errorMessage, archiveCommitted);
        };
    const MetadataCommit commitOriginal =
        [projectData, &plan](QString *errorMessage,
                             bool *archiveCommitted)
        {
            projectData->updateMetadata(plan.originalMetadata, false);
            return projectData->commitResourceCleanupMetadata(
                plan.originalMetadata, errorMessage, archiveCommitted);
        };
    return executeWithMetadataCommits(plan,
                                      result,
                                      commitUpdated,
                                      commitOriginal,
                                      false);
}

bool executePreparedResourceCleanupPlan(
    const ProjectResourceCleanupPersistence &persistence,
    const ResourceCleanupPlan &plan,
    ResourceCleanupResult *result)
{
    return executeWithMetadataCommits(
        plan,
        result,
        [&persistence](QString *errorMessage, bool *archiveCommitted)
        {
            return persistence.commitUpdated(
                errorMessage, archiveCommitted);
        },
        [&persistence](QString *errorMessage, bool *archiveCommitted)
        {
            return persistence.commitOriginal(
                errorMessage, archiveCommitted);
        },
        true);
}

} // namespace xjw::core::project::detail
