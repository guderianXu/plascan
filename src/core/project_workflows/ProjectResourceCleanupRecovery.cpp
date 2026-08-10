#include "ProjectResourceCleanupTransaction.h"

#include "ProjectResourceCleanupArtifacts.h"

#include "Logger.h"
#include "project/ProjectChunkStore.h"
#include "project/ProjectIO.h"
#include "project/ProjectPackageLayout.h"
#include "project/ProjectSessionModel.h"
#include "project/ProjectWorkspaceStore.h"
#include "project/PortableProjectFormat.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>

namespace xjw::core::project::detail
{
namespace
{

bool isFilesystemLink(const QFileInfo &info)
{
    bool link = info.isSymLink();
#ifdef Q_OS_WIN
    link = link || info.isJunction();
#endif
    return link;
}

CleanupTransactionSession sessionForProjectData(ProjectData *projectData)
{
    CleanupTransactionSession session;
    if (!projectData)
    {
        return session;
    }
    session.projectPath = QDir::cleanPath(
        QFileInfo(projectData->currentProjectPath()).absoluteFilePath());
    session.chunkId = projectData->activeChunkId();
    session.chunkDirectory = projectData->activeChunkDirectory();
    session.managedRoot =
        xjw::common::project::ProjectPackageLayout::chunkDirectory(
            session.projectPath, session.chunkDirectory);
    session.projectRoot = session.managedRoot;
    return session;
}

bool sessionBeforeOpen(const QString &projectPath,
                       CleanupTransactionSession *session,
                       QString *errorMessage)
{
    if (!session)
    {
        return false;
    }
    session->projectPath = QDir::cleanPath(
        QFileInfo(projectPath).absoluteFilePath());
    ProjectChunkStore store(session->projectPath);
    if (!store.ensureLayout(errorMessage))
    {
        return false;
    }
    const xjw::common::project::ProjectChunkRecord chunk =
        store.defaultChunk(errorMessage);
    if (chunk.id.isEmpty())
    {
        return false;
    }
    session->chunkId = chunk.id;
    session->chunkDirectory = chunk.directory;
    session->managedRoot =
        xjw::common::project::ProjectPackageLayout::chunkDirectory(
            session->projectPath, session->chunkDirectory);
    session->projectRoot = session->managedRoot;
    return !session->managedRoot.isEmpty();
}

bool readRecoveryJson(const QString &path,
                      bool compressed,
                      QJsonObject *object,
                      bool *loaded)
{
    if (!object || !loaded)
    {
        return false;
    }
    *loaded = false;
    const QFileInfo info(path);
    if (!info.isFile() || isFilesystemLink(info))
    {
        return true;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return true;
    }
    const QByteArray bytes = file.readAll();
    QJsonDocument document;
    if (compressed)
    {
        const QByteArray uncompressed = qUncompress(bytes);
        if (!uncompressed.isEmpty())
        {
            document = QJsonDocument::fromJson(uncompressed);
        }
    }
    if (!document.isObject())
    {
        document = QJsonDocument::fromJson(bytes);
    }
    if (!document.isObject())
    {
        return true;
    }
    *object = document.object();
    *loaded = true;
    return true;
}

bool loadMetadataBeforeOpen(const CleanupTransactionSession &session,
                            QJsonObject *metadata,
                            QString *errorMessage)
{
    if (!metadata)
    {
        return false;
    }
    ProjectChunkStore store(session.projectPath);
    QJsonObject document;
    if (!store.readChunkDocument(
            session.chunkDirectory, &document, errorMessage))
    {
        return false;
    }
    QJsonObject core = document.value(
        QString::fromLatin1(
            xjw::common::project::PortableProjectFormat::
                ProjectFilesSection)).toObject();
    QJsonObject results = document.value(
        QString::fromLatin1(
            xjw::common::project::PortableProjectFormat::
                ProjectResultsSection)).toObject();

    bool loaded = false;
    QJsonObject temporary;
    readRecoveryJson(
        xjw::common::project::ProjectIO::tempFilesPath(session.projectPath),
        false,
        &temporary,
        &loaded);
    if (loaded)
    {
        core = temporary;
    }
    temporary = {};
    readRecoveryJson(
        xjw::common::project::ProjectIO::tempResultsPath(session.projectPath),
        true,
        &temporary,
        &loaded);
    if (loaded)
    {
        results = temporary;
    }

    ProjectWorkspaceStore workspace(
        session.projectPath, session.chunkDirectory);
    if (!workspace.materializeMetadataForRecovery(&core, errorMessage)
        || !workspace.materializeMetadataForRecovery(&results, errorMessage))
    {
        return false;
    }
    *metadata = core;
    for (auto it = results.constBegin(); it != results.constEnd(); ++it)
    {
        metadata->insert(it.key(), it.value());
    }
    return true;
}

bool transactionEntries(const CleanupTransactionSession &session,
                        QFileInfoList *entries,
                        QString *errorMessage)
{
    if (!entries)
    {
        return false;
    }
    entries->clear();
    const QString trashBase = cleanupTrashBase(session.managedRoot);
    const QFileInfo trashInfo(trashBase);
    if (!trashInfo.exists() && !trashInfo.isSymLink())
    {
        return true;
    }
    if (!trashInfo.isDir()
        || isFilesystemLink(trashInfo)
        || cleanupPathTraversesLink(session.managedRoot, trashBase))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务恢复根目录不安全：%1").arg(trashBase);
        }
        return false;
    }
    *entries = QDir(trashBase).entryInfoList(
        QDir::AllEntries | QDir::NoDotAndDotDot
            | QDir::Hidden | QDir::System,
        QDir::Name);
    for (const QFileInfo &entry : *entries)
    {
        if (!entry.isDir() || isFilesystemLink(entry))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "清理事务根目录包含未知或链接项：%1")
                                        .arg(entry.absoluteFilePath());
            }
            return false;
        }
    }
    return true;
}

} // namespace

bool recoverPendingResourceCleanupTransactionsBeforeOpen(
    const QString &projectPath,
    QString *errorMessage)
{
    CleanupTransactionSession session;
    if (!sessionBeforeOpen(projectPath, &session, errorMessage))
    {
        return false;
    }
    QStringList failedPurgePaths;
    QString purgeError;
    if (!purgePendingCleanupDirectories(session.managedRoot,
                                        &failedPurgePaths,
                                        &purgeError))
    {
        LOG_WARN(QStringLiteral(
            "已提交清理事务残留暂时无法续删，项目打开将继续：%1")
                     .arg(purgeError));
    }
    QFileInfoList entries;
    if (!transactionEntries(session, &entries, errorMessage))
    {
        return false;
    }
    if (entries.isEmpty())
    {
        return true;
    }

    QJsonObject currentMetadata;
    if (!loadMetadataBeforeOpen(session, &currentMetadata, errorMessage))
    {
        return false;
    }
    for (const QFileInfo &entry : entries)
    {
        CleanupTransactionManifest manifest;
        if (!loadAndValidateCleanupTransaction(
                session,
                entry.absoluteFilePath(),
                &manifest,
                errorMessage))
        {
            return false;
        }
        const bool metadataIsUpdated =
            currentMetadata == manifest.updatedMetadata;
        if (manifest.state == CleanupTransactionState::MetadataCommitted
            && metadataIsUpdated)
        {
            QStringList failedPaths;
            if (!purgeCommittedCleanupTransaction(
                    entry.absoluteFilePath(),
                    manifest,
                    &failedPaths,
                    &purgeError))
            {
                LOG_WARN(QStringLiteral(
                    "已提交清理事务暂时无法推进清除，项目打开将继续：%1")
                             .arg(purgeError));
            }
            continue;
        }
        QStringList failedPaths;
        if (!restoreStagedCleanupTransaction(
                entry.absoluteFilePath(),
                manifest,
                &failedPaths,
                errorMessage))
        {
            return false;
        }
    }
    return true;
}

bool recoverPendingResourceCleanupTransactions(
    ProjectData *projectData,
    ResourceCleanupResult *result)
{
    if (!projectData || !result)
    {
        return false;
    }
    const CleanupTransactionSession session = sessionForProjectData(
        projectData);
    if (session.projectPath.trimmed().isEmpty())
    {
        return true;
    }
    QString pendingPurgeError;
    if (!purgePendingCleanupDirectories(session.managedRoot,
                                        &result->failedPaths,
                                        &pendingPurgeError))
    {
        LOG_WARN(QStringLiteral(
            "已提交清理事务残留暂时无法续删，将保留后重试：%1")
                     .arg(pendingPurgeError));
    }
    QFileInfoList entries;
    if (!transactionEntries(session, &entries, &result->errorMessage))
    {
        appendUniqueCleanupPath(
            &result->failedPaths, cleanupTrashBase(session.managedRoot));
        return false;
    }

    for (const QFileInfo &entry : entries)
    {
        CleanupTransactionManifest manifest;
        QString recoveryError;
        if (!loadAndValidateCleanupTransaction(
                session,
                entry.absoluteFilePath(),
                &manifest,
                &recoveryError))
        {
            result->errorMessage = recoveryError.isEmpty()
                ? QStringLiteral("无法验证清理事务恢复清单：%1")
                      .arg(entry.absoluteFilePath())
                : recoveryError;
            appendUniqueCleanupPath(&result->failedPaths,
                                    entry.absoluteFilePath());
            return false;
        }

        const QJsonObject currentMetadata =
            projectData->metadataIncludingResults();
        const bool metadataIsOriginal =
            currentMetadata == manifest.originalMetadata;
        const bool metadataIsUpdated =
            currentMetadata == manifest.updatedMetadata;
        bool recovered = true;
        const bool shouldRestoreArtifacts =
            manifest.state == CleanupTransactionState::Staging
            || !metadataIsUpdated;
        if (shouldRestoreArtifacts)
        {
            recovered = restoreStagedCleanupTransaction(
                entry.absoluteFilePath(),
                manifest,
                &result->failedPaths,
                &recoveryError);
            if (recovered)
            {
                const bool shouldCommitOriginal =
                    metadataIsOriginal
                    || (manifest.state == CleanupTransactionState::Staging
                        && metadataIsUpdated);
                if (shouldCommitOriginal)
                {
                    recovered = projectData->commitResourceCleanupMetadata(
                        manifest.originalMetadata,
                        &recoveryError);
                }
                else
                {
                    LOG_WARN(QStringLiteral(
                        "清理事务恢复检测到更新后的第三种元数据状态；"
                        "仅恢复物理产物，不覆盖当前项目元数据：%1")
                                 .arg(entry.absoluteFilePath()));
                }
            }
        }
        if (!recovered)
        {
            result->errorMessage = recoveryError.isEmpty()
                ? QStringLiteral("清理事务自动恢复失败：%1")
                      .arg(entry.absoluteFilePath())
                : recoveryError;
            return false;
        }

        if (!purgeCommittedCleanupTransaction(
                entry.absoluteFilePath(),
                manifest,
                &result->failedPaths,
                &recoveryError))
        {
            LOG_WARN(QStringLiteral(
                "已完成清理事务恢复，但事务残留暂时无法清除：%1")
                         .arg(recoveryError));
        }
    }
    removeEmptyCleanupTrashBase(session.managedRoot);
    return true;
}

} // namespace xjw::core::project::detail
