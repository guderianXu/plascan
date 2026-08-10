#include "ProjectResourceCleanupTransaction.h"

#include "ProjectResourceCleanupArtifacts.h"

#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>
#include <QUuid>

namespace xjw::core::project::detail
{
namespace
{

constexpr auto kCleanupPurgeDirectory = ".plascan_cleanup_purging";
constexpr auto kCleanupPurgePrefix = ".purging-";

bool isFilesystemLink(const QFileInfo &info)
{
    bool link = info.isSymLink();
#ifdef Q_OS_WIN
    link = link || info.isJunction();
#endif
    return link;
}

QString cleanupPurgeBase(const QString &managedRoot)
{
    return QDir(managedRoot).filePath(
        QString::fromLatin1(kCleanupPurgeDirectory));
}

bool pathIsMissing(const QFileInfo &info)
{
    return !info.exists() && !isFilesystemLink(info);
}

bool validatePurgeBase(const QString &managedRoot,
                       const QString &purgeBase,
                       QString *errorMessage)
{
    const QFileInfo baseInfo(purgeBase);
    const bool valid = baseInfo.isDir()
        && !isFilesystemLink(baseInfo)
        && cleanupPathIdentity(baseInfo.absolutePath())
            == cleanupPathIdentity(managedRoot)
        && cleanupPathIsInside(managedRoot, purgeBase, false)
        && !cleanupPathTraversesLink(managedRoot, purgeBase);
    if (!valid && errorMessage)
    {
        *errorMessage = QStringLiteral(
            "项目清理清除态根目录不安全：%1").arg(purgeBase);
    }
    return valid;
}

bool ensurePurgeBase(const QString &managedRoot,
                     QString *purgeBase,
                     QString *errorMessage)
{
    if (!purgeBase)
    {
        return false;
    }
    *purgeBase = cleanupPurgeBase(managedRoot);
    const QFileInfo existing(*purgeBase);
    if (pathIsMissing(existing) && !QDir().mkpath(*purgeBase))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "无法创建项目清理清除态根目录：%1").arg(*purgeBase);
        }
        return false;
    }
    return validatePurgeBase(managedRoot, *purgeBase, errorMessage);
}

bool isPurgeDirectoryName(const QString &name)
{
    static const QRegularExpression expression(QStringLiteral(
        "^\\.purging-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
        "[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"));
    return expression.match(name).hasMatch();
}

bool validatePurgeDirectory(const QString &managedRoot,
                            const QString &purgeBase,
                            const QString &purgeRoot,
                            QString *errorMessage)
{
    const QFileInfo rootInfo(purgeRoot);
    const bool valid = isPurgeDirectoryName(rootInfo.fileName())
        && rootInfo.isDir()
        && !isFilesystemLink(rootInfo)
        && cleanupPathIdentity(rootInfo.absolutePath())
            == cleanupPathIdentity(purgeBase)
        && cleanupPathIsInside(purgeBase, purgeRoot, false)
        && !cleanupPathTraversesLink(managedRoot, purgeRoot)
        && !cleanupDirectoryContainsLink(purgeRoot);
    if (!valid && errorMessage)
    {
        *errorMessage = QStringLiteral(
            "项目清理清除态目录不安全：%1").arg(purgeRoot);
    }
    return valid;
}

bool removePurgeDirectory(const QString &managedRoot,
                          const QString &purgeBase,
                          const QString &purgeRoot,
                          QString *errorMessage)
{
    const QFileInfo current(purgeRoot);
    if (pathIsMissing(current))
    {
        return true;
    }
    if (!validatePurgeDirectory(managedRoot,
                                purgeBase,
                                purgeRoot,
                                errorMessage))
    {
        return false;
    }
    if (!QDir(purgeRoot).removeRecursively()
        && !pathIsMissing(QFileInfo(purgeRoot)))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "无法继续清理已提交事务残留：%1").arg(purgeRoot);
        }
        return false;
    }
    return true;
}

void removeEmptyPurgeBase(const QString &managedRoot)
{
    const QString purgeBase = cleanupPurgeBase(managedRoot);
    const QFileInfo info(purgeBase);
    if (!info.isDir() || isFilesystemLink(info))
    {
        return;
    }
    const QDir directory(purgeBase);
    if (directory.entryList(
            QDir::AllEntries | QDir::NoDotAndDotDot
                | QDir::Hidden | QDir::System).isEmpty())
    {
        QDir().rmdir(purgeBase);
    }
}

bool validateActiveTransactionRoot(
    const QString &transactionRoot,
    const CleanupTransactionManifest &manifest,
    QString *errorMessage)
{
    const QString trashBase = cleanupTrashBase(manifest.managedRoot);
    const QFileInfo trashInfo(trashBase);
    const QFileInfo rootInfo(transactionRoot);
    const bool valid = trashInfo.isDir()
        && !isFilesystemLink(trashInfo)
        && cleanupPathIdentity(trashInfo.absolutePath())
            == cleanupPathIdentity(manifest.managedRoot)
        && !cleanupPathTraversesLink(manifest.managedRoot, trashBase)
        && rootInfo.isDir()
        && !isFilesystemLink(rootInfo)
        && rootInfo.fileName() == manifest.transactionId
        && cleanupPathIdentity(rootInfo.absolutePath())
            == cleanupPathIdentity(trashBase)
        && cleanupPathIsInside(trashBase, transactionRoot, false)
        && !cleanupPathTraversesLink(manifest.managedRoot, transactionRoot)
        && !cleanupDirectoryContainsLink(transactionRoot);
    if (!valid && errorMessage)
    {
        *errorMessage = QStringLiteral(
            "无法安全推进清理事务到清除态：%1").arg(transactionRoot);
    }
    return valid;
}

bool moveTransactionToPurge(const QString &transactionRoot,
                            const CleanupTransactionManifest &manifest,
                            QString *purgeRoot,
                            QString *errorMessage)
{
    if (!purgeRoot
        || !validateActiveTransactionRoot(transactionRoot,
                                          manifest,
                                          errorMessage))
    {
        return false;
    }
    QString purgeBase;
    if (!ensurePurgeBase(manifest.managedRoot,
                         &purgeBase,
                         errorMessage))
    {
        return false;
    }

    const QString name = QString::fromLatin1(kCleanupPurgePrefix)
        + QUuid::createUuid().toString(QUuid::WithoutBraces);
    *purgeRoot = QDir(purgeBase).filePath(name);
    const QFileInfo destinationInfo(*purgeRoot);
    if (!pathIsMissing(destinationInfo)
        || !QDir().rename(transactionRoot, *purgeRoot))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "无法原子推进清理事务到清除态：%1 -> %2")
                                .arg(transactionRoot, *purgeRoot);
        }
        return false;
    }

    // Both bases are direct children of the same managed root, so the rename
    // stays on one filesystem. From this point onward the transaction is
    // purge-only and must never be interpreted as a rollback WAL again.
    removeEmptyCleanupTrashBase(manifest.managedRoot);
    return true;
}

} // namespace

bool purgePendingCleanupDirectories(
    const QString &managedRoot,
    QStringList *failedPaths,
    QString *errorMessage)
{
    const QString purgeBase = cleanupPurgeBase(managedRoot);
    const QFileInfo baseInfo(purgeBase);
    if (pathIsMissing(baseInfo))
    {
        return true;
    }
    if (!validatePurgeBase(managedRoot, purgeBase, errorMessage))
    {
        appendUniqueCleanupPath(failedPaths, purgeBase);
        return false;
    }

    const QFileInfoList entries = QDir(purgeBase).entryInfoList(
        QDir::AllEntries | QDir::NoDotAndDotDot
            | QDir::Hidden | QDir::System,
        QDir::Name);
    for (const QFileInfo &entry : entries)
    {
        const QString purgeRoot = entry.absoluteFilePath();
        if (!removePurgeDirectory(managedRoot,
                                  purgeBase,
                                  purgeRoot,
                                  errorMessage))
        {
            appendUniqueCleanupPath(failedPaths, purgeRoot);
            return false;
        }
    }
    removeEmptyPurgeBase(managedRoot);
    return true;
}

bool purgeCommittedCleanupTransaction(
    const QString &transactionRoot,
    const CleanupTransactionManifest &manifest,
    QStringList *failedPaths,
    QString *errorMessage)
{
    if (!purgePendingCleanupDirectories(manifest.managedRoot,
                                        failedPaths,
                                        errorMessage))
    {
        return false;
    }

    QString purgeRoot;
    if (!moveTransactionToPurge(transactionRoot,
                                manifest,
                                &purgeRoot,
                                errorMessage))
    {
        appendUniqueCleanupPath(failedPaths, transactionRoot);
        return false;
    }
    if (!removePurgeDirectory(manifest.managedRoot,
                              cleanupPurgeBase(manifest.managedRoot),
                              purgeRoot,
                              errorMessage))
    {
        appendUniqueCleanupPath(failedPaths, purgeRoot);
        return false;
    }
    removeEmptyPurgeBase(manifest.managedRoot);
    return true;
}

} // namespace xjw::core::project::detail
