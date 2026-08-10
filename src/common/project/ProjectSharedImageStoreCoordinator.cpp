#include "project/ProjectSharedImageStoreInternal.h"

#include "project/PortableProjectFormat.h"
#include "project/ProjectPackageLayout.h"

#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <QLockFile>
#include <QMutexLocker>

#include <memory>

namespace xjw::common::project::detail
{
namespace
{

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

QString normalizedProjectKey(const QString &projectPath)
{
    QString key = QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath());
#ifdef Q_OS_WIN
    key = key.toCaseFolded();
#endif
    return key;
}

} // namespace

std::shared_ptr<SharedImageSynchronization> synchronizationForProject(
    const QString &projectPath)
{
    static QMutex registryMutex;
    static QHash<QString, std::shared_ptr<SharedImageSynchronization>> registry;

    QMutexLocker<QMutex> registryLock(&registryMutex);
    const QString key = normalizedProjectKey(projectPath);
    auto it = registry.find(key);
    if (it == registry.end())
    {
        it = registry.insert(
            key, std::make_shared<SharedImageSynchronization>());
    }
    return it.value();
}

QString entryForReference(const QString &projectPath,
                          const QString &reference)
{
    const QString trimmed = reference.trimmed();
    if (trimmed.isEmpty())
    {
        return {};
    }

    QString entry = PortableProjectFormat::entryPathFromResourceUri(trimmed);
    if (entry.isEmpty())
    {
        const QString dataRoot = QDir::cleanPath(
            QFileInfo(ProjectPackageLayout::dataDirectory(projectPath))
                .absoluteFilePath());
        const QString absolutePath = QDir::cleanPath(
            QFileInfo(trimmed).absoluteFilePath());
        entry = QDir::fromNativeSeparators(
            QDir(dataRoot).relativeFilePath(absolutePath));
        if (QDir::isAbsolutePath(entry)
            || entry == QStringLiteral("..")
            || entry.startsWith(QStringLiteral("../")))
        {
            return {};
        }
    }
    entry = QDir::cleanPath(QDir::fromNativeSeparators(entry));
    if (!entry.startsWith(QStringLiteral("shared/images/"))
        || entry == QStringLiteral("shared/images"))
    {
        return {};
    }
    return entry;
}

QString sharedImageLockPath(const QString &projectPath)
{
    return QDir(ProjectPackageLayout::dataDirectory(projectPath)).filePath(
        QStringLiteral(".shared-images.lock"));
}

bool ensureCrossProcessLeaseLock(
    const QString &projectPath,
    const std::shared_ptr<SharedImageSynchronization> &synchronization,
    QString *errorMessage)
{
    if (synchronization->crossProcessLeaseLock)
    {
        return true;
    }
    if (!QDir().mkpath(ProjectPackageLayout::dataDirectory(projectPath)))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建共享影像同步锁目录: %1")
                     .arg(ProjectPackageLayout::dataDirectory(projectPath)));
        return false;
    }

    auto lock = std::make_unique<QLockFile>(sharedImageLockPath(projectPath));
    // 正常写入者还受项目级 .plascan.lock 排他保护；这里沿用其超时，
    // 让异常退出遗留的辅助锁能够恢复。
    lock->setStaleLockTime(30000);
    if (!lock->tryLock(0))
    {
        setError(errorMessage,
                 QStringLiteral("共享影像库正由另一进程更新: %1")
                     .arg(sharedImageLockPath(projectPath)));
        return false;
    }
    synchronization->crossProcessLeaseLock = std::move(lock);
    return true;
}

void releaseCrossProcessLeaseLockIfUnused(
    const std::shared_ptr<SharedImageSynchronization> &synchronization)
{
    if (!synchronization->activeReservations.isEmpty()
        || !synchronization->crossProcessLeaseLock)
    {
        return;
    }
    synchronization->crossProcessLeaseLock->unlock();
    synchronization->crossProcessLeaseLock.reset();
}

} // namespace xjw::common::project::detail
