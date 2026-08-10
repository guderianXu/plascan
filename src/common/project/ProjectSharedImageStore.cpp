#include "project/ProjectSharedImageStore.h"
#include "project/ProjectSharedImageStoreInternal.h"

#include "project/PortableProjectFormat.h"
#include "project/ProjectPackageLayout.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMutex>
#include <QMutexLocker>
#include <QSaveFile>
#include <QScopeGuard>
#include <QSet>

#include <memory>

namespace xjw::common::project
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

QString fileSha256(const QString &path, QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法读取待导入影像 %1: %2")
                     .arg(path, file.errorString()));
        return {};
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    QByteArray buffer(1024 * 1024, Qt::Uninitialized);
    while (!file.atEnd())
    {
        const qint64 count = file.read(buffer.data(), buffer.size());
        if (count < 0)
        {
            setError(errorMessage,
                     QStringLiteral("计算影像 SHA-256 失败: %1").arg(path));
            return {};
        }
        hash.addData(buffer.constData(), count);
    }
    return QString::fromLatin1(hash.result().toHex());
}

QString safeFileName(const QString &path)
{
    QString name = QFileInfo(path).fileName().trimmed();
    if (name.isEmpty())
    {
        name = QStringLiteral("image.bin");
    }
    static const QString invalid = QStringLiteral("<>:\"/\\|?*");
    for (QChar &character : name)
    {
        if (invalid.contains(character) || character.unicode() < 32)
        {
            character = QLatin1Char('_');
        }
    }
    while (name.endsWith(QLatin1Char('.'))
           || name.endsWith(QLatin1Char(' ')))
    {
        name.chop(1);
    }
    return name.isEmpty() ? QStringLiteral("image.bin") : name;
}

bool copyAtomically(const QString &source,
                    const QString &destination,
                    QString *errorMessage)
{
    QFile input(source);
    if (!input.open(QIODevice::ReadOnly)
        || !QDir().mkpath(QFileInfo(destination).absolutePath()))
    {
        setError(errorMessage,
                 QStringLiteral("无法准备共享影像: %1").arg(source));
        return false;
    }
    QSaveFile output(destination);
    if (!output.open(QIODevice::WriteOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法写入共享影像: %1").arg(destination));
        return false;
    }
    QByteArray buffer(1024 * 1024, Qt::Uninitialized);
    while (!input.atEnd())
    {
        const qint64 count = input.read(buffer.data(), buffer.size());
        if (count < 0 || output.write(buffer.constData(), count) != count)
        {
            output.cancelWriting();
            setError(errorMessage,
                     QStringLiteral("复制共享影像失败: %1").arg(source));
            return false;
        }
    }
    if (!output.commit())
    {
        setError(errorMessage,
                 QStringLiteral("提交共享影像失败: %1").arg(destination));
        return false;
    }
    return true;
}

} // namespace

ProjectSharedImageStore::ProjectSharedImageStore(const QString &projectPath)
    : _projectPath(QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath()))
{
}

bool ProjectSharedImageStore::importImage(
    const QString &sourcePath,
    QString *resourceUri,
    QString *materializedPath,
    QString *errorMessage) const
{
    if (!resourceUri)
    {
        setError(errorMessage, QStringLiteral("共享影像 URI 输出参数为空"));
        return false;
    }
    const QFileInfo source(sourcePath);
    if (!source.isFile())
    {
        setError(errorMessage,
                 QStringLiteral("待导入影像不存在: %1").arg(sourcePath));
        return false;
    }
    // 外部源文件的只读哈希不触碰共享库，可并行执行；从检查目标目录开始
    // 才进入项目级发布边界。
    const QString checksum = fileSha256(source.absoluteFilePath(), errorMessage);
    if (checksum.isEmpty())
    {
        return false;
    }

    // 复制、引用发布与 GC 共用同一个项目级同步边界。reservation 与
    // 原子复制在同一临界区完成，GC 不会观察到“文件已出现但尚未保留”。
    const std::shared_ptr<detail::SharedImageSynchronization> synchronization =
        detail::synchronizationForProject(_projectPath);
    QMutexLocker<QMutex> projectLock(&synchronization->mutex);
    if (!detail::ensureCrossProcessLeaseLock(
            _projectPath, synchronization, errorMessage))
    {
        return false;
    }
    const auto releaseUnusedLock = qScopeGuard(
        [&]()
        {
            detail::releaseCrossProcessLeaseLockIfUnused(synchronization);
        });

    const QString hashDirectory =
        QDir(ProjectPackageLayout::sharedImagesDirectory(_projectPath))
            .filePath(checksum);
    QDir directory(hashDirectory);
    const QFileInfoList existing = directory.entryInfoList(
        QDir::Files | QDir::NoDotAndDotDot, QDir::Name);
    QString destination;
    if (!existing.isEmpty())
    {
        destination = existing.constFirst().absoluteFilePath();
        if (fileSha256(destination, errorMessage) != checksum)
        {
            setError(errorMessage,
                     QStringLiteral("共享影像哈希目录内容损坏: %1")
                         .arg(hashDirectory));
            return false;
        }
    }
    else
    {
        destination = QDir(hashDirectory).filePath(
            safeFileName(source.absoluteFilePath()));
        if (!copyAtomically(source.absoluteFilePath(), destination, errorMessage))
        {
            return false;
        }
        if (fileSha256(destination, errorMessage) != checksum)
        {
            QFile::remove(destination);
            QDir().rmdir(hashDirectory);
            setError(errorMessage,
                     QStringLiteral(
                         "共享影像源文件在导入期间发生变化，已取消导入: %1")
                         .arg(source.absoluteFilePath()));
            return false;
        }
    }

    const QString entry = QDir::fromNativeSeparators(
        QDir(ProjectPackageLayout::dataDirectory(_projectPath))
            .relativeFilePath(destination));
    *resourceUri = PortableProjectFormat::resourceUriForEntry(entry);
    if (resourceUri->isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("无法生成共享影像 URI: %1").arg(destination));
        return false;
    }
    ++synchronization->activeReservations[entry];
    if (materializedPath)
    {
        *materializedPath = destination;
    }
    return true;
}

bool ProjectSharedImageStore::publishReferences(
    const QStringList &references,
    QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    const std::shared_ptr<detail::SharedImageSynchronization> synchronization =
        detail::synchronizationForProject(_projectPath);
    QMutexLocker<QMutex> projectLock(&synchronization->mutex);
    if (!detail::ensureCrossProcessLeaseLock(
            _projectPath, synchronization, errorMessage))
    {
        return false;
    }
    const auto releaseUnusedLock = qScopeGuard(
        [&]()
        {
            detail::releaseCrossProcessLeaseLockIfUnused(synchronization);
        });

    QSet<QString> entries;
    for (const QString &reference : references)
    {
        const QString entry = detail::entryForReference(
            _projectPath, reference);
        if (entry.isEmpty())
        {
            setError(errorMessage,
                     QStringLiteral("共享影像引用不属于当前项目: %1")
                         .arg(reference));
            return false;
        }
        const QString path = QDir(
            ProjectPackageLayout::dataDirectory(_projectPath)).filePath(entry);
        if (!QFileInfo(path).isFile())
        {
            setError(errorMessage,
                     QStringLiteral("共享影像引用不存在: %1").arg(reference));
            return false;
        }
        entries.insert(entry);
    }

    for (const QString &entry : entries)
    {
        if (!synchronization->activeReservations.contains(entry))
        {
            synchronization->activeReservations.insert(entry, 1);
        }
    }
    return true;
}

void ProjectSharedImageStore::releaseReservations(
    const QStringList &references) const
{
    const std::shared_ptr<detail::SharedImageSynchronization> synchronization =
        detail::synchronizationForProject(_projectPath);
    QMutexLocker<QMutex> projectLock(&synchronization->mutex);
    for (const QString &reference : references)
    {
        const QString entry = detail::entryForReference(
            _projectPath, reference);
        auto it = synchronization->activeReservations.find(entry);
        if (it == synchronization->activeReservations.end())
        {
            continue;
        }
        if (*it <= 1)
        {
            synchronization->activeReservations.erase(it);
        }
        else
        {
            --(*it);
        }
    }
    detail::releaseCrossProcessLeaseLockIfUnused(synchronization);
}

QString ProjectSharedImageStore::materialize(
    const QString &resourceUri,
    QString *errorMessage) const
{
    const QString entry =
        PortableProjectFormat::entryPathFromResourceUri(resourceUri);
    if (!entry.startsWith(QStringLiteral("shared/images/")))
    {
        setError(errorMessage,
                 QStringLiteral("不是共享影像 URI: %1").arg(resourceUri));
        return {};
    }
    const QString path = PortableProjectFormat::resolveEntryPath(
        ProjectPackageLayout::dataDirectory(_projectPath),
        entry,
        errorMessage);
    if (path.isEmpty() || !QFileInfo(path).isFile())
    {
        setError(errorMessage,
                 QStringLiteral("共享影像不存在: %1").arg(resourceUri));
        return {};
    }
    return path;
}

bool ProjectSharedImageStore::pruneUnreferenced(QString *errorMessage) const
{
    return detail::pruneSharedImages(
        _projectPath,
        detail::synchronizationForProject(_projectPath),
        errorMessage);
}

} // namespace xjw::common::project
