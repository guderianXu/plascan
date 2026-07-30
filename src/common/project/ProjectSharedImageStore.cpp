#include "project/ProjectSharedImageStore.h"

#include "project/PortableProjectFormat.h"
#include "project/ProjectChunkStore.h"
#include "project/ProjectPackageLayout.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QSaveFile>
#include <QSet>

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

void collectSharedUris(const QJsonValue &value, QSet<QString> *entries)
{
    if (value.isString())
    {
        const QString entry =
            PortableProjectFormat::entryPathFromResourceUri(value.toString());
        if (entry.startsWith(QStringLiteral("shared/images/")))
        {
            entries->insert(entry);
        }
        return;
    }
    if (value.isArray())
    {
        for (const QJsonValue &item : value.toArray())
        {
            collectSharedUris(item, entries);
        }
        return;
    }
    if (value.isObject())
    {
        const QJsonObject object = value.toObject();
        for (auto it = object.constBegin(); it != object.constEnd(); ++it)
        {
            collectSharedUris(it.value(), entries);
        }
    }
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
    const QString checksum = fileSha256(source.absoluteFilePath(), errorMessage);
    if (checksum.isEmpty())
    {
        return false;
    }

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
    if (materializedPath)
    {
        *materializedPath = destination;
    }
    return true;
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
    ProjectChunkStore store(_projectPath);
    const QList<ProjectChunkRecord> chunks = store.chunks(errorMessage);
    if (chunks.isEmpty())
    {
        return false;
    }
    QSet<QString> referenced;
    for (const ProjectChunkRecord &chunk : chunks)
    {
        QJsonObject document;
        if (!store.readChunkDocument(
                chunk.directory, &document, errorMessage))
        {
            return false;
        }
        collectSharedUris(document, &referenced);
    }

    const QString root =
        ProjectPackageLayout::sharedImagesDirectory(_projectPath);
    if (!QFileInfo(root).isDir())
    {
        return true;
    }
    QDirIterator iterator(
        root,
        QDir::Files | QDir::NoDotAndDotDot,
        QDirIterator::Subdirectories);
    const QDir dataRoot(
        ProjectPackageLayout::dataDirectory(_projectPath));
    while (iterator.hasNext())
    {
        const QString path = iterator.next();
        const QString entry = QDir::fromNativeSeparators(
            dataRoot.relativeFilePath(path));
        if (!referenced.contains(entry) && !QFile::remove(path))
        {
            setError(errorMessage,
                     QStringLiteral("无法清理未引用共享影像: %1").arg(path));
            return false;
        }
    }

    QDir imagesRoot(root);
    const QFileInfoList hashDirectories = imagesRoot.entryInfoList(
        QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QFileInfo &directory : hashDirectories)
    {
        QDir hashDirectory(directory.absoluteFilePath());
        if (hashDirectory.entryList(
                QDir::AllEntries | QDir::NoDotAndDotDot).isEmpty())
        {
            imagesRoot.rmdir(directory.fileName());
        }
    }
    return true;
}

} // namespace xjw::common::project
