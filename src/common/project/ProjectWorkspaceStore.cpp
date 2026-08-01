#include "project/ProjectWorkspaceStore.h"

#include "project/PlascanArchive.h"
#include "project/ProjectChunkStore.h"
#include "project/ProjectPackageLayout.h"
#include "project/PortableProjectFormat.h"
#include "project/ProjectIO.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QSet>

namespace
{

using xjw::common::project::PortableProjectFormat;
using xjw::common::project::ProjectIO;
using xjw::common::project::ProjectPackageLayout;
using xjw::common::project::ProjectResourceIndex;
using xjw::common::project::ProjectResourceRef;

constexpr auto ChunkEntryPrefix = "chunk/";
constexpr auto LegacyWorkspaceEntryPrefix = "workspace/";

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
                 QStringLiteral("无法读取项目资源 %1: %2")
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
                     QStringLiteral("计算项目资源校验值失败: %1").arg(path));
            return {};
        }
        if (count > 0)
        {
            hash.addData(buffer.constData(), count);
        }
    }
    return QString::fromLatin1(hash.result().toHex());
}

QString stableToken(const QString &value)
{
    return QString::fromLatin1(
        QCryptographicHash::hash(
            value.toUtf8(), QCryptographicHash::Sha256)
            .toHex()
            .left(24));
}

QString diagnosticPathToken(const QString &path)
{
    const QFileInfo info(path);
    const QString label = info.fileName().isEmpty()
        ? QStringLiteral("path")
        : info.fileName();
    return QStringLiteral("plascan-diagnostic:///%1/%2")
        .arg(stableToken(info.absoluteFilePath()), label);
}

bool isPathWithin(const QString &path, const QString &root)
{
    const QString normalizedPath =
        QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    QString normalizedRoot =
        QDir::cleanPath(QFileInfo(root).absoluteFilePath());
    if (!normalizedRoot.endsWith(QLatin1Char('/')))
    {
        normalizedRoot += QLatin1Char('/');
    }
#if defined(Q_OS_WIN)
    return normalizedPath.startsWith(normalizedRoot, Qt::CaseInsensitive);
#else
    return normalizedPath.startsWith(normalizedRoot);
#endif
}

bool copyFileAtomically(const QString &source,
                        const QString &destination,
                        QString *errorMessage)
{
    if (QDir::cleanPath(source) == QDir::cleanPath(destination))
    {
        return true;
    }
    if (!QDir().mkpath(QFileInfo(destination).absolutePath()))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建工程资源目录: %1")
                     .arg(QFileInfo(destination).absolutePath()));
        return false;
    }

    QFile input(source);
    if (!input.open(QIODevice::ReadOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法读取待封装资源 %1: %2")
                     .arg(source, input.errorString()));
        return false;
    }
    QSaveFile output(destination);
    if (!output.open(QIODevice::WriteOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法写入工程资源 %1: %2")
                     .arg(destination, output.errorString()));
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
                     QStringLiteral("复制工程资源失败: %1").arg(source));
            return false;
        }
    }
    if (!output.commit())
    {
        setError(errorMessage,
                 QStringLiteral("提交工程资源失败: %1").arg(destination));
        return false;
    }
    return true;
}

bool copyTree(const QString &source,
              const QString &destination,
              QString *errorMessage)
{
    const QFileInfo sourceInfo(source);
    if (sourceInfo.isFile())
    {
        return copyFileAtomically(
            sourceInfo.absoluteFilePath(), destination, errorMessage);
    }
    if (!sourceInfo.isDir())
    {
        setError(errorMessage,
                 QStringLiteral("待封装路径不存在: %1").arg(source));
        return false;
    }
    if (!QDir().mkpath(destination))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建工程资源目录: %1").arg(destination));
        return false;
    }

    QDirIterator iterator(
        sourceInfo.absoluteFilePath(),
        QDir::Files | QDir::NoDotAndDotDot,
        QDirIterator::Subdirectories);
    const QDir sourceRoot(sourceInfo.absoluteFilePath());
    while (iterator.hasNext())
    {
        const QString sourceFile = iterator.next();
        const QString relative = sourceRoot.relativeFilePath(sourceFile);
        if (!copyFileAtomically(
                sourceFile, QDir(destination).filePath(relative), errorMessage))
        {
            return false;
        }
    }
    return true;
}

bool isChunkEntry(const QString &entryPath)
{
    return entryPath.startsWith(QString::fromLatin1(ChunkEntryPrefix))
        || entryPath.startsWith(
            QString::fromLatin1(LegacyWorkspaceEntryPrefix));
}

QString chunkRelativePath(const QString &entryPath)
{
    if (entryPath.startsWith(QString::fromLatin1(ChunkEntryPrefix)))
    {
        return entryPath.mid(
            static_cast<int>(qstrlen(ChunkEntryPrefix)));
    }
    if (entryPath.startsWith(
            QString::fromLatin1(LegacyWorkspaceEntryPrefix)))
    {
        return entryPath.mid(
            static_cast<int>(qstrlen(LegacyWorkspaceEntryPrefix)));
    }
    return {};
}

QString chunkUri(const QString &runtimeRoot, const QString &path)
{
    const QString relative =
        QDir::fromNativeSeparators(QDir(runtimeRoot).relativeFilePath(path));
    if (relative.startsWith(QStringLiteral("../"))
        || relative == QStringLiteral(".."))
    {
        return {};
    }
    return PortableProjectFormat::resourceUriForEntry(
        QStringLiteral("chunk/%1").arg(relative));
}

QString projectUri(const QString &runtimeRoot,
                   const QString &dataRoot,
                   const QString &path)
{
    if (isPathWithin(path, runtimeRoot))
    {
        return chunkUri(runtimeRoot, path);
    }
    if (isPathWithin(path, dataRoot))
    {
        const QString relative = QDir::fromNativeSeparators(
            QDir(dataRoot).relativeFilePath(path));
        if (relative.startsWith(QStringLiteral("shared/")))
        {
            return PortableProjectFormat::resourceUriForEntry(relative);
        }
    }
    return {};
}

QString stageExternalPath(const QString &source,
                          const QString &runtimeRoot,
                          const QString &projectPath,
                          QString *errorMessage)
{
    const QFileInfo info(source);
    if ((!info.isFile() && !info.isDir())
        || QDir::cleanPath(info.absoluteFilePath())
            == QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath()))
    {
        return {};
    }
    if (isPathWithin(info.absoluteFilePath(), runtimeRoot))
    {
        return info.absoluteFilePath();
    }
    // 外部目录只作为运行参数保留，不递归复制。尤其当目录是工程父目录时，
    // 递归封装会把当前 <project>.files 再次复制进 Chunk，形成自包含循环。
    // 需要便携化的真实产物文件仍按下方逻辑逐文件导入。
    if (info.isDir())
    {
        return {};
    }

    const QString bucket = stableToken(info.absoluteFilePath());
    const QString destination = QDir(runtimeRoot).filePath(
        QStringLiteral("assets/imported/%1/%2")
            .arg(bucket, info.fileName()));
    if (!copyTree(info.absoluteFilePath(), destination, errorMessage))
    {
        return {};
    }
    return destination;
}

QJsonValue portableValue(const QJsonValue &value,
                          const QString &runtimeRoot,
                          const QString &dataRoot,
                          const QString &projectPath,
                          QSet<QString> *referencedEntries,
                          QString *errorMessage,
                          bool *success)
{
    if (!*success)
    {
        return value;
    }
    if (value.isArray())
    {
        QJsonArray result;
        for (const QJsonValue &item : value.toArray())
        {
            result.append(portableValue(
                item,
                runtimeRoot,
                dataRoot,
                projectPath,
                referencedEntries,
                errorMessage,
                success));
        }
        return result;
    }
    if (value.isObject())
    {
        QJsonObject result;
        const QJsonObject object = value.toObject();
        for (auto it = object.constBegin(); it != object.constEnd(); ++it)
        {
            result.insert(
                it.key(),
                portableValue(
                    it.value(),
                    runtimeRoot,
                    dataRoot,
                    projectPath,
                    referencedEntries,
                    errorMessage,
                    success));
        }
        return result;
    }
    if (!value.isString())
    {
        return value;
    }

    const QString text = value.toString().trimmed();
    if (text.startsWith(QStringLiteral("plascan-diagnostic:///")))
    {
        return text;
    }
    const QString existingEntry =
        PortableProjectFormat::entryPathFromResourceUri(text);
    if (!existingEntry.isEmpty())
    {
        if (isChunkEntry(existingEntry))
        {
            const QString normalizedEntry =
                QStringLiteral("chunk/%1")
                    .arg(chunkRelativePath(existingEntry));
            referencedEntries->insert(normalizedEntry);
            return PortableProjectFormat::resourceUriForEntry(
                normalizedEntry);
        }
        if (existingEntry.startsWith(QStringLiteral("shared/")))
        {
            referencedEntries->insert(existingEntry);
        }
        return text;
    }

    QString candidate = text;
    if (!QFileInfo(candidate).isAbsolute()
        && (candidate.startsWith(QStringLiteral("assets/"))
            || candidate.startsWith(QStringLiteral("assets\\"))))
    {
        candidate = QDir(runtimeRoot).filePath(candidate);
    }

    if (QFileInfo(candidate).isAbsolute())
    {
        const QString uri =
            projectUri(runtimeRoot, dataRoot, candidate);
        if (!uri.isEmpty())
        {
            const QString entry =
                PortableProjectFormat::entryPathFromResourceUri(uri);
            if (QFileInfo(candidate).isFile())
            {
                referencedEntries->insert(entry);
            }
            return uri;
        }
    }

    const QString staged =
        stageExternalPath(candidate, runtimeRoot, projectPath, errorMessage);
    if (staged.isEmpty())
    {
        if (!errorMessage || errorMessage->isEmpty())
        {
            if (QFileInfo(candidate).isAbsolute())
            {
                return diagnosticPathToken(candidate);
            }
            return value;
        }
        *success = false;
        return value;
    }
    const QString uri = projectUri(runtimeRoot, dataRoot, staged);
    if (uri.isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("无法生成工程资源 URI: %1").arg(staged));
        *success = false;
        return value;
    }
    referencedEntries->insert(
        PortableProjectFormat::entryPathFromResourceUri(uri));
    return uri;
}

QJsonValue materializedValue(const QJsonValue &value,
                              const QString &runtimeRoot,
                              const QString &dataRoot,
                              QString *errorMessage,
                              bool *success)
{
    if (!*success)
    {
        return value;
    }
    if (value.isArray())
    {
        QJsonArray result;
        for (const QJsonValue &item : value.toArray())
        {
            result.append(materializedValue(
                item, runtimeRoot, dataRoot, errorMessage, success));
        }
        return result;
    }
    if (value.isObject())
    {
        QJsonObject result;
        const QJsonObject object = value.toObject();
        for (auto it = object.constBegin(); it != object.constEnd(); ++it)
        {
            result.insert(
                it.key(),
                materializedValue(
                    it.value(),
                    runtimeRoot,
                    dataRoot,
                    errorMessage,
                    success));
        }
        return result;
    }
    if (!value.isString())
    {
        return value;
    }

    const QString entry =
        PortableProjectFormat::entryPathFromResourceUri(value.toString());
    QString path;
    if (isChunkEntry(entry))
    {
        path = QDir(runtimeRoot).filePath(chunkRelativePath(entry));
    }
    else if (entry.startsWith(QStringLiteral("shared/")))
    {
        path = QDir(dataRoot).filePath(entry);
    }
    else
    {
        return value;
    }
    // 计划输出目录和尚未生成的产物也使用项目 URI。真实文件是否缺失由
    // resource_index 在 initializeRuntime() 中校验，未进入索引的计划路径
    // 仍解析为当前机器上的目标位置。
    return path;
}

} // namespace

ProjectWorkspaceStore::ProjectWorkspaceStore(const QString &projectPath,
                                             int chunkDirectory)
    : _projectPath(QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath())),
      _chunkDirectory(chunkDirectory)
{
}

bool ProjectWorkspaceStore::ensureProjectManifest(QString *errorMessage) const
{
    ProjectChunkStore chunkStore(_projectPath);
    if (!chunkStore.ensureLayout(errorMessage))
    {
        return false;
    }
    QJsonObject document;
    return chunkStore.loadProjectDocument(&document, errorMessage);
}

QString ProjectWorkspaceStore::runtimeRoot(QString *errorMessage) const
{
    if (!ensureProjectManifest(errorMessage))
    {
        return {};
    }
    ProjectChunkStore chunkStore(_projectPath);
    if (_chunkDirectory <= 0)
    {
        return chunkStore.defaultChunkDirectory(errorMessage);
    }

    bool found = false;
    for (const auto &chunk : chunkStore.chunks(errorMessage))
    {
        if (chunk.directory == _chunkDirectory)
        {
            found = true;
            break;
        }
    }
    if (!found)
    {
        setError(errorMessage,
                 QStringLiteral("项目中不存在 Chunk 数字目录: %1")
                     .arg(_chunkDirectory));
        return {};
    }
    return ProjectPackageLayout::chunkDirectory(
        _projectPath, _chunkDirectory);
}

bool ProjectWorkspaceStore::loadResourceIndex(
    ProjectResourceIndex *index,
    QString *errorMessage) const
{
    if (!index)
    {
        setError(errorMessage, QStringLiteral("资源索引输出参数为空"));
        return false;
    }

    ProjectChunkStore chunkStore(_projectPath);
    QJsonObject chunkDocument;
    const bool loaded = _chunkDirectory > 0
        ? chunkStore.readChunkDocument(
              _chunkDirectory, &chunkDocument, errorMessage)
        : chunkStore.readDefaultChunkDocument(
              &chunkDocument, errorMessage);
    if (!loaded)
    {
        return false;
    }

    QString indexError;
    const ProjectResourceIndex parsed =
        ProjectResourceIndex::fromJson(
            chunkDocument.value(
                QString::fromLatin1(
                    PortableProjectFormat::ResourceIndexSection))
                .toObject(),
            &indexError);
    if (!indexError.isEmpty())
    {
        setError(errorMessage, indexError);
        return false;
    }
    *index = parsed;
    return true;
}

bool ProjectWorkspaceStore::initializeRuntime(QString *runtimeRootOut,
                                              QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    const QString root = runtimeRoot(errorMessage);
    if (root.isEmpty() || !QDir().mkpath(root))
    {
        if (errorMessage && errorMessage->isEmpty())
        {
            *errorMessage =
                QStringLiteral("无法创建项目运行工作区: %1").arg(root);
        }
        return false;
    }

    ProjectResourceIndex index;
    if (!loadResourceIndex(&index, errorMessage))
    {
        return false;
    }

    const QString dataRoot =
        ProjectPackageLayout::dataDirectory(_projectPath);
    for (const ProjectResourceRef &resource : index.resources())
    {
        if (resource.kind != QStringLiteral("chunk")
            && resource.kind != QStringLiteral("workspace")
            && resource.kind != QStringLiteral("shared_image"))
        {
            continue;
        }
        const bool shared =
            resource.entryPath.startsWith(QStringLiteral("shared/"));
        if (!shared && !isChunkEntry(resource.entryPath))
        {
            setError(errorMessage,
                     QStringLiteral("工程资源路径无效: %1")
                          .arg(resource.entryPath));
            return false;
        }
        const QString destination = PortableProjectFormat::resolveEntryPath(
            shared ? dataRoot : root,
            shared ? resource.entryPath
                   : chunkRelativePath(resource.entryPath),
            errorMessage);
        if (destination.isEmpty())
        {
            return false;
        }
        bool validExisting = QFileInfo(destination).isFile()
            && QFileInfo(destination).size() == resource.size;
        const qint64 modifiedMs = QFileInfo(destination)
            .lastModified().toMSecsSinceEpoch();
        const bool unchanged =
            resource.modifiedMs > 0
            && resource.modifiedMs == modifiedMs;
        if (validExisting && !unchanged && !resource.sha256.isEmpty())
        {
            QString hashError;
            validExisting =
                fileSha256(destination, &hashError) == resource.sha256;
        }
        if (!validExisting)
        {
            setError(errorMessage,
                     QStringLiteral("工程资源缺失或校验失败: %1")
                         .arg(resource.entryPath));
            return false;
        }
    }

    ProjectIO::registerRuntimeRoot(_projectPath, root);
    if (runtimeRootOut)
    {
        *runtimeRootOut = root;
    }
    return true;
}

void ProjectWorkspaceStore::releaseRuntime() const
{
    ProjectIO::unregisterRuntimeRoot(_projectPath);
}

bool ProjectWorkspaceStore::prepareSplitMetadata(
    QJsonObject *core,
    QJsonObject *results,
    QJsonObject *resourceIndex,
    QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!core || !resourceIndex)
    {
        setError(errorMessage,
                 QStringLiteral("项目核心元数据或资源索引输出参数为空"));
        return false;
    }

    const QString root = runtimeRoot(errorMessage);
    if (root.isEmpty())
    {
        return false;
    }
    if (!QDir().mkpath(root))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建当前 Chunk 目录: %1").arg(root));
        return false;
    }

    const QString dataRoot =
        ProjectPackageLayout::dataDirectory(_projectPath);
    QSet<QString> referencedEntries;
    bool success = true;
    *core = portableValue(
                *core,
                root,
                dataRoot,
                _projectPath,
                &referencedEntries,
                errorMessage,
                &success)
                .toObject();
    if (results)
    {
        *results = portableValue(
                       *results,
                       root,
                       dataRoot,
                       _projectPath,
                       &referencedEntries,
                       errorMessage,
                       &success)
                       .toObject();
    }
    if (!success)
    {
        return false;
    }

    ProjectResourceIndex oldIndex;
    if (!loadResourceIndex(&oldIndex, errorMessage))
    {
        return false;
    }

    QMap<QString, ProjectResourceRef> oldProjectFiles;
    for (const ProjectResourceRef &resource : oldIndex.resources())
    {
        if (resource.kind == QStringLiteral("chunk")
            || resource.kind == QStringLiteral("workspace")
            || resource.kind == QStringLiteral("shared_image"))
        {
            oldProjectFiles.insert(resource.entryPath, resource);
        }
    }

    ProjectResourceIndex nextIndex = oldIndex;
    for (const ProjectResourceRef &resource : oldProjectFiles)
    {
        nextIndex.remove(resource.id);
    }

    for (const QString &entryPath : referencedEntries)
    {
        const bool shared =
            entryPath.startsWith(QStringLiteral("shared/"));
        const QString relative = shared
            ? entryPath
            : chunkRelativePath(entryPath);
        const QString path = PortableProjectFormat::resolveEntryPath(
            shared ? dataRoot : root,
            relative,
            errorMessage);
        const QFileInfo info(path);
        if (path.isEmpty() || !info.isFile())
        {
            continue;
        }

        const qint64 modifiedMs =
            info.lastModified().toMSecsSinceEpoch();
        const ProjectResourceRef old = oldProjectFiles.value(entryPath);
        QString checksum;
        if (!old.id.isEmpty()
            && old.size == info.size()
            && old.modifiedMs == modifiedMs
            && !old.sha256.isEmpty())
        {
            checksum = old.sha256;
        }
        else
        {
            checksum = fileSha256(path, errorMessage);
        }
        if (checksum.isEmpty())
        {
            return false;
        }

        ProjectResourceRef resource;
        resource.id = QStringLiteral("%1-%2")
            .arg(shared ? QStringLiteral("shared-image")
                        : QStringLiteral("chunk"),
                 stableToken(entryPath));
        resource.kind = shared
            ? QStringLiteral("shared_image")
            : QStringLiteral("chunk");
        resource.name = info.fileName();
        resource.entryPath = entryPath;
        resource.sha256 = checksum;
        resource.size = info.size();
        resource.modifiedMs = modifiedMs;
        if (!nextIndex.upsert(resource, errorMessage))
        {
            return false;
        }
    }
    *resourceIndex = nextIndex.toJson();
    return true;
}

bool ProjectWorkspaceStore::validateProjectLayout(
    QString *errorMessage) const
{
    return ensureProjectManifest(errorMessage);
}

bool ProjectWorkspaceStore::materializeMetadata(
    QJsonObject *metadata,
    QString *errorMessage) const
{
    if (!metadata)
    {
        setError(errorMessage, QStringLiteral("待解析项目元数据为空"));
        return false;
    }
    QString root;
    if (!initializeRuntime(&root, errorMessage))
    {
        return false;
    }
    bool success = true;
    const QString dataRoot =
        ProjectPackageLayout::dataDirectory(_projectPath);
    *metadata = materializedValue(
                    *metadata,
                    root,
                    dataRoot,
                    errorMessage,
                    &success)
                    .toObject();
    return success;
}
