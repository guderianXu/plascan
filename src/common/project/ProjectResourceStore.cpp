#include "project/ProjectResourceStore.h"

#include "project/ProjectChunkStore.h"
#include "project/ProjectPackageLayout.h"

#include <QByteArrayView>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMimeDatabase>
#include <QRegularExpression>
#include <QSaveFile>
#include <QUuid>

namespace
{

using xjw::common::project::PortableProjectFormat;
using xjw::common::project::ProjectPackageLayout;
using xjw::common::project::ProjectResourceIndex;
using xjw::common::project::ProjectResourceRef;

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

QString sha256ForFile(const QString &path, QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法读取资源 %1: %2")
                     .arg(path, file.errorString()));
        return {};
    }

    QCryptographicHash hash(QCryptographicHash::Sha256);
    QByteArray buffer(1024 * 1024, Qt::Uninitialized);
    while (!file.atEnd())
    {
        const qint64 readCount = file.read(buffer.data(), buffer.size());
        if (readCount < 0)
        {
            setError(errorMessage,
                     QStringLiteral("计算资源校验值失败 %1: %2")
                         .arg(path, file.errorString()));
            return {};
        }
        if (readCount > 0)
        {
            hash.addData(QByteArrayView(buffer.constData(), readCount));
        }
    }
    return QString::fromLatin1(hash.result().toHex());
}

bool isSafeSegment(const QString &segment)
{
    static const QRegularExpression expression(
        QStringLiteral("^[A-Za-z0-9][A-Za-z0-9._-]*$"));
    return expression.match(segment).hasMatch();
}

QString normalizedFileName(const QString &sourcePath,
                           const QString &displayName)
{
    QString fileName = QFileInfo(displayName.trimmed()).fileName();
    if (fileName.isEmpty())
    {
        fileName = QFileInfo(sourcePath).fileName();
    }
    if (fileName.isEmpty())
    {
        fileName = QStringLiteral("resource.bin");
    }
    fileName.replace(QLatin1Char('/'), QLatin1Char('_'));
    fileName.replace(QLatin1Char('\\'), QLatin1Char('_'));
    return fileName;
}

bool copyFileAtomically(const QString &sourcePath,
                        const QString &destinationPath,
                        QString *errorMessage)
{
    QFile source(sourcePath);
    if (!source.open(QIODevice::ReadOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法读取资源 %1: %2")
                     .arg(sourcePath, source.errorString()));
        return false;
    }
    if (!QDir().mkpath(QFileInfo(destinationPath).absolutePath()))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建资源目录: %1")
                     .arg(QFileInfo(destinationPath).absolutePath()));
        return false;
    }

    QSaveFile destination(destinationPath);
    if (!destination.open(QIODevice::WriteOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法写入资源 %1: %2")
                     .arg(destinationPath, destination.errorString()));
        return false;
    }
    QByteArray buffer(1024 * 1024, Qt::Uninitialized);
    while (!source.atEnd())
    {
        const qint64 count = source.read(buffer.data(), buffer.size());
        if (count < 0
            || destination.write(buffer.constData(), count) != count)
        {
            setError(errorMessage,
                     QStringLiteral("复制项目资源失败: %1")
                         .arg(source.errorString().isEmpty()
                                  ? destination.errorString()
                                  : source.errorString()));
            return false;
        }
    }
    if (!destination.commit())
    {
        setError(errorMessage,
                 QStringLiteral("提交项目资源失败: %1")
                     .arg(destination.errorString()));
        return false;
    }
    return true;
}

QString chunkRelativeEntryPath(const QString &entryPath)
{
    if (entryPath.startsWith(QStringLiteral("chunk/")))
    {
        return entryPath.mid(QStringLiteral("chunk/").size());
    }
    if (entryPath.startsWith(QStringLiteral("workspace/")))
    {
        return entryPath.mid(QStringLiteral("workspace/").size());
    }
    return entryPath;
}

} // namespace

ProjectResourceStore::ProjectResourceStore(const QString &projectPath)
    : _projectPath(QDir::cleanPath(projectPath))
{
}

bool ProjectResourceStore::loadIndex(ProjectResourceIndex *index,
                                     QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!index)
    {
        setError(errorMessage, QStringLiteral("资源索引输出参数为空"));
        return false;
    }

    QJsonObject object;
    ProjectChunkStore chunkStore(_projectPath);
    if (!chunkStore.readDefaultChunkSection(
            QString::fromLatin1(
                PortableProjectFormat::ResourceIndexSection),
            &object,
            errorMessage))
    {
        return false;
    }

    QString indexError;
    const ProjectResourceIndex parsed =
        ProjectResourceIndex::fromJson(object, &indexError);
    if (!indexError.isEmpty())
    {
        setError(errorMessage, indexError);
        return false;
    }
    *index = parsed;
    return true;
}

bool ProjectResourceStore::saveIndex(const ProjectResourceIndex &index,
                                     QString *errorMessage) const
{
    if (!ProjectPackageLayout::ensureSplitLayout(
            _projectPath, errorMessage))
    {
        return false;
    }
    return ProjectChunkStore(_projectPath).writeDefaultChunkSections(
        {
            qMakePair(
                QString::fromLatin1(
                    PortableProjectFormat::ResourceIndexSection),
                index.toJson())
        },
        errorMessage);
}

bool ProjectResourceStore::importFile(
    const QString &sourcePath,
    const ProjectResourceImportOptions &options,
    ProjectResourceRef *resource,
    QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!resource)
    {
        setError(errorMessage, QStringLiteral("资源输出参数为空"));
        return false;
    }
    if (!ProjectPackageLayout::ensureSplitLayout(
            _projectPath, errorMessage))
    {
        return false;
    }

    const QFileInfo sourceInfo(sourcePath);
    if (!sourceInfo.isFile())
    {
        setError(errorMessage,
                 QStringLiteral("待导入资源不存在或不是文件: %1")
                     .arg(sourcePath));
        return false;
    }
    if (!isSafeSegment(options.kind))
    {
        setError(errorMessage,
                 QStringLiteral("资源类型无效: %1").arg(options.kind));
        return false;
    }

    const QString resourceId = options.resourceId.trimmed().isEmpty()
        ? QUuid::createUuid().toString(QUuid::WithoutBraces)
        : options.resourceId.trimmed();
    if (!isSafeSegment(resourceId))
    {
        setError(errorMessage,
                 QStringLiteral("资源 ID 无效: %1").arg(resourceId));
        return false;
    }

    const QString fileName =
        normalizedFileName(sourcePath, options.displayName);
    const QString entryPath = QStringLiteral("resources/%1/%2/%3")
                                  .arg(options.kind, resourceId, fileName);
    if (!PortableProjectFormat::isSafeEntryPath(entryPath))
    {
        setError(errorMessage,
                 QStringLiteral("生成的资源归档路径不安全: %1")
                     .arg(entryPath));
        return false;
    }

    QString hashError;
    const QString checksum =
        sha256ForFile(sourceInfo.absoluteFilePath(), &hashError);
    if (checksum.isEmpty())
    {
        setError(errorMessage, hashError);
        return false;
    }

    ProjectResourceRef imported;
    imported.id = resourceId;
    imported.kind = options.kind;
    imported.name = fileName;
    imported.entryPath = entryPath;
    imported.mediaType = options.mediaType.trimmed();
    if (imported.mediaType.isEmpty())
    {
        imported.mediaType =
            QMimeDatabase().mimeTypeForFile(sourceInfo).name();
    }
    imported.sha256 = checksum;
    imported.size = sourceInfo.size();

    QString validationError;
    if (!imported.isValid(&validationError))
    {
        setError(errorMessage, validationError);
        return false;
    }

    ProjectResourceIndex index;
    if (!loadIndex(&index, errorMessage))
    {
        return false;
    }
    if (index.contains(resourceId))
    {
        setError(errorMessage,
                 QStringLiteral("项目中已存在资源 ID: %1").arg(resourceId));
        return false;
    }

    Q_UNUSED(options.compression);
    const QString chunkRoot =
        ProjectChunkStore(_projectPath).defaultChunkDirectory(errorMessage);
    if (chunkRoot.isEmpty())
    {
        return false;
    }
    const QString destination = PortableProjectFormat::resolveEntryPath(
        chunkRoot,
        entryPath,
        errorMessage);
    if (destination.isEmpty())
    {
        return false;
    }
    if (!copyFileAtomically(
            sourceInfo.absoluteFilePath(), destination, errorMessage))
    {
        return false;
    }

    if (!index.upsert(imported, errorMessage)
        || !saveIndex(index, errorMessage))
    {
        QFile::remove(destination);
        return false;
    }

    *resource = imported;
    return true;
}

QString ProjectResourceStore::projectPath() const
{
    return _projectPath;
}

ProjectResourceResolver::ProjectResourceResolver(const QString &projectPath)
    : _projectPath(QDir::cleanPath(projectPath))
{
}

bool ProjectResourceResolver::materialize(
    const ProjectResourceRef &resource,
    QString *materializedPath,
    QString *errorMessage,
    const QString &cacheRoot) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!materializedPath)
    {
        setError(errorMessage, QStringLiteral("资源解析输出参数为空"));
        return false;
    }

    QString validationError;
    if (!resource.isValid(&validationError))
    {
        setError(errorMessage, validationError);
        return false;
    }
    if (!ProjectPackageLayout::ensureSplitLayout(
            _projectPath, errorMessage))
    {
        return false;
    }

    const QString chunkRoot =
        ProjectChunkStore(_projectPath).defaultChunkDirectory(errorMessage);
    if (chunkRoot.isEmpty())
    {
        return false;
    }
    const bool shared =
        resource.entryPath.startsWith(QStringLiteral("shared/"));
    const QString authoritativePath = PortableProjectFormat::resolveEntryPath(
        shared
            ? ProjectPackageLayout::dataDirectory(_projectPath)
            : chunkRoot,
        shared
            ? resource.entryPath
            : chunkRelativeEntryPath(resource.entryPath),
        errorMessage);
    if (authoritativePath.isEmpty())
    {
        return false;
    }
    bool authoritativeValid =
        QFileInfo(authoritativePath).isFile()
        && QFileInfo(authoritativePath).size() == resource.size;
    const bool unchanged =
        resource.modifiedMs > 0
        && resource.modifiedMs
            == QFileInfo(authoritativePath)
                   .lastModified().toMSecsSinceEpoch();
    if (authoritativeValid && !unchanged && !resource.sha256.isEmpty())
    {
        QString hashError;
        authoritativeValid =
            sha256ForFile(authoritativePath, &hashError) == resource.sha256;
    }
    if (!authoritativeValid)
    {
        setError(errorMessage,
                 QStringLiteral("项目资源缺失或损坏: %1")
                     .arg(resource.entryPath));
        return false;
    }

    authoritativeValid =
        QFileInfo(authoritativePath).isFile()
        && QFileInfo(authoritativePath).size() == resource.size;
    QString authoritativeHashError;
    if (authoritativeValid && !unchanged && !resource.sha256.isEmpty())
    {
        authoritativeValid =
            sha256ForFile(authoritativePath, &authoritativeHashError)
            == resource.sha256;
    }
    if (!authoritativeValid)
    {
        setError(
            errorMessage,
            authoritativeHashError.isEmpty()
                ? QStringLiteral("项目资源校验失败: %1")
                      .arg(resource.entryPath)
                : authoritativeHashError);
        return false;
    }

    if (cacheRoot.trimmed().isEmpty())
    {
        *materializedPath = authoritativePath;
        return true;
    }

    const QString destination = QDir(cacheRoot).filePath(
        QStringLiteral("%1/%2").arg(resource.id, resource.name));
    const QFileInfo existing(destination);
    if (existing.isFile() && existing.size() == resource.size)
    {
        QString hashError;
        if (resource.sha256.isEmpty()
            || sha256ForFile(destination, &hashError) == resource.sha256)
        {
            *materializedPath = destination;
            return true;
        }
    }

    if (!copyFileAtomically(
            authoritativePath, destination, errorMessage))
    {
        return false;
    }

    if (!resource.sha256.isEmpty())
    {
        QString hashError;
        const QString extractedHash = sha256ForFile(destination, &hashError);
        if (extractedHash != resource.sha256)
        {
            QFile::remove(destination);
            setError(errorMessage,
                     hashError.isEmpty()
                         ? QStringLiteral("资源校验失败: %1").arg(resource.name)
                         : hashError);
            return false;
        }
    }

    *materializedPath = destination;
    return true;
}

bool ProjectResourceResolver::materialize(
    const QString &resourceId,
    QString *materializedPath,
    QString *errorMessage,
    const QString &cacheRoot) const
{
    ProjectResourceIndex index;
    ProjectResourceStore store(_projectPath);
    if (!store.loadIndex(&index, errorMessage))
    {
        return false;
    }
    if (!index.contains(resourceId))
    {
        setError(errorMessage,
                 QStringLiteral("项目资源不存在: %1").arg(resourceId));
        return false;
    }
    return materialize(
        index.resource(resourceId), materializedPath, errorMessage, cacheRoot);
}

QString ProjectResourceResolver::defaultCacheRoot(QString *errorMessage) const
{
    if (!ProjectPackageLayout::ensureSplitLayout(
            _projectPath, errorMessage))
    {
        return {};
    }
    const QString root =
        ProjectChunkStore(_projectPath).defaultChunkDirectory(errorMessage);
    if (root.isEmpty())
    {
        return {};
    }
    if (!QDir().mkpath(root))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建项目资源缓存目录: %1").arg(root));
        return {};
    }
    return root;
}

QString ProjectResourceResolver::projectId(QString *errorMessage) const
{
    QJsonObject document;
    if (!ProjectChunkStore(_projectPath).loadProjectDocument(
            &document, errorMessage))
    {
        return {};
    }

    const QString id =
        document.value(QStringLiteral("project_id")).toString();
    if (id.isEmpty())
    {
        setError(errorMessage, QStringLiteral("项目 doc.json 缺少 project_id"));
    }
    return id;
}
