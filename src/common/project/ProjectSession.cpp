#include "project/ProjectSession.h"

#include "project/PlascanArchive.h"
#include "project/PortableProjectFormat.h"
#include "project/ProjectChunkStore.h"
#include "project/ProjectPackageLayout.h"
#include "project/ProjectLock.h"
#include "project/ProjectResourceStore.h"
#include "project/ProjectSharedImageStore.h"
#include "project/ProjectWorkspaceStore.h"
#include "Logger.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSet>
#include <QUuid>

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

QByteArray compactJson(const QJsonObject &object)
{
    return QJsonDocument(object).toJson(QJsonDocument::Compact);
}

QJsonObject defaultUiState()
{
    return QJsonObject{
        {QStringLiteral("schema_version"), 1},
        {QStringLiteral("display_settings"), QJsonObject{}}
    };
}

QJsonObject versionedResultRecord(const QJsonObject &record)
{
    QJsonObject versioned = record;
    if (!versioned.contains(QStringLiteral("schema_version")))
    {
        versioned[QStringLiteral("schema_version")] = 1;
    }
    return versioned;
}

QString canonicalPath(const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty()
        || trimmed.startsWith(QStringLiteral("plascan:///")))
    {
        return QDir::cleanPath(trimmed);
    }
    return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
}

} // namespace

ProjectSession::~ProjectSession()
{
    close();
}

bool ProjectSession::create(const QString &projectPath,
                            const QString &projectName,
                            QString *errorMessage)
{
    close();
    if (errorMessage)
    {
        errorMessage->clear();
    }

    const QString absolutePath =
        QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath());
    const QString dataDirectory =
        ProjectPackageLayout::dataDirectory(absolutePath);
    if (QFileInfo::exists(absolutePath)
        || QFileInfo::exists(dataDirectory))
    {
        setError(errorMessage,
                 QStringLiteral("项目文件或同名 .files 目录已存在: %1")
                     .arg(absolutePath));
        return false;
    }
    if (!QDir().mkpath(dataDirectory))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建项目数据目录: %1")
                     .arg(dataDirectory));
        return false;
    }

    const auto cleanup = [&]()
    {
        QFile::remove(absolutePath);
        QDir(dataDirectory).removeRecursively();
    };

    const QString projectId = PortableProjectFormat::createProjectId();
    const ProjectChunkIndex index = ProjectChunkIndex::createInitial();
    const ProjectChunkRecord chunk = index.defaultChunk();
    const QString chunkDirectory =
        ProjectPackageLayout::chunkDirectory(absolutePath, chunk.directory);
    if (!QDir().mkpath(chunkDirectory))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建默认 Chunk 目录: %1")
                     .arg(chunkDirectory));
        cleanup();
        return false;
    }
    QJsonObject config{
        {QStringLiteral("project_name"),
         projectName.trimmed().isEmpty()
             ? QFileInfo(absolutePath).completeBaseName()
             : projectName.trimmed()},
        {QStringLiteral("created_at"),
         QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},
        {QStringLiteral("version"),
         QString::fromLatin1(PortableProjectFormat::CurrentFormatVersion)},
        {QStringLiteral("schema_version"), 2},
        {QStringLiteral("project_id"), projectId}
    };
    const QJsonObject projectDocument =
        PortableProjectFormat::createProjectDocument(
            projectId, index, defaultUiState());
    const QJsonObject chunkDocument =
        PortableProjectFormat::createChunkDocument(
            chunk,
            QJsonObject{{QStringLiteral("images"), QJsonArray{}}},
            QJsonObject{},
            config,
            ProjectResourceIndex().toJson());

    QString error;
    if (!PlascanArchive::createArchive(
            ProjectPackageLayout::chunkArchivePath(
                absolutePath, chunk.directory),
            {
                qMakePair(
                    QString::fromLatin1(
                        PortableProjectFormat::DocumentEntry),
                    compactJson(chunkDocument))
            },
            &error)
        || !PlascanArchive::createArchive(
            ProjectPackageLayout::metadataArchivePath(absolutePath),
            {
                qMakePair(
                    QString::fromLatin1(
                        PortableProjectFormat::DocumentEntry),
                    compactJson(projectDocument))
            },
            &error)
        || !ProjectPackageLayout::writeDescriptor(absolutePath, &error))
    {
        cleanup();
        setError(errorMessage,
                 QStringLiteral("创建 4.0 Chunk 工程失败: %1").arg(error));
        return false;
    }

    if (!open(absolutePath, &error))
    {
        cleanup();
        setError(errorMessage, error);
        return false;
    }
    return true;
}

bool ProjectSession::open(const QString &projectPath,
                          QString *errorMessage)
{
    close();
    if (errorMessage)
    {
        errorMessage->clear();
    }

    _projectPath =
        QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath());
    ProjectChunkStore chunkStore(_projectPath);
    if (!chunkStore.ensureLayout(errorMessage))
    {
        _projectPath.clear();
        return false;
    }
    _lock = std::make_unique<ProjectLock>();
    if (!_lock->acquire(_projectPath, errorMessage)
        || !chunkStore.ensureLayout(errorMessage))
    {
        close();
        return false;
    }
    if (!load(errorMessage))
    {
        close();
        return false;
    }
    return true;
}

bool ProjectSession::openOrCreate(const QString &projectPath,
                                  const QString &projectName,
                                  QString *errorMessage)
{
    const QString absolutePath =
        QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath());
    if (QFileInfo::exists(absolutePath)
        || QFileInfo::exists(ProjectPackageLayout::dataDirectory(absolutePath)))
    {
        return open(absolutePath, errorMessage);
    }
    return create(absolutePath, projectName, errorMessage);
}

bool ProjectSession::selectChunk(const QString &chunkId,
                                 const QString &chunkName,
                                 QString *errorMessage)
{
    if (!isOpen())
    {
        setError(errorMessage, QStringLiteral("项目会话未打开"));
        return false;
    }
    ProjectChunkStore store(_projectPath);
    ProjectChunkIndex index;
    if (!store.loadIndex(&index, errorMessage))
    {
        return false;
    }

    ProjectChunkRecord selected;
    if (!chunkId.trimmed().isEmpty())
    {
        selected = index.chunk(chunkId.trimmed());
    }
    else if (!chunkName.trimmed().isEmpty())
    {
        for (const ProjectChunkRecord &candidate : index.chunks())
        {
            if (candidate.name == chunkName.trimmed())
            {
                if (!selected.id.isEmpty())
                {
                    setError(errorMessage,
                             QStringLiteral("Chunk 名称不唯一: %1")
                                 .arg(chunkName));
                    return false;
                }
                selected = candidate;
            }
        }
    }
    else
    {
        return true;
    }
    if (selected.id.isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("找不到指定 Chunk: %1")
                     .arg(chunkId.isEmpty() ? chunkName : chunkId));
        return false;
    }
    if (selected.id == _activeChunk.id)
    {
        return true;
    }

    if (!store.setDefaultChunk(selected.id, errorMessage))
    {
        return false;
    }
    if (_runtimeRegistered)
    {
        ProjectWorkspaceStore(_projectPath).releaseRuntime();
        _runtimeRegistered = false;
    }
    _activeChunk = {};
    _activeChunkRoot.clear();
    return load(errorMessage);
}

void ProjectSession::close()
{
    if (_runtimeRegistered && !_projectPath.isEmpty())
    {
        ProjectWorkspaceStore(_projectPath).releaseRuntime();
    }
    _projectPath.clear();
    _projectId.clear();
    _activeChunk = {};
    _activeChunkRoot.clear();
    _projectFiles = {};
    _projectResults = {};
    _projectConfig = {};
    _runtimeRegistered = false;
    if (_lock)
    {
        _lock->release();
        _lock.reset();
    }
}

bool ProjectSession::isOpen() const
{
    return !_projectPath.isEmpty() && !_activeChunk.id.isEmpty();
}

QString ProjectSession::projectPath() const
{
    return _projectPath;
}

QString ProjectSession::projectId() const
{
    return _projectId;
}

ProjectChunkRecord ProjectSession::activeChunk() const
{
    return _activeChunk;
}

QString ProjectSession::activeChunkRoot() const
{
    return _activeChunkRoot;
}

QJsonObject ProjectSession::projectFiles() const
{
    return _projectFiles;
}

QJsonObject ProjectSession::projectResults() const
{
    return _projectResults;
}

QJsonObject ProjectSession::projectConfig() const
{
    return _projectConfig;
}

QJsonObject ProjectSession::mergedMetadata() const
{
    QJsonObject merged = _projectFiles;
    for (auto it = _projectResults.constBegin();
         it != _projectResults.constEnd();
         ++it)
    {
        merged.insert(it.key(), it.value());
    }
    return merged;
}

bool ProjectSession::mergeImages(const QJsonArray &images,
                                 QString *errorMessage)
{
    if (!isOpen())
    {
        setError(errorMessage, QStringLiteral("项目会话未打开"));
        return false;
    }

    QJsonArray merged = _projectFiles.value(
        QStringLiteral("images")).toArray();
    QMap<QString, int> existingByPath;
    QSet<QString> usedIds;
    QStringList importedSharedPaths;
    for (int index = 0; index < merged.size(); ++index)
    {
        QJsonObject image = merged.at(index).toObject();
        QString imageId =
            image.value(QStringLiteral("image_uuid")).toString().trimmed();
        if (imageId.isEmpty() || usedIds.contains(imageId))
        {
            do
            {
                imageId = QUuid::createUuid().toString(
                    QUuid::WithoutBraces);
            }
            while (usedIds.contains(imageId));
            image[QStringLiteral("image_uuid")] = imageId;
            merged[index] = image;
        }
        usedIds.insert(imageId);
        existingByPath.insert(
            normalizedImagePath(
                image.value(QStringLiteral("path")).toString()),
            index);
    }

    for (const QJsonValue &value : images)
    {
        QJsonObject incoming = value.toObject();
        QString path = normalizedImagePath(
            incoming.value(QStringLiteral("path")).toString());
        if (path.isEmpty())
        {
            continue;
        }
        if (QFileInfo(path).isFile())
        {
            QString sharedUri;
            QString sharedPath;
            if (!ProjectSharedImageStore(_projectPath).importImage(
                    path,
                    &sharedUri,
                    &sharedPath,
                    errorMessage))
            {
                ProjectSharedImageStore(_projectPath)
                    .releaseReservations(importedSharedPaths);
                return false;
            }
            path = normalizedImagePath(sharedPath);
            importedSharedPaths.append(path);
        }
        incoming[QStringLiteral("path")] = path;
        if (!incoming.contains(QStringLiteral("name")))
        {
            incoming[QStringLiteral("name")] = QFileInfo(path).fileName();
        }

        const int existingIndex = existingByPath.value(path, -1);
        if (existingIndex >= 0)
        {
            QJsonObject combined = merged.at(existingIndex).toObject();
            for (auto it = incoming.constBegin();
                 it != incoming.constEnd();
                 ++it)
            {
                combined.insert(it.key(), it.value());
            }
            merged[existingIndex] = combined;
            continue;
        }

        QString imageId =
            incoming.value(QStringLiteral("image_uuid")).toString().trimmed();
        if (imageId.isEmpty() || usedIds.contains(imageId))
        {
            do
            {
                imageId = QUuid::createUuid().toString(
                    QUuid::WithoutBraces);
            }
            while (usedIds.contains(imageId));
        }
        incoming[QStringLiteral("image_uuid")] = imageId;
        usedIds.insert(imageId);
        existingByPath.insert(path, merged.size());
        merged.append(incoming);
    }

    ProjectSharedImageStore sharedImageStore(_projectPath);
    if (!sharedImageStore.publishReferences(
            importedSharedPaths, errorMessage))
    {
        sharedImageStore.releaseReservations(importedSharedPaths);
        return false;
    }
    _projectFiles[QStringLiteral("images")] = merged;
    return true;
}

bool ProjectSession::updateImageCameras(
    const QMap<QString, QJsonObject> &cameraMetaByImage,
    int *updatedCount,
    QString *errorMessage)
{
    if (updatedCount)
    {
        *updatedCount = 0;
    }
    if (!isOpen())
    {
        setError(errorMessage, QStringLiteral("项目会话未打开"));
        return false;
    }

    QMap<QString, QJsonObject> normalizedUpdates;
    QMap<QString, QJsonObject> uniqueUpdatesByFileName;
    QSet<QString> ambiguousFileNames;
    for (auto it = cameraMetaByImage.constBegin();
         it != cameraMetaByImage.constEnd();
         ++it)
    {
        const QString normalizedPath = normalizedImagePath(it.key());
        normalizedUpdates.insert(normalizedPath, it.value());

        // CLI 会把源影像导入共享资源目录，而空三结果仍以源路径为键。
        // 精确路径无法命中时，只允许通过唯一文件名建立别名；同名影像保持
        // 未匹配状态，避免把一个相机解误写到另一个影像记录。
        const QString fileName = QFileInfo(normalizedPath)
                                     .fileName()
                                     .toCaseFolded();
        if (fileName.isEmpty() || ambiguousFileNames.contains(fileName))
        {
            continue;
        }
        if (uniqueUpdatesByFileName.contains(fileName))
        {
            uniqueUpdatesByFileName.remove(fileName);
            ambiguousFileNames.insert(fileName);
            continue;
        }
        uniqueUpdatesByFileName.insert(fileName, it.value());
    }

    QJsonArray images =
        _projectFiles.value(QStringLiteral("images")).toArray();
    int count = 0;
    for (int index = 0; index < images.size(); ++index)
    {
        QJsonObject image = images.at(index).toObject();
        const QString imagePath = normalizedImagePath(
            image.value(QStringLiteral("path")).toString());
        const auto exactUpdate = normalizedUpdates.constFind(imagePath);
        QJsonObject cameraUpdate;
        if (exactUpdate != normalizedUpdates.constEnd())
        {
            cameraUpdate = exactUpdate.value();
        }
        else
        {
            const QString fileName = QFileInfo(imagePath)
                                         .fileName()
                                         .toCaseFolded();
            const auto uniqueUpdate = uniqueUpdatesByFileName.constFind(
                fileName);
            if (uniqueUpdate != uniqueUpdatesByFileName.constEnd())
            {
                cameraUpdate = uniqueUpdate.value();
            }
        }
        if (cameraUpdate.isEmpty())
        {
            continue;
        }
        image[QStringLiteral("camera")] = cameraUpdate;
        images[index] = image;
        ++count;
    }
    _projectFiles[QStringLiteral("images")] = images;
    if (updatedCount)
    {
        *updatedCount = count;
    }
    return true;
}

void ProjectSession::appendResult(const QString &arrayKey,
                                  const QJsonObject &record)
{
    QJsonArray records = _projectResults.value(arrayKey).toArray();
    records.append(versionedResultRecord(record));
    _projectResults[arrayKey] = records;
}

void ProjectSession::upsertResultByPath(
    const QString &arrayKey,
    const QString &pathKey,
    const QJsonObject &record)
{
    const QString targetPath = canonicalPath(
        record.value(pathKey).toString());
    const QString targetName = QFileInfo(targetPath).fileName();
    QJsonArray records;
    for (const QJsonValue &value :
         _projectResults.value(arrayKey).toArray())
    {
        const QString existingPath = canonicalPath(
            value.toObject().value(pathKey).toString());
        if (existingPath == targetPath
            || (!targetName.isEmpty()
                && QFileInfo(existingPath).fileName() == targetName))
        {
            continue;
        }
        records.append(value);
    }
    records.append(versionedResultRecord(record));
    _projectResults[arrayKey] = records;
}

bool ProjectSession::save(QString *errorMessage)
{
    if (!isOpen())
    {
        setError(errorMessage, QStringLiteral("项目会话未打开"));
        return false;
    }

    QJsonObject portableFiles = _projectFiles;
    QJsonObject portableResults = _projectResults;
    QJsonObject resourceIndex;
    ProjectWorkspaceStore workspace(
        _projectPath, _activeChunk.directory);
    if (!workspace.prepareSplitMetadata(
            &portableFiles,
            &portableResults,
            &resourceIndex,
            errorMessage))
    {
        return false;
    }

    ProjectChunkStore chunkStore(_projectPath);
    if (!chunkStore.writeChunkSections(
            _activeChunk.directory,
            {
                qMakePair(
                    QString::fromLatin1(
                        PortableProjectFormat::ProjectFilesSection),
                    portableFiles),
                qMakePair(
                    QString::fromLatin1(
                        PortableProjectFormat::ProjectResultsSection),
                    portableResults),
                qMakePair(
                    QString::fromLatin1(
                        PortableProjectFormat::ProjectConfigSection),
                    _projectConfig),
                qMakePair(
                    QString::fromLatin1(
                        PortableProjectFormat::ResourceIndexSection),
                    resourceIndex)
            },
            errorMessage))
    {
        return false;
    }
    QString garbageCollectionError;
    if (!ProjectSharedImageStore(_projectPath)
             .pruneUnreferenced(&garbageCollectionError))
    {
        LOG_WARN(
            QStringLiteral(
                "共享影像 GC 暂未完成，将在后续保存重试（项目 %1）: %2")
                .arg(_projectPath, garbageCollectionError));
    }
    return ProjectPackageLayout::pruneEmptyOptionalDirectories(
        _projectPath, _activeChunk.directory, errorMessage);
}

bool ProjectSession::load(QString *errorMessage)
{
    ProjectChunkStore chunkStore(_projectPath);
    QJsonObject projectDocument;
    if (!chunkStore.loadProjectDocument(&projectDocument, errorMessage))
    {
        return false;
    }
    _projectId =
        projectDocument.value(QStringLiteral("project_id")).toString();
    _activeChunk = chunkStore.defaultChunk(errorMessage);
    if (_activeChunk.id.isEmpty())
    {
        return false;
    }

    QJsonObject chunkDocument;
    if (!chunkStore.readChunkDocument(
            _activeChunk.directory, &chunkDocument, errorMessage))
    {
        return false;
    }
    _projectFiles = chunkDocument.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectFilesSection)).toObject();
    _projectResults = chunkDocument.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectResultsSection)).toObject();
    _projectConfig = chunkDocument.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectConfigSection)).toObject();

    ProjectWorkspaceStore workspace(
        _projectPath, _activeChunk.directory);
    if (!workspace.initializeRuntime(&_activeChunkRoot, errorMessage))
    {
        return false;
    }
    _runtimeRegistered = true;
    if (!workspace.materializeMetadata(&_projectFiles, errorMessage)
        || !workspace.materializeMetadata(&_projectResults, errorMessage))
    {
        return false;
    }
    return true;
}

QString ProjectSession::normalizedImagePath(const QString &path) const
{
    return canonicalPath(path);
}

} // namespace xjw::common::project
