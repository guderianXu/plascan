#include "project/ProjectChunkStore.h"

#include "project/PlascanArchive.h"
#include "project/ProjectPackageLayout.h"
#include "project/ProjectSharedImageStore.h"
#include "project/PortableProjectFormat.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QUuid>

namespace
{

using xjw::common::project::PortableProjectFormat;
using xjw::common::project::ProjectChunkIndex;
using xjw::common::project::ProjectChunkRecord;
using xjw::common::project::ProjectPackageLayout;
using xjw::common::project::ProjectSharedImageStore;

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

QJsonObject readJsonObject(PlascanArchive &archive,
                           const QString &entryName,
                           QString *errorMessage)
{
    QString readError;
    const QByteArray data = archive.readEntry(entryName, &readError);
    if (!readError.isEmpty())
    {
        setError(errorMessage, readError);
        return {};
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        setError(errorMessage,
                 QStringLiteral("归档条目不是有效 JSON 对象 %1: %2")
                     .arg(entryName, parseError.errorString()));
        return {};
    }
    return document.object();
}

bool isWritableChunkSection(const QString &sectionName)
{
    return sectionName == QString::fromLatin1(
               PortableProjectFormat::ProjectFilesSection)
        || sectionName == QString::fromLatin1(
               PortableProjectFormat::ProjectResultsSection)
        || sectionName == QString::fromLatin1(
               PortableProjectFormat::ProjectConfigSection)
        || sectionName == QString::fromLatin1(
               PortableProjectFormat::ResourceIndexSection);
}

} // namespace

ProjectChunkStore::ProjectChunkStore(const QString &projectPath)
    : _projectPath(QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath()))
{
}

bool ProjectChunkStore::ensureLayout(QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!ProjectPackageLayout::ensureSplitLayout(_projectPath, errorMessage))
    {
        return false;
    }

    QJsonObject document;
    if (!loadProjectDocument(&document, errorMessage))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "不支持旧版工程格式；请新建 4.0 Chunk 工程后重新导入数据");
        }
        return false;
    }
    return validateCurrentLayout(errorMessage);
}

bool ProjectChunkStore::loadProjectDocument(
    QJsonObject *document,
    QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!document)
    {
        setError(errorMessage, QStringLiteral("项目文档输出参数为空"));
        return false;
    }
    PlascanArchive archive(_projectPath);
    if (!archive.isValid())
    {
        setError(errorMessage,
                 QStringLiteral("无法打开项目归档: %1").arg(_projectPath));
        return false;
    }
    const QString entry =
        QString::fromLatin1(PortableProjectFormat::DocumentEntry);
    if (!archive.containsEntry(entry))
    {
        setError(errorMessage,
                 QStringLiteral("项目归档缺少必要条目: %1").arg(entry));
        return false;
    }
    const QJsonObject parsed = readJsonObject(archive, entry, errorMessage);
    if (!PortableProjectFormat::isCurrentProjectDocument(parsed))
    {
        if (!errorMessage || errorMessage->isEmpty())
        {
            setError(errorMessage,
                     QStringLiteral("项目 doc.json 类型或版本无效"));
        }
        return false;
    }
    ProjectChunkIndex index;
    if (!PortableProjectFormat::readProjectIndex(
            parsed, &index, errorMessage))
    {
        return false;
    }
    *document = parsed;
    return true;
}

bool ProjectChunkStore::writeProjectDocument(
    const QJsonObject &document,
    QString *errorMessage) const
{
    if (!PortableProjectFormat::isCurrentProjectDocument(document))
    {
        setError(errorMessage, QStringLiteral("拒绝写入无效的项目 doc.json"));
        return false;
    }
    ProjectChunkIndex index;
    if (!PortableProjectFormat::readProjectIndex(
            document, &index, errorMessage))
    {
        return false;
    }
    PlascanArchive archive(_projectPath);
    if (!archive.isValid())
    {
        setError(errorMessage,
                 QStringLiteral("无法打开项目归档: %1").arg(_projectPath));
        return false;
    }
    return archive.writeEntry(
        QString::fromLatin1(PortableProjectFormat::DocumentEntry),
        QJsonDocument(document).toJson(QJsonDocument::Compact),
        errorMessage);
}

bool ProjectChunkStore::loadIndex(ProjectChunkIndex *index,
                                  QString *errorMessage) const
{
    QJsonObject document;
    if (!loadProjectDocument(&document, errorMessage))
    {
        return false;
    }
    return PortableProjectFormat::readProjectIndex(
        document, index, errorMessage);
}

bool ProjectChunkStore::saveIndex(const ProjectChunkIndex &index,
                                  QString *errorMessage) const
{
    QString validationError;
    if (!index.isValid(&validationError))
    {
        setError(errorMessage, validationError);
        return false;
    }
    QJsonObject document;
    if (!loadProjectDocument(&document, errorMessage))
    {
        return false;
    }
    document[QString::fromLatin1(
        PortableProjectFormat::ChunkIndexSection)] = index.toJson();
    return writeProjectDocument(document, errorMessage);
}

bool ProjectChunkStore::saveProjectUiState(
    const QJsonObject &uiState,
    QString *errorMessage) const
{
    QJsonObject document;
    if (!loadProjectDocument(&document, errorMessage))
    {
        return false;
    }
    document[QString::fromLatin1(
        PortableProjectFormat::ProjectUiStateSection)] = uiState;
    return writeProjectDocument(document, errorMessage);
}

QList<ProjectChunkRecord> ProjectChunkStore::chunks(
    QString *errorMessage) const
{
    ProjectChunkIndex index;
    return loadIndex(&index, errorMessage)
        ? index.chunks()
        : QList<ProjectChunkRecord>();
}

bool ProjectChunkStore::createChunk(
    const QString &name,
    const QJsonObject &projectFiles,
    const QJsonObject &projectResults,
    const QJsonObject &projectConfig,
    const QJsonObject &resourceIndex,
    ProjectChunkRecord *createdChunk,
    QString *errorMessage) const
{
    ProjectChunkIndex index;
    if (!loadIndex(&index, errorMessage))
    {
        return false;
    }

    QString indexError;
    const QString chunkId = index.appendChunk(name, &indexError);
    if (chunkId.isEmpty())
    {
        setError(errorMessage, indexError);
        return false;
    }
    const ProjectChunkRecord chunk = index.chunk(chunkId);
    const QString directory = ProjectPackageLayout::chunkDirectory(
        _projectPath, chunk.directory);
    if (QFileInfo::exists(directory))
    {
        setError(errorMessage,
                 QStringLiteral("Chunk 数字目录已存在，拒绝覆盖: %1")
                     .arg(directory));
        return false;
    }
    if (!QDir().mkpath(directory))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建 Chunk 数字目录: %1")
                     .arg(directory));
        return false;
    }
    const QJsonObject document =
        PortableProjectFormat::createChunkDocument(
            chunk,
            projectFiles,
            projectResults,
            projectConfig,
            resourceIndex);
    const QString archivePath = ProjectPackageLayout::chunkArchivePath(
        _projectPath, chunk.directory);
    QString archiveError;
    if (!PlascanArchive::createArchive(
            archivePath,
            {
                qMakePair(
                    QString::fromLatin1(
                        PortableProjectFormat::DocumentEntry),
                    QJsonDocument(document)
                        .toJson(QJsonDocument::Compact))
            },
            &archiveError))
    {
        QDir(directory).removeRecursively();
        setError(errorMessage, archiveError);
        return false;
    }
    if (!saveIndex(index, errorMessage))
    {
        QDir(directory).removeRecursively();
        return false;
    }
    if (createdChunk)
    {
        *createdChunk = chunk;
    }
    return true;
}

bool ProjectChunkStore::renameChunk(
    const QString &chunkId,
    const QString &name,
    QString *errorMessage) const
{
    ProjectChunkIndex index;
    if (!loadIndex(&index, errorMessage))
    {
        return false;
    }
    const ProjectChunkRecord previous = index.chunk(chunkId);
    QString indexError;
    if (previous.id.isEmpty()
        || !index.renameChunk(chunkId, name, &indexError))
    {
        setError(errorMessage,
                 previous.id.isEmpty()
                     ? QStringLiteral("Chunk 不存在: %1").arg(chunkId)
                     : indexError);
        return false;
    }

    const ProjectChunkRecord renamed = index.chunk(chunkId);
    QJsonObject document;
    if (!readChunkDocument(
            renamed.directory, &document, errorMessage))
    {
        return false;
    }
    document[QString::fromLatin1(
        PortableProjectFormat::ChunkRecordSection)] = renamed.toJson();
    if (!writeChunkDocument(
            renamed.directory, document, errorMessage))
    {
        return false;
    }
    if (!saveIndex(index, errorMessage))
    {
        document[QString::fromLatin1(
            PortableProjectFormat::ChunkRecordSection)] = previous.toJson();
        QString rollbackError;
        writeChunkDocument(
            previous.directory, document, &rollbackError);
        return false;
    }
    return true;
}

bool ProjectChunkStore::removeChunk(
    const QString &chunkId,
    QString *errorMessage) const
{
    ProjectChunkIndex index;
    if (!loadIndex(&index, errorMessage))
    {
        return false;
    }
    const ProjectChunkRecord removed = index.chunk(chunkId);
    if (removed.id.isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("Chunk 不存在: %1").arg(chunkId));
        return false;
    }
    ProjectChunkIndex updated = index;
    QString indexError;
    if (!updated.removeChunk(chunkId, &indexError))
    {
        setError(errorMessage, indexError);
        return false;
    }

    const QString directory = ProjectPackageLayout::chunkDirectory(
        _projectPath, removed.directory);
    const QString dataDirectory =
        ProjectPackageLayout::dataDirectory(_projectPath);
    const QString tombstone = QDir(dataDirectory).filePath(
        QStringLiteral(".chunk-%1-deleting-%2")
            .arg(removed.directory)
            .arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
    if (!QFileInfo(directory).isDir()
        || !QDir(dataDirectory).rename(directory, tombstone))
    {
        setError(errorMessage,
                 QStringLiteral("无法隔离待删除 Chunk 目录: %1")
                     .arg(directory));
        return false;
    }
    if (!saveIndex(updated, errorMessage))
    {
        QDir(dataDirectory).rename(tombstone, directory);
        return false;
    }
    if (!QDir(tombstone).removeRecursively())
    {
        QString rollbackError;
        saveIndex(index, &rollbackError);
        QDir(dataDirectory).rename(tombstone, directory);
        setError(errorMessage,
                 QStringLiteral("无法删除 Chunk 目录: %1")
                     .arg(directory));
        return false;
    }
    return ProjectSharedImageStore(_projectPath)
        .pruneUnreferenced(errorMessage);
}

bool ProjectChunkStore::setDefaultChunk(
    const QString &chunkId,
    QString *errorMessage) const
{
    ProjectChunkIndex index;
    if (!loadIndex(&index, errorMessage))
    {
        return false;
    }
    QString indexError;
    if (!index.setDefaultChunk(chunkId, &indexError))
    {
        setError(errorMessage, indexError);
        return false;
    }
    return saveIndex(index, errorMessage);
}

ProjectChunkRecord ProjectChunkStore::defaultChunk(
    QString *errorMessage) const
{
    ProjectChunkIndex index;
    if (!loadIndex(&index, errorMessage))
    {
        return {};
    }
    return index.defaultChunk();
}

QString ProjectChunkStore::defaultChunkDirectory(
    QString *errorMessage) const
{
    const ProjectChunkRecord chunk = defaultChunk(errorMessage);
    return chunk.id.isEmpty()
        ? QString()
        : ProjectPackageLayout::chunkDirectory(
              _projectPath, chunk.directory);
}

QString ProjectChunkStore::defaultChunkArchivePath(
    QString *errorMessage) const
{
    const ProjectChunkRecord chunk = defaultChunk(errorMessage);
    return chunk.id.isEmpty()
        ? QString()
        : ProjectPackageLayout::chunkArchivePath(
              _projectPath, chunk.directory);
}

bool ProjectChunkStore::readChunkDocument(
    int chunkDirectory,
    QJsonObject *document,
    QString *errorMessage) const
{
    if (!document)
    {
        setError(errorMessage, QStringLiteral("Chunk 文档输出参数为空"));
        return false;
    }
    const QString archivePath = ProjectPackageLayout::chunkArchivePath(
        _projectPath, chunkDirectory);
    PlascanArchive archive(
        archivePath, PlascanArchivePathType::DirectArchive);
    if (!archive.isValid())
    {
        setError(errorMessage,
                 QStringLiteral("无法打开 Chunk 归档: %1")
                     .arg(archivePath));
        return false;
    }
    const QString entry =
        QString::fromLatin1(PortableProjectFormat::DocumentEntry);
    if (!archive.containsEntry(entry))
    {
        setError(errorMessage,
                 QStringLiteral("Chunk 归档缺少必要条目: %1").arg(entry));
        return false;
    }
    const QJsonObject parsed = readJsonObject(
        archive, entry, errorMessage);
    if (!PortableProjectFormat::isCurrentChunkDocument(
            parsed, nullptr, errorMessage))
    {
        return false;
    }
    *document = parsed;
    return true;
}

bool ProjectChunkStore::writeChunkDocument(
    int chunkDirectory,
    const QJsonObject &document,
    QString *errorMessage) const
{
    if (!PortableProjectFormat::isCurrentChunkDocument(
            document, nullptr, errorMessage))
    {
        return false;
    }
    const QString archivePath = ProjectPackageLayout::chunkArchivePath(
        _projectPath, chunkDirectory);
    PlascanArchive archive(
        archivePath, PlascanArchivePathType::DirectArchive);
    if (!archive.isValid())
    {
        setError(errorMessage,
                 QStringLiteral("无法打开 Chunk 归档: %1")
                     .arg(archivePath));
        return false;
    }
    return archive.writeEntry(
        QString::fromLatin1(PortableProjectFormat::DocumentEntry),
        QJsonDocument(document).toJson(QJsonDocument::Compact),
        errorMessage);
}

bool ProjectChunkStore::readDefaultChunkDocument(
    QJsonObject *document,
    QString *errorMessage) const
{
    const ProjectChunkRecord chunk = defaultChunk(errorMessage);
    if (chunk.id.isEmpty()
        || !readChunkDocument(
            chunk.directory, document, errorMessage))
    {
        return false;
    }
    return PortableProjectFormat::isCurrentChunkDocument(
        *document, &chunk, errorMessage);
}

bool ProjectChunkStore::readDefaultChunkSection(
    const QString &sectionName,
    QJsonObject *section,
    QString *errorMessage) const
{
    if (!section)
    {
        setError(errorMessage, QStringLiteral("Chunk 字段输出参数为空"));
        return false;
    }
    QJsonObject document;
    if (!readDefaultChunkDocument(&document, errorMessage))
    {
        return false;
    }
    if (!document.value(sectionName).isObject())
    {
        setError(errorMessage,
                 QStringLiteral("Chunk doc.json 缺少对象字段: %1")
                     .arg(sectionName));
        return false;
    }
    *section = document.value(sectionName).toObject();
    return true;
}

bool ProjectChunkStore::writeDefaultChunkSections(
    const QVector<QPair<QString, QJsonObject>> &sections,
    QString *errorMessage) const
{
    const ProjectChunkRecord chunk = defaultChunk(errorMessage);
    return !chunk.id.isEmpty()
        && writeChunkSections(
            chunk.directory, sections, errorMessage);
}

bool ProjectChunkStore::writeChunkSections(
    int chunkDirectory,
    const QVector<QPair<QString, QJsonObject>> &sections,
    QString *errorMessage) const
{
    QJsonObject document;
    if (!readChunkDocument(
            chunkDirectory, &document, errorMessage))
    {
        return false;
    }
    for (const auto &section : sections)
    {
        if (!isWritableChunkSection(section.first))
        {
            setError(errorMessage,
                     QStringLiteral("不允许更新 Chunk doc.json 字段: %1")
                         .arg(section.first));
            return false;
        }
        document[section.first] =
            section.first == QString::fromLatin1(
                PortableProjectFormat::ProjectResultsSection)
            ? PortableProjectFormat::normalizeProjectResults(
                  section.second)
            : section.second;
    }

    ProjectChunkIndex index;
    if (!loadIndex(&index, errorMessage))
    {
        return false;
    }
    ProjectChunkRecord chunk;
    for (const ProjectChunkRecord &candidate : index.chunks())
    {
        if (candidate.directory == chunkDirectory)
        {
            chunk = candidate;
            break;
        }
    }
    if (chunk.id.isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("Chunk 数字目录不在根索引中: %1")
                     .arg(chunkDirectory));
        return false;
    }

    QString revisionError;
    if (!index.touchChunk(chunk.id, &revisionError))
    {
        setError(errorMessage, revisionError);
        return false;
    }
    const ProjectChunkRecord updated = index.chunk(chunk.id);
    document[QString::fromLatin1(
        PortableProjectFormat::ChunkRecordSection)] = updated.toJson();

    const QJsonObject previousDocument = [&]()
    {
        QJsonObject previous;
        readChunkDocument(chunkDirectory, &previous, nullptr);
        return previous;
    }();
    if (!writeChunkDocument(chunkDirectory, document, errorMessage))
    {
        return false;
    }
    if (!saveIndex(index, errorMessage))
    {
        QString rollbackError;
        writeChunkDocument(
            chunkDirectory, previousDocument, &rollbackError);
        return false;
    }
    return true;
}

bool ProjectChunkStore::validateCurrentLayout(
    QString *errorMessage) const
{
    QJsonObject projectDocument;
    if (!loadProjectDocument(&projectDocument, errorMessage))
    {
        return false;
    }
    ProjectChunkIndex index;
    if (!PortableProjectFormat::readProjectIndex(
            projectDocument, &index, errorMessage))
    {
        return false;
    }
    for (const ProjectChunkRecord &chunk : index.chunks())
    {
        const QString directory = ProjectPackageLayout::chunkDirectory(
            _projectPath, chunk.directory);
        const QString archivePath = ProjectPackageLayout::chunkArchivePath(
            _projectPath, chunk.directory);
        if (!QFileInfo(directory).isDir()
            || !QFileInfo(archivePath).isFile())
        {
            setError(errorMessage,
                     QStringLiteral("Chunk %1 的数字目录或归档不存在: %2")
                         .arg(chunk.name, directory));
            return false;
        }
        QJsonObject chunkDocument;
        if (!readChunkDocument(
                chunk.directory, &chunkDocument, errorMessage)
            || !PortableProjectFormat::isCurrentChunkDocument(
                chunkDocument, &chunk, errorMessage))
        {
            return false;
        }
    }
    return true;
}
