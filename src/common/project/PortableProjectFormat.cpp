#include "PortableProjectFormat.h"

#include <QJsonArray>
#include <QRegularExpression>
#include <QUuid>

namespace xjw::common::project
{

namespace
{

constexpr auto ResourceUriPrefix = "plascan:///";

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

bool hasValidResourceId(const QString &resourceId)
{
    static const QRegularExpression expression(
        QStringLiteral("^[A-Za-z0-9][A-Za-z0-9._-]*$"));
    return expression.match(resourceId).hasMatch();
}

} // namespace

bool ProjectResourceRef::isValid(QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!hasValidResourceId(id))
    {
        setError(errorMessage, QStringLiteral("资源 ID 无效: %1").arg(id));
        return false;
    }
    if (!hasValidResourceId(kind))
    {
        setError(errorMessage, QStringLiteral("资源类型无效: %1").arg(kind));
        return false;
    }
    if (name.trimmed().isEmpty())
    {
        setError(errorMessage, QStringLiteral("资源显示名称不能为空"));
        return false;
    }
    if (!PortableProjectFormat::isSafeEntryPath(entryPath))
    {
        setError(errorMessage, QStringLiteral("资源归档路径不安全: %1").arg(entryPath));
        return false;
    }
    const QString normalized = PortableProjectFormat::normalizeEntryPath(entryPath);
    if (!normalized.startsWith(QStringLiteral("resources/"))
        && !normalized.startsWith(QStringLiteral("artifacts/"))
        && !normalized.startsWith(QStringLiteral("workspace/"))
        && !normalized.startsWith(QStringLiteral("shared/")))
    {
        setError(errorMessage,
                 QStringLiteral(
                     "资源必须位于 resources/、artifacts/ 或 workspace/ 下: %1")
                     .arg(entryPath));
        return false;
    }
    if (size < 0)
    {
        setError(errorMessage, QStringLiteral("资源大小不能为负数"));
        return false;
    }
    return true;
}

QJsonObject ProjectResourceRef::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("id"), id);
    object.insert(QStringLiteral("kind"), kind);
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("resource"),
                  PortableProjectFormat::resourceUriForEntry(entryPath));
    if (!mediaType.isEmpty())
    {
        object.insert(QStringLiteral("media_type"), mediaType);
    }
    if (!sha256.isEmpty())
    {
        object.insert(QStringLiteral("sha256"), sha256);
    }
    object.insert(QStringLiteral("size"), size);
    if (modifiedMs > 0)
    {
        object.insert(QStringLiteral("modified_ms"), modifiedMs);
    }
    return object;
}

ProjectResourceRef ProjectResourceRef::fromJson(const QJsonObject &object,
                                                QString *errorMessage)
{
    ProjectResourceRef resource;
    resource.id = object.value(QStringLiteral("id")).toString();
    resource.kind = object.value(QStringLiteral("kind")).toString();
    resource.name = object.value(QStringLiteral("name")).toString();
    resource.entryPath = PortableProjectFormat::entryPathFromResourceUri(
        object.value(QStringLiteral("resource")).toString());
    resource.mediaType = object.value(QStringLiteral("media_type")).toString();
    resource.sha256 = object.value(QStringLiteral("sha256")).toString();
    resource.size = object.value(QStringLiteral("size")).toVariant().toLongLong();
    resource.modifiedMs =
        object.value(QStringLiteral("modified_ms")).toVariant().toLongLong();
    if (!resource.isValid(errorMessage))
    {
        return {};
    }
    return resource;
}

bool ProjectResourceIndex::isEmpty() const
{
    return _resources.isEmpty();
}

int ProjectResourceIndex::size() const
{
    return _resources.size();
}

bool ProjectResourceIndex::contains(const QString &resourceId) const
{
    return _resources.contains(resourceId);
}

ProjectResourceRef ProjectResourceIndex::resource(const QString &resourceId) const
{
    return _resources.value(resourceId);
}

QList<ProjectResourceRef> ProjectResourceIndex::resources() const
{
    return _resources.values();
}

bool ProjectResourceIndex::upsert(const ProjectResourceRef &resource,
                                  QString *errorMessage)
{
    if (!resource.isValid(errorMessage))
    {
        return false;
    }
    const QString normalized_path =
        PortableProjectFormat::normalizeEntryPath(resource.entryPath).toCaseFolded();
    for (auto it = _resources.constBegin(); it != _resources.constEnd(); ++it)
    {
        if (it.key() != resource.id
            && PortableProjectFormat::normalizeEntryPath(
                   it.value().entryPath).toCaseFolded() == normalized_path)
        {
            setError(errorMessage,
                     QStringLiteral("资源归档路径冲突: %1 与 %2")
                         .arg(it.value().entryPath, resource.entryPath));
            return false;
        }
    }
    _resources.insert(resource.id, resource);
    return true;
}

bool ProjectResourceIndex::remove(const QString &resourceId)
{
    return _resources.remove(resourceId) > 0;
}

QJsonObject ProjectResourceIndex::toJson() const
{
    QJsonArray resourcesArray;
    for (const ProjectResourceRef &resource : _resources)
    {
        resourcesArray.append(resource.toJson());
    }

    return QJsonObject{
        {QStringLiteral("schema_version"), CurrentSchemaVersion},
        {QStringLiteral("resources"), resourcesArray}
    };
}

ProjectResourceIndex ProjectResourceIndex::fromJson(const QJsonObject &object,
                                                    QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }

    ProjectResourceIndex index;
    if (object.isEmpty())
    {
        return index;
    }

    const int version = object.value(QStringLiteral("schema_version")).toInt(0);
    if (version != CurrentSchemaVersion)
    {
        setError(errorMessage,
                 QStringLiteral("不支持的资源索引版本: %1").arg(version));
        return {};
    }

    const QJsonValue resourcesValue = object.value(QStringLiteral("resources"));
    if (!resourcesValue.isArray())
    {
        setError(errorMessage, QStringLiteral("资源索引缺少 resources 数组"));
        return {};
    }

    for (const QJsonValue &value : resourcesValue.toArray())
    {
        if (!value.isObject())
        {
            setError(errorMessage, QStringLiteral("资源索引包含非对象条目"));
            return {};
        }

        QString resourceError;
        const ProjectResourceRef resource =
            ProjectResourceRef::fromJson(value.toObject(), &resourceError);
        if (!resourceError.isEmpty())
        {
            setError(errorMessage, resourceError);
            return {};
        }
        if (index.contains(resource.id))
        {
            setError(errorMessage,
                     QStringLiteral("资源索引包含重复 ID: %1").arg(resource.id));
            return {};
        }
        index._resources.insert(resource.id, resource);
    }
    return index;
}

QString PortableProjectFormat::createProjectId()
{
    return QUuid::createUuid().toString(QUuid::WithoutBraces);
}

QJsonObject PortableProjectFormat::createProjectDocument(
    const QString &projectId,
    const ProjectChunkIndex &chunkIndex,
    const QJsonObject &uiState,
    const QString &createdWith)
{
    return QJsonObject{
        {QStringLiteral("type"), QString::fromLatin1(ProjectType)},
        {QStringLiteral("format_version"), QString::fromLatin1(CurrentFormatVersion)},
        {QStringLiteral("minimum_reader_version"),
         QString::fromLatin1(MinimumReaderVersion)},
        {QStringLiteral("project_id"), projectId},
        {QStringLiteral("created_with"), createdWith},
        {QString::fromLatin1(ChunkIndexSection), chunkIndex.toJson()},
        {QString::fromLatin1(ProjectUiStateSection), uiState}
    };
}

bool PortableProjectFormat::isCurrentProjectDocument(
    const QJsonObject &document)
{
    return document.value(QStringLiteral("type")).toString()
               == QString::fromLatin1(ProjectType)
        && document.value(QStringLiteral("format_version")).toString()
               == QString::fromLatin1(CurrentFormatVersion)
        && !document.value(QStringLiteral("project_id")).toString().isEmpty()
        && document.value(QString::fromLatin1(ChunkIndexSection)).isObject()
        && document.value(QString::fromLatin1(ProjectUiStateSection)).isObject();
}

bool PortableProjectFormat::readProjectIndex(
    const QJsonObject &document,
    ProjectChunkIndex *chunkIndex,
    QString *errorMessage)
{
    if (!chunkIndex)
    {
        setError(errorMessage, QStringLiteral("Chunk 索引输出参数为空"));
        return false;
    }
    if (!isCurrentProjectDocument(document))
    {
        setError(errorMessage, QStringLiteral("项目 doc.json 类型或版本无效"));
        return false;
    }
    QString indexError;
    const ProjectChunkIndex parsed = ProjectChunkIndex::fromJson(
        document.value(QString::fromLatin1(ChunkIndexSection)).toObject(),
        &indexError);
    if (!indexError.isEmpty())
    {
        setError(errorMessage, indexError);
        return false;
    }
    *chunkIndex = parsed;
    return true;
}

QJsonObject PortableProjectFormat::createChunkDocument(
    const ProjectChunkRecord &chunk,
    const QJsonObject &projectFiles,
    const QJsonObject &projectResults,
    const QJsonObject &projectConfig,
    const QJsonObject &resourceIndex)
{
    return QJsonObject{
        {QStringLiteral("type"), QString::fromLatin1(ChunkType)},
        {QStringLiteral("format_version"),
         QString::fromLatin1(ChunkFormatVersion)},
        {QString::fromLatin1(ChunkRecordSection), chunk.toJson()},
        {QString::fromLatin1(ProjectFilesSection), projectFiles},
        {QString::fromLatin1(ProjectResultsSection),
         normalizeProjectResults(projectResults)},
        {QString::fromLatin1(ProjectConfigSection), projectConfig},
        {QString::fromLatin1(ResourceIndexSection), resourceIndex}
    };
}

QJsonObject PortableProjectFormat::normalizeProjectResults(
    const QJsonObject &projectResults)
{
    QJsonObject normalized = projectResults;
    for (auto it = normalized.begin(); it != normalized.end(); ++it)
    {
        if (!it.value().isArray())
        {
            continue;
        }
        QJsonArray records;
        for (const QJsonValue &value : it.value().toArray())
        {
            if (!value.isObject())
            {
                records.append(value);
                continue;
            }
            QJsonObject record = value.toObject();
            if (!record.contains(QStringLiteral("schema_version")))
            {
                record[QStringLiteral("schema_version")] = 1;
            }
            records.append(record);
        }
        it.value() = records;
    }
    return normalized;
}

bool PortableProjectFormat::isCurrentChunkDocument(
    const QJsonObject &document,
    const ProjectChunkRecord *expectedChunk,
    QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (document.value(QStringLiteral("type")).toString()
            != QString::fromLatin1(ChunkType)
        || document.value(QStringLiteral("format_version")).toString()
            != QString::fromLatin1(ChunkFormatVersion))
    {
        setError(errorMessage, QStringLiteral("Chunk doc.json 类型或版本无效"));
        return false;
    }

    const QStringList requiredSections{
        QString::fromLatin1(ChunkRecordSection),
        QString::fromLatin1(ProjectFilesSection),
        QString::fromLatin1(ProjectResultsSection),
        QString::fromLatin1(ProjectConfigSection),
        QString::fromLatin1(ResourceIndexSection)
    };
    for (const QString &section : requiredSections)
    {
        if (!document.value(section).isObject())
        {
            setError(errorMessage,
                     QStringLiteral("Chunk doc.json 缺少对象字段: %1")
                         .arg(section));
            return false;
        }
    }

    QString chunkError;
    const ProjectChunkRecord chunk = ProjectChunkRecord::fromJson(
        document.value(QString::fromLatin1(ChunkRecordSection)).toObject(),
        &chunkError);
    if (!chunkError.isEmpty())
    {
        setError(errorMessage, chunkError);
        return false;
    }
    if (expectedChunk
        && (chunk.id != expectedChunk->id
            || chunk.directory != expectedChunk->directory))
    {
        setError(errorMessage,
                 QStringLiteral("Chunk 索引与 doc.json 身份不一致: %1")
                     .arg(expectedChunk->name));
        return false;
    }
    return true;
}

QString PortableProjectFormat::resourceUriForEntry(const QString &entryPath)
{
    if (!isSafeEntryPath(entryPath))
    {
        return {};
    }
    return QString::fromLatin1(ResourceUriPrefix) + normalizeEntryPath(entryPath);
}

QString PortableProjectFormat::entryPathFromResourceUri(const QString &resourceUri)
{
    if (!resourceUri.startsWith(QString::fromLatin1(ResourceUriPrefix)))
    {
        return {};
    }
    const QString entryPath =
        resourceUri.mid(static_cast<int>(qstrlen(ResourceUriPrefix)));
    return isSafeEntryPath(entryPath) ? normalizeEntryPath(entryPath) : QString();
}

} // namespace xjw::common::project
