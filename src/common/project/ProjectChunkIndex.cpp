#include "ProjectChunkIndex.h"

#include <QJsonArray>
#include <QRegularExpression>
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

bool isValidId(const QString &id)
{
    static const QRegularExpression expression(
        QStringLiteral("^[A-Za-z0-9][A-Za-z0-9._-]*$"));
    return expression.match(id).hasMatch();
}

QString createChunkId()
{
    return QUuid::createUuid().toString(QUuid::WithoutBraces);
}

} // namespace

bool ProjectChunkRecord::isValid(QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!isValidId(id))
    {
        setError(errorMessage, QStringLiteral("Chunk ID 无效: %1").arg(id));
        return false;
    }
    if (name.trimmed().isEmpty())
    {
        setError(errorMessage, QStringLiteral("Chunk 名称不能为空"));
        return false;
    }
    if (directory <= 0)
    {
        setError(errorMessage,
                 QStringLiteral("Chunk 数字目录必须为正整数: %1")
                     .arg(directory));
        return false;
    }
    if (order < 0)
    {
        setError(errorMessage,
                 QStringLiteral("Chunk 排序值不能为负数: %1").arg(order));
        return false;
    }
    if (revision < 0)
    {
        setError(errorMessage,
                 QStringLiteral("Chunk revision 不能为负数: %1")
                     .arg(revision));
        return false;
    }
    return true;
}

QJsonObject ProjectChunkRecord::toJson() const
{
    return QJsonObject{
        {QStringLiteral("id"), id},
        {QStringLiteral("name"), name},
        {QStringLiteral("directory"), QString::number(directory)},
        {QStringLiteral("storage"),
         QStringLiteral("%1/chunk.zip").arg(directory)},
        {QStringLiteral("order"), order},
        {QStringLiteral("revision"), revision}
    };
}

ProjectChunkRecord ProjectChunkRecord::fromJson(
    const QJsonObject &object,
    QString *errorMessage)
{
    ProjectChunkRecord record;
    record.id = object.value(QStringLiteral("id")).toString().trimmed();
    record.name = object.value(QStringLiteral("name")).toString().trimmed();
    record.directory =
        object.value(QStringLiteral("directory")).toString().toInt();
    if (record.directory <= 0)
    {
        const QString storage =
            object.value(QStringLiteral("storage")).toString();
        const qsizetype slash = storage.indexOf(QLatin1Char('/'));
        record.directory = storage.left(slash).toInt();
    }
    record.order = object.value(QStringLiteral("order")).toInt(-1);
    record.revision =
        object.value(QStringLiteral("revision")).toVariant().toLongLong();
    if (!record.isValid(errorMessage))
    {
        return {};
    }
    return record;
}

ProjectChunkIndex ProjectChunkIndex::createInitial(const QString &chunkName)
{
    ProjectChunkIndex index;
    ProjectChunkRecord record;
    record.id = createChunkId();
    record.name = chunkName.trimmed().isEmpty()
        ? QStringLiteral("区块 1")
        : chunkName.trimmed();
    record.directory = 1;
    record.order = 0;
    record.revision = 0;
    index._chunks.append(record);
    index._defaultChunkId = record.id;
    index._nextChunkDirectory = 2;
    return index;
}

bool ProjectChunkIndex::isValid(QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (_chunks.isEmpty())
    {
        setError(errorMessage, QStringLiteral("项目至少需要一个 Chunk"));
        return false;
    }

    QSet<QString> ids;
    QSet<int> directories;
    int maximumDirectory = 0;
    bool hasDefault = false;
    for (const ProjectChunkRecord &record : _chunks)
    {
        QString recordError;
        if (!record.isValid(&recordError))
        {
            setError(errorMessage, recordError);
            return false;
        }
        if (ids.contains(record.id))
        {
            setError(errorMessage,
                     QStringLiteral("Chunk 索引包含重复 ID: %1")
                         .arg(record.id));
            return false;
        }
        if (directories.contains(record.directory))
        {
            setError(errorMessage,
                     QStringLiteral("Chunk 索引包含重复数字目录: %1")
                         .arg(record.directory));
            return false;
        }
        ids.insert(record.id);
        directories.insert(record.directory);
        maximumDirectory = qMax(maximumDirectory, record.directory);
        hasDefault = hasDefault || record.id == _defaultChunkId;
    }
    if (!hasDefault)
    {
        setError(errorMessage,
                 QStringLiteral("默认 Chunk 不在 Chunk 索引中: %1")
                     .arg(_defaultChunkId));
        return false;
    }
    if (_nextChunkDirectory <= maximumDirectory)
    {
        setError(errorMessage,
                 QStringLiteral(
                     "next_chunk_directory 必须大于全部已分配目录: %1")
                     .arg(_nextChunkDirectory));
        return false;
    }
    return true;
}

bool ProjectChunkIndex::isEmpty() const
{
    return _chunks.isEmpty();
}

int ProjectChunkIndex::size() const
{
    return _chunks.size();
}

QList<ProjectChunkRecord> ProjectChunkIndex::chunks() const
{
    return _chunks;
}

QString ProjectChunkIndex::defaultChunkId() const
{
    return _defaultChunkId;
}

ProjectChunkRecord ProjectChunkIndex::defaultChunk() const
{
    return chunk(_defaultChunkId);
}

ProjectChunkRecord ProjectChunkIndex::chunk(const QString &chunkId) const
{
    for (const ProjectChunkRecord &record : _chunks)
    {
        if (record.id == chunkId)
        {
            return record;
        }
    }
    return {};
}

int ProjectChunkIndex::nextChunkDirectory() const
{
    return _nextChunkDirectory;
}

QString ProjectChunkIndex::appendChunk(const QString &name,
                                       QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (_nextChunkDirectory <= 0)
    {
        setError(errorMessage,
                 QStringLiteral("下一个 Chunk 数字目录无效"));
        return {};
    }

    ProjectChunkRecord record;
    record.id = createChunkId();
    record.directory = _nextChunkDirectory;
    record.name = name.trimmed().isEmpty()
        ? QStringLiteral("区块 %1").arg(record.directory)
        : name.trimmed();
    int maximumOrder = -1;
    for (const ProjectChunkRecord &existing : _chunks)
    {
        maximumOrder = qMax(maximumOrder, existing.order);
    }
    record.order = maximumOrder + 1;
    record.revision = 0;
    _chunks.append(record);
    ++_nextChunkDirectory;
    if (_defaultChunkId.isEmpty())
    {
        _defaultChunkId = record.id;
    }
    return record.id;
}

bool ProjectChunkIndex::removeChunk(const QString &chunkId,
                                    QString *errorMessage)
{
    if (_chunks.size() <= 1)
    {
        setError(errorMessage,
                 QStringLiteral("项目至少需要保留一个 Chunk"));
        return false;
    }
    for (qsizetype index = 0; index < _chunks.size(); ++index)
    {
        if (_chunks[index].id != chunkId)
        {
            continue;
        }
        _chunks.removeAt(index);
        if (_defaultChunkId == chunkId)
        {
            _defaultChunkId = _chunks.constFirst().id;
        }
        return true;
    }
    setError(errorMessage,
             QStringLiteral("Chunk 不存在: %1").arg(chunkId));
    return false;
}

bool ProjectChunkIndex::renameChunk(const QString &chunkId,
                                    const QString &name,
                                    QString *errorMessage)
{
    const QString trimmedName = name.trimmed();
    if (trimmedName.isEmpty())
    {
        setError(errorMessage, QStringLiteral("Chunk 名称不能为空"));
        return false;
    }
    for (ProjectChunkRecord &record : _chunks)
    {
        if (record.id == chunkId)
        {
            record.name = trimmedName;
            ++record.revision;
            return true;
        }
    }
    setError(errorMessage,
             QStringLiteral("Chunk 不存在: %1").arg(chunkId));
    return false;
}

bool ProjectChunkIndex::touchChunk(const QString &chunkId,
                                   QString *errorMessage)
{
    for (ProjectChunkRecord &record : _chunks)
    {
        if (record.id == chunkId)
        {
            ++record.revision;
            return true;
        }
    }
    setError(errorMessage,
             QStringLiteral("Chunk 不存在: %1").arg(chunkId));
    return false;
}

bool ProjectChunkIndex::setDefaultChunk(const QString &chunkId,
                                        QString *errorMessage)
{
    if (chunk(chunkId).id.isEmpty())
    {
        setError(errorMessage,
                 QStringLiteral("Chunk 不存在: %1").arg(chunkId));
        return false;
    }
    _defaultChunkId = chunkId;
    return true;
}

QJsonObject ProjectChunkIndex::toJson() const
{
    QJsonArray chunksArray;
    for (const ProjectChunkRecord &record : _chunks)
    {
        chunksArray.append(record.toJson());
    }
    return QJsonObject{
        {QStringLiteral("schema_version"), CurrentSchemaVersion},
        {QStringLiteral("default_chunk_id"), _defaultChunkId},
        {QStringLiteral("next_chunk_directory"), _nextChunkDirectory},
        {QStringLiteral("chunks"), chunksArray}
    };
}

ProjectChunkIndex ProjectChunkIndex::fromJson(const QJsonObject &object,
                                              QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    ProjectChunkIndex index;
    if (object.value(QStringLiteral("schema_version")).toInt(0)
        != CurrentSchemaVersion)
    {
        setError(errorMessage,
                 QStringLiteral("不支持的 Chunk 索引版本: %1")
                     .arg(object.value(QStringLiteral("schema_version"))
                              .toInt(0)));
        return {};
    }
    const QJsonValue chunksValue = object.value(QStringLiteral("chunks"));
    if (!chunksValue.isArray())
    {
        setError(errorMessage, QStringLiteral("Chunk 索引缺少 chunks 数组"));
        return {};
    }
    for (const QJsonValue &value : chunksValue.toArray())
    {
        if (!value.isObject())
        {
            setError(errorMessage,
                     QStringLiteral("Chunk 索引包含非对象条目"));
            return {};
        }
        QString recordError;
        const ProjectChunkRecord record =
            ProjectChunkRecord::fromJson(value.toObject(), &recordError);
        if (!recordError.isEmpty())
        {
            setError(errorMessage, recordError);
            return {};
        }
        index._chunks.append(record);
    }
    index._defaultChunkId =
        object.value(QStringLiteral("default_chunk_id")).toString();
    index._nextChunkDirectory =
        object.value(QStringLiteral("next_chunk_directory")).toInt(0);
    QString validationError;
    if (!index.isValid(&validationError))
    {
        setError(errorMessage, validationError);
        return {};
    }
    return index;
}

} // namespace xjw::common::project
