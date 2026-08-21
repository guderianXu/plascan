#include "project/ProjectSharedImageStoreInternal.h"

#include "project/PortableProjectFormat.h"
#include "project/ProjectChunkStore.h"
#include "project/ProjectIO.h"
#include "project/ProjectPackageLayout.h"

#include <QByteArrayView>
#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QMutexLocker>
#include <QSaveFile>
#include <QScopeGuard>
#include <QSet>

#include <algorithm>
#include <limits>

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

void collectSharedReferences(const QJsonValue &value,
                             const QString &projectPath,
                             QSet<QString> *entries)
{
    if (value.isString())
    {
        const QString text = value.toString();
        const QString entry =
            PortableProjectFormat::entryPathFromResourceUri(text);
        if (entry.startsWith(QStringLiteral("shared/images/")))
        {
            entries->insert(entry);
            return;
        }
        const QString dataRoot = ProjectPackageLayout::dataDirectory(
            projectPath);
        const QString absolutePath = QDir::cleanPath(
            QFileInfo(text).absoluteFilePath());
        const QString relative = QDir::fromNativeSeparators(
            QDir(dataRoot).relativeFilePath(absolutePath));
        if (relative.startsWith(QStringLiteral("shared/images/"))
            && !relative.startsWith(QStringLiteral("../")))
        {
            entries->insert(relative);
        }
        return;
    }
    if (value.isArray())
    {
        for (const QJsonValue &item : value.toArray())
        {
            collectSharedReferences(item, projectPath, entries);
        }
        return;
    }
    if (value.isObject())
    {
        const QJsonObject object = value.toObject();
        for (auto it = object.constBegin(); it != object.constEnd(); ++it)
        {
            collectSharedReferences(it.value(), projectPath, entries);
        }
    }
}

bool collectTemporarySharedReferences(const QString &projectPath,
                                      QSet<QString> *entries,
                                      QString *errorMessage)
{
    const QString path = ProjectIO::tempFilesPath(projectPath);
    const QFileInfo info(path);
    constexpr qint64 MaximumTemporaryMetadataBytes = 64 * 1024 * 1024;
    if (!info.exists() && !info.isSymLink())
    {
        return true;
    }
    if (!info.isFile() || info.isSymLink())
    {
        setError(errorMessage,
                 QStringLiteral(
                     "临时项目元数据不是可信的普通文件，已停止共享影像 GC: %1")
                     .arg(path));
        return false;
    }
    if (info.size() <= 0 || info.size() > MaximumTemporaryMetadataBytes)
    {
        setError(errorMessage,
                 QStringLiteral(
                     "临时项目元数据大小无效，已停止共享影像 GC: %1")
                     .arg(path));
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage,
                 QStringLiteral(
                     "无法读取临时项目元数据，已停止共享影像 GC %1: %2")
                     .arg(path, file.errorString()));
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError
        || !document.isObject())
    {
        setError(errorMessage,
                 QStringLiteral(
                     "临时项目元数据 JSON 无效，已停止共享影像 GC %1: %2")
                     .arg(path, parseError.errorString()));
        return false;
    }
    collectSharedReferences(document.object(), projectPath, entries);
    return true;
}

QString garbageCollectionStatePath(const QString &projectPath)
{
    return QDir(ProjectPackageLayout::dataDirectory(projectPath)).filePath(
        QStringLiteral(".shared-image-gc.json"));
}

struct GarbageCollectionTombstone
{
    quint64 firstGeneration = 0;
    quint64 lastGeneration = 0;
};

struct GarbageCollectionState
{
    quint64 generation = 0;
    QString commitToken;
    QHash<QString, GarbageCollectionTombstone> tombstones;
};

QString commitTokenForChunks(QList<ProjectChunkRecord> chunks)
{
    std::sort(
        chunks.begin(),
        chunks.end(),
        [](const ProjectChunkRecord &left, const ProjectChunkRecord &right)
        {
            if (left.id != right.id)
            {
                return left.id < right.id;
            }
            return left.revision < right.revision;
        });
    QCryptographicHash hash(QCryptographicHash::Sha256);
    for (const ProjectChunkRecord &chunk : chunks)
    {
        hash.addData(chunk.id.toUtf8());
        hash.addData(QByteArrayView("\0", 1));
        hash.addData(QByteArray::number(chunk.revision));
        hash.addData(QByteArrayView("\n", 1));
    }
    return QString::fromLatin1(hash.result().toHex());
}

bool parseGeneration(const QJsonValue &value, quint64 *generation)
{
    if (!generation)
    {
        return false;
    }
    bool ok = false;
    const quint64 parsed = value.toString().toULongLong(&ok);
    if (!ok)
    {
        return false;
    }
    *generation = parsed;
    return true;
}

bool loadGarbageCollectionState(const QString &path,
                                GarbageCollectionState *state,
                                QString *errorMessage)
{
    if (!state)
    {
        setError(errorMessage, QStringLiteral("共享影像 GC 状态输出为空"));
        return false;
    }
    *state = {};
    if (!QFileInfo::exists(path))
    {
        return true;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage,
                 QStringLiteral("无法读取共享影像 GC 状态 %1: %2")
                     .arg(path, file.errorString()));
        return false;
    }
    constexpr qint64 MaximumStateBytes = 4 * 1024 * 1024;
    if (file.size() < 0 || file.size() > MaximumStateBytes)
    {
        setError(errorMessage,
                 QStringLiteral("共享影像 GC 状态大小无效: %1").arg(path));
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
    {
        setError(errorMessage,
                 QStringLiteral("共享影像 GC 状态 JSON 无效 %1: %2")
                     .arg(path, parseError.errorString()));
        return false;
    }

    const QJsonObject object = document.object();
    if (object.value(QStringLiteral("schema_version")).toInt() != 1
        || !parseGeneration(
            object.value(QStringLiteral("committed_generation")),
            &state->generation)
        || object.value(QStringLiteral("commit_token"))
               .toString()
               .trimmed()
               .isEmpty()
        || !object.value(QStringLiteral("tombstones")).isArray())
    {
        setError(errorMessage,
                 QStringLiteral("共享影像 GC 状态版本或代次无效: %1")
                     .arg(path));
        return false;
    }
    state->commitToken = object.value(
        QStringLiteral("commit_token")).toString();

    const QJsonArray tombstones = object.value(
        QStringLiteral("tombstones")).toArray();
    for (const QJsonValue &value : tombstones)
    {
        const QJsonObject tombstone = value.toObject();
        const QString entry = QDir::cleanPath(QDir::fromNativeSeparators(
            tombstone.value(QStringLiteral("entry")).toString()));
        GarbageCollectionTombstone parsed;
        if (!entry.startsWith(QStringLiteral("shared/images/"))
            || !parseGeneration(
                tombstone.value(QStringLiteral("first_generation")),
                &parsed.firstGeneration)
            || !parseGeneration(
                tombstone.value(QStringLiteral("last_generation")),
                &parsed.lastGeneration)
            || parsed.firstGeneration == 0
            || parsed.lastGeneration < parsed.firstGeneration
            || parsed.lastGeneration > state->generation)
        {
            setError(errorMessage,
                     QStringLiteral("共享影像 GC tombstone 无效: %1")
                         .arg(path));
            return false;
        }
        state->tombstones.insert(entry, parsed);
    }
    return true;
}

bool writeGarbageCollectionState(const QString &path,
                                 const GarbageCollectionState &state,
                                 QString *errorMessage)
{
    if (!QDir().mkpath(QFileInfo(path).absolutePath()))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建共享影像 GC 状态目录: %1")
                     .arg(QFileInfo(path).absolutePath()));
        return false;
    }

    QStringList entries = state.tombstones.keys();
    entries.sort();
    QJsonArray tombstones;
    for (const QString &entry : entries)
    {
        const GarbageCollectionTombstone tombstone =
            state.tombstones.value(entry);
        tombstones.append(QJsonObject{
            {QStringLiteral("entry"), entry},
            {QStringLiteral("first_generation"),
             QString::number(tombstone.firstGeneration)},
            {QStringLiteral("last_generation"),
             QString::number(tombstone.lastGeneration)}
        });
    }
    const QJsonObject object{
        {QStringLiteral("schema_version"), 1},
        {QStringLiteral("committed_generation"),
         QString::number(state.generation)},
        {QStringLiteral("commit_token"), state.commitToken},
        {QStringLiteral("tombstones"), tombstones}
    };

    const QByteArray data =
        QJsonDocument(object).toJson(QJsonDocument::Compact);
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)
        || file.write(data) != data.size()
        || !file.commit())
    {
        setError(errorMessage,
                 QStringLiteral("无法原子提交共享影像 GC 状态 %1: %2")
                     .arg(path, file.errorString()));
        return false;
    }
    return true;
}

bool removeStateWhenImageRootIsMissing(const QString &statePath,
                                       QString *errorMessage)
{
    if (!QFileInfo::exists(statePath) || QFile::remove(statePath))
    {
        return true;
    }
    setError(errorMessage,
             QStringLiteral("无法清理过期共享影像 GC 状态: %1")
                 .arg(statePath));
    return false;
}

} // namespace

bool pruneSharedImages(
    const QString &projectPath,
    const std::shared_ptr<SharedImageSynchronization> &synchronization,
    QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    QMutexLocker<QMutex> projectLock(&synchronization->mutex);
    if (!ensureCrossProcessLeaseLock(
            projectPath, synchronization, errorMessage))
    {
        return false;
    }
    const auto releaseUnusedLock = qScopeGuard(
        [&]()
        {
            releaseCrossProcessLeaseLockIfUnused(synchronization);
        });

    ProjectChunkStore store(projectPath);
    QString chunkError;
    const QList<ProjectChunkRecord> chunks = store.chunks(&chunkError);
    if (!chunkError.isEmpty())
    {
        setError(errorMessage, chunkError);
        return false;
    }
    const QString commitToken = commitTokenForChunks(chunks);
    QSet<QString> archivedReferences;
    for (const ProjectChunkRecord &chunk : chunks)
    {
        QJsonObject document;
        if (!store.readChunkDocument(
                chunk.directory, &document, errorMessage))
        {
            return false;
        }
        collectSharedReferences(
            document, projectPath, &archivedReferences);
    }
    QSet<QString> referenced = archivedReferences;
    if (!collectTemporarySharedReferences(
            projectPath, &referenced, errorMessage))
    {
        return false;
    }

    // 成功读取到的归档快照是已提交代次。只有它确实包含某个 URI 时，
    // 才结束该实体的所有导入 reservation。
    for (const QString &entry : archivedReferences)
    {
        synchronization->activeReservations.remove(entry);
    }

    const QString root = ProjectPackageLayout::sharedImagesDirectory(
        projectPath);
    const QString statePath = garbageCollectionStatePath(projectPath);
    if (!QFileInfo(root).isDir())
    {
        synchronization->activeReservations.clear();
        return removeStateWhenImageRootIsMissing(statePath, errorMessage);
    }

    GarbageCollectionState previousState;
    if (!loadGarbageCollectionState(
            statePath, &previousState, errorMessage))
    {
        return false;
    }
    const bool advancesGeneration =
        previousState.commitToken != commitToken;
    if (advancesGeneration
        && previousState.generation == std::numeric_limits<quint64>::max())
    {
        setError(errorMessage, QStringLiteral("共享影像 GC 代次已达到上限"));
        return false;
    }

    GarbageCollectionState nextState;
    nextState.generation = previousState.generation
        + (advancesGeneration ? 1 : 0);
    nextState.commitToken = commitToken;
    QStringList eligibleForDeletion;
    QDirIterator iterator(
        root,
        QDir::Files | QDir::NoDotAndDotDot,
        QDirIterator::Subdirectories);
    const QDir dataRoot(ProjectPackageLayout::dataDirectory(projectPath));
    while (iterator.hasNext())
    {
        const QString path = iterator.next();
        const QString entry = QDir::fromNativeSeparators(
            dataRoot.relativeFilePath(path));
        if (referenced.contains(entry)
            || synchronization->activeReservations.contains(entry))
        {
            continue;
        }

        GarbageCollectionTombstone tombstone =
            previousState.tombstones.value(entry);
        if (!advancesGeneration && tombstone.firstGeneration == 0)
        {
            // 当前 token 已经处理过；此时才解除的 lease 不能倒算为该代
            // “无 lease”，必须等待下一个真实提交代次再建立 tombstone。
            continue;
        }
        const bool consecutive =
            advancesGeneration
            && tombstone.lastGeneration == previousState.generation
            && tombstone.firstGeneration > 0;
        if (tombstone.firstGeneration == 0)
        {
            tombstone.firstGeneration = nextState.generation;
        }
        if (advancesGeneration || tombstone.lastGeneration == 0)
        {
            tombstone.lastGeneration = nextState.generation;
        }
        nextState.tombstones.insert(entry, tombstone);
        if (consecutive)
        {
            eligibleForDeletion.append(path);
        }
    }

    // 先原子记录当前代次的 tombstone，再执行不可逆删除。
    if (!writeGarbageCollectionState(statePath, nextState, errorMessage))
    {
        return false;
    }

    bool deletionFailed = false;
    QStringList deletionErrors;
    for (const QString &path : eligibleForDeletion)
    {
        const QString entry = QDir::fromNativeSeparators(
            dataRoot.relativeFilePath(path));
        if (QFile::remove(path))
        {
            nextState.tombstones.remove(entry);
        }
        else
        {
            deletionFailed = true;
            deletionErrors.append(path);
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
    if (!eligibleForDeletion.isEmpty()
        && !writeGarbageCollectionState(
            statePath, nextState, errorMessage))
    {
        return false;
    }
    if (deletionFailed)
    {
        setError(errorMessage,
                 QStringLiteral("无法清理未引用共享影像: %1")
                     .arg(deletionErrors.join(QStringLiteral(", "))));
        return false;
    }
    return true;
}

} // namespace xjw::common::project::detail
