#include "ProjectResourceCleanupTransaction.h"

#include "ProjectResourceCleanupArtifacts.h"

#include <QDir>
#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>
#include <QSet>
#include <QUuid>

namespace xjw::core::project::detail
{
namespace
{

constexpr int kCleanupTransactionSchemaVersion = 1;
constexpr qint64 kMaximumManifestBytes = 4 * 1024 * 1024;
constexpr qint64 kMaximumMetadataSnapshotBytes = 64 * 1024 * 1024;
constexpr int kMaximumManifestMoves = 10000;
constexpr auto kOriginalMetadataFile = "original_metadata.json";
constexpr auto kUpdatedMetadataFile = "updated_metadata.json";

QString manifestPath(const QString &transactionRoot)
{
    return QDir(transactionRoot).filePath(
        QStringLiteral("transaction.json"));
}

bool isFilesystemLink(const QFileInfo &info);

QByteArray metadataBytes(const QJsonObject &metadata)
{
    return QJsonDocument(metadata).toJson(QJsonDocument::Compact);
}

QByteArray metadataHash(const QByteArray &bytes)
{
    return QCryptographicHash::hash(
        bytes, QCryptographicHash::Sha256);
}

bool writeBytesAtomically(const QString &path,
                          const QByteArray &bytes,
                          QString *errorMessage)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)
        || file.write(bytes) != bytes.size()
        || !file.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "无法原子写入清理事务元数据快照：%1").arg(path);
        }
        return false;
    }
    return true;
}

bool loadMetadataSnapshot(const QString &transactionRoot,
                          const QString &fileName,
                          const QByteArray &expectedHash,
                          QJsonObject *metadata,
                          QString *errorMessage)
{
    if (!metadata
        || (fileName != QLatin1String(kOriginalMetadataFile)
            && fileName != QLatin1String(kUpdatedMetadataFile))
        || expectedHash.size() != 32)
    {
        return false;
    }
    const QString path = QDir(transactionRoot).filePath(fileName);
    const QFileInfo info(path);
    if (!info.isFile()
        || isFilesystemLink(info)
        || cleanupPathIdentity(info.absolutePath())
            != cleanupPathIdentity(transactionRoot)
        || info.size() <= 0
        || info.size() > kMaximumMetadataSnapshotBytes)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务元数据快照缺失或不安全：%1").arg(path);
        }
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }
    const QByteArray bytes = file.readAll();
    if (metadataHash(bytes) != expectedHash)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务元数据快照哈希不匹配：%1").arg(path);
        }
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        bytes, &parseError);
    if (parseError.error != QJsonParseError::NoError
        || !document.isObject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务元数据快照格式无效：%1").arg(path);
        }
        return false;
    }
    *metadata = document.object();
    return true;
}

bool isFilesystemLink(const QFileInfo &info)
{
    bool link = info.isSymLink();
#ifdef Q_OS_WIN
    link = link || info.isJunction();
#endif
    return link;
}

bool hasParentTraversal(const QString &path)
{
    return QDir::fromNativeSeparators(path)
        .split(QLatin1Char('/'), Qt::SkipEmptyParts)
        .contains(QLatin1String(".."));
}

QString transactionStateName(CleanupTransactionState state)
{
    return state == CleanupTransactionState::MetadataCommitted
        ? QStringLiteral("metadata_committed")
        : QStringLiteral("staging");
}

QString moveStateName(CleanupMoveState state)
{
    return state == CleanupMoveState::Staged
        ? QStringLiteral("staged")
        : QStringLiteral("planned");
}

bool parseTransactionState(const QString &text,
                           CleanupTransactionState *state)
{
    if (!state)
    {
        return false;
    }
    if (text == QLatin1String("staging"))
    {
        *state = CleanupTransactionState::Staging;
        return true;
    }
    if (text == QLatin1String("metadata_committed"))
    {
        *state = CleanupTransactionState::MetadataCommitted;
        return true;
    }
    return false;
}

bool parseMoveState(const QString &text, CleanupMoveState *state)
{
    if (!state)
    {
        return false;
    }
    if (text == QLatin1String("planned"))
    {
        *state = CleanupMoveState::Planned;
        return true;
    }
    if (text == QLatin1String("staged"))
    {
        *state = CleanupMoveState::Staged;
        return true;
    }
    return false;
}

bool parseSha256(const QString &text, QByteArray *hash)
{
    if (!hash || text.size() != 64)
    {
        return false;
    }
    for (const QChar character : text)
    {
        const ushort value = character.unicode();
        if (!((value >= '0' && value <= '9')
              || (value >= 'a' && value <= 'f')
              || (value >= 'A' && value <= 'F')))
        {
            return false;
        }
    }
    *hash = QByteArray::fromHex(text.toLatin1());
    return hash->size() == 32;
}

QJsonObject manifestJson(const CleanupTransactionManifest &manifest)
{
    QJsonArray moves;
    for (const CleanupTransactionMove &move : manifest.moves)
    {
        moves.append(QJsonObject{
            {QStringLiteral("source"), move.source},
            {QStringLiteral("destination"), move.destination},
            {QStringLiteral("kind"), move.directory
                 ? QStringLiteral("directory")
                 : QStringLiteral("file")},
            {QStringLiteral("state"), moveStateName(move.state)}
        });
    }
    return QJsonObject{
        {QStringLiteral("schema_version"),
         kCleanupTransactionSchemaVersion},
        {QStringLiteral("type"),
         QStringLiteral("plascan_resource_cleanup_transaction")},
        {QStringLiteral("transaction_id"), manifest.transactionId},
        {QStringLiteral("project_path"), manifest.projectPath},
        {QStringLiteral("chunk_id"), manifest.chunkId},
        {QStringLiteral("chunk_directory"), manifest.chunkDirectory},
        {QStringLiteral("project_root"), manifest.projectRoot},
        {QStringLiteral("managed_root"), manifest.managedRoot},
        {QStringLiteral("metadata_wal"), QJsonObject{
             {QStringLiteral("original_file"),
              QString::fromLatin1(kOriginalMetadataFile)},
             {QStringLiteral("original_sha256"),
              QString::fromLatin1(
                  manifest.originalMetadataHash.toHex())},
             {QStringLiteral("updated_file"),
              QString::fromLatin1(kUpdatedMetadataFile)},
             {QStringLiteral("updated_sha256"),
              QString::fromLatin1(
                  manifest.updatedMetadataHash.toHex())}
         }},
        {QStringLiteral("state"), transactionStateName(manifest.state)},
        {QStringLiteral("moves"), moves}
    };
}

bool writeManifest(const QString &transactionRoot,
                   const CleanupTransactionManifest &manifest,
                   QString *errorMessage)
{
    const QFileInfo rootInfo(transactionRoot);
    if (!rootInfo.isDir() || isFilesystemLink(rootInfo))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务目录不是安全的实体目录：%1")
                                .arg(transactionRoot);
        }
        return false;
    }

    const QByteArray bytes = QJsonDocument(manifestJson(manifest)).toJson(
        QJsonDocument::Indented);
    QSaveFile file(manifestPath(transactionRoot));
    if (!file.open(QIODevice::WriteOnly)
        || file.write(bytes) != bytes.size()
        || !file.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "无法原子写入清理事务恢复清单：%1")
                                .arg(manifestPath(transactionRoot));
        }
        return false;
    }
    return true;
}

bool validateMove(const CleanupTransactionManifest &manifest,
                  const QString &transactionRoot,
                  const CleanupTransactionMove &move,
                  QString *errorMessage)
{
    const QFileInfo sourceInfo(move.source);
    const QFileInfo destinationInfo(move.destination);
    const bool pathsValid = sourceInfo.isAbsolute()
        && destinationInfo.isAbsolute()
        && !hasParentTraversal(move.source)
        && !hasParentTraversal(move.destination)
        && cleanupPathIsInside(manifest.managedRoot, move.source, false)
        && !cleanupPathIsProtected(manifest.managedRoot, move.source)
        && !cleanupPathTraversesLink(manifest.managedRoot, move.source)
        && cleanupPathIsInside(transactionRoot, move.destination, false)
        && cleanupPathIdentity(destinationInfo.absolutePath())
            == cleanupPathIdentity(transactionRoot)
        && cleanupPathIdentity(move.source)
            != cleanupPathIdentity(move.destination);
    if (!pathsValid)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务包含越界或不安全路径：%1 -> %2")
                                .arg(move.source, move.destination);
        }
        return false;
    }

    const auto validateExisting = [&](const QFileInfo &info)
    {
        if (!info.exists() && !info.isSymLink())
        {
            return true;
        }
        if (isFilesystemLink(info)
            || info.isDir() != move.directory)
        {
            return false;
        }
        return !move.directory
            || !cleanupDirectoryContainsLink(info.absoluteFilePath());
    };
    if (!validateExisting(sourceInfo)
        || !validateExisting(destinationInfo))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务路径类型已变化或包含链接：%1 -> %2")
                                .arg(move.source, move.destination);
        }
        return false;
    }
    return true;
}

bool removeTransactionDirectory(const QString &transactionRoot,
                                QString *errorMessage)
{
    const QFileInfo rootInfo(transactionRoot);
    if (!rootInfo.isDir()
        || isFilesystemLink(rootInfo)
        || cleanupDirectoryContainsLink(transactionRoot)
        || !QDir(transactionRoot).removeRecursively())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "无法安全清理事务目录：%1").arg(transactionRoot);
        }
        return false;
    }
    return true;
}

} // namespace

QString cleanupTrashBase(const QString &managedRoot)
{
    return QDir(managedRoot).filePath(
        QStringLiteral(".plascan_cleanup_trash"));
}

bool createCleanupTransaction(
    const ResourceCleanupPlan &plan,
    QString *transactionRoot,
    CleanupTransactionManifest *manifest,
    QString *errorMessage)
{
    if (!transactionRoot || !manifest)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("清理事务输出参数为空");
        }
        return false;
    }
    const QString trashBase = cleanupTrashBase(plan.managedRoot);
    if (!QDir().mkpath(trashBase))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "无法创建项目清理事务根目录：%1").arg(trashBase);
        }
        return false;
    }
    const QFileInfo trashInfo(trashBase);
    if (!trashInfo.isDir()
        || isFilesystemLink(trashInfo)
        || cleanupPathTraversesLink(plan.managedRoot, trashBase))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "项目清理事务根目录无效：%1").arg(trashBase);
        }
        return false;
    }

    manifest->transactionId = QUuid::createUuid().toString(
        QUuid::WithoutBraces);
    manifest->projectPath = plan.projectPath;
    manifest->chunkId = plan.chunkId;
    manifest->chunkDirectory = plan.chunkDirectory;
    manifest->projectRoot = plan.projectRoot;
    manifest->managedRoot = plan.managedRoot;
    manifest->originalMetadata = plan.originalMetadata;
    manifest->updatedMetadata = plan.updatedMetadata;
    const QByteArray originalBytes = metadataBytes(
        manifest->originalMetadata);
    const QByteArray updatedBytes = metadataBytes(
        manifest->updatedMetadata);
    manifest->originalMetadataHash = metadataHash(originalBytes);
    manifest->updatedMetadataHash = metadataHash(updatedBytes);
    if (originalBytes.size() > kMaximumMetadataSnapshotBytes
        || updatedBytes.size() > kMaximumMetadataSnapshotBytes)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务元数据快照超过 64 MiB 安全上限");
        }
        removeEmptyCleanupTrashBase(plan.managedRoot);
        return false;
    }
    manifest->state = CleanupTransactionState::Staging;
    manifest->moves.clear();
    *transactionRoot = QDir(trashBase).filePath(manifest->transactionId);
    const bool transactionCreated = QDir(trashBase).mkdir(
        manifest->transactionId);
    const bool originalWritten = transactionCreated
        && writeBytesAtomically(
            QDir(*transactionRoot).filePath(
                QString::fromLatin1(kOriginalMetadataFile)),
            originalBytes,
            errorMessage);
    const bool updatedWritten = originalWritten
        && writeBytesAtomically(
            QDir(*transactionRoot).filePath(
                QString::fromLatin1(kUpdatedMetadataFile)),
            updatedBytes,
            errorMessage);
    if (!updatedWritten
        || !writeManifest(*transactionRoot, *manifest, errorMessage))
    {
        removeTransactionDirectory(*transactionRoot, nullptr);
        removeEmptyCleanupTrashBase(plan.managedRoot);
        return false;
    }
    return true;
}

bool appendPlannedCleanupMove(
    const QString &transactionRoot,
    CleanupTransactionManifest *manifest,
    const CleanupTransactionMove &move,
    QString *errorMessage)
{
    if (!manifest
        || !validateMove(*manifest, transactionRoot, move, errorMessage))
    {
        return false;
    }
    CleanupTransactionMove planned = move;
    planned.state = CleanupMoveState::Planned;
    manifest->moves.append(planned);
    if (!writeManifest(transactionRoot, *manifest, errorMessage))
    {
        manifest->moves.removeLast();
        return false;
    }
    return true;
}

bool markCleanupMoveStaged(
    const QString &transactionRoot,
    CleanupTransactionManifest *manifest,
    int moveIndex,
    QString *errorMessage)
{
    if (!manifest
        || moveIndex < 0
        || moveIndex >= manifest->moves.size())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("清理事务移动索引无效");
        }
        return false;
    }
    manifest->moves[moveIndex].state = CleanupMoveState::Staged;
    return writeManifest(transactionRoot, *manifest, errorMessage);
}

bool markCleanupMetadataCommitted(
    const QString &transactionRoot,
    CleanupTransactionManifest *manifest,
    QString *errorMessage)
{
    if (!manifest)
    {
        return false;
    }
    manifest->state = CleanupTransactionState::MetadataCommitted;
    return writeManifest(transactionRoot, *manifest, errorMessage);
}

bool loadAndValidateCleanupTransaction(
    const CleanupTransactionSession &session,
    const QString &transactionRoot,
    CleanupTransactionManifest *manifest,
    QString *errorMessage)
{
    if (!manifest)
    {
        return false;
    }
    const QFileInfo rootInfo(transactionRoot);
    const QFileInfo fileInfo(manifestPath(transactionRoot));
    if (!rootInfo.isDir()
        || isFilesystemLink(rootInfo)
        || !fileInfo.isFile()
        || isFilesystemLink(fileInfo)
        || fileInfo.size() <= 0
        || fileInfo.size() > kMaximumManifestBytes)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务恢复清单缺失或不安全：%1")
                                .arg(transactionRoot);
        }
        return false;
    }

    QFile file(fileInfo.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        file.readAll(), &parseError);
    const QJsonObject object = document.object();
    if (parseError.error != QJsonParseError::NoError
        || !document.isObject()
        || object.value(QStringLiteral("schema_version")).toInt()
            != kCleanupTransactionSchemaVersion
        || object.value(QStringLiteral("type")).toString()
            != QStringLiteral("plascan_resource_cleanup_transaction"))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务恢复清单格式无效：%1")
                                .arg(fileInfo.absoluteFilePath());
        }
        return false;
    }

    manifest->transactionId = object.value(
        QStringLiteral("transaction_id")).toString();
    manifest->projectPath = object.value(
        QStringLiteral("project_path")).toString();
    manifest->chunkId = object.value(QStringLiteral("chunk_id")).toString();
    manifest->chunkDirectory = object.value(
        QStringLiteral("chunk_directory")).toInt(-1);
    manifest->projectRoot = object.value(
        QStringLiteral("project_root")).toString();
    manifest->managedRoot = object.value(
        QStringLiteral("managed_root")).toString();
    if (!parseTransactionState(
            object.value(QStringLiteral("state")).toString(),
            &manifest->state))
    {
        return false;
    }

    const QString currentProjectPath = QDir::cleanPath(
        QFileInfo(session.projectPath).absoluteFilePath());
    const QString trashBase = cleanupTrashBase(session.managedRoot);
    const bool identityMatches = manifest->transactionId == rootInfo.fileName()
        && cleanupPathIdentity(rootInfo.absolutePath())
            == cleanupPathIdentity(trashBase)
        && cleanupPathIdentity(manifest->projectPath)
            == cleanupPathIdentity(currentProjectPath)
        && manifest->chunkId == session.chunkId
        && manifest->chunkDirectory == session.chunkDirectory
        && cleanupPathIdentity(manifest->projectRoot)
            == cleanupPathIdentity(session.projectRoot)
        && cleanupPathIdentity(manifest->managedRoot)
            == cleanupPathIdentity(session.managedRoot);
    if (!identityMatches)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral(
                "清理事务项目或 Chunk 身份不匹配：%1")
                                .arg(transactionRoot);
        }
        return false;
    }

    const QJsonObject metadataWal = object.value(
        QStringLiteral("metadata_wal")).toObject();
    const QString originalFile = metadataWal.value(
        QStringLiteral("original_file")).toString();
    const QString updatedFile = metadataWal.value(
        QStringLiteral("updated_file")).toString();
    if (!parseSha256(
            metadataWal.value(
                QStringLiteral("original_sha256")).toString(),
            &manifest->originalMetadataHash)
        || !parseSha256(
            metadataWal.value(
                QStringLiteral("updated_sha256")).toString(),
            &manifest->updatedMetadataHash)
        || !loadMetadataSnapshot(
            transactionRoot,
            originalFile,
            manifest->originalMetadataHash,
            &manifest->originalMetadata,
            errorMessage)
        || !loadMetadataSnapshot(
            transactionRoot,
            updatedFile,
            manifest->updatedMetadataHash,
            &manifest->updatedMetadata,
            errorMessage))
    {
        if (errorMessage && errorMessage->isEmpty())
        {
            *errorMessage = QStringLiteral(
                "清理事务元数据 WAL 无效：%1").arg(transactionRoot);
        }
        return false;
    }

    const QJsonArray moves = object.value(QStringLiteral("moves")).toArray();
    if (moves.size() > kMaximumManifestMoves)
    {
        return false;
    }
    manifest->moves.clear();
    QSet<QString> sourceIdentities;
    QSet<QString> destinationIdentities;
    for (const QJsonValue &value : moves)
    {
        const QJsonObject moveObject = value.toObject();
        CleanupTransactionMove move;
        move.source = moveObject.value(QStringLiteral("source")).toString();
        move.destination = moveObject.value(
            QStringLiteral("destination")).toString();
        const QString kind = moveObject.value(
            QStringLiteral("kind")).toString();
        move.directory = kind == QLatin1String("directory");
        if ((kind != QLatin1String("file") && !move.directory)
            || !parseMoveState(
                moveObject.value(QStringLiteral("state")).toString(),
                &move.state)
            || !validateMove(*manifest,
                             transactionRoot,
                             move,
                             errorMessage))
        {
            return false;
        }
        const QString sourceIdentity = cleanupPathIdentity(move.source);
        const QString destinationIdentity = cleanupPathIdentity(
            move.destination);
        if (sourceIdentities.contains(sourceIdentity)
            || destinationIdentities.contains(destinationIdentity))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "清理事务恢复清单包含重复路径：%1")
                                    .arg(transactionRoot);
            }
            return false;
        }
        sourceIdentities.insert(sourceIdentity);
        destinationIdentities.insert(destinationIdentity);
        manifest->moves.append(move);
    }
    return true;
}

bool restoreStagedCleanupTransaction(
    const QString &transactionRoot,
    const CleanupTransactionManifest &manifest,
    QStringList *failedPaths,
    QString *errorMessage)
{
    for (auto it = manifest.moves.crbegin(); it != manifest.moves.crend(); ++it)
    {
        if (!validateMove(manifest, transactionRoot, *it, errorMessage))
        {
            appendUniqueCleanupPath(failedPaths, it->source);
            return false;
        }
        const QFileInfo sourceInfo(it->source);
        const QFileInfo destinationInfo(it->destination);
        const bool sourceExists = sourceInfo.exists()
            || sourceInfo.isSymLink();
        const bool destinationExists = destinationInfo.exists()
            || destinationInfo.isSymLink();
        if (sourceExists && !destinationExists)
        {
            continue;
        }
        if (sourceExists == destinationExists)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "清理事务无法判定恢复来源：%1 -> %2")
                                    .arg(it->source, it->destination);
            }
            appendUniqueCleanupPath(failedPaths, it->source);
            return false;
        }
        if (!QDir().mkpath(sourceInfo.absolutePath())
            || cleanupPathTraversesLink(manifest.managedRoot, it->source)
            || !QDir().rename(it->destination, it->source))
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral(
                    "无法从清理事务区恢复产物：%1")
                                    .arg(it->source);
            }
            appendUniqueCleanupPath(failedPaths, it->source);
            return false;
        }
    }

    return true;
}

void removeEmptyCleanupTrashBase(const QString &managedRoot)
{
    const QString trashBase = cleanupTrashBase(managedRoot);
    const QFileInfo info(trashBase);
    if (!info.isDir() || isFilesystemLink(info))
    {
        return;
    }
    const QDir directory(trashBase);
    if (directory.entryList(
            QDir::AllEntries | QDir::NoDotAndDotDot
                | QDir::Hidden | QDir::System).isEmpty())
    {
        QDir().rmdir(trashBase);
    }
}

} // namespace xjw::core::project::detail
