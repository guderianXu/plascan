#include "ImageMatchIndexFile.h"

#include <QCryptographicHash>
#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QMutex>
#include <QMutexLocker>
#include <QSaveFile>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace xjw::image_matching
{
namespace
{

constexpr char kIndexMagic[] = {'P', 'L', 'I', 'I', 'D', 'X', '0', '1'};
constexpr int kIndexMagicSize = static_cast<int>(sizeof(kIndexMagic));
constexpr std::uint32_t kIndexFormatVersion = 3;
constexpr std::uint32_t kMaximumIndexBytes = 16U * 1024U * 1024U;
constexpr std::uint64_t kMaximumIndexPayloadBytes = 512ULL * 1024ULL * 1024ULL;
constexpr std::uint32_t kMaximumIndexNeighbors = 10U * 1000U * 1000U;
constexpr int kMaximumMemoryCacheEntries = 4096;

struct CachedIndex
{
    ImageMatchFileSignature signature;
    ImageMatchFileIndex index;
};

QMutex &cacheMutex()
{
    static QMutex mutex;
    return mutex;
}

QHash<QString, CachedIndex> &memoryCache()
{
    static QHash<QString, CachedIndex> cache;
    return cache;
}

QString cacheKey(const QString &path)
{
    QString key = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
#if defined(Q_OS_WIN)
    key = key.toLower();
#endif
    return key;
}

void configureStream(QDataStream *stream)
{
    stream->setVersion(QDataStream::Qt_5_15);
    stream->setByteOrder(QDataStream::LittleEndian);
    stream->setFloatingPointPrecision(QDataStream::SinglePrecision);
}

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

bool writeBytes(QDataStream *stream, const QByteArray &value)
{
    if (value.size() < 0 || static_cast<std::uint64_t>(value.size()) > kMaximumIndexBytes)
    {
        return false;
    }
    *stream << static_cast<quint32>(value.size());
    return value.isEmpty() || stream->writeRawData(value.constData(), value.size()) == value.size();
}

bool readBytes(QDataStream *stream, QByteArray *value)
{
    quint32 length = 0;
    *stream >> length;
    if (stream->status() != QDataStream::Ok || length > kMaximumIndexBytes)
    {
        return false;
    }
    value->resize(static_cast<int>(length));
    return length == 0 || stream->readRawData(value->data(), static_cast<int>(length)) ==
        static_cast<int>(length);
}

bool writeString(QDataStream *stream, const QString &value)
{
    return writeBytes(stream, value.toUtf8());
}

bool readString(QDataStream *stream, QString *value)
{
    QByteArray utf8;
    if (!readBytes(stream, &utf8))
    {
        return false;
    }
    *value = QString::fromUtf8(utf8);
    return true;
}

bool writeIdentity(QDataStream *stream, const ImageIdentity &identity)
{
    return writeString(stream, identity.stableId) &&
        writeString(stream, identity.path) &&
        writeString(stream, identity.displayName) &&
        ((*stream) << static_cast<quint32>(identity.width)
                   << static_cast<quint32>(identity.height)
                   << static_cast<quint64>(identity.fileSize)
                   << static_cast<qint64>(identity.modifiedTimeMs),
         stream->status() == QDataStream::Ok);
}

bool readIdentity(QDataStream *stream, ImageIdentity *identity)
{
    quint32 width = 0;
    quint32 height = 0;
    quint64 fileSize = 0;
    qint64 modifiedTimeMs = 0;
    if (!readString(stream, &identity->stableId) ||
        !readString(stream, &identity->path) ||
        !readString(stream, &identity->displayName))
    {
        return false;
    }
    *stream >> width >> height >> fileSize >> modifiedTimeMs;
    identity->width = width;
    identity->height = height;
    identity->fileSize = fileSize;
    identity->modifiedTimeMs = modifiedTimeMs;
    return stream->status() == QDataStream::Ok;
}

double geometricGridCoverage(const ImageMatchShard &shard,
                             const NeighborMatchBlock &block)
{
    constexpr int gridColumns = 4;
    constexpr int gridRows = 4;
    std::array<bool, gridColumns * gridRows> ownerOccupied{};
    std::array<bool, gridColumns * gridRows> peerOccupied{};
    const float ownerWidth = static_cast<float>(std::max<std::uint32_t>(1U, shard.owner.width));
    const float ownerHeight = static_cast<float>(std::max<std::uint32_t>(1U, shard.owner.height));
    const float peerWidth = static_cast<float>(std::max<std::uint32_t>(1U, block.peer.width));
    const float peerHeight = static_cast<float>(std::max<std::uint32_t>(1U, block.peer.height));

    auto occupy = [](float x, float y, float width, float height, auto *grid)
    {
        if (!std::isfinite(x) || !std::isfinite(y))
        {
            return;
        }
        const int column = std::clamp(
            static_cast<int>(std::floor(x / width * gridColumns)), 0, gridColumns - 1);
        const int row = std::clamp(
            static_cast<int>(std::floor(y / height * gridRows)), 0, gridRows - 1);
        (*grid)[static_cast<std::size_t>(row * gridColumns + column)] = true;
    };

    for (const MatchRecord &match : block.matches)
    {
        if (!hasFlag(match.flags, MatchRecordFlag::GeometryInlier))
        {
            continue;
        }
        const KeypointObservation *owner = block.findOwnerObservation(match.ownerFeatureId);
        if (!owner)
        {
            continue;
        }
        occupy(owner->x, owner->y, ownerWidth, ownerHeight, &ownerOccupied);
        occupy(match.peerX, match.peerY, peerWidth, peerHeight, &peerOccupied);
    }

    const auto ratio = [](const auto &grid)
    {
        return static_cast<double>(std::count(grid.cbegin(), grid.cend(), true)) /
            static_cast<double>(grid.size());
    };
    return 0.5 * (ratio(ownerOccupied) + ratio(peerOccupied));
}

ImageMatchFileIndex makeIndex(const ImageMatchShard &shard,
                              const ImageMatchFileSignature &signature)
{
    ImageMatchFileIndex index;
    index.sourceSignature = signature;
    index.owner = shard.owner;
    index.neighbors.reserve(shard.neighbors.size());
    for (const NeighborMatchBlock &block : shard.neighbors)
    {
        ImageMatchNeighborIndex neighbor;
        neighbor.peer = block.peer;
        neighbor.algorithmId = block.algorithmId;
        neighbor.algorithmVersion = block.algorithmVersion;
        neighbor.configFingerprint = block.configFingerprint;
        neighbor.modelFingerprint = block.modelFingerprint;
        neighbor.createdTimeMs = block.createdTimeMs;
        neighbor.rawMatchCount = block.rawMatchCount;
        neighbor.geometryInlierCount = block.geometryInlierCount;
        neighbor.tiePointMatchCount = block.tiePointMatchCount;
        neighbor.geometryPassed = block.geometryPassed;
        neighbor.geometryModel = block.geometryModel;
        neighbor.geometricCoverage = geometricGridCoverage(shard, block);
        index.neighbors.push_back(std::move(neighbor));
    }
    return index;
}

bool writeIndexFile(const QString &path,
                    const ImageMatchFileIndex &index,
                    QString *errorMessage)
{
    QByteArray payload;
    QDataStream stream(&payload, QIODevice::WriteOnly);
    configureStream(&stream);
    stream << static_cast<quint32>(index.sourceSignature.formatVersion)
           << static_cast<quint64>(index.sourceSignature.payloadBytes)
           << static_cast<quint64>(index.sourceSignature.containerBytes)
           << static_cast<qint64>(index.sourceSignature.modifiedTimeMs);
    if (!writeBytes(&stream, index.sourceSignature.payloadSha256) ||
        !writeIdentity(&stream, index.owner) ||
        index.neighbors.size() > kMaximumIndexNeighbors)
    {
        setError(errorMessage, QStringLiteral("影像匹配轻量索引字段无效: %1").arg(path));
        return false;
    }
    stream << static_cast<quint32>(index.neighbors.size());
    for (const ImageMatchNeighborIndex &neighbor : index.neighbors)
    {
        if (!writeIdentity(&stream, neighbor.peer) ||
            !writeString(&stream, neighbor.algorithmId) ||
            !writeBytes(&stream, neighbor.configFingerprint) ||
            !writeBytes(&stream, neighbor.modelFingerprint))
        {
            setError(errorMessage, QStringLiteral("影像匹配轻量索引邻接字段无效: %1").arg(path));
            return false;
        }
        stream << static_cast<quint32>(neighbor.algorithmVersion)
               << static_cast<qint64>(neighbor.createdTimeMs)
               << static_cast<quint32>(neighbor.rawMatchCount)
               << static_cast<quint32>(neighbor.geometryInlierCount)
               << static_cast<quint32>(neighbor.tiePointMatchCount)
               << static_cast<quint8>(neighbor.geometryPassed ? 1 : 0)
               << static_cast<quint8>(neighbor.geometryModel);
        stream.setFloatingPointPrecision(QDataStream::DoublePrecision);
        stream << neighbor.geometricCoverage;
        stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    }
    if (stream.status() != QDataStream::Ok ||
        static_cast<std::uint64_t>(payload.size()) > kMaximumIndexPayloadBytes)
    {
        setError(errorMessage, QStringLiteral("序列化影像匹配轻量索引失败: %1").arg(path));
        return false;
    }

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        setError(errorMessage, QStringLiteral("无法写入影像匹配轻量索引: %1").arg(path));
        return false;
    }
    QDataStream container(&file);
    configureStream(&container);
    container.writeRawData(kIndexMagic, kIndexMagicSize);
    container << static_cast<quint32>(kIndexFormatVersion)
              << static_cast<quint64>(payload.size());
    const QByteArray digest = QCryptographicHash::hash(payload, QCryptographicHash::Sha256);
    container.writeRawData(digest.constData(), digest.size());
    container.writeRawData(payload.constData(), payload.size());
    if (container.status() != QDataStream::Ok || !file.commit())
    {
        setError(errorMessage, QStringLiteral("提交影像匹配轻量索引失败: %1").arg(path));
        return false;
    }
    return true;
}

bool readIndexFile(const QString &path,
                   const ImageMatchFileSignature &sourceSignature,
                   ImageMatchFileIndex *index)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
    {
        return false;
    }
    QDataStream container(&file);
    configureStream(&container);
    char magic[kIndexMagicSize] = {};
    quint32 indexVersion = 0;
    quint64 indexPayloadBytes = 0;
    QByteArray storedDigest(32, Qt::Uninitialized);
    if (container.readRawData(magic, kIndexMagicSize) != kIndexMagicSize ||
        std::memcmp(magic, kIndexMagic, kIndexMagicSize) != 0)
    {
        return false;
    }
    container >> indexVersion >> indexPayloadBytes;
    if (container.status() != QDataStream::Ok || indexVersion != kIndexFormatVersion ||
        indexPayloadBytes > kMaximumIndexPayloadBytes ||
        indexPayloadBytes > static_cast<quint64>(file.bytesAvailable()) ||
        container.readRawData(storedDigest.data(), storedDigest.size()) != storedDigest.size())
    {
        return false;
    }
    QByteArray payload(static_cast<int>(indexPayloadBytes), Qt::Uninitialized);
    if ((indexPayloadBytes > 0 &&
         container.readRawData(payload.data(), payload.size()) != payload.size()) ||
        file.bytesAvailable() != 0 ||
        QCryptographicHash::hash(payload, QCryptographicHash::Sha256) != storedDigest)
    {
        return false;
    }

    QDataStream stream(payload);
    configureStream(&stream);
    quint32 matchVersion = 0;
    quint64 payloadBytes = 0;
    quint64 containerBytes = 0;
    qint64 modifiedTimeMs = 0;
    QByteArray digest;
    stream >> matchVersion >> payloadBytes >> containerBytes >> modifiedTimeMs;
    if (stream.status() != QDataStream::Ok || !readBytes(&stream, &digest))
    {
        return false;
    }

    ImageMatchFileIndex loaded;
    loaded.sourceSignature.valid = true;
    loaded.sourceSignature.formatVersion = matchVersion;
    loaded.sourceSignature.payloadBytes = payloadBytes;
    loaded.sourceSignature.containerBytes = containerBytes;
    loaded.sourceSignature.modifiedTimeMs = modifiedTimeMs;
    loaded.sourceSignature.payloadSha256 = std::move(digest);
    if (!(loaded.sourceSignature == sourceSignature) ||
        !readIdentity(&stream, &loaded.owner) || !loaded.owner.isValid())
    {
        return false;
    }

    quint32 neighborCount = 0;
    stream >> neighborCount;
    if (stream.status() != QDataStream::Ok || neighborCount > kMaximumIndexNeighbors)
    {
        return false;
    }
    loaded.neighbors.resize(neighborCount);
    for (ImageMatchNeighborIndex &neighbor : loaded.neighbors)
    {
        quint32 algorithmVersion = 0;
        qint64 createdTimeMs = 0;
        quint32 rawCount = 0;
        quint32 inlierCount = 0;
        quint32 tiePointCount = 0;
        quint8 geometryPassed = 0;
        quint8 geometryModel = 0;
        if (!readIdentity(&stream, &neighbor.peer) ||
            !readString(&stream, &neighbor.algorithmId) ||
            !readBytes(&stream, &neighbor.configFingerprint) ||
            !readBytes(&stream, &neighbor.modelFingerprint))
        {
            return false;
        }
        stream >> algorithmVersion >> createdTimeMs >> rawCount >> inlierCount >> tiePointCount
               >> geometryPassed >> geometryModel;
        stream.setFloatingPointPrecision(QDataStream::DoublePrecision);
        stream >> neighbor.geometricCoverage;
        stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
        if (stream.status() != QDataStream::Ok || !neighbor.peer.isValid() ||
            geometryModel > static_cast<quint8>(GeometryModel::Affine) ||
            !std::isfinite(neighbor.geometricCoverage))
        {
            return false;
        }
        neighbor.algorithmVersion = algorithmVersion;
        neighbor.createdTimeMs = createdTimeMs;
        neighbor.rawMatchCount = rawCount;
        neighbor.geometryInlierCount = inlierCount;
        neighbor.tiePointMatchCount = tiePointCount;
        neighbor.geometryPassed = geometryPassed != 0;
        neighbor.geometryModel = static_cast<GeometryModel>(geometryModel);
    }
    if (stream.status() != QDataStream::Ok || !stream.atEnd())
    {
        return false;
    }
    *index = std::move(loaded);
    return true;
}

void remember(const QString &key, const ImageMatchFileIndex &index)
{
    QMutexLocker lock(&cacheMutex());
    auto &cache = memoryCache();
    if (cache.size() >= kMaximumMemoryCacheEntries && !cache.contains(key))
    {
        cache.clear();
    }
    cache.insert(key, CachedIndex{index.sourceSignature, index});
}

} // namespace

QString ImageMatchIndexFile::pathForMatchFile(const QString &matchFilePath)
{
    return matchFilePath + QString::fromLatin1(kImageMatchIndexFileSuffix);
}

bool ImageMatchIndexFile::load(const QString &matchFilePath,
                               ImageMatchFileIndex *index,
                               ImageMatchIndexLoadSource *source,
                               QString *errorMessage)
{
    if (!index)
    {
        setError(errorMessage, QStringLiteral("影像匹配轻量索引读取目标为空"));
        return false;
    }

    ImageMatchFileSignature signature;
    if (!ImageMatchFile::readSignature(matchFilePath, &signature, errorMessage))
    {
        return false;
    }
    const QString key = cacheKey(matchFilePath);
    {
        QMutexLocker lock(&cacheMutex());
        const auto it = memoryCache().constFind(key);
        if (it != memoryCache().constEnd() && it->signature == signature)
        {
            *index = it->index;
            if (source)
            {
                *source = ImageMatchIndexLoadSource::MemoryCache;
            }
            return true;
        }
    }

    ImageMatchFileIndex loaded;
    if (readIndexFile(pathForMatchFile(matchFilePath), signature, &loaded))
    {
        remember(key, loaded);
        *index = std::move(loaded);
        if (source)
        {
            *source = ImageMatchIndexLoadSource::PersistentIndex;
        }
        return true;
    }

    for (int attempt = 0; attempt < 2; ++attempt)
    {
        const ImageMatchFileSignature before = signature;
        ImageMatchShard shard;
        if (!ImageMatchFile::read(matchFilePath, &shard, errorMessage))
        {
            return false;
        }
        ImageMatchFileSignature after;
        if (!ImageMatchFile::readSignature(matchFilePath, &after, errorMessage))
        {
            return false;
        }
        if (!(before == after))
        {
            signature = std::move(after);
            continue;
        }

        loaded = makeIndex(shard, after);
        QString ignoredWriteError;
        writeIndexFile(pathForMatchFile(matchFilePath), loaded, &ignoredWriteError);
        remember(key, loaded);
        *index = std::move(loaded);
        if (source)
        {
            *source = ImageMatchIndexLoadSource::RebuiltFromMatchFile;
        }
        return true;
    }

    setError(errorMessage, QStringLiteral("影像匹配文件在建立轻量索引期间持续变化: %1")
                               .arg(matchFilePath));
    return false;
}

bool ImageMatchIndexFile::writeForShard(const QString &matchFilePath,
                                        const ImageMatchShard &shard,
                                        QString *errorMessage)
{
    ImageMatchFileSignature signature;
    if (!ImageMatchFile::readSignature(matchFilePath, &signature, errorMessage))
    {
        return false;
    }
    const ImageMatchFileIndex index = makeIndex(shard, signature);
    if (!writeIndexFile(pathForMatchFile(matchFilePath), index, errorMessage))
    {
        return false;
    }
    remember(cacheKey(matchFilePath), index);
    return true;
}

bool ImageMatchIndexFile::removeForMatchFile(const QString &matchFilePath)
{
    {
        QMutexLocker lock(&cacheMutex());
        memoryCache().remove(cacheKey(matchFilePath));
    }
    const QString path = pathForMatchFile(matchFilePath);
    return !QFileInfo::exists(path) || QFile::remove(path);
}

void ImageMatchIndexFile::clearMemoryCache()
{
    QMutexLocker lock(&cacheMutex());
    memoryCache().clear();
}

} // namespace xjw::image_matching
