#include "ImageMatchFile.h"

#include <QCryptographicHash>
#include <QDataStream>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSaveFile>

#include <algorithm>
#include <cstring>
#include <limits>

namespace xjw::image_matching
{
namespace
{

constexpr char kMagic[] = {'P', 'L', 'I', 'M', 'A', 'T', 'C', 'H'};
constexpr int kMagicSize = static_cast<int>(sizeof(kMagic));
// 当前实现将 payload 一次性装入 QByteArray，并在 QDataStream 原始读写接口中使用
// int 长度。上限必须与真实分配/调用类型一致，不能宣称支持 32 GiB 后再发生窄化。
constexpr std::uint64_t kMaximumPayloadBytes =
    static_cast<std::uint64_t>(std::numeric_limits<int>::max());
constexpr std::uint32_t kMaximumStringBytes = 16U * 1024U * 1024U;
constexpr std::uint32_t kMaximumObservations = 100U * 1000U * 1000U;
constexpr std::uint32_t kMaximumNeighborVariants = 10U * 1000U * 1000U;
constexpr std::uint32_t kMaximumMatchesPerVariant = 100U * 1000U * 1000U;

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

bool writeString(QDataStream *stream, const QString &value)
{
    const QByteArray utf8 = value.toUtf8();
    if (utf8.size() < 0 || static_cast<std::uint64_t>(utf8.size()) > kMaximumStringBytes)
    {
        return false;
    }
    *stream << static_cast<quint32>(utf8.size());
    return utf8.isEmpty() || stream->writeRawData(utf8.constData(), utf8.size()) == utf8.size();
}

bool readString(QDataStream *stream, QString *value)
{
    quint32 length = 0;
    *stream >> length;
    if (stream->status() != QDataStream::Ok || length > kMaximumStringBytes)
    {
        return false;
    }
    QByteArray utf8(static_cast<int>(length), Qt::Uninitialized);
    if (length > 0 && stream->readRawData(utf8.data(), static_cast<int>(length)) !=
                          static_cast<int>(length))
    {
        return false;
    }
    *value = QString::fromUtf8(utf8);
    return true;
}

bool writeBytes(QDataStream *stream, const QByteArray &value)
{
    if (value.size() < 0 || static_cast<std::uint64_t>(value.size()) > kMaximumStringBytes)
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
    if (stream->status() != QDataStream::Ok || length > kMaximumStringBytes)
    {
        return false;
    }
    value->resize(static_cast<int>(length));
    return length == 0 || stream->readRawData(value->data(), static_cast<int>(length)) ==
                              static_cast<int>(length);
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
    qint64 modifiedTime = 0;
    if (!readString(stream, &identity->stableId) ||
        !readString(stream, &identity->path) ||
        !readString(stream, &identity->displayName))
    {
        return false;
    }
    *stream >> width >> height >> fileSize >> modifiedTime;
    identity->width = width;
    identity->height = height;
    identity->fileSize = fileSize;
    identity->modifiedTimeMs = modifiedTime;
    return stream->status() == QDataStream::Ok;
}

bool writeObservation(QDataStream *stream, const KeypointObservation &observation)
{
    *stream << static_cast<quint32>(observation.featureId)
            << observation.x
            << observation.y
            << observation.scale
            << observation.orientation
            << observation.response;
    return stream->status() == QDataStream::Ok;
}

bool readObservation(QDataStream *stream, KeypointObservation *observation)
{
    quint32 featureId = 0;
    *stream >> featureId
            >> observation->x
            >> observation->y
            >> observation->scale
            >> observation->orientation
            >> observation->response;
    observation->featureId = featureId;
    return stream->status() == QDataStream::Ok;
}

bool writeMatch(QDataStream *stream, const MatchRecord &match)
{
    *stream << static_cast<quint32>(match.ownerFeatureId)
            << static_cast<quint32>(match.peerFeatureId)
            << match.peerX
            << match.peerY
            << match.confidence
            << match.residualPixels
            << static_cast<quint32>(match.flags);
    return stream->status() == QDataStream::Ok;
}

bool readMatch(QDataStream *stream, MatchRecord *match)
{
    quint32 ownerFeatureId = 0;
    quint32 peerFeatureId = 0;
    quint32 flags = 0;
    *stream >> ownerFeatureId
            >> peerFeatureId
            >> match->peerX
            >> match->peerY
            >> match->confidence
            >> match->residualPixels
            >> flags;
    match->ownerFeatureId = ownerFeatureId;
    match->peerFeatureId = peerFeatureId;
    match->flags = static_cast<MatchRecordFlag>(flags);
    return stream->status() == QDataStream::Ok;
}

bool writeNeighbor(QDataStream *stream, const NeighborMatchBlock &block)
{
    if (block.ownerObservations.size() > kMaximumObservations ||
        block.matches.size() > kMaximumMatchesPerVariant)
    {
        return false;
    }
    if (!writeIdentity(stream, block.peer) ||
        !writeString(stream, block.algorithmId) ||
        !writeBytes(stream, block.configFingerprint) ||
        !writeBytes(stream, block.modelFingerprint))
    {
        return false;
    }

    *stream << static_cast<quint32>(block.algorithmVersion)
            << static_cast<qint64>(block.createdTimeMs)
            << static_cast<quint32>(block.rawMatchCount)
            << static_cast<quint32>(block.geometryInlierCount)
            << static_cast<quint32>(block.tiePointMatchCount)
            << static_cast<quint8>(block.geometryPassed ? 1 : 0)
            << static_cast<quint8>(block.geometryModel);
    for (const double value : block.geometryMatrix)
    {
        // QDataStream 的精度开关会影响 double，因此矩阵显式按双精度切换写入。
        stream->setFloatingPointPrecision(QDataStream::DoublePrecision);
        *stream << value;
        stream->setFloatingPointPrecision(QDataStream::SinglePrecision);
    }
    // 观测表放在算法变体内部，避免不同算法或不同配置复用 featureId 时串点。
    *stream << static_cast<quint32>(block.ownerObservations.size());
    if (stream->status() != QDataStream::Ok)
    {
        return false;
    }
    for (const KeypointObservation &observation : block.ownerObservations)
    {
        if (!writeObservation(stream, observation))
        {
            return false;
        }
    }

    *stream << static_cast<quint32>(block.matches.size());
    if (stream->status() != QDataStream::Ok)
    {
        return false;
    }
    for (const MatchRecord &match : block.matches)
    {
        if (!writeMatch(stream, match))
        {
            return false;
        }
    }
    return true;
}

bool readNeighbor(QDataStream *stream, NeighborMatchBlock *block)
{
    quint32 algorithmVersion = 0;
    qint64 createdTime = 0;
    quint32 rawCount = 0;
    quint32 inlierCount = 0;
    quint32 tiePointCount = 0;
    quint8 geometryPassed = 0;
    quint8 geometryModel = 0;
    quint32 observationCount = 0;
    quint32 matchCount = 0;
    if (!readIdentity(stream, &block->peer) ||
        !readString(stream, &block->algorithmId) ||
        !readBytes(stream, &block->configFingerprint) ||
        !readBytes(stream, &block->modelFingerprint))
    {
        return false;
    }

    *stream >> algorithmVersion >> createdTime >> rawCount >> inlierCount >> tiePointCount
            >> geometryPassed >> geometryModel;
    for (double &value : block->geometryMatrix)
    {
        stream->setFloatingPointPrecision(QDataStream::DoublePrecision);
        *stream >> value;
        stream->setFloatingPointPrecision(QDataStream::SinglePrecision);
    }
    *stream >> observationCount;
    if (stream->status() != QDataStream::Ok || observationCount > kMaximumObservations ||
        geometryModel > static_cast<quint8>(GeometryModel::Affine))
    {
        return false;
    }

    block->algorithmVersion = algorithmVersion;
    block->createdTimeMs = createdTime;
    block->rawMatchCount = rawCount;
    block->geometryInlierCount = inlierCount;
    block->tiePointMatchCount = tiePointCount;
    block->geometryPassed = geometryPassed != 0;
    block->geometryModel = static_cast<GeometryModel>(geometryModel);
    block->ownerObservations.resize(observationCount);
    for (KeypointObservation &observation : block->ownerObservations)
    {
        if (!readObservation(stream, &observation))
        {
            return false;
        }
    }

    *stream >> matchCount;
    if (stream->status() != QDataStream::Ok || matchCount > kMaximumMatchesPerVariant)
    {
        return false;
    }
    block->matches.resize(matchCount);
    for (MatchRecord &match : block->matches)
    {
        if (!readMatch(stream, &match))
        {
            return false;
        }
    }
    return true;
}

bool serializePayload(const ImageMatchShard &shard, QByteArray *payload, QString *errorMessage)
{
    payload->clear();
    QDataStream stream(payload, QIODevice::WriteOnly);
    configureStream(&stream);
    if (!writeIdentity(&stream, shard.owner))
    {
        setError(errorMessage, QStringLiteral("影像匹配分片包含无效的影像身份字段"));
        return false;
    }
    if (shard.neighbors.size() > kMaximumNeighborVariants)
    {
        setError(errorMessage, QStringLiteral("影像匹配分片的算法变体数量超过格式上限"));
        return false;
    }

    stream << static_cast<quint32>(shard.neighbors.size());
    for (const NeighborMatchBlock &block : shard.neighbors)
    {
        if (!writeNeighbor(&stream, block))
        {
            setError(errorMessage, QStringLiteral("序列化相邻影像匹配块失败"));
            return false;
        }
    }
    return stream.status() == QDataStream::Ok;
}

bool deserializePayload(const QByteArray &payload,
                        ImageMatchShard *shard,
                        ImageMatchFileSummary *summary,
                        bool summaryOnly,
                        QString *errorMessage)
{
    QDataStream stream(payload);
    configureStream(&stream);
    ImageIdentity owner;
    if (!readIdentity(&stream, &owner) || !owner.isValid())
    {
        setError(errorMessage, QStringLiteral("影像匹配分片的所属影像身份无效"));
        return false;
    }

    quint32 neighborCount = 0;
    stream >> neighborCount;
    if (stream.status() != QDataStream::Ok || neighborCount > kMaximumNeighborVariants)
    {
        setError(errorMessage, QStringLiteral("相邻影像匹配块数量无效"));
        return false;
    }

    std::vector<NeighborMatchBlock> neighbors;
    neighbors.resize(neighborCount);
    for (NeighborMatchBlock &block : neighbors)
    {
        if (!readNeighbor(&stream, &block))
        {
            setError(errorMessage, QStringLiteral("读取相邻影像匹配块失败"));
            return false;
        }
    }

    std::uint64_t observationCount = 0;
    for (const NeighborMatchBlock &block : neighbors)
    {
        const std::uint64_t blockCount =
            static_cast<std::uint64_t>(block.ownerObservations.size());
        if (observationCount > std::numeric_limits<std::uint64_t>::max() - blockCount)
        {
            setError(errorMessage, QStringLiteral("影像匹配观测总数溢出"));
            return false;
        }
        observationCount += blockCount;
    }

    if (stream.status() != QDataStream::Ok || !stream.atEnd())
    {
        setError(errorMessage, QStringLiteral("影像匹配分片尾部包含无效或未识别数据"));
        return false;
    }

    if (summary)
    {
        summary->valid = true;
        summary->formatVersion = kImageMatchFormatVersion;
        summary->owner = owner;
        summary->observationCount = observationCount;
        summary->neighborVariantCount = neighborCount;
        summary->payloadBytes = static_cast<std::uint64_t>(payload.size());
    }
    if (!summaryOnly && shard)
    {
        shard->owner = std::move(owner);
        shard->neighbors = std::move(neighbors);
        shard->normalize();
    }
    return true;
}

bool readContainer(const QString &filePath,
                   QByteArray *payload,
                   std::uint32_t *formatVersion,
                   QString *errorMessage)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage, QStringLiteral("无法打开影像匹配文件: %1").arg(filePath));
        return false;
    }
    QDataStream stream(&file);
    configureStream(&stream);

    char magic[kMagicSize] = {};
    if (stream.readRawData(magic, kMagicSize) != kMagicSize ||
        std::memcmp(magic, kMagic, kMagicSize) != 0)
    {
        setError(errorMessage, QStringLiteral("不是 PlaScan 影像匹配分片: %1").arg(filePath));
        return false;
    }

    quint32 version = 0;
    quint64 payloadSize = 0;
    stream >> version >> payloadSize;
    if (stream.status() != QDataStream::Ok || version != kImageMatchFormatVersion)
    {
        setError(errorMessage,
                 QStringLiteral("不支持的影像匹配格式版本 %1，当前仅支持版本 %2")
                     .arg(version)
                     .arg(kImageMatchFormatVersion));
        return false;
    }
    if (payloadSize > kMaximumPayloadBytes || payloadSize > static_cast<quint64>(file.bytesAvailable()))
    {
        setError(errorMessage, QStringLiteral("影像匹配 payload 长度无效: %1").arg(filePath));
        return false;
    }

    QByteArray storedDigest(32, Qt::Uninitialized);
    if (stream.readRawData(storedDigest.data(), storedDigest.size()) != storedDigest.size())
    {
        setError(errorMessage, QStringLiteral("影像匹配文件缺少 SHA-256 校验值"));
        return false;
    }
    payload->resize(static_cast<int>(payloadSize));
    if (payloadSize > 0 && stream.readRawData(payload->data(), static_cast<int>(payloadSize)) !=
                               static_cast<int>(payloadSize))
    {
        setError(errorMessage, QStringLiteral("影像匹配文件内容不完整: %1").arg(filePath));
        return false;
    }
    if (file.bytesAvailable() != 0)
    {
        setError(errorMessage, QStringLiteral("影像匹配文件包含未识别的尾部数据: %1").arg(filePath));
        return false;
    }

    const QByteArray actualDigest = QCryptographicHash::hash(*payload, QCryptographicHash::Sha256);
    if (actualDigest != storedDigest)
    {
        setError(errorMessage, QStringLiteral("影像匹配文件 SHA-256 校验失败: %1").arg(filePath));
        return false;
    }
    if (formatVersion)
    {
        *formatVersion = version;
    }
    return true;
}

QString sanitizedStem(const QString &imagePath)
{
    QString stem = QFileInfo(imagePath).completeBaseName();
    for (QChar &character : stem)
    {
        if (!character.isLetterOrNumber() && character != QLatin1Char('-') &&
            character != QLatin1Char('_'))
        {
            character = QLatin1Char('_');
        }
    }
    if (stem.isEmpty())
    {
        stem = QStringLiteral("image");
    }
    return stem.left(80);
}

} // namespace

QString ImageMatchFile::stableImageId(const QString &imagePath)
{
    const QString canonical = QDir::cleanPath(QFileInfo(imagePath).absoluteFilePath());
    return QString::fromLatin1(
        QCryptographicHash::hash(canonical.toUtf8(), QCryptographicHash::Sha256).toHex());
}

ImageIdentity ImageMatchFile::identityForImage(const QString &imagePath, int width, int height)
{
    const QFileInfo info(imagePath);
    ImageIdentity identity;
    identity.path = QDir::cleanPath(info.absoluteFilePath());
    identity.stableId = stableImageId(identity.path);
    identity.displayName = info.fileName();
    identity.width = static_cast<std::uint32_t>(std::max(0, width));
    identity.height = static_cast<std::uint32_t>(std::max(0, height));
    identity.fileSize = info.exists() ? static_cast<std::uint64_t>(std::max<qint64>(0, info.size())) : 0;
    identity.modifiedTimeMs = info.exists() ? info.lastModified().toMSecsSinceEpoch() : 0;
    return identity;
}

QString ImageMatchFile::filePathForImage(const QString &directory, const QString &imagePath)
{
    const QString idPrefix = stableImageId(imagePath).left(20);
    return QDir(directory).filePath(
        sanitizedStem(imagePath) + QLatin1Char('_') + idPrefix +
        QString::fromLatin1(kImageMatchFileSuffix));
}

bool ImageMatchFile::write(const QString &filePath,
                           ImageMatchShard shard,
                           QString *errorMessage)
{
    if (!shard.owner.isValid())
    {
        setError(errorMessage, QStringLiteral("无法写入没有所属影像身份的匹配分片"));
        return false;
    }
    shard.normalize();

    QByteArray payload;
    if (!serializePayload(shard, &payload, errorMessage))
    {
        return false;
    }
    if (static_cast<std::uint64_t>(payload.size()) > kMaximumPayloadBytes)
    {
        setError(errorMessage, QStringLiteral("影像匹配分片超过当前 QByteArray 安全上限"));
        return false;
    }

    const QFileInfo outputInfo(filePath);
    if (!QDir().mkpath(outputInfo.absolutePath()))
    {
        setError(errorMessage,
                 QStringLiteral("无法创建影像匹配目录: %1").arg(outputInfo.absolutePath()));
        return false;
    }

    QSaveFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        setError(errorMessage, QStringLiteral("无法写入影像匹配文件: %1").arg(filePath));
        return false;
    }
    QDataStream stream(&file);
    configureStream(&stream);
    stream.writeRawData(kMagic, kMagicSize);
    stream << static_cast<quint32>(kImageMatchFormatVersion)
           << static_cast<quint64>(payload.size());
    const QByteArray digest = QCryptographicHash::hash(payload, QCryptographicHash::Sha256);
    stream.writeRawData(digest.constData(), digest.size());
    stream.writeRawData(payload.constData(), payload.size());
    if (stream.status() != QDataStream::Ok || !file.commit())
    {
        setError(errorMessage, QStringLiteral("提交影像匹配文件失败: %1").arg(filePath));
        return false;
    }
    return true;
}

bool ImageMatchFile::read(const QString &filePath,
                          ImageMatchShard *shard,
                          QString *errorMessage)
{
    if (!shard)
    {
        setError(errorMessage, QStringLiteral("影像匹配读取目标为空"));
        return false;
    }
    QByteArray payload;
    std::uint32_t version = 0;
    if (!readContainer(filePath, &payload, &version, errorMessage))
    {
        return false;
    }
    Q_UNUSED(version)
    return deserializePayload(payload, shard, nullptr, false, errorMessage);
}

bool ImageMatchFile::readSignature(const QString &filePath,
                                   ImageMatchFileSignature *signature,
                                   QString *errorMessage)
{
    if (!signature)
    {
        setError(errorMessage, QStringLiteral("影像匹配签名读取目标为空"));
        return false;
    }
    *signature = ImageMatchFileSignature{};

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly))
    {
        setError(errorMessage, QStringLiteral("无法打开影像匹配文件: %1").arg(filePath));
        return false;
    }
    QDataStream stream(&file);
    configureStream(&stream);

    char magic[kMagicSize] = {};
    quint32 version = 0;
    quint64 payloadSize = 0;
    QByteArray digest(32, Qt::Uninitialized);
    if (stream.readRawData(magic, kMagicSize) != kMagicSize ||
        std::memcmp(magic, kMagic, kMagicSize) != 0)
    {
        setError(errorMessage, QStringLiteral("不是 PlaScan 影像匹配分片: %1").arg(filePath));
        return false;
    }
    stream >> version >> payloadSize;
    if (stream.status() != QDataStream::Ok || version != kImageMatchFormatVersion ||
        payloadSize > kMaximumPayloadBytes ||
        stream.readRawData(digest.data(), digest.size()) != digest.size() ||
        static_cast<quint64>(file.bytesAvailable()) != payloadSize)
    {
        setError(errorMessage, QStringLiteral("影像匹配容器头或 payload 长度无效: %1").arg(filePath));
        return false;
    }

    signature->valid = true;
    signature->formatVersion = version;
    signature->payloadBytes = payloadSize;
    signature->containerBytes = static_cast<std::uint64_t>(std::max<qint64>(0, file.size()));
    signature->modifiedTimeMs = QFileInfo(filePath).lastModified().toMSecsSinceEpoch();
    signature->payloadSha256 = std::move(digest);
    return true;
}

bool ImageMatchFile::readSummary(const QString &filePath,
                                 ImageMatchFileSummary *summary,
                                 QString *errorMessage)
{
    if (!summary)
    {
        setError(errorMessage, QStringLiteral("影像匹配摘要读取目标为空"));
        return false;
    }
    *summary = ImageMatchFileSummary{};
    QByteArray payload;
    std::uint32_t version = 0;
    if (!readContainer(filePath, &payload, &version, errorMessage))
    {
        return false;
    }
    const bool ok = deserializePayload(payload, nullptr, summary, true, errorMessage);
    if (ok)
    {
        summary->formatVersion = version;
    }
    return ok;
}

} // namespace xjw::image_matching
