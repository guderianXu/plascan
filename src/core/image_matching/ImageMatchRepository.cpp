#include "ImageMatchRepository.h"

#include "ImageMatchIndexFile.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

namespace xjw::image_matching
{

struct ImageMatchRepositoryReadCache
{
    struct Entry
    {
        ImageMatchFileSignature signature;
        std::shared_ptr<const ImageMatchShard> shard;
        std::uint64_t accessSerial = 0;
    };

    std::mutex mutex;
    QHash<QString, Entry> entries;
    std::uint64_t nextAccessSerial = 0;
    std::uint64_t cachedPayloadBytes = 0;
};

namespace
{

constexpr qsizetype kMaximumCachedMatchShards = 16;
constexpr std::uint64_t kMaximumCachedMatchPayloadBytes = 256ull * 1024ull * 1024ull;
constexpr std::uint64_t kMaximumSingleCachedMatchPayloadBytes = 64ull * 1024ull * 1024ull;

QString canonicalPath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

QString variantKey(const NeighborMatchBlock &block)
{
    return block.peer.stableId + QLatin1Char('\n') + block.algorithmId.toLower() +
        QLatin1Char('\n') + QString::number(block.algorithmVersion) + QLatin1Char('\n') +
        QString::fromLatin1(block.configFingerprint.toHex()) + QLatin1Char('\n') +
        QString::fromLatin1(block.modelFingerprint.toHex());
}

QString variantKey(const PairMatchData &pair, const ImageIdentity &peer)
{
    return peer.stableId + QLatin1Char('\n') + pair.algorithmId.toLower() +
        QLatin1Char('\n') + QString::number(pair.algorithmVersion) + QLatin1Char('\n') +
        QString::fromLatin1(pair.configFingerprint.toHex()) + QLatin1Char('\n') +
        QString::fromLatin1(pair.modelFingerprint.toHex());
}

std::array<double, 9> transposeMatrix(const std::array<double, 9> &matrix)
{
    std::array<double, 9> result{};
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            result[static_cast<std::size_t>(row * 3 + column)] =
                matrix[static_cast<std::size_t>(column * 3 + row)];
        }
    }
    return result;
}

bool invertMatrix(const std::array<double, 9> &matrix, std::array<double, 9> *inverse)
{
    const double a = matrix[0];
    const double b = matrix[1];
    const double c = matrix[2];
    const double d = matrix[3];
    const double e = matrix[4];
    const double f = matrix[5];
    const double g = matrix[6];
    const double h = matrix[7];
    const double i = matrix[8];
    const double determinant = a * (e * i - f * h) -
        b * (d * i - f * g) + c * (d * h - e * g);
    const double scale = std::max({std::abs(a), std::abs(b), std::abs(c),
                                   std::abs(d), std::abs(e), std::abs(f),
                                   std::abs(g), std::abs(h), std::abs(i)});
    const double tolerance = std::numeric_limits<double>::epsilon() * 64.0 *
        std::max(1.0, scale * scale * scale);
    if (!inverse || !std::isfinite(determinant) || std::abs(determinant) <= tolerance)
    {
        return false;
    }

    const double reciprocal = 1.0 / determinant;
    *inverse = {
        (e * i - f * h) * reciprocal,
        (c * h - b * i) * reciprocal,
        (b * f - c * e) * reciprocal,
        (f * g - d * i) * reciprocal,
        (a * i - c * g) * reciprocal,
        (c * d - a * f) * reciprocal,
        (d * h - e * g) * reciprocal,
        (b * g - a * h) * reciprocal,
        (a * e - b * d) * reciprocal};
    return std::all_of(inverse->begin(), inverse->end(),
                       [](double value)
                       {
                           return std::isfinite(value);
                       });
}

bool reverseGeometryMatrix(GeometryModel model,
                           const std::array<double, 9> &matrix,
                           std::array<double, 9> *reversed)
{
    if (!reversed)
    {
        return false;
    }
    if (model == GeometryModel::Fundamental || model == GeometryModel::Essential)
    {
        *reversed = transposeMatrix(matrix);
        return true;
    }
    if (model == GeometryModel::Homography || model == GeometryModel::Affine)
    {
        return invertMatrix(matrix, reversed);
    }
    *reversed = matrix;
    return true;
}

NeighborMatchBlock directedBlock(const PairMatchData &pair, bool reverse)
{
    NeighborMatchBlock block;
    block.peer = reverse ? pair.image0 : pair.image1;
    block.algorithmId = pair.algorithmId;
    block.algorithmVersion = pair.algorithmVersion;
    block.configFingerprint = pair.configFingerprint;
    block.modelFingerprint = pair.modelFingerprint;
    block.createdTimeMs = pair.createdTimeMs;
    block.rawMatchCount = pair.rawMatchCount;
    block.geometryInlierCount = pair.geometryInlierCount;
    block.tiePointMatchCount = pair.tiePointMatchCount;
    block.geometryPassed = pair.geometryPassed;
    block.geometryModel = pair.geometryModel;
    block.geometryMatrix = pair.geometryMatrix;
    if (reverse)
    {
        // 极线模型反向时取转置；映射模型反向时必须取逆矩阵，二者不能混用。
        if (!reverseGeometryMatrix(pair.geometryModel,
                                   pair.geometryMatrix,
                                   &block.geometryMatrix))
        {
            block.geometryPassed = false;
            block.geometryModel = GeometryModel::None;
            block.geometryMatrix = {};
        }
    }

    block.ownerObservations.reserve(pair.correspondences.size());
    block.matches.reserve(pair.correspondences.size());
    for (const PairCorrespondence &correspondence : pair.correspondences)
    {
        const KeypointObservation &owner = reverse ? correspondence.observation1
                                                   : correspondence.observation0;
        const KeypointObservation &peer = reverse ? correspondence.observation0
                                                  : correspondence.observation1;
        MatchRecord record;
        record.ownerFeatureId = owner.featureId;
        record.peerFeatureId = peer.featureId;
        record.peerX = peer.x;
        record.peerY = peer.y;
        record.confidence = correspondence.confidence;
        record.residualPixels = correspondence.residualPixels;
        record.flags = correspondence.flags;
        block.ownerObservations.push_back(owner);
        block.matches.push_back(record);
    }
    return block;
}

void upsertBlock(ImageMatchShard *shard, NeighborMatchBlock block)
{
    const QString key = variantKey(block);
    const auto it = std::find_if(shard->neighbors.begin(), shard->neighbors.end(),
                                 [&](const NeighborMatchBlock &existing)
                                 {
                                     return variantKey(existing) == key;
                                 });
    if (it == shard->neighbors.end())
    {
        shard->neighbors.push_back(std::move(block));
    }
    else
    {
        *it = std::move(block);
    }
}

bool identityMatchesCurrentFile(const ImageIdentity &stored, const QString &imagePath)
{
    const ImageIdentity current = ImageMatchFile::identityForImage(
        imagePath, static_cast<int>(stored.width), static_cast<int>(stored.height));
    return stored.stableId == current.stableId &&
        stored.fileSize == current.fileSize &&
        stored.modifiedTimeMs == current.modifiedTimeMs;
}

const NeighborMatchBlock *findNeighborWithConfigPrefix(
    const ImageMatchShard &shard,
    const QString &peerStableId,
    const QString &algorithmId,
    std::uint32_t algorithmVersion,
    const QByteArray &configFingerprintPrefix,
    const QByteArray &modelFingerprint)
{
    const NeighborMatchBlock *selected = nullptr;
    for (const NeighborMatchBlock &block : shard.neighbors)
    {
        const bool compatible = block.peer.stableId == peerStableId &&
            block.algorithmId.compare(algorithmId, Qt::CaseInsensitive) == 0 &&
            block.algorithmVersion == algorithmVersion &&
            block.configFingerprint.startsWith(configFingerprintPrefix) &&
            (modelFingerprint.isEmpty() || block.modelFingerprint == modelFingerprint);
        if (compatible && (!selected || block.createdTimeMs > selected->createdTimeMs))
        {
            selected = &block;
        }
    }
    return selected;
}

PairMatchData pairFromDirectedBlock(const ImageMatchShard &ownerShard,
                                    const NeighborMatchBlock &block,
                                    bool reverseOutput)
{
    PairMatchData pair;
    pair.image0 = reverseOutput ? block.peer : ownerShard.owner;
    pair.image1 = reverseOutput ? ownerShard.owner : block.peer;
    pair.algorithmId = block.algorithmId;
    pair.algorithmVersion = block.algorithmVersion;
    pair.configFingerprint = block.configFingerprint;
    pair.modelFingerprint = block.modelFingerprint;
    pair.createdTimeMs = block.createdTimeMs;
    pair.rawMatchCount = block.rawMatchCount;
    pair.geometryInlierCount = block.geometryInlierCount;
    pair.tiePointMatchCount = block.tiePointMatchCount;
    pair.geometryPassed = block.geometryPassed;
    pair.geometryModel = block.geometryModel;
    pair.geometryMatrix = block.geometryMatrix;
    if (reverseOutput)
    {
        if (!reverseGeometryMatrix(block.geometryModel,
                                   block.geometryMatrix,
                                   &pair.geometryMatrix))
        {
            pair.geometryPassed = false;
            pair.geometryModel = GeometryModel::None;
            pair.geometryMatrix = {};
        }
    }

    pair.correspondences.reserve(block.matches.size());
    for (const MatchRecord &match : block.matches)
    {
        const KeypointObservation *ownerObservation =
            block.findOwnerObservation(match.ownerFeatureId);
        if (!ownerObservation)
        {
            continue;
        }

        KeypointObservation peerObservation;
        peerObservation.featureId = match.peerFeatureId;
        peerObservation.x = match.peerX;
        peerObservation.y = match.peerY;
        PairCorrespondence correspondence;
        correspondence.observation0 = reverseOutput ? peerObservation : *ownerObservation;
        correspondence.observation1 = reverseOutput ? *ownerObservation : peerObservation;
        correspondence.confidence = match.confidence;
        correspondence.residualPixels = match.residualPixels;
        correspondence.flags = match.flags;
        pair.correspondences.push_back(correspondence);
    }
    return pair;
}

void mergePeerObservationMetadata(PairMatchData *pair,
                                   const ImageMatchShard &peerShard)
{
    if (!pair)
    {
        return;
    }

    const bool peerIsImage0 = peerShard.owner.stableId == pair->image0.stableId;
    const bool peerIsImage1 = peerShard.owner.stableId == pair->image1.stableId;
    if (!peerIsImage0 && !peerIsImage1)
    {
        return;
    }

    const QString otherStableId = peerIsImage0 ? pair->image1.stableId
                                               : pair->image0.stableId;
    const NeighborMatchBlock *peerBlock = peerShard.findNeighbor(
        otherStableId,
        pair->algorithmId,
        pair->algorithmVersion,
        pair->configFingerprint,
        pair->modelFingerprint);
    if (!peerBlock)
    {
        return;
    }

    for (PairCorrespondence &correspondence : pair->correspondences)
    {
        KeypointObservation &target = peerIsImage0
            ? correspondence.observation0
            : correspondence.observation1;
        if (const KeypointObservation *stored =
                peerBlock->findOwnerObservation(target.featureId))
        {
            target = *stored;
        }
    }
}

const NeighborMatchBlock *findRequestedBlock(
    const ImageMatchShard &shard,
    const QString &peerStableId,
    const QString &algorithmId,
    std::uint32_t algorithmVersion,
    const QByteArray &configFingerprint,
    const QByteArray &modelFingerprint,
    bool matchConfigPrefix)
{
    return matchConfigPrefix
        ? findNeighborWithConfigPrefix(shard,
                                       peerStableId,
                                       algorithmId,
                                       algorithmVersion,
                                       configFingerprint,
                                       modelFingerprint)
        : shard.findNeighbor(peerStableId,
                             algorithmId,
                             algorithmVersion,
                             configFingerprint,
                             modelFingerprint);
}

template <typename ShardLoader>
bool loadPairImpl(ShardLoader &&loadShard,
                  const QString &image0Path,
                  const QString &image1Path,
                  const QString &algorithmId,
                  std::uint32_t algorithmVersion,
                  const QByteArray &configFingerprint,
                  const QByteArray &modelFingerprint,
                  bool matchConfigPrefix,
                  PairMatchData *pair,
                  QString *errorMessage)
{
    const QString id0 = ImageMatchFile::stableImageId(image0Path);
    const QString id1 = ImageMatchFile::stableImageId(image1Path);
    QString firstError;
    const std::shared_ptr<const ImageMatchShard> shard = loadShard(image0Path, &firstError);
    if (shard)
    {
        const NeighborMatchBlock *block = findRequestedBlock(
            *shard,
            id1,
            algorithmId,
            algorithmVersion,
            configFingerprint,
            modelFingerprint,
            matchConfigPrefix);
        if (block && identityMatchesCurrentFile(block->peer, image1Path))
        {
            *pair = pairFromDirectedBlock(*shard, *block, false);
            // owner 邻接块为快速单文件浏览保存了对端坐标，但尺度、方向和响应值只在
            // 对端对应算法变体中保存一份。缓存复用时补读对端分片，避免再次提交
            // 结果时把这些 SIFT 属性退化为默认值。
            QString ignoredError;
            const std::shared_ptr<const ImageMatchShard> peerShard =
                loadShard(image1Path, &ignoredError);
            if (peerShard)
            {
                mergePeerObservationMetadata(pair, *peerShard);
            }
            return true;
        }
    }
    QString secondError;
    const std::shared_ptr<const ImageMatchShard> reverseShard =
        loadShard(image1Path, &secondError);
    if (reverseShard)
    {
        const NeighborMatchBlock *block = findRequestedBlock(
            *reverseShard,
            id0,
            algorithmId,
            algorithmVersion,
            configFingerprint,
            modelFingerprint,
            matchConfigPrefix);
        if (block && identityMatchesCurrentFile(block->peer, image0Path))
        {
            *pair = pairFromDirectedBlock(*reverseShard, *block, true);
            QString ignoredError;
            const std::shared_ptr<const ImageMatchShard> peerShard =
                loadShard(image0Path, &ignoredError);
            if (peerShard)
            {
                mergePeerObservationMetadata(pair, *peerShard);
            }
            return true;
        }
    }
    if (errorMessage)
    {
        // 单侧分片可能被外部破坏；只要对侧权威分片仍完整，就应允许按反向
        // 邻接块恢复像对。双方都无法提供结果时再报告首个真实读取错误。
        *errorMessage = !firstError.isEmpty() ? firstError : secondError;
    }
    return false;
}

} // namespace

ImageMatchRepository::ImageMatchRepository(QString directory)
    : _directory(QDir::cleanPath(std::move(directory))),
      _readCache(std::make_shared<ImageMatchRepositoryReadCache>())
{
}

const QString &ImageMatchRepository::directory() const
{
    return _directory;
}

QString ImageMatchRepository::shardPath(const QString &imagePath) const
{
    return ImageMatchFile::filePathForImage(_directory, imagePath);
}

bool ImageMatchRepository::loadShard(const QString &imagePath,
                                     ImageMatchShard *shard,
                                     QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!shard)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("影像匹配分片读取目标为空");
        }
        return false;
    }
    const std::shared_ptr<const ImageMatchShard> cached =
        loadShardCached(imagePath, errorMessage);
    if (!cached)
    {
        return false;
    }
    *shard = *cached;
    return true;
}

std::shared_ptr<const ImageMatchShard> ImageMatchRepository::loadShardCached(
    const QString &imagePath,
    QString *errorMessage) const
{
    const QString path = shardPath(imagePath);
    if (!QFileInfo::exists(path))
    {
        return {};
    }

    ImageMatchFileSignature signature;
    if (!ImageMatchFile::readSignature(path, &signature, errorMessage))
    {
        return {};
    }
    const QString cacheKey = ImageMatchFile::stableImageId(imagePath);
    {
        std::lock_guard lock(_readCache->mutex);
        auto iterator = _readCache->entries.find(cacheKey);
        if (iterator != _readCache->entries.end() &&
            iterator->signature == signature && iterator->shard &&
            identityMatchesCurrentFile(iterator->shard->owner, imagePath))
        {
            iterator->accessSerial = ++_readCache->nextAccessSerial;
            return iterator->shard;
        }
    }

    auto loaded = std::make_shared<ImageMatchShard>();
    if (!ImageMatchFile::read(path, loaded.get(), errorMessage))
    {
        return {};
    }
    if (!identityMatchesCurrentFile(loaded->owner, imagePath))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("影像内容或修改时间已变化，匹配分片不能复用: %1")
                .arg(imagePath);
        }
        return {};
    }

    {
        std::lock_guard lock(_readCache->mutex);
        auto existing = _readCache->entries.find(cacheKey);
        if (existing != _readCache->entries.end())
        {
            _readCache->cachedPayloadBytes -= std::min(
                _readCache->cachedPayloadBytes,
                existing->signature.payloadBytes);
            _readCache->entries.erase(existing);
        }

        if (signature.payloadBytes > kMaximumSingleCachedMatchPayloadBytes)
        {
            return loaded;
        }

        while (!_readCache->entries.isEmpty() &&
               (_readCache->entries.size() >= kMaximumCachedMatchShards ||
                _readCache->cachedPayloadBytes + signature.payloadBytes >
                    kMaximumCachedMatchPayloadBytes))
        {
            auto oldest = _readCache->entries.begin();
            for (auto iterator = _readCache->entries.begin();
                 iterator != _readCache->entries.end();
                 ++iterator)
            {
                if (iterator->accessSerial < oldest->accessSerial)
                {
                    oldest = iterator;
                }
            }
            _readCache->cachedPayloadBytes -= std::min(
                _readCache->cachedPayloadBytes,
                oldest->signature.payloadBytes);
            _readCache->entries.erase(oldest);
        }
        _readCache->entries.insert(
            cacheKey,
            {signature, loaded, ++_readCache->nextAccessSerial});
        _readCache->cachedPayloadBytes += signature.payloadBytes;
    }
    return loaded;
}

void ImageMatchRepository::invalidateCachedShard(const QString &imagePath) const
{
    std::lock_guard lock(_readCache->mutex);
    const QString cacheKey = ImageMatchFile::stableImageId(imagePath);
    const auto iterator = _readCache->entries.find(cacheKey);
    if (iterator == _readCache->entries.end())
    {
        return;
    }
    _readCache->cachedPayloadBytes -= std::min(
        _readCache->cachedPayloadBytes,
        iterator->signature.payloadBytes);
    _readCache->entries.erase(iterator);
}

bool ImageMatchRepository::loadPair(const QString &image0Path,
                                    const QString &image1Path,
                                    const QString &algorithmId,
                                    std::uint32_t algorithmVersion,
                                    const QByteArray &configFingerprint,
                                    const QByteArray &modelFingerprint,
                                    PairMatchData *pair,
                                    QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!pair)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("像对缓存读取目标为空");
        }
        return false;
    }

    return loadPairImpl([this](const QString &path, QString *error)
                        {
                            return loadShardCached(path, error);
                        },
                        image0Path,
                        image1Path,
                        algorithmId,
                        algorithmVersion,
                        configFingerprint,
                        modelFingerprint,
                        false,
                        pair,
                        errorMessage);
}

bool ImageMatchRepository::loadPairWithConfigPrefix(
    const QString &image0Path,
    const QString &image1Path,
    const QString &algorithmId,
    std::uint32_t algorithmVersion,
    const QByteArray &configFingerprintPrefix,
    const QByteArray &modelFingerprint,
    PairMatchData *pair,
    QString *errorMessage) const
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!pair || configFingerprintPrefix.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = pair
                ? QStringLiteral("原始匹配配置指纹前缀为空")
                : QStringLiteral("像对缓存读取目标为空");
        }
        return false;
    }

    return loadPairImpl([this](const QString &path, QString *error)
                        {
                            return loadShardCached(path, error);
                        },
                        image0Path,
                        image1Path,
                        algorithmId,
                        algorithmVersion,
                        configFingerprintPrefix,
                        modelFingerprint,
                        true,
                        pair,
                        errorMessage);
}

ImageMatchWriteResult ImageMatchRepository::writePairs(
    const std::vector<PairMatchData> &pairs,
    bool preserveOtherVariants) const
{
    std::vector<const PairMatchData *> references;
    references.reserve(pairs.size());
    for (const PairMatchData &pair : pairs)
    {
        references.push_back(&pair);
    }
    return writePairReferences(references, preserveOtherVariants);
}

ImageMatchWriteResult ImageMatchRepository::writePairReferences(
    const std::vector<const PairMatchData *> &pairs,
    bool preserveOtherVariants) const
{
    ImageMatchWriteResult result;
    if (_directory.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("影像匹配目录为空");
        return result;
    }
    if (!QDir().mkpath(_directory))
    {
        result.errorMessage = QStringLiteral("无法创建影像匹配目录: %1").arg(_directory);
        return result;
    }

    std::map<QString, ImageMatchShard> shards;
    for (const PairMatchData *pairPointer : pairs)
    {
        if (!pairPointer)
        {
            result.errorMessage = QStringLiteral("待写入结果包含空像对指针");
            return result;
        }
        const PairMatchData &pair = *pairPointer;
        if (!pair.image0.isValid() || !pair.image1.isValid() ||
            pair.image0.stableId == pair.image1.stableId)
        {
            result.errorMessage = QStringLiteral("待写入结果包含无效像对身份");
            return result;
        }

        for (const ImageIdentity *identity : {&pair.image0, &pair.image1})
        {
            ImageMatchShard &shard = shards[identity->stableId];
            if (!shard.owner.isValid())
            {
                shard.owner = *identity;
                if (preserveOtherVariants)
                {
                    ImageMatchShard existing;
                    QString loadError;
                    if (loadShard(identity->path, &existing, &loadError))
                    {
                        shard = std::move(existing);
                        shard.owner = *identity;
                    }
                    else if (!loadError.isEmpty())
                    {
                        result.errorMessage = loadError;
                        return result;
                    }
                }
            }
        }

        ImageMatchShard &shard0 = shards[pair.image0.stableId];
        ImageMatchShard &shard1 = shards[pair.image1.stableId];
        upsertBlock(&shard0, directedBlock(pair, false));
        upsertBlock(&shard1, directedBlock(pair, true));
    }

    for (auto &[stableId, shard] : shards)
    {
        Q_UNUSED(stableId)
        shard.normalize();
        const QString path = shardPath(shard.owner.path);
        QString writeError;
        bool changed = false;
        if (!ImageMatchFile::writeIfChanged(path, shard, &changed, &writeError))
        {
            result.errorMessage = writeError;
            return result;
        }
        QString indexError;
        const bool indexMissing = !QFileInfo::exists(
            ImageMatchIndexFile::pathForMatchFile(path));
        if ((changed || indexMissing) &&
            !ImageMatchIndexFile::writeForShard(path, shard, &indexError))
        {
            // 索引是可重建的性能缓存，不能让已经原子提交成功的权威匹配结果失败。
            // 删除可能残留的旧索引，使下一次目录扫描从 `.pimatch` 安全重建。
            ImageMatchIndexFile::removeForMatchFile(path);
        }
        if (changed)
        {
            invalidateCachedShard(shard.owner.path);
            result.writtenFiles.append(path);
        }
    }

    result.success = true;
    result.imageCount = static_cast<int>(shards.size());
    result.pairCount = static_cast<int>(pairs.size());
    return result;
}

bool ImageMatchRepository::clear(QString *errorMessage) const
{
    QDir directory(_directory);
    if (!directory.exists())
    {
        return true;
    }
    const QString pattern = QStringLiteral("*") + QString::fromLatin1(kImageMatchFileSuffix);
    const QStringList files = directory.entryList({pattern}, QDir::Files, QDir::Name);
    QStringList failed;
    for (const QString &fileName : files)
    {
        const QString path = directory.filePath(fileName);
        const bool matchRemoved = QFile::remove(path);
        const bool indexRemoved = ImageMatchIndexFile::removeForMatchFile(path);
        if (!matchRemoved)
        {
            failed.append(path);
        }
        if (!indexRemoved)
        {
            failed.append(ImageMatchIndexFile::pathForMatchFile(path));
        }
    }
    const QString indexPattern = QStringLiteral("*") +
        QString::fromLatin1(kImageMatchFileSuffix) +
        QString::fromLatin1(kImageMatchIndexFileSuffix);
    const QStringList orphanIndexes = directory.entryList(
        {indexPattern}, QDir::Files, QDir::Name);
    for (const QString &fileName : orphanIndexes)
    {
        const QString path = directory.filePath(fileName);
        if (!QFile::remove(path))
        {
            if (!failed.contains(path))
            {
                failed.append(path);
            }
        }
        else
        {
            failed.removeAll(path);
        }
    }
    if (failed.isEmpty())
    {
        std::lock_guard lock(_readCache->mutex);
        _readCache->entries.clear();
        _readCache->cachedPayloadBytes = 0;
        return true;
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("无法删除以下影像匹配分片:\n%1")
            .arg(failed.join(QLatin1Char('\n')));
    }
    return false;
}

} // namespace xjw::image_matching
