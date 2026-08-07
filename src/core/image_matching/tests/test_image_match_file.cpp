#include "ImageMatchFile.h"
#include "ImageMatchIndexFile.h"
#include "ImageMatchRepository.h"

#include <gtest/gtest.h>

#include <QFile>
#include <QDataStream>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QTemporaryDir>

#include <limits>

namespace xjw::image_matching
{
namespace
{

KeypointObservation observation(std::uint32_t id, float x, float y)
{
    KeypointObservation value;
    value.featureId = id;
    value.x = x;
    value.y = y;
    value.scale = 3.0f;
    value.orientation = 45.0f;
    value.response = 0.8f;
    return value;
}

PairMatchData samplePair(const QString &image0, const QString &image1)
{
    PairMatchData pair;
    pair.image0 = ImageMatchFile::identityForImage(image0, 640, 480);
    pair.image1 = ImageMatchFile::identityForImage(image1, 640, 480);
    pair.algorithmId = QStringLiteral("sift_lightglue");
    pair.algorithmVersion = 1;
    pair.configFingerprint = QByteArray::fromHex("01020304");
    pair.modelFingerprint = QByteArray::fromHex("05060708");
    pair.createdTimeMs = 123456;
    pair.rawMatchCount = 2;
    pair.geometryInlierCount = 1;
    pair.tiePointMatchCount = 1;
    pair.geometryPassed = true;
    pair.geometryModel = GeometryModel::Fundamental;
    pair.geometryMatrix = {0.0, 0.0, 1.0,
                           0.0, 0.0, 2.0,
                           -1.0, -2.0, 0.0};

    PairCorrespondence first;
    first.observation0 = observation(4, 10.0f, 20.0f);
    first.observation1 = observation(9, 30.0f, 40.0f);
    first.confidence = 0.95f;
    first.residualPixels = 0.25f;
    first.flags = MatchRecordFlag::GeometryInlier | MatchRecordFlag::InTiePointTrack;
    pair.correspondences.push_back(first);

    PairCorrespondence second;
    second.observation0 = observation(6, 50.0f, 60.0f);
    second.observation1 = observation(11, 70.0f, 80.0f);
    second.confidence = 0.4f;
    second.residualPixels = 4.0f;
    pair.correspondences.push_back(second);
    return pair;
}

bool createImageFile(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        return false;
    }
    return file.write("image", 5) == 5;
}

TEST(ImageMatchFileTest, RoundTripsOneShardWithResidualAndFlags)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    const PairMatchData pair = samplePair(image0, image1);
    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    const ImageMatchWriteResult writeResult = repository.writePairs({pair}, false);
    ASSERT_TRUE(writeResult.success) << writeResult.errorMessage.toStdString();
    EXPECT_EQ(writeResult.imageCount, 2);
    EXPECT_EQ(writeResult.pairCount, 1);

    ImageMatchShard shard;
    QString error;
    ASSERT_TRUE(repository.loadShard(image0, &shard, &error)) << error.toStdString();
    ASSERT_EQ(shard.neighbors.size(), 1U);
    const NeighborMatchBlock &block = shard.neighbors.front();
    ASSERT_EQ(block.ownerObservations.size(), 2U);
    EXPECT_EQ(block.rawMatchCount, 2U);
    EXPECT_EQ(block.geometryInlierCount, 1U);
    ASSERT_EQ(block.matches.size(), 2U);
    EXPECT_FLOAT_EQ(block.matches.front().residualPixels, 0.25f);
    EXPECT_TRUE(hasFlag(block.matches.front().flags, MatchRecordFlag::GeometryInlier));
    EXPECT_TRUE(hasFlag(block.matches.front().flags, MatchRecordFlag::InTiePointTrack));
}

TEST(ImageMatchIndexFileTest, RepositoryWritesCompactPersistentIndex)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    ASSERT_TRUE(repository.writePairs({samplePair(image0, image1)}, false).success);
    const QString matchPath = repository.shardPath(image0);
    const QString indexPath = ImageMatchIndexFile::pathForMatchFile(matchPath);
    ASSERT_TRUE(QFileInfo::exists(indexPath));
    EXPECT_LT(QFileInfo(indexPath).size(), QFileInfo(matchPath).size());

    ImageMatchIndexFile::clearMemoryCache();
    ImageMatchFileIndex index;
    ImageMatchIndexLoadSource source = ImageMatchIndexLoadSource::RebuiltFromMatchFile;
    QString error;
    ASSERT_TRUE(ImageMatchIndexFile::load(matchPath, &index, &source, &error))
        << error.toStdString();
    EXPECT_EQ(source, ImageMatchIndexLoadSource::PersistentIndex);
    EXPECT_EQ(index.owner.stableId, ImageMatchFile::stableImageId(image0));
    ASSERT_EQ(index.neighbors.size(), 1U);
    const ImageMatchNeighborIndex &neighbor = index.neighbors.front();
    EXPECT_EQ(neighbor.peer.stableId, ImageMatchFile::stableImageId(image1));
    EXPECT_EQ(neighbor.rawMatchCount, 2U);
    EXPECT_EQ(neighbor.geometryInlierCount, 1U);
    EXPECT_TRUE(neighbor.geometryPassed);
    EXPECT_DOUBLE_EQ(neighbor.geometricCoverage, 1.0 / 16.0);
}

TEST(ImageMatchIndexFileTest, RebuildsMissingIndexAndThenUsesMemoryCache)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    ASSERT_TRUE(repository.writePairs({samplePair(image0, image1)}, false).success);
    const QString matchPath = repository.shardPath(image0);
    ASSERT_TRUE(ImageMatchIndexFile::removeForMatchFile(matchPath));
    ImageMatchIndexFile::clearMemoryCache();

    ImageMatchFileIndex rebuilt;
    ImageMatchIndexLoadSource source = ImageMatchIndexLoadSource::PersistentIndex;
    QString error;
    ASSERT_TRUE(ImageMatchIndexFile::load(matchPath, &rebuilt, &source, &error))
        << error.toStdString();
    EXPECT_EQ(source, ImageMatchIndexLoadSource::RebuiltFromMatchFile);
    EXPECT_TRUE(QFileInfo::exists(ImageMatchIndexFile::pathForMatchFile(matchPath)));

    ImageMatchFileIndex cached;
    ASSERT_TRUE(ImageMatchIndexFile::load(matchPath, &cached, &source, &error))
        << error.toStdString();
    EXPECT_EQ(source, ImageMatchIndexLoadSource::MemoryCache);
    EXPECT_EQ(cached.sourceSignature, rebuilt.sourceSignature);
}

TEST(ImageMatchIndexFileTest, RebuildsIndexWithCorruptedChecksum)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    ASSERT_TRUE(repository.writePairs({samplePair(image0, image1)}, false).success);
    const QString matchPath = repository.shardPath(image0);
    QFile indexFile(ImageMatchIndexFile::pathForMatchFile(matchPath));
    ASSERT_TRUE(indexFile.open(QIODevice::ReadWrite));
    ASSERT_GT(indexFile.size(), 64);
    ASSERT_TRUE(indexFile.seek(indexFile.size() - 1));
    char byte = 0;
    ASSERT_EQ(indexFile.read(&byte, 1), 1);
    byte ^= 0x5a;
    ASSERT_TRUE(indexFile.seek(indexFile.size() - 1));
    ASSERT_EQ(indexFile.write(&byte, 1), 1);
    indexFile.close();

    ImageMatchIndexFile::clearMemoryCache();
    ImageMatchFileIndex index;
    ImageMatchIndexLoadSource source = ImageMatchIndexLoadSource::PersistentIndex;
    QString error;
    ASSERT_TRUE(ImageMatchIndexFile::load(matchPath, &index, &source, &error))
        << error.toStdString();
    EXPECT_EQ(source, ImageMatchIndexLoadSource::RebuiltFromMatchFile);
    EXPECT_EQ(index.neighbors.size(), 1U);
}

TEST(ImageMatchIndexFileTest, RefreshesIndexWhenMatchShardChanges)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    PairMatchData first = samplePair(image0, image1);
    ASSERT_TRUE(repository.writePairs({first}, false).success);
    const QString matchPath = repository.shardPath(image0);
    const QString indexPath = ImageMatchIndexFile::pathForMatchFile(matchPath);
    QFile oldIndex(indexPath);
    ASSERT_TRUE(oldIndex.open(QIODevice::ReadOnly));
    const QByteArray staleIndexBytes = oldIndex.readAll();
    oldIndex.close();
    ASSERT_FALSE(staleIndexBytes.isEmpty());

    PairMatchData second = first;
    second.configFingerprint = QByteArrayLiteral("second-config");
    second.rawMatchCount = 1;
    second.correspondences.resize(1);
    ASSERT_TRUE(repository.writePairs({second}, true).success);

    // 模拟异常退出遗留的旧索引：源分片头部 payload SHA 已变化，读取器必须
    // 拒绝旧索引并从权威 `.pimatch` 重建，而不能仅依赖文件名。
    QFile staleIndex(indexPath);
    ASSERT_TRUE(staleIndex.open(QIODevice::WriteOnly | QIODevice::Truncate));
    ASSERT_EQ(staleIndex.write(staleIndexBytes), staleIndexBytes.size());
    staleIndex.close();

    ImageMatchIndexFile::clearMemoryCache();
    ImageMatchFileIndex index;
    ImageMatchIndexLoadSource source = ImageMatchIndexLoadSource::RebuiltFromMatchFile;
    QString error;
    ASSERT_TRUE(ImageMatchIndexFile::load(matchPath,
                                          &index,
                                          &source,
                                          &error)) << error.toStdString();
    EXPECT_EQ(source, ImageMatchIndexLoadSource::RebuiltFromMatchFile);
    EXPECT_EQ(index.neighbors.size(), 2U);
}

TEST(ImageMatchIndexFileTest, RepositoryClearRemovesMatchAndIndexFiles)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    ASSERT_TRUE(repository.writePairs({samplePair(image0, image1)}, false).success);
    const QString matchPath = repository.shardPath(image0);
    const QString indexPath = ImageMatchIndexFile::pathForMatchFile(matchPath);
    ASSERT_TRUE(QFileInfo::exists(matchPath));
    ASSERT_TRUE(QFileInfo::exists(indexPath));

    QString error;
    ASSERT_TRUE(repository.clear(&error)) << error.toStdString();
    EXPECT_FALSE(QFileInfo::exists(matchPath));
    EXPECT_FALSE(QFileInfo::exists(indexPath));
}

TEST(ImageMatchFileTest, ReadsPairFromEitherDirectedShard)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("same_name.png"));
    const QString nested = temporary.filePath(QStringLiteral("nested"));
    ASSERT_TRUE(QDir().mkpath(nested));
    const QString image1 = QDir(nested).filePath(QStringLiteral("same_name.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    const PairMatchData expected = samplePair(image0, image1);
    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    ASSERT_TRUE(repository.writePairs({expected}, false).success);

    PairMatchData loaded;
    QString error;
    ASSERT_TRUE(repository.loadPair(image0,
                                    image1,
                                    expected.algorithmId,
                                    expected.algorithmVersion,
                                    expected.configFingerprint,
                                    expected.modelFingerprint,
                                    &loaded,
                                    &error)) << error.toStdString();
    ASSERT_EQ(loaded.correspondences.size(), 2U);
    EXPECT_EQ(loaded.correspondences.front().observation0.featureId, 4U);
    EXPECT_EQ(loaded.correspondences.front().observation1.featureId, 9U);
    EXPECT_NE(repository.shardPath(image0), repository.shardPath(image1));
}

TEST(ImageMatchFileTest, RejectsCorruptedPayload)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    ASSERT_TRUE(repository.writePairs({samplePair(image0, image1)}, false).success);
    const QString path = repository.shardPath(image0);

    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::ReadWrite));
    ASSERT_GT(file.size(), 64);
    ASSERT_TRUE(file.seek(file.size() - 1));
    char byte = 0;
    ASSERT_EQ(file.read(&byte, 1), 1);
    byte ^= 0x5a;
    ASSERT_TRUE(file.seek(file.size() - 1));
    ASSERT_EQ(file.write(&byte, 1), 1);
    file.close();

    // 部分 Linux 文件系统会把两次相邻写入折叠到同一毫秒时间戳，而轻量索引
    // 的快速失效键按毫秒保存。显式推进修改时间，使测试稳定覆盖“签名变化后
    // 回退完整 SHA 校验”的契约，而不依赖主机文件系统的时间分辨率。
    QFile timestampFile(path);
    ASSERT_TRUE(timestampFile.open(QIODevice::ReadWrite));
    ASSERT_TRUE(timestampFile.setFileTime(
        QDateTime::currentDateTimeUtc().addSecs(1),
        QFileDevice::FileModificationTime));
    timestampFile.close();

    ImageMatchShard shard;
    QString error;
    EXPECT_FALSE(ImageMatchFile::read(path, &shard, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("SHA-256")));

    // 轻量索引仍以源文件大小和修改时间参与失效判断。payload 被外部改写后
    // 不得继续信任旧 `.pidx`，回退到完整校验时应报告同一 SHA 错误。
    ImageMatchIndexFile::clearMemoryCache();
    ImageMatchFileIndex index;
    error.clear();
    EXPECT_FALSE(ImageMatchIndexFile::load(path, &index, nullptr, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("SHA-256")));
}

TEST(ImageMatchFileTest, RejectsPayloadLengthThatCannotFitReaderBuffer)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString path = temporary.filePath(QStringLiteral("oversized.pimatch"));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));

    // 只伪造容器头，不分配巨量数据。读取器必须在 QByteArray resize 前拒绝该长度。
    QDataStream stream(&file);
    stream.setVersion(QDataStream::Qt_5_15);
    stream.setByteOrder(QDataStream::LittleEndian);
    ASSERT_EQ(stream.writeRawData("PLIMATCH", 8), 8);
    stream << static_cast<quint32>(kImageMatchFormatVersion)
           << std::numeric_limits<quint64>::max();
    ASSERT_EQ(stream.status(), QDataStream::Ok);
    file.close();

    ImageMatchShard shard;
    QString error;
    EXPECT_FALSE(ImageMatchFile::read(path, &shard, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("payload")));
}

TEST(ImageMatchFileTest, ReplacesOnlyMatchingAlgorithmVariant)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    PairMatchData first = samplePair(image0, image1);
    ASSERT_TRUE(repository.writePairs({first}, false).success);

    PairMatchData second = first;
    second.configFingerprint = QByteArrayLiteral("another-config");
    second.rawMatchCount = 1;
    second.correspondences.resize(1);
    ASSERT_TRUE(repository.writePairs({second}, true).success);

    ImageMatchShard shard;
    QString error;
    ASSERT_TRUE(repository.loadShard(image0, &shard, &error)) << error.toStdString();
    EXPECT_EQ(shard.neighbors.size(), 2U);
}

TEST(ImageMatchFileTest, KeepsFeatureIdNamespacesSeparateAcrossVariants)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    PairMatchData first = samplePair(image0, image1);
    first.configFingerprint = QByteArrayLiteral("first-config");
    first.correspondences.resize(1);
    first.correspondences.front().observation0 = observation(0, 10.0f, 20.0f);
    first.correspondences.front().observation1 = observation(0, 30.0f, 40.0f);
    ASSERT_TRUE(repository.writePairs({first}, false).success);

    PairMatchData second = first;
    second.configFingerprint = QByteArrayLiteral("second-config");
    second.correspondences.front().observation0 = observation(0, 110.0f, 120.0f);
    second.correspondences.front().observation1 = observation(0, 130.0f, 140.0f);
    ASSERT_TRUE(repository.writePairs({second}, true).success);

    ImageMatchShard shard;
    QString error;
    ASSERT_TRUE(repository.loadShard(image0, &shard, &error)) << error.toStdString();
    const NeighborMatchBlock *first_block = shard.findNeighbor(
        first.image1.stableId,
        first.algorithmId,
        first.algorithmVersion,
        first.configFingerprint);
    const NeighborMatchBlock *second_block = shard.findNeighbor(
        second.image1.stableId,
        second.algorithmId,
        second.algorithmVersion,
        second.configFingerprint);
    ASSERT_NE(first_block, nullptr);
    ASSERT_NE(second_block, nullptr);
    ASSERT_NE(first_block->findOwnerObservation(0), nullptr);
    ASSERT_NE(second_block->findOwnerObservation(0), nullptr);
    EXPECT_FLOAT_EQ(first_block->findOwnerObservation(0)->x, 10.0f);
    EXPECT_FLOAT_EQ(second_block->findOwnerObservation(0)->x, 110.0f);

    PairMatchData loaded_first;
    ASSERT_TRUE(repository.loadPair(image0,
                                    image1,
                                    first.algorithmId,
                                    first.algorithmVersion,
                                    first.configFingerprint,
                                    first.modelFingerprint,
                                    &loaded_first,
                                    &error)) << error.toStdString();
    ASSERT_EQ(loaded_first.correspondences.size(), 1U);
    EXPECT_FLOAT_EQ(loaded_first.correspondences.front().observation0.x, 10.0f);
    EXPECT_FLOAT_EQ(loaded_first.correspondences.front().observation1.x, 30.0f);

    PairMatchData loaded_second;
    ASSERT_TRUE(repository.loadPair(image0,
                                    image1,
                                    second.algorithmId,
                                    second.algorithmVersion,
                                    second.configFingerprint,
                                    second.modelFingerprint,
                                    &loaded_second,
                                    &error)) << error.toStdString();
    ASSERT_EQ(loaded_second.correspondences.size(), 1U);
    EXPECT_FLOAT_EQ(loaded_second.correspondences.front().observation0.x, 110.0f);
    EXPECT_FLOAT_EQ(loaded_second.correspondences.front().observation1.x, 130.0f);
}

TEST(ImageMatchFileTest, RejectsCacheEntryFromDifferentModelFingerprint)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    const PairMatchData expected = samplePair(image0, image1);
    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    ASSERT_TRUE(repository.writePairs({expected}, false).success);

    PairMatchData loaded;
    QString error;
    EXPECT_FALSE(repository.loadPair(image0,
                                     image1,
                                     expected.algorithmId,
                                     expected.algorithmVersion,
                                     expected.configFingerprint,
                                     QByteArrayLiteral("different-model"),
                                     &loaded,
                                     &error));
    EXPECT_TRUE(error.isEmpty());
}

TEST(ImageMatchFileTest, RestoresHomographyDirectionWhenOnlyReverseShardExists)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image0 = temporary.filePath(QStringLiteral("left.png"));
    const QString image1 = temporary.filePath(QStringLiteral("right.png"));
    ASSERT_TRUE(createImageFile(image0));
    ASSERT_TRUE(createImageFile(image1));

    PairMatchData expected = samplePair(image0, image1);
    expected.geometryModel = GeometryModel::Homography;
    expected.geometryMatrix = {2.0, 0.0, 10.0,
                               0.0, 3.0, 20.0,
                               0.0, 0.0, 1.0};
    ImageMatchRepository repository(temporary.filePath(QStringLiteral("matches")));
    ASSERT_TRUE(repository.writePairs({expected}, false).success);
    ASSERT_TRUE(QFile::remove(repository.shardPath(image0)));

    PairMatchData loaded;
    QString error;
    ASSERT_TRUE(repository.loadPair(image0,
                                    image1,
                                    expected.algorithmId,
                                    expected.algorithmVersion,
                                    expected.configFingerprint,
                                    expected.modelFingerprint,
                                    &loaded,
                                    &error)) << error.toStdString();
    EXPECT_EQ(loaded.geometryModel, GeometryModel::Homography);
    EXPECT_TRUE(loaded.geometryPassed);
    for (std::size_t index = 0; index < expected.geometryMatrix.size(); ++index)
    {
        EXPECT_NEAR(loaded.geometryMatrix[index], expected.geometryMatrix[index], 1.0e-10);
    }
}

} // namespace
} // namespace xjw::image_matching
