#include "ImageMatchRepository.h"
#include "ImageMatchIndexFile.h"
#include "preparation/MatchResultCatalog.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <gtest/gtest.h>

namespace
{

QString createImage(const QString &directory, const QString &name)
{
    const QString path = QDir(directory).filePath(name);
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly));
    EXPECT_EQ(file.write(name.toUtf8()), name.toUtf8().size());
    file.close();
    return path;
}

xjw::image_matching::PairMatchData makePair(const QString &image0,
                                             const QString &image1,
                                             const QByteArray &fingerprint,
                                             std::uint32_t raw_count,
                                             std::uint32_t inlier_count,
                                             std::int64_t created_time)
{
    using namespace xjw::image_matching;

    PairMatchData pair;
    pair.image0 = ImageMatchFile::identityForImage(image0, 640, 480);
    pair.image1 = ImageMatchFile::identityForImage(image1, 640, 480);
    pair.algorithmId = QStringLiteral("sift_lightglue");
    pair.algorithmVersion = 1;
    pair.configFingerprint = fingerprint;
    pair.modelFingerprint = QByteArrayLiteral("engine-sha256");
    pair.createdTimeMs = created_time;
    pair.rawMatchCount = raw_count;
    pair.geometryInlierCount = inlier_count;
    pair.tiePointMatchCount = inlier_count;
    pair.geometryPassed = inlier_count > 0;
    pair.geometryModel = GeometryModel::Fundamental;

    PairCorrespondence correspondence;
    correspondence.observation0 = {7, 10.0f, 20.0f, 1.5f, 0.2f, 0.8f};
    correspondence.observation1 = {9, 12.0f, 21.0f, 1.7f, 0.3f, 0.9f};
    correspondence.confidence = 0.95f;
    correspondence.residualPixels = 0.4f;
    correspondence.flags = MatchRecordFlag::GeometryInlier |
        MatchRecordFlag::InTiePointTrack;
    pair.correspondences.push_back(correspondence);
    return pair;
}

} // namespace

TEST(MatchResultCatalogTest, ScansPerImageShardsWithoutDuplicatingSymmetricPair)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image_a = createImage(temporary.path(), QStringLiteral("A.png"));
    const QString image_b = createImage(temporary.path(), QStringLiteral("B.png"));
    const QString match_directory = QDir(temporary.path()).filePath(QStringLiteral("matches"));

    xjw::image_matching::ImageMatchRepository repository(match_directory);
    const auto write_result = repository.writePairs(
        {makePair(image_a, image_b, QByteArrayLiteral("config-a"), 120, 90, 1000)},
        false);
    ASSERT_TRUE(write_result.success) << write_result.errorMessage.toStdString();
    ASSERT_EQ(write_result.imageCount, 2);

    int last_processed = 0;
    int last_total = 0;
    xjw::aerial_triangulation::MatchResultCatalogConfig config;
    config.matchDirectory = match_directory;
    config.progressCallback = [&](int processed, int total)
    {
        last_processed = processed;
        last_total = total;
    };

    xjw::image_matching::ImageMatchIndexFile::clearMemoryCache();

    const auto summary = xjw::aerial_triangulation::MatchResultCatalog(config).scan();
    EXPECT_EQ(summary.matchFileCount, 2);
    EXPECT_EQ(summary.pairGroupCount, 1);
    EXPECT_EQ(summary.variantCount, 1);
    EXPECT_EQ(summary.compatibleVariantCount, 1);
    EXPECT_EQ(summary.incompatibleVariantCount, 0);
    EXPECT_EQ(summary.persistentIndexHitCount, 2);
    EXPECT_EQ(summary.rebuiltIndexCount, 0);
    EXPECT_EQ(last_processed, 2);
    EXPECT_EQ(last_total, 2);

    ASSERT_EQ(summary.pairGroups.size(), 1);
    const auto &group = summary.pairGroups.first();
    ASSERT_EQ(group.variants.size(), 1);
    ASSERT_EQ(group.bestVariantIndex, 0);
    EXPECT_EQ(group.variants.first().algorithmId, QStringLiteral("sift_lightglue"));
    EXPECT_EQ(group.variants.first().totalMatches, 120);
    EXPECT_EQ(group.variants.first().geometricVerifiedInliers, 90);
    EXPECT_DOUBLE_EQ(group.variants.first().geometricCoverage, 1.0 / 16.0);
    EXPECT_TRUE(group.variants.first().geometryPassed);
    EXPECT_TRUE(group.variants.first().matchFilePath.endsWith(QStringLiteral(".pimatch")));
}

TEST(MatchResultCatalogTest, RebuildsOnlyMissingPersistentIndex)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image_a = createImage(temporary.path(), QStringLiteral("A.png"));
    const QString image_b = createImage(temporary.path(), QStringLiteral("B.png"));
    const QString match_directory = QDir(temporary.path()).filePath(QStringLiteral("matches"));
    xjw::image_matching::ImageMatchRepository repository(match_directory);
    ASSERT_TRUE(repository.writePairs(
        {makePair(image_a, image_b, QByteArrayLiteral("config-a"), 120, 90, 1000)},
        false).success);

    const QString missingIndexMatch = repository.shardPath(image_a);
    ASSERT_TRUE(xjw::image_matching::ImageMatchIndexFile::removeForMatchFile(missingIndexMatch));
    xjw::image_matching::ImageMatchIndexFile::clearMemoryCache();

    QVector<int> progress;
    xjw::aerial_triangulation::MatchResultCatalogConfig config;
    config.matchDirectory = match_directory;
    config.maxConcurrency = 4;
    config.progressCallback = [&](int processed, int total)
    {
        EXPECT_EQ(total, 2);
        progress.push_back(processed);
    };
    const auto summary = xjw::aerial_triangulation::MatchResultCatalog(config).scan();
    EXPECT_EQ(summary.persistentIndexHitCount, 1);
    EXPECT_EQ(summary.rebuiltIndexCount, 1);
    EXPECT_EQ(progress, QVector<int>({1, 2}));

    xjw::image_matching::ImageMatchIndexFile::clearMemoryCache();
    const auto cachedSummary = xjw::aerial_triangulation::MatchResultCatalog(config).scan();
    EXPECT_EQ(cachedSummary.persistentIndexHitCount, 2);
    EXPECT_EQ(cachedSummary.rebuiltIndexCount, 0);
}

TEST(MatchResultCatalogTest, KeepsVariantsInsideShardAndSelectsStrongestGeometry)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image_a = createImage(temporary.path(), QStringLiteral("A.png"));
    const QString image_b = createImage(temporary.path(), QStringLiteral("B.png"));
    const QString match_directory = QDir(temporary.path()).filePath(QStringLiteral("matches"));
    xjw::image_matching::ImageMatchRepository repository(match_directory);

    ASSERT_TRUE(repository.writePairs(
        {makePair(image_a, image_b, QByteArrayLiteral("config-low"), 300, 40, 1000)},
        false).success);
    ASSERT_TRUE(repository.writePairs(
        {makePair(image_a, image_b, QByteArrayLiteral("config-high"), 180, 95, 2000)},
        true).success);

    xjw::aerial_triangulation::MatchResultCatalogConfig config;
    config.matchDirectory = match_directory;
    const auto summary = xjw::aerial_triangulation::MatchResultCatalog(config).scan();
    ASSERT_EQ(summary.pairGroups.size(), 1);
    const auto &group = summary.pairGroups.first();
    ASSERT_EQ(group.variants.size(), 2);
    ASSERT_GE(group.bestVariantIndex, 0);
    EXPECT_EQ(group.variants.at(group.bestVariantIndex).geometricVerifiedInliers, 95);
}

TEST(MatchResultCatalogTest, AppliesTargetImageFilterAndRejectsCorruptedShard)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString image_a = createImage(temporary.path(), QStringLiteral("A.png"));
    const QString image_b = createImage(temporary.path(), QStringLiteral("B.png"));
    const QString image_c = createImage(temporary.path(), QStringLiteral("C.png"));
    const QString match_directory = QDir(temporary.path()).filePath(QStringLiteral("matches"));
    xjw::image_matching::ImageMatchRepository repository(match_directory);
    ASSERT_TRUE(repository.writePairs(
        {makePair(image_a, image_b, QByteArrayLiteral("config-a"), 100, 70, 1000)},
        false).success);

    QFile corrupted(QDir(match_directory).filePath(QStringLiteral("corrupted.pimatch")));
    ASSERT_TRUE(corrupted.open(QIODevice::WriteOnly));
    ASSERT_GT(corrupted.write("not-a-pimatch"), 0);
    corrupted.close();

    xjw::aerial_triangulation::MatchResultCatalogConfig included;
    included.matchDirectory = match_directory;
    included.targetImagePath = image_a;
    const auto included_summary =
        xjw::aerial_triangulation::MatchResultCatalog(included).scan();
    EXPECT_EQ(included_summary.matchFileCount, 3);
    EXPECT_EQ(included_summary.incompatibleVariantCount, 1);
    EXPECT_EQ(included_summary.pairGroupCount, 1);

    xjw::aerial_triangulation::MatchResultCatalogConfig excluded = included;
    excluded.targetImagePath = image_c;
    const auto excluded_summary =
        xjw::aerial_triangulation::MatchResultCatalog(excluded).scan();
    EXPECT_EQ(excluded_summary.pairGroupCount, 0);
    EXPECT_EQ(excluded_summary.incompatibleVariantCount, 1);
}

TEST(MatchResultCatalogTest, FormatsRegisteredAlgorithmLabel)
{
    xjw::aerial_triangulation::MatchVariant variant;
    variant.algorithmId = QStringLiteral("sift_lightglue");
    variant.algorithmVersion = 3;
    EXPECT_EQ(xjw::aerial_triangulation::MatchResultCatalog::algorithmDisplayLabel(variant),
              QStringLiteral("sift-lightglue v3"));
}
