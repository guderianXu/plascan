#include <gtest/gtest.h>

#include "MatchResultCatalog.h"

#include <QDataStream>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSet>
#include <QTemporaryDir>

namespace
{

QString writeSgmtMatch(const QString &directory,
                       const QString &fileName,
                       const QString &imageA,
                       const QString &imageB,
                       int matchCount,
                       const QDateTime &modifiedTime = QDateTime(),
                       quint32 version = 2)
{
    const QString path = QDir(directory).filePath(fileName);
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));

    QDataStream out(&file);
    out.setVersion(QDataStream::Qt_5_15);
    out.writeRawData("SGMT", 4);
    out << version;

    const QByteArray imageABytes = imageA.toUtf8();
    const QByteArray imageBBytes = imageB.toUtf8();
    out << quint32(imageABytes.size());
    out.writeRawData(imageABytes.constData(), imageABytes.size());
    out << quint32(imageBBytes.size());
    out.writeRawData(imageBBytes.constData(), imageBBytes.size());

    out << qint32(matchCount);
    out << qint32(matchCount);
    out << qint32(matchCount);

    for (int i = 0; i < matchCount; ++i)
    {
        out << qint32(i);
        out << float(1.0f);
    }
    for (int i = 0; i < matchCount; ++i)
    {
        out << qint32(i);
        out << float(1.0f);
    }

    file.close();
    if (modifiedTime.isValid())
    {
        QFile timeFile(path);
        EXPECT_TRUE(timeFile.open(QIODevice::ReadWrite));
        EXPECT_TRUE(timeFile.setFileTime(modifiedTime, QFileDevice::FileModificationTime));
    }

    return path;
}

QString writeInvalidMatch(const QString &directory, const QString &fileName)
{
    const QString path = QDir(directory).filePath(fileName);
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write("BAD!");
    return path;
}

void writeSidecar(const QString &matchPath,
                  const QString &imageA,
                  const QString &imageB,
                  const QString &featureAlgorithm,
                  const QString &matchAlgorithm,
                  int matchCount,
                  int geometricInliers,
                  bool includeInlierStats = true)
{
    QJsonArray indices0;
    QJsonArray indices1;
    QJsonArray scores;
    for (int i = 0; i < matchCount; ++i)
    {
        indices0.append(i);
        indices1.append(i);
        scores.append(1.0);
    }

    QJsonObject sidecar;
    sidecar.insert(QStringLiteral("match_file"), matchPath);
    sidecar.insert(QStringLiteral("image0_name"), QFileInfo(imageA).fileName());
    sidecar.insert(QStringLiteral("image1_name"), QFileInfo(imageB).fileName());
    sidecar.insert(QStringLiteral("image0_path"), imageA);
    sidecar.insert(QStringLiteral("image1_path"), imageB);
    sidecar.insert(QStringLiteral("feature0_path"), imageA + QStringLiteral(".sp"));
    sidecar.insert(QStringLiteral("feature1_path"), imageB + QStringLiteral(".sp"));
    sidecar.insert(QStringLiteral("sp0_path"), imageA + QStringLiteral(".sp"));
    sidecar.insert(QStringLiteral("sp1_path"), imageB + QStringLiteral(".sp"));
    sidecar.insert(QStringLiteral("feature_algorithm"), featureAlgorithm);
    sidecar.insert(QStringLiteral("match_algorithm"), matchAlgorithm);
    sidecar.insert(QStringLiteral("feature_format_version"), 2);
    sidecar.insert(QStringLiteral("num_matches"), matchCount);
    if (includeInlierStats)
    {
        sidecar.insert(QStringLiteral("geometric_inlier_count"), geometricInliers);
    }
    sidecar.insert(QStringLiteral("matched_indices0"), indices0);
    sidecar.insert(QStringLiteral("matched_indices1"), indices1);
    sidecar.insert(QStringLiteral("matched_scores"), scores);

    QFile file(matchPath + QStringLiteral(".json"));
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write(QJsonDocument(sidecar).toJson(QJsonDocument::Compact));
}

const xjw::pipeline::MatchPairGroup *findGroup(const xjw::pipeline::MatchResultCatalogSummary &summary,
                                               const QString &imageA,
                                               const QString &imageB)
{
    const QString key = xjw::pipeline::MatchResultCatalog::canonicalPairKey(imageA, imageB);
    const auto it = std::find_if(summary.pairGroups.begin(), summary.pairGroups.end(),
                                 [&](const xjw::pipeline::MatchPairGroup &group)
    {
        return group.pairKey == key;
    });
    return it == summary.pairGroups.end() ? nullptr : &(*it);
}

} // namespace

TEST(MatchResultCatalogTest, GroupsAlgorithmVariantsForSameImagePair)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("image_A.tif"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("image_B.tif"));
    const QString imageC = QDir(tempDir.path()).filePath(QStringLiteral("image_C.tif"));

    const QString supergluePath = writeSgmtMatch(tempDir.path(),
                                                 QStringLiteral("image_A__image_B_superglue.match"),
                                                 imageA,
                                                 imageB,
                                                 100);
    writeSidecar(supergluePath, imageA, imageB, QStringLiteral("superpoint"), QStringLiteral("superglue"), 100, 45);

    const QString lightgluePath = writeSgmtMatch(tempDir.path(),
                                                 QStringLiteral("image_B__image_A_lightglue.match"),
                                                 imageB,
                                                 imageA,
                                                 120);
    writeSidecar(lightgluePath, imageB, imageA, QStringLiteral("disk"), QStringLiteral("lightglue"), 120, 40);

    const QString otherPath = writeSgmtMatch(tempDir.path(),
                                             QStringLiteral("image_A__image_C_superglue.match"),
                                             imageA,
                                             imageC,
                                             90);
    writeSidecar(otherPath, imageA, imageC, QStringLiteral("superpoint"), QStringLiteral("superglue"), 90, 30);

    xjw::pipeline::MatchResultCatalogConfig config;
    config.matchDirectory = tempDir.path();
    const xjw::pipeline::MatchResultCatalogSummary summary =
        xjw::pipeline::MatchResultCatalog(config).scan();

    EXPECT_EQ(summary.matchFileCount, 3);
    EXPECT_EQ(summary.variantCount, 3);
    EXPECT_EQ(summary.compatibleVariantCount, 3);
    ASSERT_EQ(summary.pairGroups.size(), 2);

    const xjw::pipeline::MatchPairGroup *group = findGroup(summary, imageA, imageB);
    ASSERT_NE(group, nullptr);
    ASSERT_EQ(group->variants.size(), 2);
    ASSERT_GE(group->bestVariantIndex, 0);
    ASSERT_LT(group->bestVariantIndex, group->variants.size());

    const xjw::pipeline::MatchVariant &best = group->variants.at(group->bestVariantIndex);
    EXPECT_EQ(best.matchFilePath, QFileInfo(supergluePath).absoluteFilePath());
    EXPECT_EQ(best.featureAlgorithm, QStringLiteral("superpoint"));
    EXPECT_EQ(best.matchAlgorithm, QStringLiteral("superglue"));
    EXPECT_EQ(best.totalMatches, 100);
    EXPECT_EQ(best.geometricVerifiedInliers, 45);
    EXPECT_TRUE(best.compatible);
    EXPECT_EQ(best.status, QStringLiteral("compatible"));
}

TEST(MatchResultCatalogTest, SelectsBestVariantByInliersThenMatchesThenModifiedTime)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("left.tif"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("right.tif"));
    const QDateTime older = QDateTime::fromSecsSinceEpoch(1700000000, Qt::UTC);
    const QDateTime newer = QDateTime::fromSecsSinceEpoch(1700003600, Qt::UTC);

    const QString lowerInliers = writeSgmtMatch(tempDir.path(),
                                                QStringLiteral("left__right_low_inliers.match"),
                                                imageA,
                                                imageB,
                                                500,
                                                newer);
    writeSidecar(lowerInliers, imageA, imageB, QStringLiteral("superpoint"), QStringLiteral("superglue"), 500, 70);

    const QString fewerMatches = writeSgmtMatch(tempDir.path(),
                                                QStringLiteral("left__right_fewer_matches.match"),
                                                imageA,
                                                imageB,
                                                120,
                                                newer);
    writeSidecar(fewerMatches, imageA, imageB, QStringLiteral("disk"), QStringLiteral("lightglue"), 120, 90);

    const QString olderTie = writeSgmtMatch(tempDir.path(),
                                            QStringLiteral("left__right_old_tie.match"),
                                            imageA,
                                            imageB,
                                            140,
                                            older);
    writeSidecar(olderTie, imageA, imageB, QStringLiteral("aliked"), QStringLiteral("lightglue"), 140, 90);

    const QString expectedBest = writeSgmtMatch(tempDir.path(),
                                                QStringLiteral("left__right_new_tie.match"),
                                                imageA,
                                                imageB,
                                                140,
                                                newer);
    writeSidecar(expectedBest, imageA, imageB, QStringLiteral("sift"), QStringLiteral("bf"), 140, 90);

    xjw::pipeline::MatchResultCatalogConfig config;
    config.matchDirectory = tempDir.path();
    const xjw::pipeline::MatchResultCatalogSummary summary =
        xjw::pipeline::MatchResultCatalog(config).scan();

    ASSERT_EQ(summary.pairGroups.size(), 1);
    const xjw::pipeline::MatchPairGroup &group = summary.pairGroups.front();
    ASSERT_EQ(group.variants.size(), 4);
    ASSERT_GE(group.bestVariantIndex, 0);

    const xjw::pipeline::MatchVariant &best = group.variants.at(group.bestVariantIndex);
    EXPECT_EQ(best.matchFilePath, QFileInfo(expectedBest).absoluteFilePath());
    EXPECT_EQ(best.geometricVerifiedInliers, 90);
    EXPECT_EQ(best.totalMatches, 140);
}

TEST(MatchResultCatalogTest, PrefersExplicitZeroInlierStatsOverMissingStats)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("zero_A.tif"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("zero_B.tif"));
    const QDateTime older = QDateTime::fromSecsSinceEpoch(1700000000, Qt::UTC);
    const QDateTime newer = QDateTime::fromSecsSinceEpoch(1700003600, Qt::UTC);

    const QString explicitZeroPath = writeSgmtMatch(tempDir.path(),
                                                    QStringLiteral("zero_explicit.match"),
                                                    imageA,
                                                    imageB,
                                                    10,
                                                    older);
    writeSidecar(explicitZeroPath,
                 imageA,
                 imageB,
                 QStringLiteral("superpoint"),
                 QStringLiteral("superglue"),
                 10,
                 0);

    const QString missingStatsPath = writeSgmtMatch(tempDir.path(),
                                                    QStringLiteral("zero_missing_stats.match"),
                                                    imageA,
                                                    imageB,
                                                    500,
                                                    newer);
    writeSidecar(missingStatsPath,
                 imageA,
                 imageB,
                 QStringLiteral("disk"),
                 QStringLiteral("lightglue"),
                 500,
                 0,
                 false);

    xjw::pipeline::MatchResultCatalogConfig config;
    config.matchDirectory = tempDir.path();
    const xjw::pipeline::MatchResultCatalogSummary summary =
        xjw::pipeline::MatchResultCatalog(config).scan();

    ASSERT_EQ(summary.pairGroups.size(), 1);
    const xjw::pipeline::MatchPairGroup &group = summary.pairGroups.front();
    ASSERT_EQ(group.variants.size(), 2);
    ASSERT_GE(group.bestVariantIndex, 0);

    bool sawMissingStats = false;
    bool sawExplicitStats = false;
    for (const xjw::pipeline::MatchVariant &variant : group.variants)
    {
        if (variant.matchFilePath == QFileInfo(missingStatsPath).absoluteFilePath())
        {
            sawMissingStats = true;
            EXPECT_FALSE(variant.hasInlierStats);
            EXPECT_EQ(variant.geometricVerifiedInliers, 0);
        }
        if (variant.matchFilePath == QFileInfo(explicitZeroPath).absoluteFilePath())
        {
            sawExplicitStats = true;
            EXPECT_TRUE(variant.hasInlierStats);
            EXPECT_EQ(variant.geometricVerifiedInliers, 0);
        }
    }
    EXPECT_TRUE(sawMissingStats);
    EXPECT_TRUE(sawExplicitStats);

    const xjw::pipeline::MatchVariant &best = group.variants.at(group.bestVariantIndex);
    EXPECT_EQ(best.matchFilePath, QFileInfo(explicitZeroPath).absoluteFilePath());
    EXPECT_TRUE(best.hasInlierStats);
}

TEST(MatchResultCatalogTest, CanonicalPairKeyIsStableForReversedOrdering)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("A.tif"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("B.tif"));

    const QString keyAB = xjw::pipeline::MatchResultCatalog::canonicalPairKey(imageA, imageB);
    const QString keyBA = xjw::pipeline::MatchResultCatalog::canonicalPairKey(imageB, imageA);

    EXPECT_FALSE(keyAB.isEmpty());
    EXPECT_EQ(keyAB, keyBA);
    EXPECT_TRUE(keyAB.contains(QFileInfo(imageA).absoluteFilePath()));
    EXPECT_TRUE(keyAB.contains(QFileInfo(imageB).absoluteFilePath()));
}

TEST(MatchResultCatalogTest, MissingOrMismatchedSidecarsAreRecordedButNotSelected)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("A.tif"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("B.tif"));
    const QString imageC = QDir(tempDir.path()).filePath(QStringLiteral("C.tif"));

    const QString compatiblePath = writeSgmtMatch(tempDir.path(),
                                                  QStringLiteral("A__B_compatible.match"),
                                                  imageA,
                                                  imageB,
                                                  20);
    writeSidecar(compatiblePath, imageA, imageB, QStringLiteral("superpoint"), QStringLiteral("superglue"), 20, 12);

    const QString missingSidecarPath = writeSgmtMatch(tempDir.path(),
                                                      QStringLiteral("A__B_missing_sidecar.match"),
                                                      imageA,
                                                      imageB,
                                                      200);

    const QString mismatchedSidecarPath = writeSgmtMatch(tempDir.path(),
                                                         QStringLiteral("A__B_mismatched_sidecar.match"),
                                                         imageA,
                                                         imageB,
                                                         300);
    writeSidecar(mismatchedSidecarPath,
                 imageA,
                 imageC,
                 QStringLiteral("disk"),
                 QStringLiteral("lightglue"),
                 300,
                 250);

    xjw::pipeline::MatchResultCatalogConfig config;
    config.matchDirectory = tempDir.path();
    const xjw::pipeline::MatchResultCatalogSummary summary =
        xjw::pipeline::MatchResultCatalog(config).scan();

    EXPECT_EQ(summary.matchFileCount, 3);
    EXPECT_EQ(summary.compatibleVariantCount, 1);
    EXPECT_EQ(summary.incompatibleVariantCount, 2);
    ASSERT_EQ(summary.pairGroups.size(), 1);

    const xjw::pipeline::MatchPairGroup &group = summary.pairGroups.front();
    ASSERT_EQ(group.variants.size(), 3);
    ASSERT_GE(group.bestVariantIndex, 0);
    EXPECT_EQ(group.variants.at(group.bestVariantIndex).matchFilePath, QFileInfo(compatiblePath).absoluteFilePath());

    QSet<QString> statuses;
    for (const xjw::pipeline::MatchVariant &variant : group.variants)
    {
        statuses.insert(variant.status);
        if (variant.matchFilePath == QFileInfo(missingSidecarPath).absoluteFilePath())
        {
            EXPECT_FALSE(variant.compatible);
            EXPECT_EQ(variant.status, QStringLiteral("missing_sidecar"));
            EXPECT_EQ(variant.reason, QStringLiteral("sidecar_json_missing"));
            EXPECT_EQ(variant.totalMatches, 200);
        }
        if (variant.matchFilePath == QFileInfo(mismatchedSidecarPath).absoluteFilePath())
        {
            EXPECT_FALSE(variant.compatible);
            EXPECT_EQ(variant.status, QStringLiteral("mismatched_image_names"));
            EXPECT_EQ(variant.reason, QStringLiteral("sidecar_images_do_not_match_sgmt_header"));
            EXPECT_EQ(variant.geometricVerifiedInliers, 250);
        }
    }

    EXPECT_TRUE(statuses.contains(QStringLiteral("compatible")));
    EXPECT_TRUE(statuses.contains(QStringLiteral("missing_sidecar")));
    EXPECT_TRUE(statuses.contains(QStringLiteral("mismatched_image_names")));
}

TEST(MatchResultCatalogTest, InvalidMatchHeaderUsesReadableSidecarPairForGrouping)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString imageA = QDir(tempDir.path()).filePath(QStringLiteral("invalid_A.tif"));
    const QString imageB = QDir(tempDir.path()).filePath(QStringLiteral("invalid_B.tif"));
    const QString validPath = writeSgmtMatch(tempDir.path(),
                                             QStringLiteral("valid_for_pair.match"),
                                             imageA,
                                             imageB,
                                             30);
    writeSidecar(validPath, imageA, imageB, QStringLiteral("superpoint"), QStringLiteral("superglue"), 30, 12);

    const QString invalidPath = writeInvalidMatch(tempDir.path(), QStringLiteral("invalid_header.match"));
    writeSidecar(invalidPath, imageA, imageB, QStringLiteral("disk"), QStringLiteral("lightglue"), 200, 80);

    xjw::pipeline::MatchResultCatalogConfig config;
    config.matchDirectory = tempDir.path();
    const xjw::pipeline::MatchResultCatalogSummary summary =
        xjw::pipeline::MatchResultCatalog(config).scan();

    ASSERT_EQ(summary.pairGroups.size(), 1);
    const xjw::pipeline::MatchPairGroup &group = summary.pairGroups.front();
    EXPECT_EQ(group.pairKey, xjw::pipeline::MatchResultCatalog::canonicalPairKey(imageA, imageB));
    ASSERT_EQ(group.variants.size(), 2);

    const auto invalidIt = std::find_if(group.variants.begin(), group.variants.end(),
                                        [&](const xjw::pipeline::MatchVariant &variant)
    {
        return variant.matchFilePath == QFileInfo(invalidPath).absoluteFilePath();
    });
    ASSERT_NE(invalidIt, group.variants.end());
    EXPECT_FALSE(invalidIt->compatible);
    EXPECT_EQ(invalidIt->status, QStringLiteral("invalid_match_file"));
    EXPECT_EQ(invalidIt->reason, QStringLiteral("sgmt_magic_missing"));
    EXPECT_EQ(invalidIt->imageA, imageA);
    EXPECT_EQ(invalidIt->imageB, imageB);
}

TEST(MatchResultCatalogTest, ReadsSgmtV2MatchCount)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString path = writeSgmtMatch(tempDir.path(),
                                        QStringLiteral("count_test.match"),
                                        QStringLiteral("A.tif"),
                                        QStringLiteral("B.tif"),
                                        37);

    EXPECT_EQ(xjw::pipeline::MatchResultCatalog::readSgmtMatchCount(path), 37);
}

TEST(MatchResultCatalogTest, ReadsSgmtV1MatchCount)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString path = writeSgmtMatch(tempDir.path(),
                                        QStringLiteral("count_test_v1.match"),
                                        QStringLiteral("A.tif"),
                                        QStringLiteral("B.tif"),
                                        41,
                                        QDateTime(),
                                        1);

    EXPECT_EQ(xjw::pipeline::MatchResultCatalog::readSgmtMatchCount(path), 41);
}
