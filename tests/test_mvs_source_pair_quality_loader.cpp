#include "MvsSourcePairQualityLoader.h"

#include "ImageMatchIndexFile.h"
#include "ImageMatchRepository.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

namespace
{

QString createImage(const QString &directory, const QString &name)
{
    const QString path = QDir(directory).filePath(name);
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::WriteOnly));
    EXPECT_EQ(file.write(name.toUtf8()), name.toUtf8().size());
    return path;
}

xjw::image_matching::PairMatchData makePair(
    const QString &imageA,
    const QString &imageB,
    const QByteArray &fingerprint,
    std::uint32_t matchCount,
    std::uint32_t inlierCount,
    std::int64_t createdTime)
{
    using namespace xjw::image_matching;

    PairMatchData pair;
    pair.image0 = ImageMatchFile::identityForImage(imageA, 640, 480);
    pair.image1 = ImageMatchFile::identityForImage(imageB, 640, 480);
    pair.algorithmId = QStringLiteral("sift_lightglue");
    pair.algorithmVersion = 1;
    pair.configFingerprint = fingerprint;
    pair.modelFingerprint = QByteArrayLiteral("engine-sha256");
    pair.createdTimeMs = createdTime;
    pair.rawMatchCount = matchCount;
    pair.geometryInlierCount = inlierCount;
    pair.tiePointMatchCount = inlierCount;
    pair.geometryPassed = inlierCount > 0;
    pair.geometryModel = GeometryModel::Fundamental;

    PairCorrespondence correspondence;
    correspondence.observation0 = {7, 10.0f, 20.0f, 1.5f, 0.2f, 0.8f};
    correspondence.observation1 = {9, 12.0f, 21.0f, 1.7f, 0.3f, 0.9f};
    correspondence.confidence = 0.95f;
    correspondence.residualPixels = 0.4f;
    correspondence.flags = inlierCount > 0
        ? MatchRecordFlag::GeometryInlier | MatchRecordFlag::InTiePointTrack
        : MatchRecordFlag::None;
    pair.correspondences.push_back(correspondence);
    return pair;
}

const xjw::mvs::MvsSourcePairQuality *findPair(
    const std::vector<xjw::mvs::MvsSourcePairQuality> &qualities,
    const QString &imageA,
    const QString &imageB)
{
    const auto normalized = [](const std::string &path)
    {
        return QDir::cleanPath(QString::fromUtf8(path.c_str()));
    };
    const auto it = std::find_if(
        qualities.cbegin(),
        qualities.cend(),
        [&](const xjw::mvs::MvsSourcePairQuality &quality)
        {
            const QString left = normalized(quality.imageA);
            const QString right = normalized(quality.imageB);
            return (left == QDir::cleanPath(imageA) &&
                    right == QDir::cleanPath(imageB)) ||
                (left == QDir::cleanPath(imageB) &&
                 right == QDir::cleanPath(imageA));
        });
    return it == qualities.cend() ? nullptr : &*it;
}

} // namespace

TEST(MvsSourcePairQualityLoaderTest, MapsBestPimatchVariantsAndAppliesVerifiedPolicy)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString imageA = createImage(temporary.path(), QStringLiteral("A.png"));
    const QString imageB = createImage(temporary.path(), QStringLiteral("B.png"));
    const QString imageC = createImage(temporary.path(), QStringLiteral("C.png"));
    const QString imageD = createImage(temporary.path(), QStringLiteral("D.png"));
    const QString imageE = createImage(temporary.path(), QStringLiteral("E.png"));
    const QString matchDirectory =
        QDir(temporary.path()).filePath(QStringLiteral("matches"));

    xjw::image_matching::ImageMatchRepository repository(matchDirectory);
    ASSERT_TRUE(repository.writePairs(
        {makePair(imageA, imageB, QByteArrayLiteral("config-low"), 200, 30, 1000)},
        false).success);
    ASSERT_TRUE(repository.writePairs(
        {makePair(imageA, imageB, QByteArrayLiteral("config-high"), 180, 95, 2000),
         makePair(imageA, imageC, QByteArrayLiteral("config-failed"), 140, 0, 1500),
         makePair(imageD, imageE, QByteArrayLiteral("config-excluded"), 160, 80, 1500)},
        true).success);
    xjw::image_matching::ImageMatchIndexFile::clearMemoryCache();

    auto result = xjw::core::project::loadMvsSourcePairQualities(
        matchDirectory, {imageA, imageB, imageC});

    EXPECT_EQ(result.catalogPairCount, 2);
    EXPECT_EQ(result.qualities.size(), 2);
    EXPECT_EQ(result.verifiedPairCount, 1);
    EXPECT_EQ(result.failedPairCount, 1);
    EXPECT_EQ(result.missingStatisticsPairCount, 0);

    const auto *verified = findPair(result.qualities, imageA, imageB);
    ASSERT_NE(verified, nullptr);
    EXPECT_TRUE(verified->verified);
    EXPECT_TRUE(verified->hasVerificationStatistics);
    EXPECT_EQ(verified->totalMatches, 180);
    EXPECT_EQ(verified->geometricInliers, 95);
    EXPECT_EQ(verified->verificationReason, "verified_from_pimatch");

    const auto *failed = findPair(result.qualities, imageA, imageC);
    ASSERT_NE(failed, nullptr);
    EXPECT_FALSE(failed->verified);
    EXPECT_TRUE(failed->hasVerificationStatistics);
    EXPECT_EQ(failed->verificationReason, "pimatch_geometry_gate_failed");
    EXPECT_EQ(findPair(result.qualities, imageD, imageE), nullptr);

    xjw::mvs::DepthGenConfig config;
    xjw::core::project::applyMvsSourcePairQualities(
        &config, std::move(result));
    EXPECT_TRUE(config.requireVerifiedSourcePairs);
    EXPECT_EQ(config.sourcePairQualities.size(), 2);
}

TEST(MvsSourcePairQualityLoaderTest, EmptyCatalogDisablesVerifiedPolicy)
{
    xjw::mvs::DepthGenConfig config;
    config.requireVerifiedSourcePairs = true;
    config.sourcePairQualities.push_back(xjw::mvs::MvsSourcePairQuality{});

    auto result = xjw::core::project::loadMvsSourcePairQualities(QString());
    xjw::core::project::applyMvsSourcePairQualities(
        &config, std::move(result));

    EXPECT_FALSE(config.requireVerifiedSourcePairs);
    EXPECT_TRUE(config.sourcePairQualities.empty());
}
