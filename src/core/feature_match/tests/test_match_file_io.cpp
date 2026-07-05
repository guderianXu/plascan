#include "MatchFileIO.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QTemporaryDir>

namespace
{

xjw::feature_match::MatchResult makeIndexedResult()
{
    xjw::feature_match::MatchResult result;
    result.matches0 = {1, -1, 0};
    result.matches1 = {2, 0};
    result.matchingScores0 = {0.90f, 0.0f, 0.75f};
    result.matchingScores1 = {0.75f, 0.90f};
    result.numMatches = 2;
    result.sourceAlgorithm = "lightglue";
    return result;
}

} // namespace

TEST(MatchFileIOTest, IndexedMatchRoundTrips)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());

    const QString path = QDir(tempDir.path()).filePath(QStringLiteral("pair.match"));
    const xjw::feature_match::MatchResult input = makeIndexedResult();

    ASSERT_TRUE(xjw::feature_match::writeIndexedMatchFile(
        path,
        QStringLiteral("image_a.png"),
        QStringLiteral("image_b.png"),
        input));

    QString image0Name;
    QString image1Name;
    xjw::feature_match::MatchResult loaded;
    ASSERT_TRUE(xjw::feature_match::readIndexedMatchFile(path, image0Name, image1Name, loaded));

    EXPECT_EQ(image0Name, QStringLiteral("image_a.png"));
    EXPECT_EQ(image1Name, QStringLiteral("image_b.png"));
    EXPECT_EQ(loaded.matches0, input.matches0);
    EXPECT_EQ(loaded.matches1, input.matches1);
    ASSERT_EQ(loaded.matchingScores0.size(), input.matchingScores0.size());
    ASSERT_EQ(loaded.matchingScores1.size(), input.matchingScores1.size());
    EXPECT_FLOAT_EQ(loaded.matchingScores0[0], input.matchingScores0[0]);
    EXPECT_FLOAT_EQ(loaded.matchingScores0[2], input.matchingScores0[2]);
    EXPECT_FLOAT_EQ(loaded.matchingScores1[0], input.matchingScores1[0]);
    EXPECT_FLOAT_EQ(loaded.matchingScores1[1], input.matchingScores1[1]);

    ASSERT_EQ(loaded.cvMatches.size(), 2);
    EXPECT_EQ(loaded.numMatches, 2);
    EXPECT_EQ(loaded.cvMatches[0].queryIdx, 0);
    EXPECT_EQ(loaded.cvMatches[0].trainIdx, 1);
    EXPECT_FLOAT_EQ(loaded.cvMatches[0].distance, 0.10f);
    EXPECT_EQ(loaded.cvMatches[1].queryIdx, 2);
    EXPECT_EQ(loaded.cvMatches[1].trainIdx, 0);
    EXPECT_FLOAT_EQ(loaded.cvMatches[1].distance, 0.25f);
}
