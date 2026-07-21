#include "project/ProjectMatchCatalog.h"

#include "project/ProjectIO.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <gtest/gtest.h>

namespace
{

using namespace xjw::common::project;

void writeJson(const QString &path, const QJsonDocument &document)
{
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_EQ(file.write(document.toJson(QJsonDocument::Compact)),
              document.toJson(QJsonDocument::Compact).size());
}

QJsonObject projectMetadata()
{
    return QJsonObject{
        {QStringLiteral("images"),
         QJsonArray{
             QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/a.png")}},
             QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/b.png")}},
             QJsonObject{{QStringLiteral("path"), QStringLiteral("E:/data/c.png")}}}}};
}

TEST(ProjectMatchCatalogTest, ReadsAndDeduplicatesMatchedPairs)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    const QString project_path = QDir(temp_dir.path()).filePath(QStringLiteral("test.plascan"));
    const QString match_dir = ProjectIO::ipmatchOutputDir(project_path);
    ASSERT_TRUE(QDir().mkpath(match_dir));

    QFile match_file(QDir(match_dir).filePath(QStringLiteral("a__b_lightglue.match")));
    ASSERT_TRUE(match_file.open(QIODevice::WriteOnly));
    match_file.close();
    writeJson(match_file.fileName() + QStringLiteral(".json"),
              QJsonDocument(QJsonObject{{QStringLiteral("image0_name"), QStringLiteral("b.png")},
                                        {QStringLiteral("image1_name"), QStringLiteral("a.png")}}));

    QJsonObject metadata = projectMetadata();
    metadata[QStringLiteral("ipmatch_results")] =
        QJsonArray{QJsonObject{{QStringLiteral("image0_name"), QStringLiteral("a.png")},
                               {QStringLiteral("image1_name"), QStringLiteral("b.png")}}};

    const auto pairs = collectMatchedImageNamePairs(project_path, metadata);
    ASSERT_EQ(pairs.size(), 1);
    EXPECT_EQ(pairs.front(), qMakePair(QStringLiteral("a.png"), QStringLiteral("b.png")));
}

TEST(ProjectMatchCatalogTest, ReadsSettledNoMatchPairsInStableOrder)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());
    const QString project_path = QDir(temp_dir.path()).filePath(QStringLiteral("test.plascan"));
    const QString match_dir = ProjectIO::ipmatchOutputDir(project_path);
    ASSERT_TRUE(QDir().mkpath(match_dir));

    writeJson(QDir(match_dir).filePath(QStringLiteral("no_match_pairs.json")),
              QJsonDocument(QJsonArray{
                  QJsonObject{{QStringLiteral("image0"), QStringLiteral("c")},
                              {QStringLiteral("image1"), QStringLiteral("b")}},
                  QJsonObject{{QStringLiteral("image0"), QStringLiteral("a")},
                              {QStringLiteral("image1"), QStringLiteral("c")}}}));

    const auto pairs = collectSettledNoMatchImageNamePairs(project_path, projectMetadata());
    ASSERT_EQ(pairs.size(), 2);
    EXPECT_EQ(pairs.at(0), qMakePair(QStringLiteral("a.png"), QStringLiteral("c.png")));
    EXPECT_EQ(pairs.at(1), qMakePair(QStringLiteral("b.png"), QStringLiteral("c.png")));
}

TEST(ProjectMatchCatalogTest, EncodesCanonicalAndOrderedPairKeys)
{
    EXPECT_EQ(canonicalImagePairKey(QStringLiteral("b"), QStringLiteral("a"),
                                    QStringLiteral("\n")),
              QStringLiteral("a\nb"));
    EXPECT_EQ(encodeImagePairKey(QStringLiteral("b"), QStringLiteral("a"),
                                 QStringLiteral("|")),
              QStringLiteral("b|a"));
    EXPECT_TRUE(canonicalImagePairKey(QStringLiteral("a"), QStringLiteral("a"),
                                     QStringLiteral("__")).isEmpty());
}

TEST(ProjectMatchCatalogTest, DecodesExistingPairKeyFormats)
{
    QString first;
    QString second;
    EXPECT_TRUE(decodeImagePairKey(QStringLiteral("a\nb"), QStringLiteral("\n"),
                                   &first, &second));
    EXPECT_EQ(first, QStringLiteral("a"));
    EXPECT_EQ(second, QStringLiteral("b"));

    EXPECT_TRUE(decodeImagePairKey(QStringLiteral("a__b"), QStringLiteral("__"),
                                   &first, &second));
    EXPECT_TRUE(decodeImagePairKey(QStringLiteral("a|b"), QStringLiteral("|"),
                                   &first, &second));
    EXPECT_FALSE(decodeImagePairKey(QStringLiteral("a|b|c"), QStringLiteral("|"),
                                    &first, &second));
    EXPECT_FALSE(decodeImagePairKey(QStringLiteral("a|"), QStringLiteral("|"),
                                    &first, &second));
}

} // namespace
