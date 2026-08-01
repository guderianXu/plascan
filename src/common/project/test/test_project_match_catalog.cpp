#include "project/ProjectMatchCatalog.h"

#include "project/ProjectIO.h"

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>

#include <gtest/gtest.h>

namespace
{

using namespace xjw::common::project;

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

    QJsonObject metadata = projectMetadata();
    metadata[QStringLiteral("image_match_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("image"), QStringLiteral("E:/data/a.png")},
                    {QStringLiteral("neighbors"),
                     QJsonArray{QStringLiteral("E:/data/b.png"),
                                QStringLiteral("E:/data/b.png")}}},
        QJsonObject{{QStringLiteral("image"), QStringLiteral("E:/data/b.png")},
                    {QStringLiteral("neighbors"),
                     QJsonArray{QStringLiteral("E:/data/a.png")}}}};

    const auto pairs = collectMatchedImageNamePairs(project_path, metadata);
    ASSERT_EQ(pairs.size(), 1);
    EXPECT_EQ(pairs.front(), qMakePair(QStringLiteral("a.png"), QStringLiteral("b.png")));
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
