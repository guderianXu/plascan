#include "project/ProjectMetadata.h"

#include <QJsonArray>
#include <QJsonObject>

#include <gtest/gtest.h>

namespace
{

using namespace xjw::common::project;

QJsonObject makeImage(const QString &path)
{
    return QJsonObject{{QStringLiteral("path"), path}};
}

TEST(ProjectMetadataTest, ReadsTopLevelAndNestedProjectFiles)
{
    const QJsonArray images{makeImage(QStringLiteral("images/a.tif")),
                            makeImage(QStringLiteral("images/b.tif"))};
    const QJsonObject top_level{{QStringLiteral("images"), images}};
    const QJsonObject nested{{QStringLiteral("project_files"), top_level}};

    EXPECT_EQ(projectImageEntries(top_level), images);
    EXPECT_EQ(projectImageEntries(nested), images);
}

TEST(ProjectMetadataTest, IgnoresEmptyImagePaths)
{
    const QJsonObject metadata{
        {QStringLiteral("images"),
         QJsonArray{makeImage(QStringLiteral("images/a.tif")),
                    makeImage(QString()),
                    makeImage(QStringLiteral("images/b.tif"))}}};

    EXPECT_EQ(projectImagePaths(metadata),
              (QStringList{QStringLiteral("images/a.tif"),
                           QStringLiteral("images/b.tif")}));
}

TEST(ProjectMetadataTest, ResolvesImageByPathFileNameOrStem)
{
    const QString image_path = QStringLiteral("E:/dataset/templeSR0001.png");
    const QJsonObject metadata{
        {QStringLiteral("images"), QJsonArray{makeImage(image_path)}}};

    EXPECT_EQ(resolveProjectImagePathFromToken(image_path, metadata), image_path);
    EXPECT_EQ(resolveProjectImagePathFromToken(QStringLiteral("templeSR0001.png"), metadata),
              image_path);
    EXPECT_EQ(resolveProjectImagePathFromToken(QStringLiteral("templeSR0001"), metadata),
              image_path);
    EXPECT_TRUE(resolveProjectImagePathFromToken(QStringLiteral("missing"), metadata).isEmpty());
}

TEST(ProjectMetadataTest, NormalizesImageTokensAcrossSeparatorsAndCase)
{
    EXPECT_EQ(normalizedImageToken(QStringLiteral("  Folder\\Sub\\IMAGE.PNG  ")),
              QStringLiteral("folder/sub/image.png"));
    EXPECT_EQ(imageBaseToken(QStringLiteral("Folder/Sub/IMAGE.PNG")),
              QStringLiteral("image"));
}

TEST(ProjectMetadataTest, MatchesImageTokensByPathFileNameOrStem)
{
    const QString image_path = QStringLiteral("E:/Dataset/TempleSR0001.PNG");

    EXPECT_TRUE(imageTokensReferToSameImage(QStringLiteral("e:\\dataset\\templesr0001.png"),
                                            image_path));
    EXPECT_TRUE(imageTokensReferToSameImage(QStringLiteral("TempleSR0001.PNG"), image_path));
    EXPECT_TRUE(imageTokensReferToSameImage(QStringLiteral("templesr0001"), image_path));
    EXPECT_FALSE(imageTokensReferToSameImage(QStringLiteral("templesr0002"), image_path));
    EXPECT_FALSE(imageTokensReferToSameImage(QString(), image_path));
}

TEST(ProjectMetadataTest, MatchesImageReferencePathOrNameAgainstDisplayToken)
{
    EXPECT_TRUE(imageReferenceMatchesToken(QStringLiteral("E:/data/a.png"),
                                           QStringLiteral("a.png"),
                                           QStringLiteral("A")));
    EXPECT_TRUE(imageReferenceMatchesToken(QString(),
                                           QStringLiteral("a.png"),
                                           QStringLiteral("A.PNG")));
    EXPECT_FALSE(imageReferenceMatchesToken(QStringLiteral("E:/data/a.png"),
                                            QStringLiteral("a.png"),
                                            QStringLiteral("b.png")));
}

} // namespace
