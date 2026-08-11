#include "project/ProjectMetadata.h"
#include "project/ProjectConfigManager.h"

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

TEST(ProjectMetadataTest, ResolvesOnlyImagesStillReferencedByCurrentProjectList)
{
    const QString retained_image = QStringLiteral("E:/dataset/retained.png");
    const QString removed_image = QStringLiteral("E:/dataset/removed.png");
    const QStringList current_images{retained_image};

    EXPECT_EQ(resolveProjectImagePathFromToken(QStringLiteral("retained"), current_images),
              retained_image);
    EXPECT_TRUE(resolveProjectImagePathFromToken(removed_image, current_images).isEmpty());
    EXPECT_TRUE(resolveProjectImagePathFromToken(QStringLiteral("removed.png"), current_images).isEmpty());
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
    EXPECT_FALSE(imageTokensReferToSameImage(QStringLiteral("TempleSR0001.PNG"), image_path));
    EXPECT_FALSE(imageTokensReferToSameImage(QStringLiteral("templesr0001"), image_path));
    EXPECT_FALSE(imageTokensReferToSameImage(QStringLiteral("templesr0002"), image_path));
    EXPECT_FALSE(imageTokensReferToSameImage(QString(), image_path));
}

TEST(ProjectMetadataTest, MatchesImageReferencePathOrNameAgainstDisplayToken)
{
    EXPECT_TRUE(imageReferenceMatchesToken(QStringLiteral("E:/data/a.png"),
                                           QStringLiteral("a.png"),
                                           QStringLiteral("e:/data/a.png")));
    EXPECT_TRUE(imageReferenceMatchesToken(QString(),
                                           QStringLiteral("a.png"),
                                           QStringLiteral("A.PNG")));
    EXPECT_FALSE(imageReferenceMatchesToken(QStringLiteral("E:/data/a.png"),
                                            QStringLiteral("a.png"),
                                            QStringLiteral("b.png")));
}

TEST(ProjectConfigManagerTest, DefaultsToFramePinholeCameraModel)
{
    ProjectConfigManager config;
    config.setData(ProjectConfigManager::defaultConfig());

    ASSERT_TRUE(config.cameraModelPolicy().has_value());
    EXPECT_EQ(config.cameraModelPolicy().value(),
              ProjectCameraModelPolicy::FramePinhole);
    EXPECT_EQ(config.data().value(QStringLiteral("camera_model_policy")).toString(),
              QStringLiteral("frame_pinhole"));
}

TEST(ProjectConfigManagerTest, SuppliesFramePinholeDefaultToLegacyConfig)
{
    ProjectConfigManager config;
    config.setData(ProjectConfigManager::mergeWithDefaults(QJsonObject{}));

    ASSERT_TRUE(config.cameraModelPolicy().has_value());
    EXPECT_EQ(config.cameraModelPolicy().value(),
              ProjectCameraModelPolicy::FramePinhole);
}

TEST(ProjectConfigManagerTest, PreservesExplicitLineScanCameraModel)
{
    const QJsonObject input{
        {QStringLiteral("camera_model_policy"),
         QStringLiteral("isis_usgscsm_linescan")}};
    ProjectConfigManager config;
    config.setData(ProjectConfigManager::mergeWithDefaults(input));

    ASSERT_TRUE(config.cameraModelPolicy().has_value());
    EXPECT_EQ(config.cameraModelPolicy().value(),
              ProjectCameraModelPolicy::IsisUsgsCsmLineScan);
}

TEST(ProjectConfigManagerTest, RejectsUnknownNonEmptyCameraModelToken)
{
    ProjectConfigManager config;
    config.setData(QJsonObject{
        {QStringLiteral("camera_model_policy"), QStringLiteral("unknown")}});

    EXPECT_FALSE(config.cameraModelPolicy().has_value());
}

TEST(ProjectMetadataTest, RejectsAmbiguousFileNameAndStemTokens)
{
    const QString left = QStringLiteral("E:/dataset/left/frame001.tif");
    const QString right = QStringLiteral("E:/dataset/right/frame001.tif");
    const QStringList images{left, right};

    const ImageResolveResult file_name =
        resolveProjectImageToken(QStringLiteral("frame001.tif"), images);
    EXPECT_EQ(file_name.status, ImageResolveStatus::Ambiguous);
    EXPECT_TRUE(file_name.path.isEmpty());
    EXPECT_EQ(file_name.candidates.size(), 2);

    const ImageResolveResult stem =
        resolveProjectImageToken(QStringLiteral("frame001"), images);
    EXPECT_EQ(stem.status, ImageResolveStatus::Ambiguous);
    EXPECT_TRUE(resolveProjectImagePathFromToken(
                    QStringLiteral("frame001"), images).isEmpty());
    EXPECT_EQ(resolveProjectImagePathFromToken(left, images), left);
}

TEST(ProjectMetadataTest, ResolvesStableImageUuidBeforeDisplayTokens)
{
    const QString image_path = QStringLiteral("E:/dataset/left/frame001.tif");
    const QJsonObject metadata{
        {QStringLiteral("images"),
         QJsonArray{QJsonObject{
             {QStringLiteral("image_uuid"), QStringLiteral("stable-image-id")},
             {QStringLiteral("path"), image_path}}}}};

    const ImageResolveResult resolved =
        resolveProjectImageToken(QStringLiteral("stable-image-id"), metadata);
    EXPECT_EQ(resolved.status, ImageResolveStatus::Found);
    EXPECT_EQ(resolved.path, image_path);
}

} // namespace
