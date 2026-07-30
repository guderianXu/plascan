#include "project/ProjectIO.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <gtest/gtest.h>

namespace
{

using xjw::common::project::ProjectIO;

TEST(ProjectIOTest, ResolvesCanonicalProjectDirectories)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    const QString project_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("摄影测量工程.plascan"));

    EXPECT_EQ(ProjectIO::projectRootFromPlascan(project_path),
              QDir::cleanPath(temp_dir.path()));
    EXPECT_EQ(ProjectIO::projectAssetsDir(project_path),
              QDir(temp_dir.path()).filePath(QStringLiteral("assets")));
    EXPECT_EQ(ProjectIO::projectBundleAdjustDir(project_path),
              QDir(temp_dir.path()).filePath(
                  QStringLiteral("bundle_adjust")));
    EXPECT_EQ(ProjectIO::ipfindOutputDir(project_path),
              QDir(temp_dir.path()).filePath(QStringLiteral("assets/ip")));
    EXPECT_EQ(ProjectIO::ipmatchOutputDir(project_path),
              QDir(temp_dir.path()).filePath(QStringLiteral("assets/matches")));
    EXPECT_EQ(ProjectIO::maskOutputDir(project_path),
              QDir(temp_dir.path()).filePath(QStringLiteral("assets/masks")));
}

TEST(ProjectIOTest, ResolvesRelativeResourcesAgainstProjectRoot)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    const QString project_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("project.plascan"));
    const QString relative_path = QStringLiteral("assets/images/image 01.tif");

    EXPECT_EQ(ProjectIO::resolveProjectResourcePath(project_path, relative_path),
              QDir(temp_dir.path()).absoluteFilePath(relative_path));
}

TEST(ProjectIOTest, PreservesAbsoluteResourcePaths)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    const QString project_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("project.plascan"));
    const QString absolute_path =
        QDir(temp_dir.path()).absoluteFilePath(QStringLiteral("external/image.tif"));

    EXPECT_EQ(ProjectIO::resolveProjectResourcePath(project_path, absolute_path),
              QDir::cleanPath(absolute_path));
}

TEST(ProjectIOTest, UsesDistinctArtifactPathsForDuplicateBaseNames)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    const QString project_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("project.plascan"));
    const QString left_image =
        QDir(temp_dir.path()).filePath(QStringLiteral("left/frame001.tif"));
    const QString right_image =
        QDir(temp_dir.path()).filePath(QStringLiteral("right/frame001.tif"));

    const QString left_feature =
        ProjectIO::featureOutputPathForImage(project_path, left_image, QStringLiteral(".sp"));
    const QString right_feature =
        ProjectIO::featureOutputPathForImage(project_path, right_image, QStringLiteral(".sp"));
    const QString left_mask =
        ProjectIO::maskOutputPathForImage(project_path, left_image);
    const QString right_mask =
        ProjectIO::maskOutputPathForImage(project_path, right_image);

    EXPECT_NE(left_feature, right_feature);
    EXPECT_NE(left_mask, right_mask);
    EXPECT_TRUE(QFileInfo(left_feature).fileName().startsWith(QStringLiteral("frame001-")));
    EXPECT_TRUE(QFileInfo(right_feature).fileName().startsWith(QStringLiteral("frame001-")));
    EXPECT_EQ(left_feature,
              ProjectIO::featureOutputPathForImage(
                  project_path, left_image, QStringLiteral(".sp")));
}

TEST(ProjectIOTest, MaskLookupDoesNotExposeAmbiguousBaseNameAliases)
{
    QTemporaryDir temp_dir;
    ASSERT_TRUE(temp_dir.isValid());

    const QString project_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("project.plascan"));
    const QString image_path =
        QDir(temp_dir.path()).filePath(QStringLiteral("left/frame001.tif"));
    const QString mask_path =
        ProjectIO::maskOutputPathForImage(project_path, image_path);
    ASSERT_TRUE(QDir().mkpath(QFileInfo(mask_path).absolutePath()));
    QFile mask(mask_path);
    ASSERT_TRUE(mask.open(QIODevice::WriteOnly));
    mask.close();

    const QMap<QString, QString> masks =
        ProjectIO::maskPathsForImages(project_path, {image_path});
    EXPECT_EQ(masks.value(image_path), mask_path);
    EXPECT_FALSE(masks.contains(QStringLiteral("frame001.tif")));
    EXPECT_FALSE(masks.contains(QStringLiteral("frame001")));
}

} // namespace
