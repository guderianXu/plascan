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

} // namespace
