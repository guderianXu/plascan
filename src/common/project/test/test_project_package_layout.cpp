#include "project/ProjectPackageLayout.h"

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

#include <gtest/gtest.h>

using xjw::common::project::ProjectPackageLayout;

TEST(ProjectPackageLayoutTest, WritesMetashapeStyleDescriptor)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("中文项目.plascan"));
    ASSERT_TRUE(QDir().mkpath(ProjectPackageLayout::dataDirectory(projectPath)));
    QFile archive(ProjectPackageLayout::metadataArchivePath(projectPath));
    ASSERT_TRUE(archive.open(QIODevice::WriteOnly));
    archive.write("PK");
    archive.close();

    QString error;
    ASSERT_TRUE(ProjectPackageLayout::writeDescriptor(projectPath, &error))
        << error.toStdString();
    EXPECT_TRUE(ProjectPackageLayout::isDescriptor(projectPath, &error))
        << error.toStdString();
    EXPECT_EQ(ProjectPackageLayout::resolveMetadataArchive(projectPath, &error),
              ProjectPackageLayout::metadataArchivePath(projectPath));
}

TEST(ProjectPackageLayoutTest, UsesStableNumericChunkDirectories)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("月球工程.plascan"));

    EXPECT_EQ(
        ProjectPackageLayout::chunkDirectory(projectPath, 1),
        QDir(ProjectPackageLayout::dataDirectory(projectPath))
            .filePath(QStringLiteral("1")));
    EXPECT_EQ(
        ProjectPackageLayout::chunkArchivePath(projectPath, 3),
        QDir(ProjectPackageLayout::dataDirectory(projectPath))
            .filePath(QStringLiteral("3/chunk.zip")));
    EXPECT_EQ(ProjectPackageLayout::workspaceDirectory(projectPath),
              ProjectPackageLayout::chunkDirectory(projectPath, 1));
    EXPECT_TRUE(ProjectPackageLayout::isChunkDirectoryName(
        QStringLiteral("1")));
    EXPECT_TRUE(ProjectPackageLayout::isChunkDirectoryName(
        QStringLiteral("123")));
    EXPECT_FALSE(ProjectPackageLayout::isChunkDirectoryName(
        QStringLiteral("0")));
    EXPECT_FALSE(ProjectPackageLayout::isChunkDirectoryName(
        QStringLiteral("02")));
    EXPECT_FALSE(ProjectPackageLayout::isChunkDirectoryName(
        QStringLiteral("workspace")));
}

TEST(ProjectPackageLayoutTest, RejectsLegacyMonolithicProject)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("legacy.plascan"));
    QFile legacy(projectPath);
    ASSERT_TRUE(legacy.open(QIODevice::WriteOnly));
    const QByteArray contents("PK legacy archive");
    ASSERT_EQ(legacy.write(contents), contents.size());
    legacy.close();

    QString error;
    EXPECT_FALSE(ProjectPackageLayout::ensureSplitLayout(projectPath, &error));
    EXPECT_FALSE(error.isEmpty());
    EXPECT_TRUE(error.contains(QStringLiteral("仅支持版本")));

    QFile original(projectPath);
    ASSERT_TRUE(original.open(QIODevice::ReadOnly));
    EXPECT_EQ(original.readAll(), contents);
    EXPECT_FALSE(QFileInfo::exists(
        ProjectPackageLayout::dataDirectory(projectPath)));
}

TEST(ProjectPackageLayoutTest, RejectsDescriptorTraversal)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("unsafe.plascan"));
    QFile descriptor(projectPath);
    ASSERT_TRUE(descriptor.open(QIODevice::WriteOnly));
    descriptor.write(
        "<?xml version=\"1.0\"?><document version=\"3.0.0\" "
        "type=\"plascan_project\" path=\"../other.zip\"/>");
    descriptor.close();

    QString error;
    EXPECT_FALSE(ProjectPackageLayout::isDescriptor(projectPath, &error));
    EXPECT_FALSE(error.isEmpty());
    EXPECT_TRUE(ProjectPackageLayout::resolveMetadataArchive(
                    projectPath, &error).isEmpty());
}

TEST(ProjectPackageLayoutTest, DescriptorRequiresItsMatchingDataArchive)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("missing.plascan"));

    QString error;
    ASSERT_TRUE(ProjectPackageLayout::writeDescriptor(projectPath, &error))
        << error.toStdString();
    EXPECT_TRUE(ProjectPackageLayout::isDescriptor(projectPath, &error));
    EXPECT_TRUE(ProjectPackageLayout::resolveMetadataArchive(
                    projectPath, &error).isEmpty());
    EXPECT_FALSE(error.isEmpty());
    EXPECT_FALSE(ProjectPackageLayout::ensureSplitLayout(
        projectPath, &error));
}

TEST(ProjectPackageLayoutTest, RequiresExactDescriptorTypeAndVersion)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    const auto writeDescriptor = [&temporary](const QString &name,
                                               const QString &attributes)
    {
        const QString path =
            QDir(temporary.path()).filePath(name + QStringLiteral(".plascan"));
        QFile descriptor(path);
        EXPECT_TRUE(descriptor.open(QIODevice::WriteOnly));
        descriptor.write(
            QStringLiteral("<?xml version=\"1.0\"?><document %1/>")
                .arg(attributes).toUtf8());
        descriptor.close();
        return path;
    };

    QString error;
    const QString missingType = writeDescriptor(
        QStringLiteral("missing-type"),
        QStringLiteral("version=\"4.0.0\" path=\"{projectname}.files/project.zip\""));
    EXPECT_FALSE(ProjectPackageLayout::isDescriptor(missingType, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("类型")));

    const QString futureVersion = writeDescriptor(
        QStringLiteral("future"),
        QStringLiteral("version=\"5.0.0\" type=\"plascan_project\" "
                       "path=\"{projectname}.files/project.zip\""));
    EXPECT_FALSE(ProjectPackageLayout::isDescriptor(futureVersion, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("版本")));

    const QString truncated =
        QDir(temporary.path()).filePath(QStringLiteral("truncated.plascan"));
    QFile descriptor(truncated);
    ASSERT_TRUE(descriptor.open(QIODevice::WriteOnly));
    descriptor.write(
        "<?xml version=\"1.0\"?><document version=\"4.0.0\" "
        "type=\"plascan_project\" path=\"{projectname}.files/project.zip\">");
    descriptor.close();
    EXPECT_FALSE(ProjectPackageLayout::isDescriptor(truncated, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("XML")));
}
