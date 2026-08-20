#include <gtest/gtest.h>

#include "project/PortableProjectFormat.h"

#include <QDir>
#include <QJsonArray>
#include <QTemporaryDir>

using xjw::common::project::PortableProjectFormat;
using xjw::common::project::ProjectChunkIndex;
using xjw::common::project::ProjectResourceIndex;
using xjw::common::project::ProjectResourceRef;

TEST(PortableProjectFormatTest, CreatesLayeredProjectAndChunkDocuments)
{
    const QString projectId = PortableProjectFormat::createProjectId();
    ASSERT_FALSE(projectId.isEmpty());

    const ProjectChunkIndex sourceIndex =
        ProjectChunkIndex::createInitial();
    const QJsonObject uiState{
        {QStringLiteral("schema_version"), 1}
    };
    const QJsonObject projectDocument =
        PortableProjectFormat::createProjectDocument(
            projectId, sourceIndex, uiState);
    EXPECT_TRUE(
        PortableProjectFormat::isCurrentProjectDocument(projectDocument));
    EXPECT_EQ(projectDocument.value(QStringLiteral("project_id")).toString(),
              projectId);
    EXPECT_EQ(
        projectDocument.value(QStringLiteral("format_version")).toString(),
        QStringLiteral("4.0"));
    EXPECT_TRUE(projectDocument.value(
        QString::fromLatin1(
            PortableProjectFormat::ChunkIndexSection)).isObject());

    ProjectChunkIndex restoredIndex;
    QString error;
    ASSERT_TRUE(PortableProjectFormat::readProjectIndex(
        projectDocument, &restoredIndex, &error)) << qPrintable(error);
    EXPECT_EQ(restoredIndex.defaultChunkId(), sourceIndex.defaultChunkId());

    const auto defaultChunk = sourceIndex.defaultChunk();
    const QJsonObject chunkDocument =
        PortableProjectFormat::createChunkDocument(
            defaultChunk,
            QJsonObject{{QStringLiteral("images"), QJsonArray{}}},
            QJsonObject{},
            QJsonObject{},
            ProjectResourceIndex().toJson());
    EXPECT_TRUE(PortableProjectFormat::isCurrentChunkDocument(
        chunkDocument, &defaultChunk, &error))
        << qPrintable(error);
}

TEST(PortableProjectFormatTest, RejectsUnsafeArchivePaths)
{
    EXPECT_TRUE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources/images/id/影像.tif")));
    EXPECT_TRUE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("artifacts/models/id/model.ply")));

    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(QString()));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("../outside.tif")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources/../outside.tif")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("/absolute/path.tif")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("C:/images/source.tif")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources\\images\\source.tif")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("\\\\server\\share\\source.tif")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources//images/source.tif")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources/id/file:stream.tif")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources/CON/file.tif")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources/id/NUL.txt")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources/id/file.")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources/id/file ")));
    EXPECT_FALSE(PortableProjectFormat::isSafeEntryPath(
        QStringLiteral("resources/id/file\x01.tif")));
}

TEST(PortableProjectFormatTest, ResourceUriRoundTripsUnicodeEntry)
{
    const QString entry =
        QStringLiteral("resources/images/image-1/月球影像 01.tif");
    const QString uri = PortableProjectFormat::resourceUriForEntry(entry);
    ASSERT_FALSE(uri.isEmpty());
    EXPECT_EQ(PortableProjectFormat::entryPathFromResourceUri(uri), entry);
    EXPECT_TRUE(PortableProjectFormat::entryPathFromResourceUri(
                    QStringLiteral("file:///tmp/source.tif"))
                    .isEmpty());
}

TEST(PortableProjectFormatTest, ResourceIndexRoundTripsValidatedRecords)
{
    ProjectResourceRef resource;
    resource.id = QStringLiteral("image-1");
    resource.kind = QStringLiteral("images");
    resource.name = QStringLiteral("月球影像.tif");
    resource.entryPath =
        QStringLiteral("resources/images/image-1/月球影像.tif");
    resource.mediaType = QStringLiteral("image/tiff");
    resource.sha256 = QStringLiteral("0123456789abcdef");
    resource.size = 1024;

    ProjectResourceIndex index;
    QString error;
    ASSERT_TRUE(index.upsert(resource, &error)) << qPrintable(error);

    const ProjectResourceIndex restored =
        ProjectResourceIndex::fromJson(index.toJson(), &error);
    ASSERT_TRUE(error.isEmpty()) << qPrintable(error);
    ASSERT_EQ(restored.size(), 1);

    const ProjectResourceRef restoredResource =
        restored.resource(QStringLiteral("image-1"));
    EXPECT_EQ(restoredResource.entryPath, resource.entryPath);
    EXPECT_EQ(restoredResource.name, resource.name);
    EXPECT_EQ(restoredResource.size, resource.size);
}

TEST(PortableProjectFormatTest, RejectsInvalidResourceIndex)
{
    const QJsonObject invalid{
        {QStringLiteral("schema_version"), 99},
        {QStringLiteral("resources"), QJsonArray{}}
    };

    QString error;
    const ProjectResourceIndex index =
        ProjectResourceIndex::fromJson(invalid, &error);
    EXPECT_TRUE(index.isEmpty());
    EXPECT_FALSE(error.isEmpty());
}

TEST(PortableProjectFormatTest, RejectsCaseInsensitiveArchivePathCollisions)
{
    ProjectResourceIndex index;
    ProjectResourceRef first;
    first.id = QStringLiteral("image-1");
    first.kind = QStringLiteral("images");
    first.name = QStringLiteral("Frame.tif");
    first.entryPath = QStringLiteral("resources/images/image-1/Frame.tif");

    ProjectResourceRef second = first;
    second.id = QStringLiteral("image-2");
    second.entryPath = QStringLiteral("resources/images/image-1/frame.tif");

    QString error;
    ASSERT_TRUE(index.upsert(first, &error)) << qPrintable(error);
    EXPECT_FALSE(index.upsert(second, &error));
    EXPECT_FALSE(error.isEmpty());
}

TEST(PortableProjectFormatTest, ResolvesEntryOnlyInsideExistingRoot)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    QString error;
    const QString resolved = PortableProjectFormat::resolveEntryPath(
        temporary.path(), QStringLiteral("resources/images/frame.tif"), &error);
    EXPECT_EQ(resolved,
              QDir(QDir(temporary.path()).canonicalPath()).filePath(
                  QStringLiteral("resources/images/frame.tif")));
    EXPECT_TRUE(error.isEmpty());

    EXPECT_TRUE(PortableProjectFormat::resolveEntryPath(
                    temporary.path(), QStringLiteral("../outside.tif"), &error)
                    .isEmpty());
    EXPECT_FALSE(error.isEmpty());
}
