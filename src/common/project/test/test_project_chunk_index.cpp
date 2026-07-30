#include "project/ProjectChunkIndex.h"

#include <gtest/gtest.h>

#include <QJsonArray>

using xjw::common::project::ProjectChunkIndex;

TEST(ProjectChunkIndexTest, CreatesDefaultChunkInDirectoryOne)
{
    const ProjectChunkIndex index =
        ProjectChunkIndex::createInitial(QStringLiteral("月球区块"));

    ASSERT_EQ(index.size(), 1);
    EXPECT_EQ(index.defaultChunk().name, QStringLiteral("月球区块"));
    EXPECT_EQ(index.defaultChunk().directory, 1);
    EXPECT_EQ(index.nextChunkDirectory(), 2);
}

TEST(ProjectChunkIndexTest, NeverReusesDeletedDirectoryNumbers)
{
    ProjectChunkIndex index = ProjectChunkIndex::createInitial();
    QString error;
    const QString second = index.appendChunk(QStringLiteral("区块 2"), &error);
    ASSERT_FALSE(second.isEmpty()) << qPrintable(error);
    const QString third = index.appendChunk(QStringLiteral("区块 3"), &error);
    ASSERT_FALSE(third.isEmpty()) << qPrintable(error);

    ASSERT_TRUE(index.removeChunk(second, &error)) << qPrintable(error);
    const QString fourth = index.appendChunk(QStringLiteral("区块 4"), &error);
    ASSERT_FALSE(fourth.isEmpty()) << qPrintable(error);

    EXPECT_EQ(index.chunk(third).directory, 3);
    EXPECT_EQ(index.chunk(fourth).directory, 4);
    EXPECT_EQ(index.nextChunkDirectory(), 5);
}

TEST(ProjectChunkIndexTest, RoundTripsUuidDirectoryMapping)
{
    ProjectChunkIndex source = ProjectChunkIndex::createInitial();
    QString error;
    ASSERT_FALSE(source.appendChunk(QStringLiteral("第二处理方案"), &error).isEmpty())
        << qPrintable(error);

    const ProjectChunkIndex restored =
        ProjectChunkIndex::fromJson(source.toJson(), &error);

    ASSERT_TRUE(error.isEmpty()) << qPrintable(error);
    ASSERT_EQ(restored.size(), 2);
    EXPECT_EQ(restored.defaultChunkId(), source.defaultChunkId());
    EXPECT_EQ(restored.chunks()[1].directory, 2);
    EXPECT_EQ(restored.nextChunkDirectory(), 3);
}

TEST(ProjectChunkIndexTest, RejectsDuplicateNumericDirectories)
{
    QJsonObject object = ProjectChunkIndex::createInitial().toJson();
    QJsonArray chunks = object.value(QStringLiteral("chunks")).toArray();
    QJsonObject duplicate = chunks.first().toObject();
    duplicate[QStringLiteral("id")] = QStringLiteral("another-chunk");
    chunks.append(duplicate);
    object[QStringLiteral("chunks")] = chunks;
    object[QStringLiteral("next_chunk_directory")] = 2;

    QString error;
    const ProjectChunkIndex restored =
        ProjectChunkIndex::fromJson(object, &error);

    EXPECT_TRUE(restored.isEmpty());
    EXPECT_TRUE(error.contains(QStringLiteral("重复数字目录")))
        << qPrintable(error);
}
