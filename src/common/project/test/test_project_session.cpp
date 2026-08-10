#include "project/PlascanArchive.h"
#include "project/PortableProjectFormat.h"
#include "project/ProjectChunkStore.h"
#include "project/ProjectIO.h"
#include "project/ProjectPackageLayout.h"
#include "project/ProjectSession.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>

namespace
{

using xjw::common::project::PortableProjectFormat;
using xjw::common::project::ProjectIO;
using xjw::common::project::ProjectPackageLayout;
using xjw::common::project::ProjectSession;

QJsonObject readChunkDocument(const QString &projectPath,
                              int directory)
{
    const QString archivePath =
        ProjectPackageLayout::chunkArchivePath(projectPath, directory);
    PlascanArchive archive(
        archivePath, PlascanArchivePathType::DirectArchive);
    EXPECT_TRUE(archive.isValid());
    QString error;
    const QByteArray bytes = archive.readEntry(
        QString::fromLatin1(PortableProjectFormat::DocumentEntry),
        &error);
    EXPECT_TRUE(error.isEmpty()) << qPrintable(error);
    return QJsonDocument::fromJson(bytes).object();
}

} // namespace

TEST(ProjectSessionTest, CreatesCurrentChunkProjectAndRoundTripsUris)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("cli.plascan"));
    const QString imagePath =
        QDir(temporary.path()).filePath(QStringLiteral("a.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    ASSERT_EQ(imageFile.write("image"), 5);
    imageFile.close();

    ProjectSession session;
    QString error;
    ASSERT_TRUE(session.create(
        projectPath, QStringLiteral("CLI 工程"), &error))
        << qPrintable(error);
    EXPECT_EQ(session.activeChunk().directory, 1);
    EXPECT_EQ(ProjectIO::projectBundleAdjustDir(projectPath),
              QDir(session.activeChunkRoot()).filePath(
                  QStringLiteral("bundle_adjust")));
    EXPECT_TRUE(QFileInfo(projectPath).isFile());
    EXPECT_TRUE(QFileInfo(
        ProjectPackageLayout::metadataArchivePath(projectPath)).isFile());
    EXPECT_TRUE(QFileInfo(
        ProjectPackageLayout::chunkArchivePath(projectPath, 1)).isFile());
    const QString chunkRoot =
        ProjectPackageLayout::chunkDirectory(projectPath, 1);
    for (const QString &optionalDirectory :
         {QStringLiteral("assets"),
          QStringLiteral("bundle_adjust"),
          QStringLiteral("reconstruction"),
          QStringLiteral("reports")})
    {
        EXPECT_FALSE(QFileInfo::exists(
            QDir(chunkRoot).filePath(optionalDirectory)));
    }
    EXPECT_FALSE(QFileInfo::exists(
        ProjectPackageLayout::sharedDirectory(projectPath)));

    const QString artifact = QDir(session.activeChunkRoot())
        .filePath(QStringLiteral("assets/image_matches/a.pimatch"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(artifact).absolutePath()));
    QFile artifactFile(artifact);
    ASSERT_TRUE(artifactFile.open(QIODevice::WriteOnly));
    ASSERT_EQ(artifactFile.write("match"), 5);
    artifactFile.close();

    ASSERT_TRUE(session.mergeImages(
        QJsonArray{
            QJsonObject{
                {QStringLiteral("path"),
                 QDir(temporary.path()).filePath(
                     QStringLiteral("a.png"))}
            }
        },
        &error)) << qPrintable(error);
    session.appendResult(
        QStringLiteral("image_match_results"),
        QJsonObject{
            {QStringLiteral("image"),
             QDir(temporary.path()).filePath(QStringLiteral("a.png"))},
            {QStringLiteral("output"), artifact}
        });
    session.appendResult(
        QStringLiteral("report_results"),
        QJsonObject{
            {QStringLiteral("path"), artifact},
            {QStringLiteral("planned_dir"),
             QDir(session.activeChunkRoot()).filePath(
                 QStringLiteral("reconstruction/future/run"))},
            {QStringLiteral("external_output_dir"), temporary.path()}
        });
    ASSERT_TRUE(session.save(&error)) << qPrintable(error);
    session.close();

    const QJsonObject stored = readChunkDocument(projectPath, 1);
    const QJsonObject storedResults = stored.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectResultsSection)).toObject();
    const QString storedArtifact = storedResults.value(
        QStringLiteral("image_match_results")).toArray().first().toObject()
        .value(QStringLiteral("output")).toString();
    EXPECT_EQ(storedArtifact,
              QStringLiteral(
                   "plascan:///chunk/assets/image_matches/a.pimatch"));
    const QJsonObject storedImage = stored.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectFilesSection)).toObject()
        .value(QStringLiteral("images")).toArray().first().toObject();
    EXPECT_TRUE(storedImage.value(QStringLiteral("path")).toString()
                    .startsWith(QStringLiteral(
                        "plascan:///shared/images/")));
    EXPECT_EQ(storedResults.value(QStringLiteral("image_match_results"))
                  .toArray().first().toObject()
                  .value(QStringLiteral("schema_version")).toInt(),
              1);
    EXPECT_GT(stored.value(QStringLiteral("chunk")).toObject()
                  .value(QStringLiteral("revision")).toInt(),
              0);
    const QJsonObject storedReport = storedResults.value(
        QStringLiteral("report_results")).toArray().first().toObject();
    EXPECT_EQ(storedReport.value(QStringLiteral("planned_dir")).toString(),
              QStringLiteral(
                  "plascan:///chunk/reconstruction/future/run"));
    EXPECT_TRUE(storedReport.value(
        QStringLiteral("external_output_dir")).toString()
                    .startsWith(QStringLiteral(
                        "plascan-diagnostic:///")));

    ASSERT_TRUE(session.open(projectPath, &error)) << qPrintable(error);
    const QString materialized = session.projectResults().value(
        QStringLiteral("image_match_results")).toArray().first().toObject()
        .value(QStringLiteral("output")).toString();
    EXPECT_EQ(QDir::cleanPath(materialized), QDir::cleanPath(artifact));
}

TEST(ProjectSessionTest, UpdatesImportedImageCameraFromOriginalSourcePath)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("camera.plascan"));
    const QString imagePath =
        QDir(temporary.path()).filePath(QStringLiteral("source.png"));
    QFile imageFile(imagePath);
    ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
    ASSERT_EQ(imageFile.write("image"), 5);
    imageFile.close();

    ProjectSession session;
    QString error;
    ASSERT_TRUE(session.create(
        projectPath, QStringLiteral("相机写回工程"), &error))
        << qPrintable(error);
    ASSERT_TRUE(session.mergeImages(
        QJsonArray{QJsonObject{{QStringLiteral("path"), imagePath}}},
        &error)) << qPrintable(error);

    const QString importedPath = session.projectFiles()
                                     .value(QStringLiteral("images"))
                                     .toArray()
                                     .first()
                                     .toObject()
                                     .value(QStringLiteral("path"))
                                     .toString();
    ASSERT_NE(QDir::cleanPath(importedPath), QDir::cleanPath(imagePath));

    const QJsonObject camera{
        {QStringLiteral("fx"), 1200.0},
        {QStringLiteral("registered"), true}
    };
    int updatedCount = 0;
    ASSERT_TRUE(session.updateImageCameras(
        QMap<QString, QJsonObject>{{imagePath, camera}},
        &updatedCount,
        &error)) << qPrintable(error);
    EXPECT_EQ(updatedCount, 1);
    EXPECT_EQ(session.projectFiles()
                  .value(QStringLiteral("images"))
                  .toArray()
                  .first()
                  .toObject()
                  .value(QStringLiteral("camera"))
                  .toObject(),
              camera);
}

TEST(ProjectSessionTest, SavePrunesOnlyEmptyLegacyWorkflowDirectories)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("lazy.plascan"));

    ProjectSession session;
    QString error;
    ASSERT_TRUE(session.create(
        projectPath, QStringLiteral("按需目录"), &error))
        << qPrintable(error);

    const QString chunkRoot = session.activeChunkRoot();
    const QString bundleAdjustDir =
        QDir(chunkRoot).filePath(QStringLiteral("bundle_adjust"));
    const QString reconstructionModelDir =
        QDir(chunkRoot).filePath(QStringLiteral("reconstruction/model"));
    const QString reportsDir =
        QDir(chunkRoot).filePath(QStringLiteral("reports"));
    ASSERT_TRUE(QDir().mkpath(bundleAdjustDir));
    ASSERT_TRUE(QDir().mkpath(reconstructionModelDir));
    ASSERT_TRUE(QDir().mkpath(reportsDir));

    const QString reportPath =
        QDir(reportsDir).filePath(QStringLiteral("keep.json"));
    QFile report(reportPath);
    ASSERT_TRUE(report.open(QIODevice::WriteOnly));
    ASSERT_EQ(report.write("{}"), 2);
    report.close();

    ASSERT_TRUE(session.save(&error)) << qPrintable(error);
    EXPECT_FALSE(QFileInfo::exists(bundleAdjustDir));
    EXPECT_FALSE(QFileInfo::exists(reconstructionModelDir));
    EXPECT_FALSE(QFileInfo::exists(
        QDir(chunkRoot).filePath(QStringLiteral("reconstruction"))));
    EXPECT_TRUE(QFileInfo(reportsDir).isDir());
    EXPECT_TRUE(QFileInfo(reportPath).isFile());
}

TEST(ProjectSessionTest, SharesIdenticalImagesAcrossChunks)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("shared.plascan"));
    const QString imagePath =
        QDir(temporary.path()).filePath(QStringLiteral("moon.png"));
    QFile image(imagePath);
    ASSERT_TRUE(image.open(QIODevice::WriteOnly));
    ASSERT_EQ(image.write("same-image-content"), 18);
    image.close();
    const QString duplicatePath =
        QDir(temporary.path()).filePath(QStringLiteral("renamed.png"));
    QFile duplicate(duplicatePath);
    ASSERT_TRUE(duplicate.open(QIODevice::WriteOnly));
    ASSERT_EQ(duplicate.write("same-image-content"), 18);
    duplicate.close();

    QString error;
    ProjectSession first;
    ASSERT_TRUE(first.create(projectPath, QStringLiteral("共享"), &error))
        << qPrintable(error);
    ASSERT_TRUE(first.mergeImages(
        QJsonArray{QJsonObject{{QStringLiteral("path"), imagePath}}},
        &error)) << qPrintable(error);
    ASSERT_TRUE(first.save(&error)) << qPrintable(error);
    first.close();

    ProjectChunkStore store(projectPath);
    xjw::common::project::ProjectChunkRecord second;
    ASSERT_TRUE(store.createChunk(
        QStringLiteral("区块 2"),
        QJsonObject{{QStringLiteral("images"), QJsonArray{}}},
        QJsonObject{},
        QJsonObject{},
        xjw::common::project::ProjectResourceIndex().toJson(),
        &second,
        &error)) << qPrintable(error);
    ASSERT_TRUE(store.setDefaultChunk(second.id, &error))
        << qPrintable(error);

    ProjectSession secondSession;
    ASSERT_TRUE(secondSession.open(projectPath, &error))
        << qPrintable(error);
    ASSERT_TRUE(secondSession.mergeImages(
        QJsonArray{QJsonObject{{QStringLiteral("path"), duplicatePath}}},
        &error)) << qPrintable(error);
    ASSERT_TRUE(secondSession.save(&error)) << qPrintable(error);
    secondSession.close();

    const QString firstUri = readChunkDocument(projectPath, 1)
        .value(QStringLiteral("project_files")).toObject()
        .value(QStringLiteral("images")).toArray().first().toObject()
        .value(QStringLiteral("path")).toString();
    const QString secondUri = readChunkDocument(projectPath, 2)
        .value(QStringLiteral("project_files")).toObject()
        .value(QStringLiteral("images")).toArray().first().toObject()
        .value(QStringLiteral("path")).toString();
    EXPECT_EQ(firstUri, secondUri);

    QDirIterator sharedFiles(
        ProjectPackageLayout::sharedImagesDirectory(projectPath),
        QDir::Files | QDir::NoDotAndDotDot,
        QDirIterator::Subdirectories);
    int fileCount = 0;
    while (sharedFiles.hasNext())
    {
        sharedFiles.next();
        ++fileCount;
    }
    EXPECT_EQ(fileCount, 1);

    xjw::common::project::ProjectChunkIndex index;
    ASSERT_TRUE(store.loadIndex(&index, &error)) << qPrintable(error);
    const QString firstChunkId = index.chunks().constFirst().id;
    ASSERT_TRUE(store.removeChunk(firstChunkId, &error))
        << qPrintable(error);
    EXPECT_EQ(
        QDir(ProjectPackageLayout::sharedImagesDirectory(projectPath))
            .entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot).size(),
              1);

    xjw::common::project::ProjectChunkRecord third;
    ASSERT_TRUE(store.createChunk(
        QStringLiteral("区块 3"),
        QJsonObject{{QStringLiteral("images"), QJsonArray{}}},
        QJsonObject{},
        QJsonObject{},
        xjw::common::project::ProjectResourceIndex().toJson(),
        &third,
        &error)) << qPrintable(error);
    ASSERT_TRUE(store.setDefaultChunk(third.id, &error))
        << qPrintable(error);
    ASSERT_TRUE(store.removeChunk(second.id, &error))
        << qPrintable(error);
    EXPECT_EQ(
        QDir(ProjectPackageLayout::sharedImagesDirectory(projectPath))
            .entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot).size(),
        1);

    // The first committed generation without references only establishes the
    // tombstone. A later durable chunk revision must confirm that the image is
    // still unreachable before the shared-image GC may delete it.
    ProjectSession thirdSession;
    ASSERT_TRUE(thirdSession.open(projectPath, &error))
        << qPrintable(error);
    ASSERT_TRUE(thirdSession.save(&error)) << qPrintable(error);
    thirdSession.close();
    EXPECT_TRUE(
        QDir(ProjectPackageLayout::sharedImagesDirectory(projectPath))
            .entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot).isEmpty());
}

TEST(ProjectSessionTest, RejectsConcurrentWriters)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("locked.plascan"));
    QString error;
    ProjectSession first;
    ASSERT_TRUE(first.create(projectPath, QStringLiteral("锁"), &error))
        << qPrintable(error);

    ProjectSession second;
    EXPECT_FALSE(second.open(projectPath, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("其他进程")))
        << qPrintable(error);
}

TEST(ProjectSessionTest, OpensTheDefaultChunkInsteadOfAssumingDirectoryOne)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("chunks.plascan"));

    ProjectSession creator;
    QString error;
    ASSERT_TRUE(creator.create(
        projectPath, QStringLiteral("Chunks"), &error))
        << qPrintable(error);
    creator.close();

    ProjectChunkStore store(projectPath);
    xjw::common::project::ProjectChunkRecord second;
    ASSERT_TRUE(store.createChunk(
        QStringLiteral("区块 2"),
        QJsonObject{{QStringLiteral("images"), QJsonArray{}}},
        QJsonObject{},
        QJsonObject{},
        xjw::common::project::ProjectResourceIndex().toJson(),
        &second,
        &error)) << qPrintable(error);
    ASSERT_TRUE(store.setDefaultChunk(second.id, &error))
        << qPrintable(error);

    ProjectSession session;
    ASSERT_TRUE(session.open(projectPath, &error)) << qPrintable(error);
    EXPECT_EQ(session.activeChunk().directory, 2);
    EXPECT_EQ(
        QDir::cleanPath(session.activeChunkRoot()),
        QDir::cleanPath(
            ProjectPackageLayout::chunkDirectory(projectPath, 2)));
}

TEST(ProjectSessionTest, SelectsChunkByNameAndUpdatesDefault)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("select.plascan"));
    QString error;
    ProjectSession creator;
    ASSERT_TRUE(creator.create(
        projectPath, QStringLiteral("选择"), &error))
        << qPrintable(error);
    creator.close();

    ProjectChunkStore store(projectPath);
    xjw::common::project::ProjectChunkRecord second;
    ASSERT_TRUE(store.createChunk(
        QStringLiteral("处理方案 B"),
        QJsonObject{{QStringLiteral("images"), QJsonArray{}}},
        QJsonObject{},
        QJsonObject{},
        xjw::common::project::ProjectResourceIndex().toJson(),
        &second,
        &error)) << qPrintable(error);
    const QString secondRoot =
        ProjectPackageLayout::chunkDirectory(projectPath, second.directory);
    EXPECT_TRUE(QFileInfo(
        ProjectPackageLayout::chunkArchivePath(
            projectPath, second.directory)).isFile());
    for (const QString &optionalDirectory :
         {QStringLiteral("assets"),
          QStringLiteral("bundle_adjust"),
          QStringLiteral("reconstruction"),
          QStringLiteral("reports")})
    {
        EXPECT_FALSE(QFileInfo::exists(
            QDir(secondRoot).filePath(optionalDirectory)));
    }

    ProjectSession session;
    ASSERT_TRUE(session.open(projectPath, &error)) << qPrintable(error);
    ASSERT_TRUE(session.selectChunk(
        QString(), QStringLiteral("处理方案 B"), &error))
        << qPrintable(error);
    EXPECT_EQ(session.activeChunk().id, second.id);
    EXPECT_EQ(session.activeChunk().directory, 2);
    EXPECT_EQ(
        QDir::cleanPath(ProjectIO::projectBundleAdjustDir(projectPath)),
        QDir::cleanPath(
            QDir(ProjectPackageLayout::chunkDirectory(projectPath, 2))
                .filePath(QStringLiteral("bundle_adjust"))));
    session.close();

    EXPECT_EQ(store.defaultChunk(&error).id, second.id)
        << qPrintable(error);
}

TEST(ProjectSessionTest, RejectsLegacyProjectWithoutMigration)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath =
        QDir(temporary.path()).filePath(QStringLiteral("legacy.plascan"));
    QFile file(projectPath);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly));
    ASSERT_GT(file.write("{\"format_version\":\"1.0\"}"), 0);
    file.close();

    ProjectSession session;
    QString error;
    EXPECT_FALSE(session.open(projectPath, &error));
    EXPECT_FALSE(error.isEmpty());
    EXPECT_FALSE(
        QFileInfo(ProjectPackageLayout::dataDirectory(projectPath)).exists());
}
