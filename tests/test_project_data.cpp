#include <gtest/gtest.h>

#include "project/PlascanArchive.h"
#include "project/ProjectChunkStore.h"
#include "project/ProjectResourceStore.h"
#include "project/ProjectWorkspaceStore.h"
#include "ProjectData.h"
#include "ProjectFilesManager.h"
#include "project/ProjectChunkIndex.h"
#include "project/ProjectPackageLayout.h"
#include "project/ProjectSharedImageStore.h"
#include "project/PortableProjectFormat.h"
#include "project/ProjectIO.h"

#include <QDir>
#include <QDirIterator>
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLockFile>
#include <QSemaphore>
#include <QTemporaryDir>
#include <QtConcurrent/QtConcurrentRun>

using xjw::common::project::PortableProjectFormat;
using xjw::common::project::ProjectChunkIndex;
using xjw::common::project::ProjectPackageLayout;
using xjw::common::project::ProjectIO;
using xjw::common::project::ProjectResourceIndex;
using xjw::common::project::ProjectResourceRef;
using xjw::common::project::ProjectSharedImageStore;

namespace {

QStringList allResultKeys()
{
    return {
        QStringLiteral("image_match_results"),
        QStringLiteral("intersection_results"),
        QStringLiteral("bundle_adjust_results"),
        QStringLiteral("aerial_triangulation_results"),
        QStringLiteral("observation_network_results"),
        QStringLiteral("depth_map_results"),
        QStringLiteral("dense_cloud_results"),
        QStringLiteral("model_results"),
        QStringLiteral("dem_results"),
        QStringLiteral("ortho_results"),
        QStringLiteral("report_results"),
        QStringLiteral("reference_datasets")
    };
}

QJsonArray singleRecord(const QString &key, const QString &pathKey = QString(), const QString &path = QString())
{
    QJsonObject record;
    record[QStringLiteral("kind")] = key;
    if (!pathKey.isEmpty())
    {
        record[pathKey] = path;
    }
    return QJsonArray{record};
}

QJsonObject archiveDocument(const QString &archivePath, bool projectDocument)
{
    const QString physicalArchive = projectDocument
        ? ProjectPackageLayout::metadataArchivePath(archivePath)
        : ProjectChunkStore(archivePath).defaultChunkArchivePath();
    PlascanArchive archive(
        physicalArchive, PlascanArchivePathType::DirectArchive);
    EXPECT_TRUE(archive.isValid()) << qPrintable(archivePath);

    QString error;
    const QByteArray data = archive.readEntry(
        QString::fromLatin1(PortableProjectFormat::DocumentEntry),
        &error);
    EXPECT_FALSE(data.isEmpty()) << qPrintable(error);

    const QJsonDocument doc = QJsonDocument::fromJson(data);
    EXPECT_TRUE(doc.isObject()) << data.constData();
    return doc.object();
}

QJsonObject projectDocument(const QString &projectPath)
{
    return archiveDocument(projectPath, true);
}

QJsonObject chunkDocument(const QString &projectPath)
{
    return archiveDocument(projectPath, false);
}

QJsonObject chunkDocument(const QString &projectPath, int chunkDirectory)
{
    QJsonObject document;
    QString error;
    EXPECT_TRUE(ProjectChunkStore(projectPath).readChunkDocument(
        chunkDirectory, &document, &error))
        << qPrintable(error);
    return document;
}

QJsonObject chunkSection(const QString &projectPath, const char *sectionName)
{
    return chunkDocument(projectPath)
        .value(QString::fromLatin1(sectionName))
        .toObject();
}

QString defaultChunkArchivePath(const QString &projectPath)
{
    QString error;
    const QString path =
        ProjectChunkStore(projectPath).defaultChunkArchivePath(&error);
    EXPECT_FALSE(path.isEmpty()) << qPrintable(error);
    return path;
}

QString chunkPhysicalPath(const QString &projectPath,
                          const QString &entryPath)
{
    QString relativePath = entryPath;
    if (relativePath.startsWith(QStringLiteral("shared/")))
    {
        return QDir(ProjectPackageLayout::dataDirectory(projectPath))
            .filePath(relativePath);
    }
    if (relativePath.startsWith(QStringLiteral("chunk/")))
    {
        relativePath = relativePath.mid(6);
    }
    else if (relativePath.startsWith(QStringLiteral("workspace/")))
    {
        relativePath = relativePath.mid(10);
    }
    QString error;
    const QString root =
        ProjectChunkStore(projectPath).defaultChunkDirectory(&error);
    EXPECT_FALSE(root.isEmpty()) << qPrintable(error);
    return QDir(root).filePath(relativePath);
}

QString tempProjectPath(QTemporaryDir &dir)
{
    return QDir(dir.path()).filePath(QStringLiteral("demo.plascan"));
}

void moveProjectPair(const QString &sourceProject,
                     const QString &destinationProject)
{
    const QString sourceData =
        ProjectPackageLayout::dataDirectory(sourceProject);
    const QString destinationData =
        ProjectPackageLayout::dataDirectory(destinationProject);
    ASSERT_TRUE(QDir().rename(sourceData, destinationData))
        << qPrintable(sourceData) << " -> " << qPrintable(destinationData);
    if (!QFile::rename(sourceProject, destinationProject))
    {
        QDir().rename(destinationData, sourceData);
        FAIL() << qPrintable(sourceProject)
               << " -> " << qPrintable(destinationProject);
    }
}

void writeTestFile(const QString &path, const QByteArray &content)
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()))
        << qPrintable(QFileInfo(path).absolutePath());
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly)) << qPrintable(path);
    ASSERT_EQ(file.write(content), content.size()) << qPrintable(path);
}

QByteArray readTestFile(const QString &path)
{
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly)) << qPrintable(path);
    return file.readAll();
}

QString resultPath(const QJsonObject &metadata,
                   const QString &resultKey,
                   const QString &pathKey)
{
    const QJsonArray records = metadata.value(resultKey).toArray();
    EXPECT_FALSE(records.isEmpty()) << qPrintable(resultKey);
    return records.isEmpty()
        ? QString()
        : records.at(0).toObject().value(pathKey).toString();
}

} // namespace

TEST(ProjectFilesManagerTest, SeparatesAllWorkflowResultsFromCoreMetadata)
{
    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/image.png")}}
    };
    meta[QStringLiteral("project_name")] = QStringLiteral("demo");
    for (const QString &key : allResultKeys())
    {
        meta[key] = singleRecord(key);
    }

    ProjectFilesManager manager;
    manager.setData(meta);

    const QJsonObject core = manager.coreData();
    const QJsonObject results = manager.resultsData();

    EXPECT_TRUE(core.contains(QStringLiteral("images")));
    EXPECT_TRUE(core.contains(QStringLiteral("project_name")));
    for (const QString &key : allResultKeys())
    {
        EXPECT_FALSE(core.contains(key)) << qPrintable(key);
        ASSERT_TRUE(results.contains(key)) << qPrintable(key);
        EXPECT_EQ(results.value(key).toArray().size(), 1) << qPrintable(key);
    }
}

TEST(PlascanArchiveTest, WriteEntryReleasesExistingReadHandleBeforeReplacingArchive)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    QJsonObject manifest;
    manifest[QStringLiteral("format_version")] = QStringLiteral("1.0");
    manifest[QStringLiteral("type")] = QStringLiteral("plascan_project");

    const QJsonObject initialFiles{
        {QStringLiteral("images"), QJsonArray{}}
    };

    QString error;
    ASSERT_TRUE(PlascanArchive::createArchive(
        projectPath,
        {
            qMakePair(
                QStringLiteral("manifest.json"),
                QJsonDocument(manifest).toJson(QJsonDocument::Compact)),
            qMakePair(
                QStringLiteral("project_files.json"),
                QJsonDocument(initialFiles).toJson(QJsonDocument::Compact))
        },
        &error))
        << qPrintable(error);

    PlascanArchive archive(
        projectPath, PlascanArchivePathType::DirectArchive);
    ASSERT_TRUE(archive.isValid()) << qPrintable(projectPath);

    const QJsonObject updatedFiles{
        {QStringLiteral("images"), QJsonArray{}},
        {QStringLiteral("project_note"), QStringLiteral("updated")}
    };
    ASSERT_TRUE(archive.writeEntry(QStringLiteral("doc.json"),
                                   QJsonDocument(updatedFiles).toJson(QJsonDocument::Compact),
                                   &error))
        << qPrintable(error);

    const QJsonObject storedFiles =
        QJsonDocument::fromJson(
            archive.readEntry(QStringLiteral("doc.json")))
            .object();
    EXPECT_EQ(storedFiles.value(QStringLiteral("project_note")).toString(), QStringLiteral("updated"));
}

TEST(ProjectDataTest, CreateSaveOpenSupportsChineseProjectPath)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString chineseDir = QDir(dir.path()).filePath(QStringLiteral("中文项目目录"));
    ASSERT_TRUE(QDir().mkpath(chineseDir)) << qPrintable(chineseDir);

    const QString projectPath = QDir(chineseDir).filePath(QStringLiteral("月球项目.plascan"));
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("月球项目"))) << qPrintable(projectPath);
    EXPECT_TRUE(QFileInfo::exists(projectPath)) << qPrintable(projectPath);

    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    project.closeProject();

    ProjectData reopened;
    ASSERT_TRUE(reopened.openProject(projectPath, &error)) << qPrintable(error);
    EXPECT_TRUE(reopened.hasProject());
    EXPECT_EQ(QDir::cleanPath(QFileInfo(reopened.currentProjectPath()).absoluteFilePath()),
              QDir::cleanPath(QFileInfo(projectPath).absoluteFilePath()));
}

TEST(ProjectDataTest, UiSettingsDoNotDirtySessionWithoutProject)
{
    ProjectData project;
    ASSERT_FALSE(project.hasProject());
    ASSERT_FALSE(project.isDirty());

    project.saveUiSettings(QJsonObject{
        {QStringLiteral("workspace_visible"), true},
        {QStringLiteral("photos_visible"), true}
    });

    EXPECT_FALSE(project.hasProject());
    EXPECT_FALSE(project.isDirty());
    EXPECT_TRUE(project.currentProjectPath().isEmpty());
}

TEST(ProjectDataTest, UiSettingsDoNotDirtyProjectContent)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("ui_state")));
    ASSERT_FALSE(project.isDirty());

    project.saveUiSettings(QJsonObject{
        {QStringLiteral("workspace_visible"), false},
        {QStringLiteral("dock_layout_version"), 4}
    });

    EXPECT_FALSE(project.isDirty());
    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    project.closeProject();

    ProjectData reopened;
    ASSERT_TRUE(reopened.openProject(projectPath, &error))
        << qPrintable(error);
    EXPECT_FALSE(reopened.isDirty());
    EXPECT_FALSE(reopened.loadUiSettings()
                     .value(QStringLiteral("workspace_visible"))
                     .toBool(true));
    EXPECT_EQ(reopened.loadUiSettings()
                  .value(QStringLiteral("dock_layout_version"))
                  .toInt(),
              4);
}

TEST(ProjectDataTest, CreatingProjectReplacesSessionWithoutImportingPreviousResources)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString firstProjectPath =
        QDir(dir.path()).filePath(QStringLiteral("first.plascan"));
    const QString secondProjectPath =
        QDir(dir.path()).filePath(QStringLiteral("second.plascan"));
    const QString externalResultPath =
        QDir(dir.path()).filePath(QStringLiteral("old-result.ply"));
    writeTestFile(externalResultPath, QByteArray("old-project-result"));

    ProjectData project;
    ASSERT_TRUE(project.createProject(
        firstProjectPath, QStringLiteral("first")));

    QJsonObject oldMetadata = project.metadata();
    oldMetadata[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), externalResultPath}}
    };
    oldMetadata[QStringLiteral("model_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), externalResultPath}}
    };
    project.updateMetadata(oldMetadata, true);

    QStringList lifecycleEvents;
    QObject::connect(&project, &ProjectData::projectClosed, [&]()
    {
        lifecycleEvents.append(QStringLiteral("closed"));
    });
    QObject::connect(&project, &ProjectData::projectOpened,
                     [&](const QString &)
    {
        lifecycleEvents.append(QStringLiteral("opened"));
    });

    ASSERT_TRUE(project.createProject(
        secondProjectPath, QStringLiteral("second")));
    EXPECT_EQ(lifecycleEvents,
              QStringList({QStringLiteral("closed"),
                           QStringLiteral("opened")}));
    EXPECT_EQ(project.currentProjectPath(), secondProjectPath);
    EXPECT_TRUE(project.metadata()
                    .value(QStringLiteral("images"))
                    .toArray()
                    .isEmpty());
    EXPECT_TRUE(project.metadata()
                    .value(QStringLiteral("model_results"))
                    .toArray()
                    .isEmpty());

    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    const QString importedDirectory = QDir(
        ProjectPackageLayout::chunkDirectory(secondProjectPath, 1))
        .filePath(QStringLiteral("assets/imported"));
    EXPECT_FALSE(QFileInfo::exists(importedDirectory));
    EXPECT_FALSE(QJsonDocument(chunkDocument(secondProjectPath))
                     .toJson(QJsonDocument::Compact)
                     .contains("old-result.ply"));
}

TEST(ProjectDataTest, OpeningDifferentProjectClosesPreviousSessionFirst)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString firstProjectPath =
        QDir(dir.path()).filePath(QStringLiteral("first.plascan"));
    const QString secondProjectPath =
        QDir(dir.path()).filePath(QStringLiteral("second.plascan"));
    {
        ProjectData creator;
        ASSERT_TRUE(creator.createProject(
            firstProjectPath, QStringLiteral("first")));
        creator.closeProject();
        ASSERT_TRUE(creator.createProject(
            secondProjectPath, QStringLiteral("second")));
    }

    ProjectData project;
    QString error;
    ASSERT_TRUE(project.openProject(firstProjectPath, &error))
        << qPrintable(error);

    QStringList lifecycleEvents;
    QObject::connect(&project, &ProjectData::projectClosed, [&]()
    {
        lifecycleEvents.append(QStringLiteral("closed"));
    });
    QObject::connect(&project, &ProjectData::projectOpened,
                     [&](const QString &)
    {
        lifecycleEvents.append(QStringLiteral("opened"));
    });

    ASSERT_TRUE(project.openProject(secondProjectPath, &error))
        << qPrintable(error);
    EXPECT_EQ(lifecycleEvents,
              QStringList({QStringLiteral("closed"),
                           QStringLiteral("opened")}));
    EXPECT_EQ(project.currentProjectPath(), secondProjectPath);
}

TEST(ProjectDataTest, OpenRejectsDescriptorWithoutMatchingDataDirectory)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    QString error;
    ASSERT_TRUE(ProjectPackageLayout::writeDescriptor(projectPath, &error))
        << qPrintable(error);

    ProjectData project;
    EXPECT_FALSE(project.openProject(projectPath, &error));
    EXPECT_FALSE(error.isEmpty());
    EXPECT_TRUE(
        error.contains(QStringLiteral("归档"))
        || error.contains(QStringLiteral("不存在")))
        << qPrintable(error);
}

TEST(ProjectDataTest, NewProjectCreatesMetashapeStyleSplitLayout)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("分体项目")));
        const QJsonArray chunkSummaries = project.chunks();
        ASSERT_EQ(chunkSummaries.size(), 1);
        EXPECT_EQ(chunkSummaries.first().toObject()
                      .value(QStringLiteral("image_count")).toInt(-1),
                  0);
        EXPECT_FALSE(chunkSummaries.first().toObject().contains(
            QStringLiteral("tie_point_count")));
    }

    QString layoutError;
    EXPECT_TRUE(ProjectPackageLayout::isDescriptor(
        projectPath, &layoutError)) << qPrintable(layoutError);
    EXPECT_TRUE(QFileInfo(
        ProjectPackageLayout::metadataArchivePath(projectPath)).isFile());
    EXPECT_TRUE(QFileInfo(
        ProjectPackageLayout::workspaceDirectory(projectPath)).isDir());
    const QString chunkRoot =
        ProjectPackageLayout::workspaceDirectory(projectPath);
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

    const QJsonObject manifest = projectDocument(projectPath);
    EXPECT_TRUE(
        PortableProjectFormat::isCurrentProjectDocument(manifest));
    const QString projectId =
        manifest.value(QStringLiteral("project_id")).toString();
    ASSERT_FALSE(projectId.isEmpty());

    const QJsonObject config = chunkSection(
        projectPath, PortableProjectFormat::ProjectConfigSection);
    EXPECT_EQ(config.value(QStringLiteral("project_id")).toString(), projectId);
    EXPECT_EQ(config.value(QStringLiteral("schema_version")).toInt(), 2);
    EXPECT_FALSE(config.contains(QStringLiteral("ui")));

    const QJsonObject resourceIndexObject = chunkSection(
        projectPath, PortableProjectFormat::ResourceIndexSection);
    QString indexError;
    const ProjectResourceIndex resourceIndex =
        ProjectResourceIndex::fromJson(resourceIndexObject, &indexError);
    EXPECT_TRUE(indexError.isEmpty()) << qPrintable(indexError);
    EXPECT_TRUE(resourceIndex.isEmpty());

    const QJsonObject uiState = manifest.value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectUiStateSection)).toObject();
    EXPECT_EQ(uiState.value(QStringLiteral("schema_version")).toInt(), 1);
    EXPECT_TRUE(uiState.value(QStringLiteral("display_settings")).isObject());

    PlascanArchive archive(projectPath);
    ASSERT_TRUE(archive.isValid());
    EXPECT_TRUE(archive.containsEntry(QStringLiteral("doc.json")));
    EXPECT_EQ(
        archive.listEntries(),
        QVector<QString>{QStringLiteral("doc.json")});
    PlascanArchive chunkArchive(
        defaultChunkArchivePath(projectPath),
        PlascanArchivePathType::DirectArchive);
    ASSERT_TRUE(chunkArchive.isValid());
    EXPECT_EQ(
        chunkArchive.listEntries(),
        QVector<QString>{QStringLiteral("doc.json")});
}

TEST(ProjectDataTest, CameraModelPolicyDefaultsAndPersists)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(
            projectPath, QStringLiteral("相机模型策略")));

        ASSERT_TRUE(project.cameraModelPolicy().has_value());
        EXPECT_EQ(project.cameraModelPolicy().value(),
                  ProjectCameraModelPolicy::FramePinhole);
        EXPECT_FALSE(project.isDirty());

        project.setCameraModelPolicy(ProjectCameraModelPolicy::FramePinhole);
        EXPECT_FALSE(project.isDirty());

        project.setCameraModelPolicy(
            ProjectCameraModelPolicy::IsisUsgsCsmLineScan);
        EXPECT_TRUE(project.isDirty());

        QString error;
        ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    }

    const QJsonObject archived_config = chunkSection(
        projectPath, PortableProjectFormat::ProjectConfigSection);
    EXPECT_EQ(
        archived_config.value(QStringLiteral("camera_model_policy")).toString(),
        QStringLiteral("isis_usgscsm_linescan"));

    ProjectData reopened;
    QString error;
    ASSERT_TRUE(reopened.openProject(projectPath, &error))
        << qPrintable(error);
    ASSERT_TRUE(reopened.cameraModelPolicy().has_value());
    EXPECT_EQ(reopened.cameraModelPolicy().value(),
              ProjectCameraModelPolicy::IsisUsgsCsmLineScan);
}

TEST(ProjectDataTest, FullSavePrunesOnlyEmptyLegacyWorkflowDirectories)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("按需目录")));

    const QString chunkRoot =
        ProjectPackageLayout::workspaceDirectory(projectPath);
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
    writeTestFile(reportPath, QByteArray("{}"));

    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    EXPECT_FALSE(QFileInfo::exists(bundleAdjustDir));
    EXPECT_FALSE(QFileInfo::exists(reconstructionModelDir));
    EXPECT_FALSE(QFileInfo::exists(
        QDir(chunkRoot).filePath(QStringLiteral("reconstruction"))));
    EXPECT_TRUE(QFileInfo(reportsDir).isDir());
    EXPECT_TRUE(QFileInfo(reportPath).isFile());
}

TEST(ProjectDataTest, ChunkDirectoriesAreMonotonicAndNeverReused)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("多 Chunk 项目")));

    QString error;
    QString chunk2;
    QString chunk3;
    QString chunk4;
    ASSERT_TRUE(project.createChunk(
        QStringLiteral("区块 2"), &chunk2, &error))
        << qPrintable(error);
    ASSERT_TRUE(project.createChunk(
        QStringLiteral("区块 3"), &chunk3, &error))
        << qPrintable(error);
    ASSERT_EQ(project.activeChunkDirectory(), 3);

    ASSERT_TRUE(project.removeChunk(chunk2, &error))
        << qPrintable(error);
    EXPECT_FALSE(QFileInfo(
        ProjectPackageLayout::chunkDirectory(projectPath, 2)).exists());

    ASSERT_TRUE(project.createChunk(
        QStringLiteral("区块 4"), &chunk4, &error))
        << qPrintable(error);
    EXPECT_EQ(project.activeChunkDirectory(), 4);
    EXPECT_TRUE(QFileInfo(
        ProjectPackageLayout::chunkArchivePath(projectPath, 1)).isFile());
    EXPECT_FALSE(QFileInfo(
        ProjectPackageLayout::chunkDirectory(projectPath, 2)).exists());
    EXPECT_TRUE(QFileInfo(
        ProjectPackageLayout::chunkArchivePath(projectPath, 3)).isFile());
    EXPECT_TRUE(QFileInfo(
        ProjectPackageLayout::chunkArchivePath(projectPath, 4)).isFile());

    ProjectChunkIndex index;
    ASSERT_TRUE(ProjectChunkStore(projectPath).loadIndex(&index, &error))
        << qPrintable(error);
    EXPECT_EQ(index.nextChunkDirectory(), 5);
    EXPECT_EQ(index.size(), 3);
    EXPECT_EQ(index.defaultChunk().directory, 4);
    EXPECT_FALSE(index.chunk(chunk3).id.isEmpty());
    EXPECT_FALSE(index.chunk(chunk4).id.isEmpty());
}

TEST(ProjectDataTest, SwitchingChunksKeepsMetadataIsolated)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("Chunk 隔离项目")));
    const QString chunk1 = project.activeChunkId();
    QJsonObject chunk1Meta = ProjectFilesManager::defaultFiles();
    chunk1Meta[QStringLiteral("chunk_note")] =
        QStringLiteral("first");
    project.updateMetadata(chunk1Meta);

    QString error;
    QString chunk2;
    ASSERT_TRUE(project.createChunk(
        QStringLiteral("第二处理区"), &chunk2, &error))
        << qPrintable(error);
    QJsonObject chunk2Meta = ProjectFilesManager::defaultFiles();
    chunk2Meta[QStringLiteral("chunk_note")] =
        QStringLiteral("second");
    project.updateMetadata(chunk2Meta);

    ASSERT_TRUE(project.switchChunk(chunk1, &error))
        << qPrintable(error);
    EXPECT_EQ(
        project.metadata().value(QStringLiteral("chunk_note")).toString(),
        QStringLiteral("first"));

    ASSERT_TRUE(project.switchChunk(chunk2, &error))
        << qPrintable(error);
    EXPECT_EQ(
        project.metadata().value(QStringLiteral("chunk_note")).toString(),
        QStringLiteral("second"));
}

TEST(ProjectDataTest, RejectsLegacyWorkspaceWithoutChangingIt)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString dataDirectory =
        ProjectPackageLayout::dataDirectory(projectPath);
    const QString legacyWorkspace =
        QDir(dataDirectory).filePath(QStringLiteral("workspace"));
    ASSERT_TRUE(QDir().mkpath(legacyWorkspace));
    QString error;
    ASSERT_TRUE(ProjectPackageLayout::writeDescriptor(
        projectPath, &error)) << qPrintable(error);

    const QString legacyAsset = QDir(legacyWorkspace)
        .filePath(QStringLiteral("assets/images/legacy.txt"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(legacyAsset).absolutePath()));
    QFile legacyFile(legacyAsset);
    ASSERT_TRUE(legacyFile.open(QIODevice::WriteOnly));
    ASSERT_EQ(legacyFile.write("legacy"), 6);
    legacyFile.close();

    QJsonObject legacyManifest{
        {QStringLiteral("type"),
         QString::fromLatin1(PortableProjectFormat::ProjectType)},
        {QStringLiteral("format_version"),
         QStringLiteral("2.0")},
        {QStringLiteral("project_id"), QStringLiteral("legacy-project")},
        {QStringLiteral("created_with"), QStringLiteral("PlaScan")}
    };
    ASSERT_TRUE(PlascanArchive::createArchive(
        QDir(dataDirectory).filePath(QStringLiteral("project.zip")),
        {
            qMakePair(
                QStringLiteral("manifest.json"),
                QJsonDocument(legacyManifest)
                    .toJson(QJsonDocument::Compact)),
            qMakePair(
                QStringLiteral("project_files.json"),
                QJsonDocument(ProjectFilesManager::defaultFiles())
                    .toJson(QJsonDocument::Compact)),
            qMakePair(
                QStringLiteral("project_results.json"),
                QJsonDocument(ProjectFilesManager::defaultResults())
                    .toJson(QJsonDocument::Compact)),
            qMakePair(
                QStringLiteral("project_config.json"),
                QJsonDocument(QJsonObject{
                    {QStringLiteral("project_name"),
                     QStringLiteral("旧项目")}})
                    .toJson(QJsonDocument::Compact)),
            qMakePair(
                QStringLiteral("project_ui_state.json"),
                QByteArrayLiteral(
                    "{\"schema_version\":1,\"display_settings\":{}}")),
            qMakePair(
                QStringLiteral("resource_index.json"),
                QJsonDocument(ProjectResourceIndex().toJson())
                    .toJson(QJsonDocument::Compact))
        },
        &error)) << qPrintable(error);

    const QString archivePath =
        QDir(dataDirectory).filePath(QStringLiteral("project.zip"));
    QFile originalArchive(archivePath);
    ASSERT_TRUE(originalArchive.open(QIODevice::ReadOnly));
    const QByteArray originalBytes = originalArchive.readAll();
    originalArchive.close();

    ProjectData oldProject;
    EXPECT_FALSE(oldProject.openProject(projectPath, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("不支持旧版工程格式")))
        << qPrintable(error);
    EXPECT_TRUE(QFileInfo(legacyAsset).isFile());
    EXPECT_FALSE(QFileInfo(
        ProjectPackageLayout::chunkDirectory(projectPath, 1)).exists());

    QFile unchangedArchive(archivePath);
    ASSERT_TRUE(unchangedArchive.open(QIODevice::ReadOnly));
    EXPECT_EQ(unchangedArchive.readAll(), originalBytes);
}

TEST(ProjectDataTest, ProjectUiStateAndWorkflowConfigPersistSeparatelyAfterMove)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString sourceDir =
        QDir(dir.path()).filePath(QStringLiteral("原始设备"));
    const QString movedDir =
        QDir(dir.path()).filePath(QStringLiteral("新设备"));
    ASSERT_TRUE(QDir().mkpath(sourceDir));
    ASSERT_TRUE(QDir().mkpath(movedDir));

    const QString projectPath =
        QDir(sourceDir).filePath(QStringLiteral("显示状态.plascan"));
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(
            projectPath, QStringLiteral("显示状态")));
        project.saveUiSettings(QJsonObject{
            {QStringLiteral("show_interest_points"), false},
            {QStringLiteral("feature_display"),
             QJsonObject{{QStringLiteral("pointSize"), 7}}}
        });
        project.saveImageMatchingSettings(QJsonObject{
            {QStringLiteral("algorithm"), QStringLiteral("sift_lightglue")},
            {QStringLiteral("max_features"), 2048}
        });

        QString error;
        ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    }

    const QJsonObject archivedConfig = chunkSection(
        projectPath, PortableProjectFormat::ProjectConfigSection);
    EXPECT_FALSE(archivedConfig.contains(QStringLiteral("ui")));
    EXPECT_EQ(
        archivedConfig.value(QStringLiteral("workflow"))
            .toObject()
            .value(QStringLiteral("image_matching"))
            .toObject()
            .value(QStringLiteral("algorithm"))
            .toString(),
        QStringLiteral("sift_lightglue"));

    const QJsonObject archivedUiState = projectDocument(projectPath).value(
        QString::fromLatin1(
            PortableProjectFormat::ProjectUiStateSection)).toObject();
    const QJsonObject displaySettings =
        archivedUiState.value(QStringLiteral("display_settings")).toObject();
    EXPECT_FALSE(
        displaySettings.value(QStringLiteral("show_interest_points")).toBool());
    EXPECT_EQ(
        displaySettings.value(QStringLiteral("feature_display"))
            .toObject()
            .value(QStringLiteral("pointSize"))
            .toInt(),
        7);

    const QString movedProjectPath =
        QDir(movedDir).filePath(QStringLiteral("显示状态.plascan"));
    moveProjectPair(projectPath, movedProjectPath);

    ProjectData reopened;
    QString error;
    ASSERT_TRUE(reopened.openProject(movedProjectPath, &error))
        << qPrintable(error);
    const QJsonObject reopenedUi = reopened.loadUiSettings();
    EXPECT_FALSE(
        reopenedUi.value(QStringLiteral("show_interest_points")).toBool());
    EXPECT_EQ(
        reopenedUi.value(QStringLiteral("feature_display"))
            .toObject()
            .value(QStringLiteral("pointSize"))
            .toInt(),
        7);
    EXPECT_EQ(
        reopened.loadImageMatchingSettings()
            .value(QStringLiteral("algorithm"))
            .toString(),
        QStringLiteral("sift_lightglue"));
}

TEST(ProjectDataTest, WorkspaceOnlySettingsAreIndexedInSplitProject)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const QString projectPath = tempProjectPath(dir);

    ProjectData data;
    ASSERT_TRUE(data.createProject(projectPath, QStringLiteral("workspace settings")));
    const QString dialogPath = QDir(ProjectIO::projectRootFromPlascan(projectPath))
                                   .filePath(QStringLiteral("project_dialog.json"));
    writeTestFile(
        dialogPath,
        QJsonDocument(QJsonObject{
            {QStringLiteral("generate_model"),
             QJsonObject{{QStringLiteral("quality"), QStringLiteral("high")}}}
        }).toJson(QJsonDocument::Compact));

    data.markWorkspaceDirty();
    EXPECT_TRUE(data.isDirty());
    QString error;
    ASSERT_TRUE(data.saveProject(&error)) << qPrintable(error);

    PlascanArchive archive(projectPath);
    ASSERT_TRUE(archive.isValid());
    EXPECT_FALSE(
        archive.containsEntry(QStringLiteral("workspace/project_dialog.json")));
    const QByteArray archivedDialog = readTestFile(dialogPath);
    EXPECT_EQ(
        QJsonDocument::fromJson(archivedDialog)
            .object()
            .value(QStringLiteral("generate_model"))
            .toObject()
            .value(QStringLiteral("quality"))
            .toString(),
        QStringLiteral("high"));
}

TEST(ProjectDataTest, EmbeddedUiConfigDoesNotOverrideProjectUiState)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    {
        ProjectData created;
        ASSERT_TRUE(created.createProject(
            projectPath, QStringLiteral("strict-ui-separation")));
        created.closeProject();
    }

    QString error;
    {
        PlascanArchive archive(
            defaultChunkArchivePath(projectPath),
            PlascanArchivePathType::DirectArchive);
        ASSERT_TRUE(archive.isValid());
        QJsonObject workflowConfig{
            {QStringLiteral("project_name"), QStringLiteral("strict-ui")},
            {QStringLiteral("ui"),
             QJsonObject{
                 {QStringLiteral("show_interest_points"), false},
                 {QStringLiteral("feature_display"),
                  QJsonObject{{QStringLiteral("pointSize"), 5}}}
             }}
        };
        QJsonObject document = QJsonDocument::fromJson(
            archive.readEntry(QStringLiteral("doc.json"))).object();
        document[QString::fromLatin1(
            PortableProjectFormat::ProjectConfigSection)] = workflowConfig;
        ASSERT_TRUE(archive.writeEntry(
            QStringLiteral("doc.json"),
            QJsonDocument(document).toJson(QJsonDocument::Compact),
            &error)) << qPrintable(error);
    }

    ProjectData project;
    ASSERT_TRUE(project.openProject(projectPath, &error))
        << qPrintable(error);
    EXPECT_TRUE(
        project.loadUiSettings()
            .value(QStringLiteral("show_interest_points"))
            .toBool());
    EXPECT_EQ(
        project.loadUiSettings()
            .value(QStringLiteral("feature_display"))
            .toObject()
            .value(QStringLiteral("pointSize"))
            .toInt(),
        1);
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

    const QJsonObject savedConfig = chunkSection(
        projectPath, PortableProjectFormat::ProjectConfigSection);
    EXPECT_FALSE(savedConfig.contains(QStringLiteral("ui")));
    const QJsonObject savedUi =
        projectDocument(projectPath)
            .value(QString::fromLatin1(
                PortableProjectFormat::ProjectUiStateSection))
            .toObject()
            .value(QStringLiteral("display_settings"))
            .toObject();
    EXPECT_TRUE(
        savedUi.value(QStringLiteral("show_interest_points")).toBool());
    EXPECT_EQ(
        savedUi.value(QStringLiteral("feature_display"))
            .toObject()
            .value(QStringLiteral("pointSize"))
            .toInt(),
        1);
}

TEST(ProjectDataTest, CommitsPreparedSharedImagesWithoutRepeatingImageIo)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("prepared-images")));

    const QString preparedImage = QDir(ProjectPackageLayout::sharedImagesDirectory(projectPath))
                                      .filePath(QStringLiteral("hash/prepared.png"));
    writeTestFile(preparedImage, QByteArray("already-copied-image"));

    QString message;
    ASSERT_TRUE(project.addImagesFromSharedStore(
        {preparedImage, preparedImage}, 3, &message));
    EXPECT_EQ(project.getAllImages(), QStringList{QDir::cleanPath(preparedImage)});
    EXPECT_EQ(message, QStringLiteral("已跳过 4 张重复图片"));

    const QJsonObject entry = project.coreFilesMeta()
                                  .value(QStringLiteral("images"))
                                  .toArray()
                                  .first()
                                  .toObject();
    EXPECT_EQ(entry.value(QStringLiteral("type")).toString(), QStringLiteral("shared"));
    EXPECT_FALSE(entry.value(QStringLiteral("image_uuid")).toString().isEmpty());
}

TEST(ProjectDataTest, ConcurrentSharedImageImportsKeepSingleContentAddressedFile)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("concurrent-images")));

    QStringList sourcePaths;
    const QByteArray content(2 * 1024 * 1024, 'x');
    for (int index = 0; index < 8; ++index)
    {
        const QString path = QDir(dir.path()).filePath(
            QStringLiteral("source_%1.tif").arg(index));
        writeTestFile(path, content);
        sourcePaths.append(path);
    }

    QList<QFuture<QString>> futures;
    for (const QString &sourcePath : sourcePaths)
    {
        futures.append(QtConcurrent::run([projectPath, sourcePath]()
        {
            QString resourceUri;
            QString materializedPath;
            QString error;
            if (!ProjectSharedImageStore(projectPath).importImage(
                    sourcePath, &resourceUri, &materializedPath, &error))
            {
                return QStringLiteral("ERROR: %1").arg(error);
            }
            return materializedPath;
        }));
    }

    QSet<QString> materializedPaths;
    for (QFuture<QString> &future : futures)
    {
        future.waitForFinished();
        const QString path = future.result();
        const bool failed = path.startsWith(QStringLiteral("ERROR:"));
        EXPECT_FALSE(failed) << qPrintable(path);
        if (!failed)
        {
            materializedPaths.insert(QDir::cleanPath(path));
        }
    }
    EXPECT_EQ(materializedPaths.size(), 1);

    QDirIterator files(ProjectPackageLayout::sharedImagesDirectory(projectPath),
                       QDir::Files | QDir::NoDotAndDotDot,
                       QDirIterator::Subdirectories);
    int fileCount = 0;
    while (files.hasNext())
    {
        files.next();
        ++fileCount;
    }
    EXPECT_EQ(fileCount, 1);
}

TEST(ProjectDataTest, SharedImageLeaseSurvivesOldSnapshotAndGcNeedsTwoGenerations)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("shared-image-lease")));

    const QString sourcePath = QDir(dir.path()).filePath(
        QStringLiteral("source/leased-image.tif"));
    writeTestFile(sourcePath, QByteArray("leased-image-content"));

    struct ImportResult
    {
        bool success = false;
        QString resourceUri;
        QString materializedPath;
        QString errorMessage;
    };
    QSemaphore copyCompleted;
    QSemaphore allowImportTaskToFinish;
    QFuture<ImportResult> importFuture = QtConcurrent::run(
        [&]()
        {
            ImportResult result;
            result.success = ProjectSharedImageStore(projectPath).importImage(
                sourcePath,
                &result.resourceUri,
                &result.materializedPath,
                &result.errorMessage);
            copyCompleted.release();
            allowImportTaskToFinish.acquire();
            return result;
        });

    // 固定“复制和 reservation 已完成，元数据尚未发布”的窗口，让旧归档
    // 快照执行一次 GC；不依赖 sleep 或调度概率。
    copyCompleted.acquire();
    QString oldSnapshotError;
    const bool oldSnapshotSaved = project.saveProject(&oldSnapshotError);
    allowImportTaskToFinish.release();
    importFuture.waitForFinished();
    const ImportResult imported = importFuture.result();

    ASSERT_TRUE(imported.success) << qPrintable(imported.errorMessage);
    ASSERT_TRUE(oldSnapshotSaved) << qPrintable(oldSnapshotError);
    ASSERT_TRUE(QFileInfo(imported.materializedPath).isFile());

    const QString sharedImageLockPath =
        QDir(ProjectPackageLayout::dataDirectory(projectPath)).filePath(
            QStringLiteral(".shared-images.lock"));
    QLockFile competingProcessLock(sharedImageLockPath);
    competingProcessLock.setStaleLockTime(0);
    EXPECT_FALSE(competingProcessLock.tryLock(0))
        << "active reservation 必须跨进程持有共享影像同步锁";

    QString error;
    ASSERT_TRUE(project.addImagesFromSharedStore(
        {imported.materializedPath}, 0, &error)) << qPrintable(error);
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    ASSERT_TRUE(QFileInfo(imported.materializedPath).isFile());
    QLockFile afterCommitLock(sharedImageLockPath);
    afterCommitLock.setStaleLockTime(0);
    ASSERT_TRUE(afterCommitLock.tryLock(0))
        << "包含 URI 的归档提交后应释放跨进程 lease";
    afterCommitLock.unlock();

    ASSERT_TRUE(project.removeResource(imported.materializedPath));
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    EXPECT_TRUE(QFileInfo(imported.materializedPath).isFile())
        << "第一个未引用代次只能写入 tombstone";

    // 同一个 Chunk id+revision token 反复 GC 不构成新的已提交代次。
    ASSERT_TRUE(ProjectSharedImageStore(projectPath).pruneUnreferenced(&error))
        << qPrintable(error);
    ASSERT_TRUE(ProjectSharedImageStore(projectPath).pruneUnreferenced(&error))
        << qPrintable(error);
    EXPECT_TRUE(QFileInfo(imported.materializedPath).isFile())
        << "重复处理同一个 committed generation 不得提前删除";

    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    EXPECT_FALSE(QFileInfo::exists(imported.materializedPath))
        << "连续两个未引用代次后才允许删除共享实体";
}

TEST(ProjectDataTest, SharedImageGcFailureDoesNotFailCommittedSave)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("shared-image-gc-warning")));

    // 用同名目录稳定阻断 QSaveFile 状态写入，模拟可重试 GC 失败。
    const QString statePath =
        QDir(ProjectPackageLayout::dataDirectory(projectPath)).filePath(
            QStringLiteral(".shared-image-gc.json"));
    ASSERT_TRUE(QDir().mkpath(statePath));

    const QString sourcePath = QDir(dir.path()).filePath(
        QStringLiteral("source/gc-warning-image.tif"));
    writeTestFile(sourcePath, QByteArray("gc-warning-image-content"));
    QString error;
    ASSERT_TRUE(project.addImages({sourcePath}, &error)) << qPrintable(error);

    error.clear();
    EXPECT_TRUE(project.saveProject(&error)) << qPrintable(error);
    EXPECT_TRUE(error.isEmpty()) << qPrintable(error);
    EXPECT_TRUE(QFileInfo(project.getAllImages().constFirst()).isFile());
}

TEST(ProjectDataTest, InvalidTemporaryMetadataStopsSharedImageDeletion)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("invalid-temporary-gc-guard")));

    const QString sourcePath = QDir(dir.path()).filePath(
        QStringLiteral("source/invalid-temporary-image.tif"));
    writeTestFile(sourcePath, QByteArray("invalid-temporary-image-content"));
    QString error;
    ASSERT_TRUE(project.addImages({sourcePath}, &error)) << qPrintable(error);
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    const QString managedImagePath = project.getAllImages().constFirst();

    ASSERT_TRUE(project.removeResource(managedImagePath));
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    ASSERT_TRUE(QFileInfo::exists(managedImagePath));

    writeTestFile(ProjectIO::tempFilesPath(projectPath), QByteArray("{broken"));
    QJsonObject config = chunkSection(
        projectPath, PortableProjectFormat::ProjectConfigSection);
    config[QStringLiteral("gc_guard_generation")] = 2;
    ASSERT_TRUE(ProjectChunkStore(projectPath).writeChunkSections(
        project.activeChunkDirectory(),
        {{QString::fromLatin1(
              PortableProjectFormat::ProjectConfigSection),
          config}},
        &error)) << qPrintable(error);

    error.clear();
    EXPECT_FALSE(
        ProjectSharedImageStore(projectPath).pruneUnreferenced(&error));
    EXPECT_FALSE(error.isEmpty());
    EXPECT_TRUE(QFileInfo::exists(managedImagePath))
        << "无效临时恢复元数据存在时必须保守停止 GC";
}

TEST(ProjectDataTest, DuplicateSharedImageImportReleasesItsReservation)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("duplicate-shared-image")));

    const QString sourcePath = QDir(dir.path()).filePath(
        QStringLiteral("source/duplicate-image.tif"));
    writeTestFile(sourcePath, QByteArray("duplicate-image-content"));
    QString error;
    ASSERT_TRUE(project.addImages({sourcePath}, &error)) << qPrintable(error);
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

    error.clear();
    ASSERT_TRUE(project.addImages({sourcePath}, &error)) << qPrintable(error);
    EXPECT_EQ(error, QStringLiteral("已跳过 1 张重复图片"));

    const QString lockPath =
        QDir(ProjectPackageLayout::dataDirectory(projectPath)).filePath(
            QStringLiteral(".shared-images.lock"));
    QLockFile competingProcessLock(lockPath);
    competingProcessLock.setStaleLockTime(0);
    EXPECT_TRUE(competingProcessLock.tryLock(0))
        << "被跳过的重复导入不得遗留 active reservation";
}

TEST(ProjectDataTest,
     DuplicateBeforeLaterFailureDoesNotReleasePendingReservation)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("duplicate-before-failure")));

    const QString sourcePath = QDir(dir.path()).filePath(
        QStringLiteral("source/pending-image.tif"));
    writeTestFile(sourcePath, QByteArray("pending-image-content"));
    QString error;
    ASSERT_TRUE(project.addImages({sourcePath}, &error)) << qPrintable(error);

    error.clear();
    const QString missingPath = QDir(dir.path()).filePath(
        QStringLiteral("source/missing-image.tif"));
    EXPECT_FALSE(project.addImages({sourcePath, missingPath}, &error));
    EXPECT_FALSE(error.isEmpty());

    const QString lockPath =
        QDir(ProjectPackageLayout::dataDirectory(projectPath)).filePath(
            QStringLiteral(".shared-images.lock"));
    QLockFile competingProcessLock(lockPath);
    competingProcessLock.setStaleLockTime(0);
    EXPECT_FALSE(competingProcessLock.tryLock(0))
        << "后续导入失败不得释放首轮尚未提交的 reservation";

    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    QLockFile afterCommitLock(lockPath);
    afterCommitLock.setStaleLockTime(0);
    EXPECT_TRUE(afterCommitLock.tryLock(0));
}

TEST(ProjectDataTest,
     CloseDrainsLatestSnapshotAndReleasesSharedImageLeaseWithoutEvents)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString sourcePath = QDir(dir.path()).filePath(
        QStringLiteral("source/close-barrier-image.tif"));
    writeTestFile(sourcePath, QByteArray("close-barrier-image-content"));
    QString temporaryFilesPath;
    QString managedImagePath;
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(
            projectPath, QStringLiteral("close-barrier")));
        QJsonObject metadata = project.metadata();
        metadata[QStringLiteral("close_barrier_marker")] =
            QStringLiteral("durable-before-unlock");
        project.updateMetadata(metadata, true);

        QString error;
        ASSERT_TRUE(project.addImages({sourcePath}, &error))
            << qPrintable(error);
        const QStringList images = project.getAllImages();
        ASSERT_EQ(images.size(), 1);
        managedImagePath = images.first();
        temporaryFilesPath = ProjectIO::tempFilesPath(projectPath);

        ASSERT_TRUE(project.closeProject());
        EXPECT_FALSE(project.hasProject());
    }

    QFile temporaryFiles(temporaryFilesPath);
    ASSERT_TRUE(temporaryFiles.open(QIODevice::ReadOnly));
    const QJsonDocument document = QJsonDocument::fromJson(
        temporaryFiles.readAll());
    ASSERT_TRUE(document.isObject());
    EXPECT_EQ(document.object().value(
                  QStringLiteral("close_barrier_marker")).toString(),
              QStringLiteral("durable-before-unlock"));
    EXPECT_TRUE(QFileInfo::exists(managedImagePath));

    const QString lockPath =
        QDir(ProjectPackageLayout::dataDirectory(projectPath)).filePath(
            QStringLiteral(".shared-images.lock"));
    QLockFile competingProcessLock(lockPath);
    competingProcessLock.setStaleLockTime(0);
    EXPECT_TRUE(competingProcessLock.tryLock(0))
        << "close must release this session's shared-image lease";
}

TEST(ProjectDataTest,
     DestructorDrainsLatestSnapshotAndReleasesSharedImageLeaseWithoutEvents)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString sourcePath = QDir(dir.path()).filePath(
        QStringLiteral("source/destructor-barrier-image.tif"));
    writeTestFile(sourcePath, QByteArray("destructor-barrier-image-content"));
    QString temporaryFilesPath;
    QString managedImagePath;
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(
            projectPath, QStringLiteral("destructor-barrier")));
        QJsonObject metadata = project.metadata();
        metadata[QStringLiteral("destructor_barrier_marker")] =
            QStringLiteral("durable-before-destruction");
        project.updateMetadata(metadata, true);

        QString error;
        ASSERT_TRUE(project.addImages({sourcePath}, &error))
            << qPrintable(error);
        managedImagePath = project.getAllImages().constFirst();
        temporaryFilesPath = ProjectIO::tempFilesPath(projectPath);
    }

    QFile temporaryFiles(temporaryFilesPath);
    ASSERT_TRUE(temporaryFiles.open(QIODevice::ReadOnly));
    const QJsonDocument document = QJsonDocument::fromJson(
        temporaryFiles.readAll());
    ASSERT_TRUE(document.isObject());
    EXPECT_EQ(document.object().value(
                  QStringLiteral("destructor_barrier_marker")).toString(),
              QStringLiteral("durable-before-destruction"));
    EXPECT_TRUE(QFileInfo::exists(managedImagePath));

    const QString lockPath =
        QDir(ProjectPackageLayout::dataDirectory(projectPath)).filePath(
            QStringLiteral(".shared-images.lock"));
    QLockFile competingProcessLock(lockPath);
    competingProcessLock.setStaleLockTime(0);
    EXPECT_TRUE(competingProcessLock.tryLock(0))
        << "destructor must release this session's shared-image lease";
}

TEST(ProjectDataTest,
     CloseThenImmediateReopenIgnoresLatePersistenceCallback)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("immediate-reopen")));
    QJsonObject metadata = project.metadata();
    metadata[QStringLiteral("reopen_marker")] =
        QStringLiteral("latest-session");
    project.updateMetadata(metadata, true);

    QString error;
    ASSERT_TRUE(project.closeProject(&error)) << qPrintable(error);
    ASSERT_TRUE(project.openProject(projectPath, &error)) << qPrintable(error);

    QCoreApplication::processEvents();

    EXPECT_EQ(QDir::cleanPath(project.currentProjectPath()),
              QDir::cleanPath(projectPath));
    EXPECT_EQ(project.metadata().value(
                  QStringLiteral("reopen_marker")).toString(),
              QStringLiteral("latest-session"));
    EXPECT_FALSE(project.isDirty())
        << "上一会话的 queued completion 不得污染重开后的会话";
}

TEST(ProjectDataTest, FailedCloseDoesNotReplaceActiveProjectSession)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString sourceProjectPath = QDir(dir.path()).filePath(
        QStringLiteral("source.plascan"));
    const QString targetProjectPath = QDir(dir.path()).filePath(
        QStringLiteral("target.plascan"));
    {
        ProjectData targetCreator;
        ASSERT_TRUE(targetCreator.createProject(
            targetProjectPath, QStringLiteral("target")));
        ASSERT_TRUE(targetCreator.closeProject());
    }

    ProjectData project;
    ASSERT_TRUE(project.createProject(
        sourceProjectPath, QStringLiteral("source")));
    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    QCoreApplication::processEvents();

    QJsonObject metadata = project.metadata();
    metadata[QStringLiteral("unsaved_close_marker")] = true;
    project.updateMetadata(metadata, true);

    const QString chunkArchivePath =
        defaultChunkArchivePath(sourceProjectPath);
    const QString archiveBackupPath =
        chunkArchivePath + QStringLiteral(".close-test-backup");
    ASSERT_TRUE(QFile::rename(chunkArchivePath, archiveBackupPath));
    ASSERT_TRUE(QDir().mkpath(chunkArchivePath));
    const QString temporaryFilesPath =
        ProjectIO::tempFilesPath(sourceProjectPath);
    QFile::remove(temporaryFilesPath);
    ASSERT_TRUE(QDir().mkpath(temporaryFilesPath));

    const QString failedCreatePath = QDir(dir.path()).filePath(
        QStringLiteral("must-not-replace.plascan"));
    EXPECT_FALSE(project.createProject(
        failedCreatePath, QStringLiteral("must-not-replace")));
    EXPECT_EQ(QDir::cleanPath(project.currentProjectPath()),
              QDir::cleanPath(sourceProjectPath));
    EXPECT_FALSE(QFileInfo::exists(failedCreatePath));
    EXPECT_FALSE(QFileInfo::exists(
        ProjectPackageLayout::dataDirectory(failedCreatePath)));

    error.clear();
    EXPECT_FALSE(project.openProject(targetProjectPath, &error));
    EXPECT_FALSE(error.isEmpty());
    EXPECT_EQ(QDir::cleanPath(project.currentProjectPath()),
              QDir::cleanPath(sourceProjectPath));
    EXPECT_TRUE(project.metadata().value(
        QStringLiteral("unsaved_close_marker")).toBool());

    {
        ProjectData targetReopened;
        ASSERT_TRUE(targetReopened.openProject(targetProjectPath, &error))
            << qPrintable(error);
        ASSERT_TRUE(targetReopened.closeProject(&error)) << qPrintable(error);
    }

    ASSERT_TRUE(QDir(temporaryFilesPath).removeRecursively());
    ASSERT_TRUE(QDir(chunkArchivePath).removeRecursively());
    ASSERT_TRUE(QFile::rename(archiveBackupPath, chunkArchivePath));

    ProjectData competingSourceSession;
    error.clear();
    EXPECT_FALSE(competingSourceSession.openProject(
        sourceProjectPath, &error));
    EXPECT_FALSE(error.isEmpty());

    ASSERT_TRUE(project.closeProject(&error)) << qPrintable(error);
}

TEST(ProjectDataTest, SplitProjectReopensAllWorkflowAssetsAfterPairMoves)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString sourceDir = QDir(dir.path()).filePath(QStringLiteral("原始电脑"));
    const QString movedDir = QDir(dir.path()).filePath(QStringLiteral("新电脑"));
    ASSERT_TRUE(QDir().mkpath(sourceDir));
    ASSERT_TRUE(QDir().mkpath(movedDir));

    const QString projectPath =
        QDir(sourceDir).filePath(QStringLiteral("月球工程.plascan"));
    const QString externalImage =
        QDir(sourceDir).filePath(QStringLiteral("外部影像/影像一.tif"));
    const QByteArray imageContent("split-project-image-content");
    writeTestFile(externalImage, imageContent);

    QString runtimeRoot;
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(
            projectPath, QStringLiteral("分体工程全流程")));
        ASSERT_TRUE(project.addImages({externalImage}));
        runtimeRoot = ProjectIO::projectRootFromPlascan(projectPath);
        ASSERT_FALSE(runtimeRoot.isEmpty());
        EXPECT_NE(QDir::cleanPath(runtimeRoot), QDir::cleanPath(sourceDir));

        const QString mask =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/masks/影像一_mask.png"));
        const QString match =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/image_matches/影像一.pimatch"));
        const QString tracks =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/tie_points/tracks.bin"));
        const QString depth =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/mvs/depth/影像一.exr"));
        const QString cloud =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/mvs/dense_cloud.ply"));
        const QString model =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/models/model.ply"));
        const QString texture =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/models/texture.png"));
        const QString dem =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/terrain/dem.tif"));
        const QString dom =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/terrain/dom.tif"));
        const QString report =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/reports/quality.json"));
        const QString reference =
            QDir(runtimeRoot).filePath(QStringLiteral("assets/reference/lidar.ply"));

        const QList<QPair<QString, QByteArray>> files{
            {mask, QByteArray("mask")},
            {match, QByteArray("match")},
            {tracks, QByteArray("tracks")},
            {depth, QByteArray("depth")},
            {cloud, QByteArray("cloud")},
            {model, QByteArray("model")},
            {texture, QByteArray("texture")},
            {dem, QByteArray("dem")},
            {dom, QByteArray("dom")},
            {report, QByteArray("report")},
            {reference, QByteArray("reference")}
        };
        for (const auto &file : files)
        {
            writeTestFile(file.first, file.second);
        }

        QJsonObject metadata = project.metadata();
        QJsonArray images = metadata.value(QStringLiteral("images")).toArray();
        QJsonObject imageRecord = images.at(0).toObject();
        imageRecord[QStringLiteral("mask_path")] = mask;
        imageRecord[QStringLiteral("camera")] = QJsonObject{
            {QStringLiteral("aligned"), true},
            {QStringLiteral("fx"), 1200.0},
            {QStringLiteral("pose"),
             QJsonArray{1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0}}
        };
        images[0] = imageRecord;
        metadata[QStringLiteral("images")] = images;
        metadata[QStringLiteral("image_match_results")] = QJsonArray{
            QJsonObject{
                {QStringLiteral("image"), externalImage},
                {QStringLiteral("output"), match},
                {QStringLiteral("track_file"), tracks}
            }
        };
        metadata[QStringLiteral("aerial_triangulation_results")] = QJsonArray{
            QJsonObject{
                {QStringLiteral("camera_count"), 1},
                {QStringLiteral("coordinate_system"), QStringLiteral("local")}
            }
        };
        metadata[QStringLiteral("depth_map_results")] = QJsonArray{
            QJsonObject{{QStringLiteral("depth_path"), depth}}
        };
        metadata[QStringLiteral("dense_cloud_results")] = QJsonArray{
            QJsonObject{{QStringLiteral("cloud_path"), cloud}}
        };
        metadata[QStringLiteral("model_results")] = QJsonArray{
            QJsonObject{
                {QStringLiteral("model_path"), model},
                {QStringLiteral("texture_path"), texture}
            }
        };
        metadata[QStringLiteral("dem_results")] = QJsonArray{
            QJsonObject{{QStringLiteral("output_path"), dem}}
        };
        metadata[QStringLiteral("ortho_results")] = QJsonArray{
            QJsonObject{{QStringLiteral("output_path"), dom}}
        };
        metadata[QStringLiteral("report_results")] = QJsonArray{
            QJsonObject{{QStringLiteral("path"), report}}
        };
        metadata[QStringLiteral("reference_datasets")] = QJsonArray{
            QJsonObject{{QStringLiteral("path"), reference}}
        };
        project.updateMetadata(metadata, true);

        QString error;
        ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

        const QJsonObject archivedCore = chunkSection(
            projectPath, PortableProjectFormat::ProjectFilesSection);
        const QString archivedImage = archivedCore
            .value(QStringLiteral("images"))
            .toArray()
            .at(0)
            .toObject()
            .value(QStringLiteral("path"))
            .toString();
        EXPECT_TRUE(archivedImage.startsWith(QStringLiteral("plascan:///shared/")));

        const QJsonObject archivedResults = chunkSection(
            projectPath, PortableProjectFormat::ProjectResultsSection);
        EXPECT_TRUE(resultPath(
                        archivedResults,
                        QStringLiteral("model_results"),
                        QStringLiteral("model_path"))
                        .startsWith(QStringLiteral("plascan:///chunk/")));
        EXPECT_TRUE(resultPath(
                        archivedResults,
                        QStringLiteral("report_results"),
                        QStringLiteral("path"))
                        .startsWith(QStringLiteral("plascan:///chunk/")));

        const QJsonObject indexObject = chunkSection(
            projectPath, PortableProjectFormat::ResourceIndexSection);
        QString indexError;
        const ProjectResourceIndex index =
            ProjectResourceIndex::fromJson(indexObject, &indexError);
        ASSERT_TRUE(indexError.isEmpty()) << qPrintable(indexError);
        EXPECT_GE(index.size(), 12);

        project.closeProject();
    }

    ASSERT_TRUE(QDir(QFileInfo(externalImage).absolutePath()).removeRecursively());

    const QString movedProject =
        QDir(movedDir).filePath(QStringLiteral("迁移后的月球工程.plascan"));
    moveProjectPair(projectPath, movedProject);
    EXPECT_FALSE(QFileInfo::exists(projectPath));

    ProjectData reopened;
    QString error;
    ASSERT_TRUE(reopened.openProject(movedProject, &error)) << qPrintable(error);
    const ProjectResultsSnapshot resultsSnapshot =
        ProjectData::loadProjectResultsSnapshot(movedProject);
    ASSERT_TRUE(resultsSnapshot.success)
        << qPrintable(resultsSnapshot.errorMessage);
    ASSERT_TRUE(reopened.applyResultsSnapshot(resultsSnapshot, &error))
        << qPrintable(error);
    const QStringList images = reopened.getAllImages();
    ASSERT_EQ(images.size(), 1);
    EXPECT_EQ(readTestFile(images.constFirst()), imageContent);

    const QJsonObject restored = reopened.metadata();
    const QJsonObject restoredImage = restored.value(QStringLiteral("images"))
        .toArray().at(0).toObject();
    EXPECT_TRUE(
        restoredImage.value(QStringLiteral("camera")).toObject()
            .value(QStringLiteral("aligned")).toBool());
    const QList<QPair<QString, QByteArray>> restoredFiles{
        {restoredImage.value(QStringLiteral("mask_path")).toString(),
         QByteArray("mask")},
        {resultPath(restored, QStringLiteral("image_match_results"), QStringLiteral("output")),
         QByteArray("match")},
        {restored.value(QStringLiteral("image_match_results")).toArray().at(0)
             .toObject().value(QStringLiteral("track_file")).toString(),
         QByteArray("tracks")},
        {resultPath(restored, QStringLiteral("depth_map_results"), QStringLiteral("depth_path")),
         QByteArray("depth")},
        {resultPath(restored, QStringLiteral("dense_cloud_results"), QStringLiteral("cloud_path")),
         QByteArray("cloud")},
        {resultPath(restored, QStringLiteral("model_results"), QStringLiteral("model_path")),
         QByteArray("model")},
        {restored.value(QStringLiteral("model_results")).toArray().at(0)
             .toObject().value(QStringLiteral("texture_path")).toString(),
         QByteArray("texture")},
        {resultPath(restored, QStringLiteral("dem_results"), QStringLiteral("output_path")),
         QByteArray("dem")},
        {resultPath(restored, QStringLiteral("ortho_results"), QStringLiteral("output_path")),
         QByteArray("dom")},
        {resultPath(restored, QStringLiteral("report_results"), QStringLiteral("path")),
         QByteArray("report")},
        {resultPath(restored, QStringLiteral("reference_datasets"), QStringLiteral("path")),
         QByteArray("reference")}
    };
    for (const auto &file : restoredFiles)
    {
        EXPECT_TRUE(QFileInfo(file.first).isFile()) << qPrintable(file.first);
        EXPECT_EQ(readTestFile(file.first), file.second) << qPrintable(file.first);
    }
}

TEST(ProjectDataTest, RejectsLegacyMonolithicProjectWithoutChangingIt)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath =
        QDir(dir.path()).filePath(QStringLiteral("旧工程.plascan"));
    const QJsonObject manifest{
        {QStringLiteral("format_version"), QStringLiteral("1.0")},
        {QStringLiteral("type"), QStringLiteral("plascan_project")}
    };
    QString error;
    ASSERT_TRUE(PlascanArchive::createArchive(
        projectPath,
        {
            qMakePair(
                QStringLiteral("manifest.json"),
                QJsonDocument(manifest).toJson(QJsonDocument::Compact)),
            qMakePair(
                QStringLiteral("project_files.json"),
                QJsonDocument(QJsonObject{})
                    .toJson(QJsonDocument::Compact))
        },
        &error)) << qPrintable(error);
    const QByteArray original = readTestFile(projectPath);

    ProjectData project;
    EXPECT_FALSE(project.openProject(projectPath, &error));
    EXPECT_TRUE(error.contains(QStringLiteral("仅支持版本")));
    EXPECT_EQ(readTestFile(projectPath), original);
    EXPECT_FALSE(QFileInfo::exists(
        ProjectPackageLayout::dataDirectory(projectPath)));
}

TEST(ProjectDataTest, RemovingImportedImageDeletesAfterTwoCommittedGenerations)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString imagePath =
        QDir(dir.path()).filePath(QStringLiteral("待删除影像.tif"));
    writeTestFile(imagePath, QByteArray("delete-me"));

    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("清理资源")));
    ASSERT_TRUE(project.addImages({imagePath}));
    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

    const QJsonObject archivedCore = chunkSection(
        projectPath, PortableProjectFormat::ProjectFilesSection);
    const QString archivedUri = archivedCore.value(QStringLiteral("images"))
        .toArray().at(0).toObject()
        .value(QStringLiteral("path")).toString();
    const QString archivedEntry =
        PortableProjectFormat::entryPathFromResourceUri(archivedUri);
    ASSERT_FALSE(archivedEntry.isEmpty());

    const QStringList materializedImages = project.getAllImages();
    ASSERT_EQ(materializedImages.size(), 1);
    ASSERT_TRUE(QFileInfo::exists(materializedImages.at(0)));
    ASSERT_TRUE(project.removeResource(materializedImages.at(0)));
    EXPECT_TRUE(QFileInfo::exists(materializedImages.at(0)));
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    EXPECT_TRUE(QFileInfo::exists(materializedImages.at(0)))
        << "首个未引用代次只登记 tombstone";

    {
        PlascanArchive archive(
            defaultChunkArchivePath(projectPath),
            PlascanArchivePathType::DirectArchive);
        ASSERT_TRUE(archive.isValid());
        EXPECT_FALSE(archive.containsEntry(archivedEntry));

        const QJsonObject indexObject = chunkSection(
            projectPath, PortableProjectFormat::ResourceIndexSection);
        QString indexError;
        const ProjectResourceIndex index =
            ProjectResourceIndex::fromJson(indexObject, &indexError);
        ASSERT_TRUE(indexError.isEmpty()) << qPrintable(indexError);
        EXPECT_TRUE(index.isEmpty());
    }

    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    EXPECT_FALSE(QFileInfo::exists(materializedImages.at(0)));
}

TEST(ProjectDataTest, PackResourcePersistsFilesAndDirectoriesInsideProject)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString sourceFile =
        QDir(dir.path()).filePath(QStringLiteral("资料/说明.txt"));
    const QString sourceDir =
        QDir(dir.path()).filePath(QStringLiteral("资料/控制点"));
    writeTestFile(sourceFile, QByteArray("document"));
    writeTestFile(
        QDir(sourceDir).filePath(QStringLiteral("points.csv")),
        QByteArray("x,y\n1,2\n"));
    writeTestFile(
        QDir(sourceDir).filePath(QStringLiteral("meta.json")),
        QByteArray("{\"kind\":\"control\"}"));

    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("资源打包")));
        ASSERT_TRUE(project.packResource(sourceFile));
        ASSERT_TRUE(project.packResource(sourceDir));
        QString error;
        ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
        const QJsonArray archivedPacked =
            chunkSection(
                projectPath, PortableProjectFormat::ProjectFilesSection)
                .value(QStringLiteral("packed_resources"))
                .toArray();
        ASSERT_EQ(archivedPacked.size(), 2);
        for (const QJsonValue &value : archivedPacked)
        {
            EXPECT_TRUE(
                value.toObject().value(QStringLiteral("path")).toString()
                    .startsWith(QStringLiteral("plascan:///chunk/")));
        }
        project.closeProject();
    }

    ASSERT_TRUE(QDir(QDir(dir.path()).filePath(QStringLiteral("资料")))
                    .removeRecursively());

    ProjectData reopened;
    QString error;
    ASSERT_TRUE(reopened.openProject(projectPath, &error)) << qPrintable(error);
    const QJsonArray packed = reopened.coreFilesMeta()
        .value(QStringLiteral("packed_resources")).toArray();
    ASSERT_EQ(packed.size(), 2);

    QString restoredFile;
    QString restoredDirectory;
    for (const QJsonValue &value : packed)
    {
        const QJsonObject record = value.toObject();
        if (record.value(QStringLiteral("resource_type")).toString()
            == QStringLiteral("directory"))
        {
            restoredDirectory =
                record.value(QStringLiteral("path")).toString();
        }
        else
        {
            restoredFile = record.value(QStringLiteral("path")).toString();
        }
    }
    EXPECT_EQ(readTestFile(restoredFile), QByteArray("document"));
    EXPECT_EQ(
        readTestFile(QDir(restoredDirectory).filePath(QStringLiteral("points.csv"))),
        QByteArray("x,y\n1,2\n"));
    EXPECT_EQ(
        readTestFile(QDir(restoredDirectory).filePath(QStringLiteral("meta.json"))),
        QByteArray("{\"kind\":\"control\"}"));
}

TEST(ProjectDataTest, MissingIndexedChunkEntryFailsWithResourceError)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString imagePath =
        QDir(dir.path()).filePath(QStringLiteral("缺失资源.tif"));
    writeTestFile(imagePath, QByteArray("resource"));

    QString entry;
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("损坏工程")));
        ASSERT_TRUE(project.addImages({imagePath}));
        QString error;
        ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
        const QString uri = chunkSection(
            projectPath, PortableProjectFormat::ProjectFilesSection)
            .value(QStringLiteral("images")).toArray().at(0).toObject()
            .value(QStringLiteral("path")).toString();
        entry = PortableProjectFormat::entryPathFromResourceUri(uri);
        ASSERT_FALSE(entry.isEmpty());
        project.closeProject();
    }

    const QString missingPhysicalPath =
        chunkPhysicalPath(projectPath, entry);
    ASSERT_TRUE(QFile::remove(missingPhysicalPath))
        << qPrintable(missingPhysicalPath);
    PlascanArchive archive(
        defaultChunkArchivePath(projectPath),
        PlascanArchivePathType::DirectArchive);
    QString error;
    ASSERT_TRUE(archive.updateFileEntries(
        {}, {entry}, PlascanArchiveCompression::Store, &error))
        << qPrintable(error);

    ProjectData reopened;
    EXPECT_FALSE(reopened.openProject(projectPath, &error));
    EXPECT_FALSE(error.isEmpty());
    EXPECT_TRUE(
        error.contains(QStringLiteral("资源"))
        || error.contains(QStringLiteral("归档"))
        || error.contains(QStringLiteral("提取")))
        << qPrintable(error);
}

TEST(ProjectDataTest, SavesChangedArtifactIntoActiveChunkWithoutLegacyWorkspacePath)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("当前 Chunk 保存")));

    const int initialChunkDirectory = project.activeChunkDirectory();
    QString error;
    QString activeChunkId;
    ASSERT_TRUE(project.createChunk(
        QStringLiteral("第二处理区"), &activeChunkId, &error))
        << qPrintable(error);
    const int chunkDirectory = project.activeChunkDirectory();
    ASSERT_NE(chunkDirectory, initialChunkDirectory);
    const QString artifactPath = QDir(
        ProjectPackageLayout::chunkDirectory(
            projectPath, chunkDirectory))
        .filePath(QStringLiteral(
            "assets/image_matches/a.pimatch"));
    writeTestFile(artifactPath, QByteArray("first-version"));

    QJsonObject metadata = project.metadata();
    metadata[QStringLiteral("image_match_results")] = QJsonArray{
        QJsonObject{
            {QStringLiteral("image"), QStringLiteral("a.tif")},
            {QStringLiteral("output"), artifactPath},
            {QStringLiteral("valid_match_count"), 1}
        }
    };
    project.updateMetadata(metadata, true);

    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    const QJsonObject firstDocument =
        chunkDocument(projectPath, chunkDirectory);
    const QString firstUri = firstDocument
        .value(QString::fromLatin1(
            PortableProjectFormat::ProjectResultsSection))
        .toObject()
        .value(QStringLiteral("image_match_results"))
        .toArray()
        .first()
        .toObject()
        .value(QStringLiteral("output"))
        .toString();
    EXPECT_EQ(
        firstUri,
        QStringLiteral(
            "plascan:///chunk/assets/image_matches/a.pimatch"));

    writeTestFile(artifactPath, QByteArray("newer-version"));
    ProjectWorkspaceStore(
        projectPath, chunkDirectory).releaseRuntime();
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

    const QByteArray storedJson =
        QJsonDocument(chunkDocument(projectPath, chunkDirectory))
            .toJson(QJsonDocument::Compact);
    EXPECT_FALSE(storedJson.contains("plascan:///workspace/"));
    EXPECT_TRUE(storedJson.contains("plascan:///chunk/"));

    const QByteArray initialChunkJson =
        QJsonDocument(chunkDocument(
            projectPath, initialChunkDirectory))
            .toJson(QJsonDocument::Compact);
    EXPECT_FALSE(initialChunkJson.contains("a.pimatch"));
}

TEST(PlascanArchiveTest, StreamsUnicodeFileAndRejectsUnsafeEntry)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("流式归档")));
    }

    QByteArray expected(3 * 1024 * 1024, Qt::Uninitialized);
    for (int index = 0; index < expected.size(); ++index)
    {
        expected[index] = static_cast<char>((index * 31) % 251);
    }

    const QString sourcePath =
        QDir(dir.path()).filePath(QStringLiteral("月球资源 数据.bin"));
    QFile source(sourcePath);
    ASSERT_TRUE(source.open(QIODevice::WriteOnly));
    ASSERT_EQ(source.write(expected), expected.size());
    source.close();

    PlascanArchive archive(
        defaultChunkArchivePath(projectPath),
        PlascanArchivePathType::DirectArchive);
    ASSERT_TRUE(archive.isValid());
    QString error;
    EXPECT_FALSE(archive.writeFileEntry(
        QStringLiteral("../outside.bin"),
        sourcePath,
        PlascanArchiveCompression::Store,
        &error));
    EXPECT_FALSE(error.isEmpty());

    const QString entry =
        QStringLiteral("resources/images/image-1/月球资源 数据.bin");
    ASSERT_TRUE(archive.writeFileEntry(
        entry, sourcePath, PlascanArchiveCompression::Store, &error))
        << qPrintable(error);

    const QString extractedPath =
        QDir(dir.path()).filePath(QStringLiteral("提取/恢复资源.bin"));
    ASSERT_TRUE(archive.extractEntryToFile(entry, extractedPath, &error))
        << qPrintable(error);

    QFile extracted(extractedPath);
    ASSERT_TRUE(extracted.open(QIODevice::ReadOnly));
    EXPECT_EQ(extracted.readAll(), expected);
}

TEST(ProjectResourceStoreTest, ProjectFileRemainsUsableAfterSourceRemovalAndMove)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString firstRoot =
        QDir(dir.path()).filePath(QStringLiteral("电脑A"));
    const QString secondRoot =
        QDir(dir.path()).filePath(QStringLiteral("电脑B"));
    ASSERT_TRUE(QDir().mkpath(firstRoot));
    ASSERT_TRUE(QDir().mkpath(secondRoot));

    const QString projectPath =
        QDir(firstRoot).filePath(QStringLiteral("月球.plascan"));
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("月球")));
    }

    const QByteArray expected("portable-project-resource\n");
    const QString sourcePath =
        QDir(firstRoot).filePath(QStringLiteral("原始影像.tif"));
    QFile source(sourcePath);
    ASSERT_TRUE(source.open(QIODevice::WriteOnly));
    ASSERT_EQ(source.write(expected), expected.size());
    source.close();

    ProjectResourceImportOptions options;
    options.kind = QStringLiteral("images");
    options.resourceId = QStringLiteral("image-001");
    options.displayName = QStringLiteral("原始影像.tif");
    options.mediaType = QStringLiteral("image/tiff");
    options.compression = PlascanArchiveCompression::Store;

    ProjectResourceRef imported;
    QString error;
    ProjectResourceStore store(projectPath);
    ASSERT_TRUE(store.importFile(sourcePath, options, &imported, &error))
        << qPrintable(error);
    ASSERT_TRUE(QFile::remove(sourcePath));

    const QString movedProjectPath =
        QDir(secondRoot).filePath(QStringLiteral("已移动项目.plascan"));
    moveProjectPair(projectPath, movedProjectPath);

    const QString cacheRoot =
        QDir(secondRoot).filePath(QStringLiteral("runtime-cache"));
    ProjectResourceResolver resolver(movedProjectPath);
    QString materializedPath;
    ASSERT_TRUE(resolver.materialize(
        imported.id, &materializedPath, &error, cacheRoot))
        << qPrintable(error);

    QFile materialized(materializedPath);
    ASSERT_TRUE(materialized.open(QIODevice::ReadOnly));
    EXPECT_EQ(materialized.readAll(), expected);
    materialized.close();

    QFile corrupted(materializedPath);
    ASSERT_TRUE(corrupted.open(QIODevice::WriteOnly | QIODevice::Truncate));
    ASSERT_EQ(corrupted.write("corrupt"), 7);
    corrupted.close();

    QString restoredPath;
    ASSERT_TRUE(resolver.materialize(
        imported.id, &restoredPath, &error, cacheRoot))
        << qPrintable(error);
    QFile restored(restoredPath);
    ASSERT_TRUE(restored.open(QIODevice::ReadOnly));
    EXPECT_EQ(restored.readAll(), expected);
    EXPECT_FALSE(QFileInfo::exists(sourcePath));
}

TEST(ProjectDataTest, AssignsStableUuidToImportedImages)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString imagePath = QDir(dir.path()).filePath(QStringLiteral("a.jpg"));
    writeTestFile(imagePath, QByteArray("image-identity"));
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("image_identity")));
    ASSERT_TRUE(project.addImages({imagePath}));

    const QJsonArray initialImages = project.coreFilesMeta().value(QStringLiteral("images")).toArray();
    ASSERT_EQ(initialImages.size(), 1);
    const QString firstId = initialImages[0].toObject().value(QStringLiteral("image_uuid")).toString();
    ASSERT_FALSE(firstId.isEmpty());

    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    project.closeProject();

    ProjectData reopened;
    ASSERT_TRUE(reopened.openProject(projectPath, &error)) << qPrintable(error);
    const QJsonArray reopenedImages = reopened.coreFilesMeta().value(QStringLiteral("images")).toArray();
    ASSERT_EQ(reopenedImages.size(), 1);
    EXPECT_EQ(reopenedImages[0].toObject().value(QStringLiteral("image_uuid")).toString(), firstId);
}

TEST(ProjectDataTest, OpeningLegacyImagesAssignsUuidWithoutDirtyingProject)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString imagePath = QDir(dir.path()).filePath(
        QStringLiteral("legacy.jpg"));
    writeTestFile(imagePath, QByteArray("legacy-image"));

    ProjectData project;
    ASSERT_TRUE(project.createProject(
        projectPath, QStringLiteral("legacy_image_identity")));
    ASSERT_TRUE(project.addImages({imagePath}));
    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    project.closeProject();

    QJsonObject legacyCore = chunkSection(
        projectPath, PortableProjectFormat::ProjectFilesSection);
    QJsonArray legacyImages = legacyCore.value(
        QStringLiteral("images")).toArray();
    ASSERT_EQ(legacyImages.size(), 1);
    QJsonObject legacyImage = legacyImages[0].toObject();
    legacyImage.remove(QStringLiteral("image_uuid"));
    legacyImages[0] = legacyImage;
    legacyCore[QStringLiteral("images")] = legacyImages;

    ProjectChunkStore chunkStore(projectPath);
    ASSERT_TRUE(chunkStore.writeChunkSections(
        1,
        {{QString::fromLatin1(PortableProjectFormat::ProjectFilesSection),
          legacyCore}},
        &error)) << qPrintable(error);

    ProjectData reopened;
    ASSERT_TRUE(reopened.openProject(projectPath, &error))
        << qPrintable(error);
    EXPECT_FALSE(reopened.isDirty());
    const QJsonArray migratedImages = reopened.coreFilesMeta()
        .value(QStringLiteral("images"))
        .toArray();
    ASSERT_EQ(migratedImages.size(), 1);
    EXPECT_FALSE(migratedImages[0]
                     .toObject()
                     .value(QStringLiteral("image_uuid"))
                     .toString()
                     .isEmpty());
}

TEST(ProjectDataCameraTest, ReplaceImageCamerasClearsStaleAlignmentOutsideNewSolution)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString image1 = QDir(dir.path()).filePath(QStringLiteral("image_1.png"));
    const QString image2 = QDir(dir.path()).filePath(QStringLiteral("image_2.png"));
    const QString image3 = QDir(dir.path()).filePath(QStringLiteral("image_3.png"));
    writeTestFile(image1, QByteArray("image-1"));
    writeTestFile(image2, QByteArray("image-2"));
    writeTestFile(image3, QByteArray("image-3"));
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("camera_replace")));
    ASSERT_TRUE(project.addImages({image1, image2, image3}));
    const QStringList projectImages = project.getAllImages();
    ASSERT_EQ(projectImages.size(), 3);

    const QJsonObject oldCamera{{QStringLiteral("model"), QStringLiteral("pinhole")},
                                {QStringLiteral("aligned"), true}};
    int updatedCount = 0;
    QString error;
    ASSERT_TRUE(project.setImageCameras({{projectImages[0], oldCamera},
                                         {projectImages[1], oldCamera},
                                         {projectImages[2], oldCamera}},
                                        &updatedCount,
                                        &error))
        << qPrintable(error);
    ASSERT_EQ(updatedCount, 3);

    const QJsonObject newCamera{{QStringLiteral("model"), QStringLiteral("pinhole")},
                                {QStringLiteral("aligned"), true},
                                {QStringLiteral("solution"), QStringLiteral("current")}};
    int clearedCount = 0;
    ASSERT_TRUE(project.replaceImageCameras(projectImages,
                                             {{projectImages[0], newCamera},
                                              {projectImages[1], newCamera}},
                                             &updatedCount,
                                             &clearedCount,
                                             &error))
        << qPrintable(error);
    EXPECT_EQ(updatedCount, 2);
    EXPECT_EQ(clearedCount, 1);

    const QJsonArray images = project.coreFilesMeta().value(QStringLiteral("images")).toArray();
    ASSERT_EQ(images.size(), 3);
    EXPECT_EQ(images.at(0).toObject().value(QStringLiteral("camera")).toObject()
                  .value(QStringLiteral("solution")).toString(),
              QStringLiteral("current"));
    EXPECT_EQ(images.at(1).toObject().value(QStringLiteral("camera")).toObject()
                  .value(QStringLiteral("solution")).toString(),
              QStringLiteral("current"));
    EXPECT_FALSE(images.at(2).toObject().contains(QStringLiteral("camera")));
}

TEST(ProjectDataTest, SaveProjectWritesWorkflowResultsToResultsEntryOnly)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("demo")));

    QJsonObject meta = project.metadata();
    meta[QStringLiteral("depth_map_results")] =
        singleRecord(QStringLiteral("depth"), QStringLiteral("depth_png"), QStringLiteral("/tmp/depth_0.png"));
    meta[QStringLiteral("dense_cloud_results")] =
        singleRecord(QStringLiteral("dense"), QStringLiteral("dense_cloud_xyz"), QStringLiteral("/tmp/cloud.ply"));
    meta[QStringLiteral("model_results")] =
        singleRecord(QStringLiteral("model"), QStringLiteral("model_ply"), QStringLiteral("/tmp/model.ply"));
    meta[QStringLiteral("dem_results")] =
        singleRecord(QStringLiteral("dem"), QStringLiteral("dem_tif"), QStringLiteral("/tmp/dem.tif"));
    meta[QStringLiteral("ortho_results")] =
        singleRecord(QStringLiteral("ortho"), QStringLiteral("output_path"), QStringLiteral("/tmp/ortho.tif"));

    project.updateMetadata(meta, true);
    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

    const QJsonObject core = chunkSection(
        projectPath, PortableProjectFormat::ProjectFilesSection);
    const QJsonObject results = chunkSection(
        projectPath, PortableProjectFormat::ProjectResultsSection);

    for (const QString &key : QStringList{
             QStringLiteral("depth_map_results"),
             QStringLiteral("dense_cloud_results"),
             QStringLiteral("model_results"),
             QStringLiteral("dem_results"),
             QStringLiteral("ortho_results")
         })
    {
        EXPECT_FALSE(core.contains(key)) << qPrintable(key);
        ASSERT_TRUE(results.contains(key)) << qPrintable(key);
        EXPECT_EQ(results.value(key).toArray().size(), 1) << qPrintable(key);
    }
}

TEST(ProjectDataTest, UpdateMetadataPersistsResultsWithoutPriorFullMetadataLoad)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("demo")));

    QJsonObject meta = project.coreFilesMeta();
    meta[QStringLiteral("dem_results")] =
        singleRecord(QStringLiteral("dem"), QStringLiteral("dem_tif"), QStringLiteral("/tmp/dem_from_update.tif"));

    project.updateMetadata(meta, true);
    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

    const QJsonObject core = chunkSection(
        projectPath, PortableProjectFormat::ProjectFilesSection);
    const QJsonObject results = chunkSection(
        projectPath, PortableProjectFormat::ProjectResultsSection);

    EXPECT_FALSE(core.contains(QStringLiteral("dem_results")));
    ASSERT_TRUE(results.contains(QStringLiteral("dem_results")));
    EXPECT_EQ(results.value(QStringLiteral("dem_results")).toArray().size(), 1);
}

TEST(ProjectDataTest,
     FullMetadataMutationPreservesLazilyArchivedModelAndOtherResults)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    {
        ProjectData project;
        ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("demo")));
        QJsonObject metadata = project.coreFilesMeta();
        metadata[QStringLiteral("model_results")] = QJsonArray{
            QJsonObject{{QStringLiteral("model_run_id"),
                         QStringLiteral("existing-model")},
                        {QStringLiteral("model_ply"),
                         QStringLiteral("/tmp/existing-model.ply")}}
        };
        metadata[QStringLiteral("dem_results")] = singleRecord(
            QStringLiteral("existing-dem"),
            QStringLiteral("dem_tif"),
            QStringLiteral("/tmp/existing-dem.tif"));
        project.updateMetadata(metadata, true);
        QString error;
        ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);
    }

    ProjectData reopened;
    QString error;
    ASSERT_TRUE(reopened.openProject(projectPath, &error))
        << qPrintable(error);
    EXPECT_FALSE(reopened.metadata().contains(QStringLiteral("model_results")));

    QJsonObject fullMetadata = reopened.metadataIncludingResults();
    QJsonArray models = fullMetadata.value(
        QStringLiteral("model_results")).toArray();
    ASSERT_EQ(models.size(), 1);
    models.append(QJsonObject{
        {QStringLiteral("model_run_id"), QStringLiteral("new-model")},
        {QStringLiteral("model_ply"), QStringLiteral("/tmp/new-model.ply")}
    });
    fullMetadata[QStringLiteral("model_results")] = models;
    reopened.updateMetadata(fullMetadata, true);
    ASSERT_TRUE(reopened.saveProject(&error)) << qPrintable(error);

    const QJsonObject results = chunkSection(
        projectPath, PortableProjectFormat::ProjectResultsSection);
    EXPECT_EQ(results.value(QStringLiteral("model_results")).toArray().size(),
              2);
    EXPECT_EQ(results.value(QStringLiteral("dem_results")).toArray().size(),
              1);
}

TEST(ProjectDataTest, UpsertResultRecordPersistsThroughProjectDataContract)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("demo")));

    project.upsertResultRecordByPath(
        QStringLiteral("depth_map_results"),
        QStringLiteral("depth_png"),
        QJsonObject{
            {QStringLiteral("depth_png"), QStringLiteral("/tmp/depth_0.png")},
            {QStringLiteral("grid_width"), 640},
            {QStringLiteral("grid_height"), 480}
        });

    project.upsertResultRecordByPath(
        QStringLiteral("depth_map_results"),
        QStringLiteral("depth_png"),
        QJsonObject{
            {QStringLiteral("depth_png"), QStringLiteral("/tmp/depth_0.png")},
            {QStringLiteral("grid_width"), 800},
            {QStringLiteral("grid_height"), 600}
        });

    ASSERT_TRUE(project.isDirty());
    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

    const QJsonObject core = chunkSection(
        projectPath, PortableProjectFormat::ProjectFilesSection);
    const QJsonObject results = chunkSection(
        projectPath, PortableProjectFormat::ProjectResultsSection);
    ASSERT_FALSE(core.contains(QStringLiteral("depth_map_results")));

    const QJsonArray depthResults = results.value(QStringLiteral("depth_map_results")).toArray();
    ASSERT_EQ(depthResults.size(), 1);
    EXPECT_EQ(depthResults.first().toObject().value(QStringLiteral("grid_width")).toInt(), 800);
    EXPECT_EQ(depthResults.first().toObject().value(QStringLiteral("grid_height")).toInt(), 600);
}

TEST(ProjectDataTest, UpsertResultRecordKeepsSameFileNameInDifferentDirectories)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("demo")));

    ASSERT_TRUE(project.upsertResultRecordByPath(
        QStringLiteral("ortho_results"),
        QStringLiteral("output_path"),
        QJsonObject{
            {QStringLiteral("output_path"), QStringLiteral("assets/a/dom.tif")},
            {QStringLiteral("width"), 10}}));
    ASSERT_TRUE(project.upsertResultRecordByPath(
        QStringLiteral("ortho_results"),
        QStringLiteral("output_path"),
        QJsonObject{
            {QStringLiteral("output_path"), QStringLiteral("assets/b/dom.tif")},
            {QStringLiteral("width"), 20}}));

    const QJsonArray results =
        project.metadata().value(QStringLiteral("ortho_results")).toArray();
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results.at(0).toObject().value(QStringLiteral("width")).toInt(), 10);
    EXPECT_EQ(results.at(1).toObject().value(QStringLiteral("width")).toInt(), 20);
    EXPECT_EQ(
        results.at(0).toObject().value(QStringLiteral("schema_version")).toInt(),
        1);
}

TEST(ProjectDataTest, RemoveResourcesMatchesRelativeProjectPathsWithAbsoluteRequests)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString relativeImage = QStringLiteral("assets/img/1.png");
    const QString keptImage = QStringLiteral("assets/img/2.png");

    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("demo")));
    const QString projectRoot =
        ProjectIO::projectRootFromPlascan(projectPath);
    const QString absoluteRequest =
        QDir(projectRoot).filePath(relativeImage);

    QJsonObject meta = project.metadata();
    meta[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), relativeImage}},
        QJsonObject{{QStringLiteral("path"), keptImage}}
    };
    project.updateMetadata(meta, true);

    ASSERT_TRUE(project.removeResources(QStringList{absoluteRequest}));

    const QJsonArray images = project.metadata().value(QStringLiteral("images")).toArray();
    ASSERT_EQ(images.size(), 1);
    EXPECT_EQ(images.first().toObject().value(QStringLiteral("path")).toString(), keptImage);
}

TEST(ProjectDataTest, CoreOnlyMetadataUpdatePreservesLoadedResults)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("demo")));

    ASSERT_TRUE(project.upsertResultRecordByPath(
        QStringLiteral("depth_map_results"),
        QStringLiteral("depth_png"),
        QJsonObject{{QStringLiteral("depth_png"), QStringLiteral("/tmp/depth_keep.png")}}));

    QJsonObject coreOnly = project.coreFilesMeta();
    coreOnly[QStringLiteral("project_note")] = QStringLiteral("core-only update");
    project.updateMetadata(coreOnly, true);

    QString error;
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

    const QJsonObject core = chunkSection(
        projectPath, PortableProjectFormat::ProjectFilesSection);
    const QJsonObject results = chunkSection(
        projectPath, PortableProjectFormat::ProjectResultsSection);

    EXPECT_EQ(core.value(QStringLiteral("project_note")).toString(), QStringLiteral("core-only update"));
    EXPECT_FALSE(core.contains(QStringLiteral("depth_map_results")));
    ASSERT_TRUE(results.contains(QStringLiteral("depth_map_results")));
    EXPECT_EQ(results.value(QStringLiteral("depth_map_results")).toArray().size(), 1);
}
