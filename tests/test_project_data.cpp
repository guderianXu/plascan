#include <gtest/gtest.h>

#include "PlascanArchive.h"
#include "ProjectData.h"
#include "ProjectFilesManager.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

namespace {

QStringList allResultKeys()
{
    return {
        QStringLiteral("ipfind_results"),
        QStringLiteral("ipmatch_results"),
        QStringLiteral("intersection_results"),
        QStringLiteral("bundle_adjust_results"),
        QStringLiteral("aerial_triangulation_results"),
        QStringLiteral("observation_network_results"),
        QStringLiteral("depth_map_results"),
        QStringLiteral("dense_cloud_results"),
        QStringLiteral("model_results"),
        QStringLiteral("dem_results"),
        QStringLiteral("ortho_results")
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

QJsonObject archiveObject(const QString &archivePath, const QString &entryName)
{
    PlascanArchive archive(archivePath);
    EXPECT_TRUE(archive.isValid()) << qPrintable(archivePath);

    QString error;
    const QByteArray data = archive.readEntry(entryName, &error);
    EXPECT_FALSE(data.isEmpty()) << qPrintable(entryName) << ": " << qPrintable(error);

    const QJsonDocument doc = QJsonDocument::fromJson(data);
    EXPECT_TRUE(doc.isObject()) << data.constData();
    return doc.object();
}

QString tempProjectPath(QTemporaryDir &dir)
{
    return QDir(dir.path()).filePath(QStringLiteral("demo.plascan"));
}

void createLegacyProjectArchive(const QString &projectPath, const QJsonObject &filesMeta)
{
    QJsonObject manifest;
    manifest[QStringLiteral("format_version")] = QStringLiteral("1.0");
    manifest[QStringLiteral("type")] = QStringLiteral("plascan_project");

    QString error;
    ASSERT_TRUE(PlascanArchive::createArchive(projectPath,
                                             QJsonDocument(manifest).toJson(QJsonDocument::Compact),
                                             QJsonDocument(filesMeta).toJson(QJsonDocument::Compact),
                                             &error))
        << qPrintable(error);

    PlascanArchive archive(projectPath);
    ASSERT_TRUE(archive.isValid()) << qPrintable(projectPath);
    ASSERT_TRUE(archive.writeEntry(QStringLiteral("project_files.json"),
                                   QJsonDocument(filesMeta).toJson(QJsonDocument::Compact),
                                   &error))
        << qPrintable(error);

    QJsonObject config;
    config[QStringLiteral("project_name")] = QStringLiteral("legacy");
    ASSERT_TRUE(archive.writeEntry(QStringLiteral("project_config.json"),
                                   QJsonDocument(config).toJson(QJsonDocument::Compact),
                                   &error))
        << qPrintable(error);
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

    const QJsonObject core = archiveObject(projectPath, QStringLiteral("project_files.json"));
    const QJsonObject results = archiveObject(projectPath, QStringLiteral("project_results.json"));

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

    const QJsonObject core = archiveObject(projectPath, QStringLiteral("project_files.json"));
    const QJsonObject results = archiveObject(projectPath, QStringLiteral("project_results.json"));

    EXPECT_FALSE(core.contains(QStringLiteral("dem_results")));
    ASSERT_TRUE(results.contains(QStringLiteral("dem_results")));
    EXPECT_EQ(results.value(QStringLiteral("dem_results")).toArray().size(), 1);
}

TEST(ProjectDataTest, OpeningLegacyCoreWithNewWorkflowResultKeysMigratesResultsEntry)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    QJsonObject legacyMeta;
    legacyMeta[QStringLiteral("images")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/image.png")}}
    };
    legacyMeta[QStringLiteral("depth_map_results")] =
        singleRecord(QStringLiteral("depth"), QStringLiteral("depth_png"), QStringLiteral("/tmp/depth_legacy.png"));
    legacyMeta[QStringLiteral("ortho_results")] =
        singleRecord(QStringLiteral("ortho"), QStringLiteral("output_path"), QStringLiteral("/tmp/ortho_legacy.tif"));
    createLegacyProjectArchive(projectPath, legacyMeta);

    ProjectData project;
    QString error;
    ASSERT_TRUE(project.openProject(projectPath, &error)) << qPrintable(error);
    ASSERT_TRUE(project.saveProject(&error)) << qPrintable(error);

    const QJsonObject core = archiveObject(projectPath, QStringLiteral("project_files.json"));
    const QJsonObject results = archiveObject(projectPath, QStringLiteral("project_results.json"));

    EXPECT_FALSE(core.contains(QStringLiteral("depth_map_results")));
    EXPECT_FALSE(core.contains(QStringLiteral("ortho_results")));
    ASSERT_TRUE(results.contains(QStringLiteral("depth_map_results")));
    ASSERT_TRUE(results.contains(QStringLiteral("ortho_results")));
    EXPECT_EQ(results.value(QStringLiteral("depth_map_results")).toArray().size(), 1);
    EXPECT_EQ(results.value(QStringLiteral("ortho_results")).toArray().size(), 1);
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

    const QJsonObject core = archiveObject(projectPath, QStringLiteral("project_files.json"));
    const QJsonObject results = archiveObject(projectPath, QStringLiteral("project_results.json"));
    ASSERT_FALSE(core.contains(QStringLiteral("depth_map_results")));

    const QJsonArray depthResults = results.value(QStringLiteral("depth_map_results")).toArray();
    ASSERT_EQ(depthResults.size(), 1);
    EXPECT_EQ(depthResults.first().toObject().value(QStringLiteral("grid_width")).toInt(), 800);
    EXPECT_EQ(depthResults.first().toObject().value(QStringLiteral("grid_height")).toInt(), 600);
}

TEST(ProjectDataTest, RemoveResourcesMatchesRelativeProjectPathsWithAbsoluteRequests)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString projectPath = tempProjectPath(dir);
    const QString projectRoot = QFileInfo(projectPath).absolutePath();
    const QString relativeImage = QStringLiteral("assets/img/1.png");
    const QString keptImage = QStringLiteral("assets/img/2.png");
    const QString absoluteRequest = QDir(projectRoot).filePath(relativeImage);

    ProjectData project;
    ASSERT_TRUE(project.createProject(projectPath, QStringLiteral("demo")));

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

    const QJsonObject core = archiveObject(projectPath, QStringLiteral("project_files.json"));
    const QJsonObject results = archiveObject(projectPath, QStringLiteral("project_results.json"));

    EXPECT_EQ(core.value(QStringLiteral("project_note")).toString(), QStringLiteral("core-only update"));
    EXPECT_FALSE(core.contains(QStringLiteral("depth_map_results")));
    ASSERT_TRUE(results.contains(QStringLiteral("depth_map_results")));
    EXPECT_EQ(results.value(QStringLiteral("depth_map_results")).toArray().size(), 1);
}
