#include <gtest/gtest.h>

#include "ProjectResourceCleanup.h"
#include "project/ProjectIO.h"
#include "project/ProjectSessionModel.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QSemaphore>
#include <QTemporaryDir>

#include <atomic>
#include <filesystem>
#include <system_error>
#include <thread>

namespace
{

void writeArtifact(const QString &path, const QByteArray &contents = "artifact")
{
    ASSERT_TRUE(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly)) << path.toStdString();
    ASSERT_EQ(file.write(contents), contents.size());
}

void setResultRecords(ProjectData *projectData,
                      const QString &arrayKey,
                      const QJsonArray &records)
{
    QJsonObject metadata = projectData->metadata();
    metadata[arrayKey] = records;
    projectData->updateMetadata(metadata, false);
}

QString managedRoot(const ProjectData &projectData)
{
    return xjw::common::project::ProjectIO::projectRootFromPlascan(
        projectData.currentProjectPath());
}

std::filesystem::path filesystemPath(const QString &path)
{
#ifdef Q_OS_WIN
    return std::filesystem::path(path.toStdWString());
#else
    return std::filesystem::path(path.toStdString());
#endif
}

bool createDirectoryLink(const QString &target,
                         const QString &link,
                         QString *errorMessage)
{
#ifdef Q_OS_WIN
    QProcess process;
    process.start(QStringLiteral("cmd.exe"),
                  {QStringLiteral("/D"),
                   QStringLiteral("/C"),
                   QStringLiteral("mklink"),
                   QStringLiteral("/J"),
                   QDir::toNativeSeparators(link),
                   QDir::toNativeSeparators(target)});
    if (!process.waitForFinished(10000)
        || process.exitStatus() != QProcess::NormalExit
        || process.exitCode() != 0)
    {
        if (errorMessage)
        {
            *errorMessage = QString::fromLocal8Bit(process.readAllStandardError());
        }
        return false;
    }
    return true;
#else
    std::error_code error;
    std::filesystem::create_directory_symlink(
        filesystemPath(target), filesystemPath(link), error);
    if (error && errorMessage)
    {
        *errorMessage = QString::fromStdString(error.message());
    }
    return !error;
#endif
}

QJsonObject demRecord(const QString &demPath,
                      const QString &outputDirectory,
                      const QString &runId = {})
{
    QJsonObject record{
        {QStringLiteral("dem_tif"), demPath},
        {QStringLiteral("output_dir"), outputDirectory}
    };
    if (!runId.isEmpty())
    {
        record[QStringLiteral("run_id")] = runId;
    }
    return record;
}

QJsonObject modelRunRecord(const QString &modelPath,
                           const QString &runDirectory,
                           const QString &runId,
                           const QString &diagnosticsPath)
{
    return QJsonObject{
        {QStringLiteral("final_model_path"), modelPath},
        {QStringLiteral("model_ply"), modelPath},
        {QStringLiteral("model_run_directory"), runDirectory},
        {QStringLiteral("model_run_id"), runId},
        {QStringLiteral("model_diagnostics_path"), diagnosticsPath},
        {QStringLiteral("model_property_schema_version"), 2}
    };
}

QString createRecoveryTransaction(
    ProjectData *projectData,
    const QString &transactionId,
    const QString &source,
    const QString &destination,
    const QString &state,
    const QString &moveState,
    bool directory = false,
    const QJsonObject &originalMetadata = {},
    const QJsonObject &updatedMetadata = {})
{
    const QString root = managedRoot(*projectData);
    const QString transactionRoot = QDir(root).filePath(
        QStringLiteral(".plascan_cleanup_trash/%1").arg(transactionId));
    EXPECT_TRUE(QDir().mkpath(transactionRoot));
    const QJsonObject original = originalMetadata.isEmpty()
        ? projectData->metadata()
        : originalMetadata;
    const QJsonObject updated = updatedMetadata.isEmpty()
        ? original
        : updatedMetadata;
    const QByteArray originalBytes = QJsonDocument(original).toJson(
        QJsonDocument::Compact);
    const QByteArray updatedBytes = QJsonDocument(updated).toJson(
        QJsonDocument::Compact);
    const QByteArray originalHash = QCryptographicHash::hash(
        originalBytes, QCryptographicHash::Sha256).toHex();
    const QByteArray updatedHash = QCryptographicHash::hash(
        updatedBytes, QCryptographicHash::Sha256).toHex();
    writeArtifact(
        QDir(transactionRoot).filePath(
            QStringLiteral("original_metadata.json")),
        originalBytes);
    writeArtifact(
        QDir(transactionRoot).filePath(
            QStringLiteral("updated_metadata.json")),
        updatedBytes);
    const QJsonObject manifest{
        {QStringLiteral("schema_version"), 1},
        {QStringLiteral("type"),
         QStringLiteral("plascan_resource_cleanup_transaction")},
        {QStringLiteral("transaction_id"), transactionId},
        {QStringLiteral("project_path"),
         QDir::cleanPath(QFileInfo(
             projectData->currentProjectPath()).absoluteFilePath())},
        {QStringLiteral("chunk_id"), projectData->activeChunkId()},
        {QStringLiteral("chunk_directory"),
         projectData->activeChunkDirectory()},
        {QStringLiteral("project_root"), root},
        {QStringLiteral("managed_root"), root},
        {QStringLiteral("metadata_wal"), QJsonObject{
             {QStringLiteral("original_file"),
              QStringLiteral("original_metadata.json")},
             {QStringLiteral("original_sha256"),
              QString::fromLatin1(originalHash)},
             {QStringLiteral("updated_file"),
              QStringLiteral("updated_metadata.json")},
             {QStringLiteral("updated_sha256"),
              QString::fromLatin1(updatedHash)}
         }},
        {QStringLiteral("state"), state},
        {QStringLiteral("moves"), QJsonArray{QJsonObject{
             {QStringLiteral("source"), source},
             {QStringLiteral("destination"), destination},
             {QStringLiteral("kind"), directory
                  ? QStringLiteral("directory")
                  : QStringLiteral("file")},
             {QStringLiteral("state"), moveState}
         }}}
    };
    writeArtifact(
        QDir(transactionRoot).filePath(QStringLiteral("transaction.json")),
        QJsonDocument(manifest).toJson(QJsonDocument::Indented));
    return transactionRoot;
}

} // namespace

TEST(ProjectResourceCleanupTest, ExternalAndParentTraversalPathsOnlyLoseReferences)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("external_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath,
                                          QStringLiteral("external_cleanup")));
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString absoluteExternal = temporary.filePath(
        QStringLiteral("external/absolute.tif"));
    const QString traversalExternal = temporary.filePath(
        QStringLiteral("traversal.tif"));
    writeArtifact(absoluteExternal);
    writeArtifact(traversalExternal);
    const QString traversalPath = QDir(managedRoot(projectData))
        .relativeFilePath(traversalExternal);
    ASSERT_TRUE(traversalPath.startsWith(QLatin1String("..")));

    setResultRecords(&projectData,
                     QStringLiteral("dem_results"),
                     QJsonArray{
                         demRecord(absoluteExternal,
                                   QFileInfo(absoluteExternal).absolutePath()),
                         demRecord(traversalPath,
                                   QFileInfo(traversalExternal).absolutePath())
                     });

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::cleanupGeneratedData(
            &projectData,
            QStringLiteral("DEM"),
            QStringList{absoluteExternal, traversalPath});

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_EQ(result.removedCount, 2);
    EXPECT_TRUE(QFileInfo::exists(absoluteExternal));
    EXPECT_TRUE(QFileInfo::exists(traversalExternal));
    EXPECT_TRUE(result.preservedExternalPaths.contains(absoluteExternal));
    EXPECT_TRUE(result.preservedExternalPaths.contains(
        QDir::cleanPath(traversalExternal)));
    EXPECT_TRUE(projectData.metadata().value(
        QStringLiteral("dem_results")).toArray().isEmpty());
}

TEST(ProjectResourceCleanupTest, ManagedSymlinkToExternalDirectoryIsNeverFollowed)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("symlink_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath,
                                          QStringLiteral("symlink_cleanup")));
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString externalDirectory = temporary.filePath(
        QStringLiteral("external_target"));
    const QString externalArtifact = QDir(externalDirectory).filePath(
        QStringLiteral("escaped.tif"));
    writeArtifact(externalArtifact);

    const QString linkParent = QDir(managedRoot(projectData)).filePath(
        QStringLiteral("reconstruction"));
    ASSERT_TRUE(QDir().mkpath(linkParent));
    const QString linkPath = QDir(linkParent).filePath(
        QStringLiteral("external_link"));
    QString linkError;
    ASSERT_TRUE(createDirectoryLink(externalDirectory,
                                    linkPath,
                                    &linkError))
        << linkError.toStdString();

    const QString aliasedArtifact = QDir(linkPath).filePath(
        QStringLiteral("escaped.tif"));
    setResultRecords(&projectData,
                     QStringLiteral("dem_results"),
                     QJsonArray{demRecord(aliasedArtifact, linkPath)});

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::cleanupGeneratedData(
            &projectData,
            QStringLiteral("DEM"),
            QStringList{aliasedArtifact});

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_TRUE(QFileInfo::exists(externalArtifact));
    const QFileInfo linkInfo(linkPath);
    EXPECT_TRUE(linkInfo.isSymLink() || linkInfo.isJunction());
    EXPECT_TRUE(result.preservedExternalPaths.contains(aliasedArtifact)
                || result.preservedUnsafePaths.contains(aliasedArtifact));
}

TEST(ProjectResourceCleanupTest, SharedManagedArtifactSurvivesSelectedRecord)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("shared_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath,
                                          QStringLiteral("shared_cleanup")));
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString products = QDir(managedRoot(projectData)).filePath(
        QStringLiteral("reconstruction/model"));
    const QString selectedModel = QDir(products).filePath(
        QStringLiteral("selected/model.ply"));
    const QString retainedModel = QDir(products).filePath(
        QStringLiteral("retained/model.ply"));
    const QString sharedTexture = QDir(products).filePath(
        QStringLiteral("shared/texture.png"));
    writeArtifact(selectedModel);
    writeArtifact(retainedModel);
    writeArtifact(sharedTexture);

    const QJsonObject selectedRecord{
        {QStringLiteral("final_model_path"), selectedModel},
        {QStringLiteral("texture_png"), sharedTexture}
    };
    const QJsonObject retainedRecord{
        {QStringLiteral("final_model_path"), retainedModel},
        {QStringLiteral("texture_png"), sharedTexture}
    };
    setResultRecords(&projectData,
                     QStringLiteral("model_results"),
                     QJsonArray{selectedRecord, retainedRecord});

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::cleanupGeneratedData(
            &projectData,
            QStringLiteral("3D模型"),
            QStringList{selectedModel});

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_FALSE(QFileInfo::exists(selectedModel));
    EXPECT_TRUE(QFileInfo::exists(retainedModel));
    EXPECT_TRUE(QFileInfo::exists(sharedTexture));
    EXPECT_TRUE(result.preservedSharedPaths.contains(sharedTexture));
    ASSERT_EQ(projectData.metadata().value(
        QStringLiteral("model_results")).toArray().size(), 1);
}

TEST(ProjectResourceCleanupTest, ExclusiveManagedDirectoryUsesTransactionalCleanup)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("managed_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath,
                                          QStringLiteral("managed_cleanup")));
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString outputDirectory = QDir(managedRoot(projectData)).filePath(
        QStringLiteral("reconstruction/terrain/run-001"));
    const QString demPath = QDir(outputDirectory).filePath(
        QStringLiteral("terrain.tif"));
    const QString ancillaryPath = QDir(outputDirectory).filePath(
        QStringLiteral("diagnostics.json"));
    const QString ownershipManifestPath = QDir(outputDirectory).filePath(
        QStringLiteral("ownership.json"));
    writeArtifact(demPath);
    writeArtifact(ancillaryPath);
    writeArtifact(
        ownershipManifestPath,
        QJsonDocument(QJsonObject{
            {QStringLiteral("type"),
             QStringLiteral("plascan_owned_directory")},
            {QStringLiteral("schema_version"), 1},
            {QStringLiteral("run_id"), QStringLiteral("run-001")},
            {QStringLiteral("owned_directory"), outputDirectory}
        }).toJson(QJsonDocument::Compact));
    QJsonObject ownedRecord = demRecord(demPath,
                                        outputDirectory,
                                        QStringLiteral("run-001"));
    ownedRecord[QStringLiteral("ownership_manifest_path")] =
        ownershipManifestPath;
    setResultRecords(&projectData,
                     QStringLiteral("dem_results"),
                     QJsonArray{ownedRecord});

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::cleanupGeneratedData(
            &projectData,
            QStringLiteral("DEM"),
            QStringList{demPath});

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_FALSE(QFileInfo::exists(outputDirectory));
    EXPECT_TRUE(result.failedPaths.isEmpty());
    EXPECT_FALSE(QFileInfo::exists(QDir(managedRoot(projectData)).filePath(
        QStringLiteral(".plascan_cleanup_trash"))));

    QFile temporaryResults(
        xjw::common::project::ProjectIO::tempResultsPath(projectPath));
    ASSERT_TRUE(temporaryResults.open(QIODevice::ReadOnly));
    const QJsonDocument persisted = QJsonDocument::fromJson(
        qUncompress(temporaryResults.readAll()));
    ASSERT_TRUE(persisted.isObject());
    EXPECT_TRUE(persisted.object().value(
        QStringLiteral("dem_results")).toArray().isEmpty());
}

TEST(ProjectResourceCleanupTest, LegacyWideDirectoryOnlyDeletesListedFiles)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("legacy_wide_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath,
                                          QStringLiteral("legacy_wide_cleanup")));
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString broadDirectory = QDir(managedRoot(projectData)).filePath(
        QStringLiteral("reconstruction/terrain"));
    const QString demPath = QDir(broadDirectory).filePath(
        QStringLiteral("legacy/terrain.tif"));
    const QString unrelatedPath = QDir(broadDirectory).filePath(
        QStringLiteral("other-run/must-survive.bin"));
    writeArtifact(demPath);
    writeArtifact(unrelatedPath);
    setResultRecords(&projectData,
                     QStringLiteral("dem_results"),
                     QJsonArray{demRecord(demPath,
                                          broadDirectory,
                                          QStringLiteral("terrain"))});

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::cleanupGeneratedData(
            &projectData,
            QStringLiteral("DEM"),
            QStringList{demPath});

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_FALSE(QFileInfo::exists(demPath));
    EXPECT_TRUE(QFileInfo::exists(unrelatedPath));
    EXPECT_TRUE(QFileInfo::exists(broadDirectory));
    EXPECT_TRUE(result.preservedUnsafePaths.contains(broadDirectory));
}

TEST(ProjectResourceCleanupTest, ModelRunCleanupDeletesOnlyExclusiveSafeRun)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("model_run_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath,
                                          QStringLiteral("model_run_cleanup")));
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString root = managedRoot(projectData);
    const QString modelRuns = QDir(root).filePath(
        QStringLiteral("reconstruction/model_runs"));
    const QString safeRun = QDir(modelRuns).filePath(QStringLiteral("safe-run"));
    const QString safeModel = QDir(safeRun).filePath(
        QStringLiteral("products/model.ply"));
    const QString safeDiagnostics = QDir(root).filePath(
        QStringLiteral("reports/safe-model-result.json"));
    writeArtifact(safeModel);
    writeArtifact(safeDiagnostics);
    writeArtifact(QDir(safeRun).filePath(QStringLiteral("unlisted.log")));

    const QString sharedRun = QDir(modelRuns).filePath(
        QStringLiteral("shared-run"));
    const QString sharedModel = QDir(sharedRun).filePath(
        QStringLiteral("products/model.ply"));
    const QString sharedDiagnostics = QDir(sharedRun).filePath(
        QStringLiteral("model_result.json"));
    const QString sharedTexture = QDir(sharedRun).filePath(
        QStringLiteral("products/texture.png"));
    writeArtifact(sharedModel);
    writeArtifact(sharedDiagnostics);
    writeArtifact(sharedTexture);

    const QString retainedModel = QDir(modelRuns).filePath(
        QStringLiteral("retained-run/products/model.ply"));
    writeArtifact(retainedModel);
    QJsonObject retainedRecord{
        {QStringLiteral("final_model_path"), retainedModel},
        {QStringLiteral("model_ply"), retainedModel},
        {QStringLiteral("model_run_directory"), sharedRun},
        {QStringLiteral("texture_png"), sharedTexture}
    };

    const QString externalRun = temporary.filePath(
        QStringLiteral("external/model_runs/external-run"));
    const QString externalModel = QDir(externalRun).filePath(
        QStringLiteral("products/model.ply"));
    const QString externalDiagnostics = QDir(externalRun).filePath(
        QStringLiteral("model_result.json"));
    writeArtifact(externalModel);
    writeArtifact(externalDiagnostics);

    const QString unsafeTarget = QDir(root).filePath(
        QStringLiteral("reconstruction/unsafe-model-target"));
    const QString unsafeTargetModel = QDir(unsafeTarget).filePath(
        QStringLiteral("products/model.ply"));
    const QString unsafeTargetDiagnostics = QDir(unsafeTarget).filePath(
        QStringLiteral("model_result.json"));
    writeArtifact(unsafeTargetModel);
    writeArtifact(unsafeTargetDiagnostics);
    const QString unsafeRun = QDir(modelRuns).filePath(
        QStringLiteral("unsafe-run"));
    ASSERT_TRUE(QDir().mkpath(modelRuns));
    QString linkError;
    ASSERT_TRUE(createDirectoryLink(unsafeTarget, unsafeRun, &linkError))
        << linkError.toStdString();
    const QString unsafeModel = QDir(unsafeRun).filePath(
        QStringLiteral("products/model.ply"));
    const QString unsafeDiagnostics = QDir(unsafeRun).filePath(
        QStringLiteral("model_result.json"));

    QJsonObject sharedRecord = modelRunRecord(
        sharedModel,
        sharedRun,
        QStringLiteral("shared-run"),
        sharedDiagnostics);
    sharedRecord[QStringLiteral("texture_png")] = sharedTexture;
    setResultRecords(
        &projectData,
        QStringLiteral("model_results"),
        QJsonArray{
            modelRunRecord(safeModel,
                           safeRun,
                           QStringLiteral("safe-run"),
                           safeDiagnostics),
            sharedRecord,
            retainedRecord,
            modelRunRecord(externalModel,
                           externalRun,
                           QStringLiteral("external-run"),
                           externalDiagnostics),
            modelRunRecord(unsafeModel,
                           unsafeRun,
                           QStringLiteral("unsafe-run"),
                           unsafeDiagnostics)
        });

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::cleanupGeneratedData(
            &projectData,
            QStringLiteral("3D模型"),
            QStringList{safeModel,
                        sharedModel,
                        externalModel,
                        unsafeModel});

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_EQ(result.removedCount, 4);
    EXPECT_FALSE(QFileInfo::exists(safeRun));
    EXPECT_FALSE(QFileInfo::exists(safeDiagnostics));
    EXPECT_TRUE(QFileInfo::exists(sharedModel));
    EXPECT_TRUE(QFileInfo::exists(sharedDiagnostics));
    EXPECT_TRUE(QFileInfo::exists(sharedTexture));
    EXPECT_TRUE(QFileInfo::exists(externalModel));
    EXPECT_TRUE(QFileInfo::exists(externalDiagnostics));
    EXPECT_TRUE(QFileInfo::exists(unsafeTargetModel));
    EXPECT_TRUE(QFileInfo::exists(unsafeTargetDiagnostics));
    EXPECT_TRUE(QFileInfo(unsafeRun).isSymLink()
                || QFileInfo(unsafeRun).isJunction());
    EXPECT_TRUE(result.preservedSharedPaths.contains(sharedRun));
    EXPECT_TRUE(result.preservedExternalPaths.contains(externalRun));
    EXPECT_TRUE(result.preservedUnsafePaths.contains(unsafeRun));
    const QJsonArray remaining = projectData.metadata().value(
        QStringLiteral("model_results")).toArray();
    ASSERT_EQ(remaining.size(), 1);
    EXPECT_EQ(remaining.first().toObject().value(
                  QStringLiteral("final_model_path")).toString(),
              retainedModel);
}

TEST(ProjectResourceCleanupTest, MetadataPersistenceFailureRestoresFilesAndRecord)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("rollback_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath,
                                          QStringLiteral("rollback_cleanup")));
    const QString outputDirectory = QDir(managedRoot(projectData)).filePath(
        QStringLiteral("reconstruction/terrain/run-rollback"));
    const QString demPath = QDir(outputDirectory).filePath(
        QStringLiteral("terrain.tif"));
    writeArtifact(demPath);
    setResultRecords(&projectData,
                     QStringLiteral("dem_results"),
                     QJsonArray{demRecord(demPath, outputDirectory)});
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString temporaryMetadataDirectory =
        xjw::common::project::ProjectIO::tmpDir(projectPath);
    ASSERT_TRUE(QDir(temporaryMetadataDirectory).removeRecursively());
    QFile blocker(temporaryMetadataDirectory);
    ASSERT_TRUE(blocker.open(QIODevice::WriteOnly));
    blocker.write("block temporary metadata directory");
    blocker.close();

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::cleanupGeneratedData(
            &projectData,
            QStringLiteral("DEM"),
            QStringList{demPath});

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.errorMessage.isEmpty());
    EXPECT_TRUE(QFileInfo::exists(demPath));
    EXPECT_TRUE(QFileInfo::exists(outputDirectory));
    EXPECT_EQ(projectData.metadata().value(
        QStringLiteral("dem_results")).toArray().size(), 1);
    EXPECT_FALSE(QFileInfo::exists(QDir(managedRoot(projectData)).filePath(
        QStringLiteral(".plascan_cleanup_trash"))));

    ASSERT_TRUE(QFile::remove(temporaryMetadataDirectory));
}

TEST(ProjectResourceCleanupTest,
     SupersededPersistenceWorkerCannotOverwriteNewerCommit)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString snapshotPath = temporary.filePath(
        QStringLiteral("project_results.snapshot"));
    xjw::common::project::ProjectPersistenceCommitCoordinator coordinator;
    const quint64 oldGeneration = coordinator.currentGeneration();
    QSemaphore oldWorkerCaptured;
    QSemaphore releaseOldWorker;
    std::atomic_bool oldCommitExecuted{false};
    bool oldCommitAccepted = true;

    std::thread oldWorker([&]()
    {
        oldWorkerCaptured.release();
        releaseOldWorker.acquire();
        oldCommitAccepted = coordinator.runIfCurrent(
            oldGeneration,
            [&]()
            {
                oldCommitExecuted.store(true, std::memory_order_relaxed);
                QFile file(snapshotPath);
                if (file.open(QIODevice::WriteOnly | QIODevice::Truncate))
                {
                    file.write("old");
                }
            });
    });

    oldWorkerCaptured.acquire();
    const quint64 cleanupGeneration = coordinator.advanceGeneration();
    bool cleanupWriteSucceeded = false;
    const bool cleanupCommitAccepted = coordinator.runIfCurrent(
        cleanupGeneration,
        [&]()
        {
            QFile file(snapshotPath);
            cleanupWriteSucceeded = file.open(
                QIODevice::WriteOnly | QIODevice::Truncate)
                && file.write("cleanup") == 7;
        });
    releaseOldWorker.release();
    oldWorker.join();

    EXPECT_TRUE(cleanupCommitAccepted);
    EXPECT_TRUE(cleanupWriteSucceeded);
    EXPECT_FALSE(oldCommitAccepted);
    EXPECT_FALSE(oldCommitExecuted.load(std::memory_order_relaxed));
    QFile snapshot(snapshotPath);
    ASSERT_TRUE(snapshot.open(QIODevice::ReadOnly));
    EXPECT_EQ(snapshot.readAll(), QByteArray("cleanup"));
}

TEST(ProjectResourceCleanupTest,
     GenerationAdvanceDoesNotWaitForLongFilesystemCommit)
{
    xjw::common::project::ProjectPersistenceCommitCoordinator coordinator;
    const quint64 oldGeneration = coordinator.currentGeneration();
    QSemaphore longCommitEntered;
    QSemaphore releaseLongCommit;
    QSemaphore advanceFinished;
    std::atomic_bool longCommitAccepted{true};
    std::atomic_bool waitingCommitAccepted{true};
    std::atomic_bool waitingCommitExecuted{false};
    std::atomic<quint64> newGeneration{0};

    std::thread longCommit([&]()
    {
        longCommitAccepted.store(
            coordinator.runIfCurrent(
                oldGeneration,
                [&]()
                {
                    longCommitEntered.release();
                    releaseLongCommit.acquire();
                }),
            std::memory_order_relaxed);
    });
    longCommitEntered.acquire();

    std::thread waitingCommit([&]()
    {
        waitingCommitAccepted.store(
            coordinator.runIfCurrent(
                oldGeneration,
                [&]()
                {
                    waitingCommitExecuted.store(
                        true, std::memory_order_relaxed);
                }),
            std::memory_order_relaxed);
    });
    std::thread advance([&]()
    {
        newGeneration.store(
            coordinator.advanceGeneration(),
            std::memory_order_relaxed);
        advanceFinished.release();
    });

    const bool advancedWithoutCommit = advanceFinished.tryAcquire(1, 1000);
    releaseLongCommit.release();
    advance.join();
    longCommit.join();
    waitingCommit.join();

    EXPECT_TRUE(advancedWithoutCommit)
        << "GUI 代次推进不得等待磁盘提交锁";
    EXPECT_GT(newGeneration.load(std::memory_order_relaxed), oldGeneration);
    EXPECT_FALSE(longCommitAccepted.load(std::memory_order_relaxed));
    EXPECT_FALSE(waitingCommitAccepted.load(std::memory_order_relaxed));
    EXPECT_FALSE(waitingCommitExecuted.load(std::memory_order_relaxed));
}

TEST(ProjectResourceCleanupTest,
     CleanupGenerationAdvanceWaitsForCurrentFilesystemCommit)
{
    xjw::common::project::ProjectPersistenceCommitCoordinator coordinator;
    const quint64 oldGeneration = coordinator.currentGeneration();
    QSemaphore commitEntered;
    QSemaphore releaseCommit;
    QSemaphore cleanupAdvanceFinished;
    std::atomic<quint64> cleanupGeneration{0};

    std::thread commit([&]()
    {
        coordinator.runIfCurrent(
            oldGeneration,
            [&]()
            {
                commitEntered.release();
                releaseCommit.acquire();
            });
    });
    commitEntered.acquire();

    std::thread cleanupAdvance([&]()
    {
        cleanupGeneration.store(
            coordinator.advanceGenerationAfterCurrentCommit(),
            std::memory_order_relaxed);
        cleanupAdvanceFinished.release();
    });

    EXPECT_FALSE(cleanupAdvanceFinished.tryAcquire(1, 100))
        << "cleanup must wait for the in-flight filesystem commit";
    releaseCommit.release();
    EXPECT_TRUE(cleanupAdvanceFinished.tryAcquire(1, 1000));
    cleanupAdvance.join();
    commit.join();

    EXPECT_GT(cleanupGeneration.load(std::memory_order_relaxed),
              oldGeneration);
}

TEST(ProjectResourceCleanupTest,
     StagingManifestRestoresMovedArtifactBeforeNextCleanup)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("staging_recovery.plascan"));
    ASSERT_TRUE(projectData.createProject(projectPath,
                                          QStringLiteral("staging_recovery")));
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString root = managedRoot(projectData);
    const QString retainedPath = QDir(root).filePath(
        QStringLiteral("reconstruction/terrain/run-retained/retained.tif"));
    const QString externalPath = temporary.filePath(
        QStringLiteral("external/remove-reference.tif"));
    writeArtifact(retainedPath);
    writeArtifact(externalPath);
    setResultRecords(
        &projectData,
        QStringLiteral("dem_results"),
        QJsonArray{
            demRecord(retainedPath, QFileInfo(retainedPath).absolutePath()),
            demRecord(externalPath, QFileInfo(externalPath).absolutePath())
        });

    const QString transactionId = QStringLiteral("crashed-staging");
    const QString transactionRoot = QDir(root).filePath(
        QStringLiteral(".plascan_cleanup_trash/%1").arg(transactionId));
    ASSERT_TRUE(QDir().mkpath(transactionRoot));
    const QString stagedPath = QDir(transactionRoot).filePath(
        QStringLiteral("000000_retained.tif"));
    ASSERT_TRUE(QDir().rename(retainedPath, stagedPath));
    createRecoveryTransaction(&projectData,
                              transactionId,
                              retainedPath,
                              stagedPath,
                              QStringLiteral("staging"),
                              QStringLiteral("planned"));

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::
            cleanupGeneratedData(
                &projectData,
                QStringLiteral("DEM"),
                QStringList{externalPath});

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_TRUE(QFileInfo::exists(retainedPath));
    EXPECT_FALSE(QFileInfo::exists(stagedPath));
    EXPECT_FALSE(QFileInfo::exists(transactionRoot));
    EXPECT_TRUE(QFileInfo::exists(externalPath));
    const QJsonArray records = projectData.metadata().value(
        QStringLiteral("dem_results")).toArray();
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records.first().toObject().value(
                  QStringLiteral("dem_tif")).toString(),
              retainedPath);
}

TEST(ProjectResourceCleanupTest,
     MetadataCommittedManifestPurgesStagedArtifactOnRecovery)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("committed_recovery.plascan"));
    ASSERT_TRUE(projectData.createProject(
        projectPath, QStringLiteral("committed_recovery")));
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString root = managedRoot(projectData);
    const QString removedPath = QDir(root).filePath(
        QStringLiteral("reconstruction/terrain/run-removed/removed.tif"));
    const QString externalPath = temporary.filePath(
        QStringLiteral("external/remaining-reference.tif"));
    writeArtifact(removedPath);
    writeArtifact(externalPath);
    setResultRecords(
        &projectData,
        QStringLiteral("dem_results"),
        QJsonArray{
            demRecord(externalPath, QFileInfo(externalPath).absolutePath())
        });
    const QJsonObject updatedMetadata = projectData.metadata();
    QJsonObject originalMetadata = updatedMetadata;
    originalMetadata[QStringLiteral("dem_results")] = QJsonArray{
        demRecord(removedPath, QFileInfo(removedPath).absolutePath()),
        demRecord(externalPath, QFileInfo(externalPath).absolutePath())
    };

    const QString transactionId = QStringLiteral("crashed-committed");
    const QString transactionRoot = QDir(root).filePath(
        QStringLiteral(".plascan_cleanup_trash/%1").arg(transactionId));
    ASSERT_TRUE(QDir().mkpath(transactionRoot));
    const QString stagedPath = QDir(transactionRoot).filePath(
        QStringLiteral("000000_removed.tif"));
    ASSERT_TRUE(QDir().rename(removedPath, stagedPath));
    createRecoveryTransaction(&projectData,
                              transactionId,
                              removedPath,
                              stagedPath,
                              QStringLiteral("metadata_committed"),
                              QStringLiteral("staged"),
                              false,
                              originalMetadata,
                              updatedMetadata);

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::
            cleanupGeneratedData(
                &projectData,
                QStringLiteral("DEM"),
                QStringList{externalPath});

    ASSERT_TRUE(result.success) << result.errorMessage.toStdString();
    EXPECT_FALSE(QFileInfo::exists(removedPath));
    EXPECT_FALSE(QFileInfo::exists(stagedPath));
    EXPECT_FALSE(QFileInfo::exists(transactionRoot));
    EXPECT_TRUE(QFileInfo::exists(externalPath));
}

TEST(ProjectResourceCleanupTest,
     ReopenContinuesPurgeOnlyResidueWithoutTransactionManifest)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath = temporary.filePath(
        QStringLiteral("partial_purge_recovery.plascan"));
    QString artifactPath;
    QString purgeRoot;
    {
        ProjectData projectData;
        ASSERT_TRUE(projectData.createProject(
            projectPath, QStringLiteral("partial_purge_recovery")));
        const QString root = managedRoot(projectData);
        artifactPath = QDir(root).filePath(
            QStringLiteral(
                "reconstruction/terrain/partial-purge/terrain.tif"));
        writeArtifact(artifactPath);
        setResultRecords(
            &projectData,
            QStringLiteral("dem_results"),
            QJsonArray{demRecord(
                artifactPath, QFileInfo(artifactPath).absolutePath())});
        QString saveError;
        ASSERT_TRUE(projectData.saveProject(&saveError))
            << saveError.toStdString();

        const QJsonObject originalMetadata = projectData.metadata();
        QJsonObject updatedMetadata = originalMetadata;
        updatedMetadata[QStringLiteral("dem_results")] = QJsonArray();
        const QString transactionId = QStringLiteral(
            "crashed-during-purge");
        const QString transactionRoot = QDir(root).filePath(
            QStringLiteral(".plascan_cleanup_trash/%1")
                .arg(transactionId));
        ASSERT_TRUE(QDir().mkpath(transactionRoot));
        const QString stagedPath = QDir(transactionRoot).filePath(
            QStringLiteral("000000_terrain.tif"));
        ASSERT_TRUE(QDir().rename(artifactPath, stagedPath));
        createRecoveryTransaction(
            &projectData,
            transactionId,
            artifactPath,
            stagedPath,
            QStringLiteral("metadata_committed"),
            QStringLiteral("staged"),
            false,
            originalMetadata,
            updatedMetadata);
        QString commitError;
        ASSERT_TRUE(projectData.commitResourceCleanupMetadata(
            updatedMetadata, &commitError)) << commitError.toStdString();

        const QString purgeBase = QDir(root).filePath(
            QStringLiteral(".plascan_cleanup_purging"));
        ASSERT_TRUE(QDir().mkpath(purgeBase));
        purgeRoot = QDir(purgeBase).filePath(QStringLiteral(
            ".purging-11111111-2222-4333-8444-555555555555"));
        ASSERT_TRUE(QDir().rename(transactionRoot, purgeRoot));
        ASSERT_TRUE(QFile::remove(QDir(purgeRoot).filePath(
            QStringLiteral("transaction.json"))));
        ASSERT_TRUE(QFile::remove(QDir(purgeRoot).filePath(
            QStringLiteral("original_metadata.json"))));
    }

    ASSERT_FALSE(QFileInfo::exists(artifactPath));
    ASSERT_TRUE(QFileInfo::exists(purgeRoot));
    ProjectData reopened;
    xjw::core::project::ProjectResourceCleanupService::
        installAutomaticRecovery(&reopened);
    QString openError;
    ASSERT_TRUE(reopened.openProject(projectPath, &openError))
        << openError.toStdString();

    EXPECT_FALSE(QFileInfo::exists(artifactPath));
    EXPECT_FALSE(QFileInfo::exists(purgeRoot));
    EXPECT_TRUE(reopened.metadata().value(
        QStringLiteral("dem_results")).toArray().isEmpty());
}

TEST(ProjectResourceCleanupTest,
     ReopenIgnoresLinkedPurgeResidueWithoutFollowingIt)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath = temporary.filePath(
        QStringLiteral("linked_purge_recovery.plascan"));
    QString root;
    {
        ProjectData projectData;
        ASSERT_TRUE(projectData.createProject(
            projectPath, QStringLiteral("linked_purge_recovery")));
        root = managedRoot(projectData);
        QString saveError;
        ASSERT_TRUE(projectData.saveProject(&saveError))
            << saveError.toStdString();
    }

    const QString externalDirectory = temporary.filePath(
        QStringLiteral("external-purge-target"));
    const QString externalArtifact = QDir(externalDirectory).filePath(
        QStringLiteral("must-survive.bin"));
    writeArtifact(externalArtifact);
    const QString purgeBase = QDir(root).filePath(
        QStringLiteral(".plascan_cleanup_purging"));
    ASSERT_TRUE(QDir().mkpath(purgeBase));
    const QString linkedPurgeRoot = QDir(purgeBase).filePath(QStringLiteral(
        ".purging-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"));
    QString linkError;
    ASSERT_TRUE(createDirectoryLink(externalDirectory,
                                    linkedPurgeRoot,
                                    &linkError))
        << linkError.toStdString();

    ProjectData reopened;
    xjw::core::project::ProjectResourceCleanupService::
        installAutomaticRecovery(&reopened);
    QString openError;
    EXPECT_TRUE(reopened.openProject(projectPath, &openError))
        << openError.toStdString();
    EXPECT_TRUE(QFileInfo::exists(externalArtifact));
    const QFileInfo linkInfo(linkedPurgeRoot);
    EXPECT_TRUE(linkInfo.isSymLink() || linkInfo.isJunction());
}

TEST(ProjectResourceCleanupTest,
     MetadataCommittedManifestRestoresArtifactWhenMetadataIsOriginal)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath = temporary.filePath(
        QStringLiteral("committed_original_recovery.plascan"));
    QString artifactPath;
    QString transactionRoot;
    {
        ProjectData projectData;
        ASSERT_TRUE(projectData.createProject(
            projectPath, QStringLiteral("committed_original_recovery")));
        artifactPath = QDir(managedRoot(projectData)).filePath(
            QStringLiteral(
                "reconstruction/terrain/committed-original/terrain.tif"));
        writeArtifact(artifactPath);
        setResultRecords(
            &projectData,
            QStringLiteral("dem_results"),
            QJsonArray{demRecord(
                artifactPath, QFileInfo(artifactPath).absolutePath())});
        QString saveError;
        ASSERT_TRUE(projectData.saveProject(&saveError))
            << saveError.toStdString();

        const QJsonObject originalMetadata = projectData.metadata();
        QJsonObject updatedMetadata = originalMetadata;
        updatedMetadata[QStringLiteral("dem_results")] = QJsonArray();
        const QString transactionId = QStringLiteral(
            "committed-marker-with-original-metadata");
        transactionRoot = QDir(managedRoot(projectData)).filePath(
            QStringLiteral(".plascan_cleanup_trash/%1")
                .arg(transactionId));
        ASSERT_TRUE(QDir().mkpath(transactionRoot));
        const QString stagedPath = QDir(transactionRoot).filePath(
            QStringLiteral("000000_terrain.tif"));
        ASSERT_TRUE(QDir().rename(artifactPath, stagedPath));
        createRecoveryTransaction(
            &projectData,
            transactionId,
            artifactPath,
            stagedPath,
            QStringLiteral("metadata_committed"),
            QStringLiteral("staged"),
            false,
            originalMetadata,
            updatedMetadata);
    }

    ASSERT_FALSE(QFileInfo::exists(artifactPath));
    ProjectData reopened;
    xjw::core::project::ProjectResourceCleanupService::
        installAutomaticRecovery(&reopened);
    QString openError;
    ASSERT_TRUE(reopened.openProject(projectPath, &openError))
        << openError.toStdString();

    EXPECT_TRUE(QFileInfo::exists(artifactPath));
    EXPECT_FALSE(QFileInfo::exists(transactionRoot));
    const QJsonArray records = reopened.metadata().value(
        QStringLiteral("dem_results")).toArray();
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records.first().toObject().value(
                  QStringLiteral("dem_tif")).toString(),
              artifactPath);
}

TEST(ProjectResourceCleanupTest,
     ProjectReopenImmediatelyRestoresStagingTransaction)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath = temporary.filePath(
        QStringLiteral("reopen_recovery.plascan"));
    QString retainedPath;
    QString transactionRoot;
    QString stagedPath;
    {
        ProjectData projectData;
        ASSERT_TRUE(projectData.createProject(
            projectPath, QStringLiteral("reopen_recovery")));
        retainedPath = QDir(managedRoot(projectData)).filePath(
            QStringLiteral(
                "reconstruction/terrain/run-reopen/retained.tif"));
        writeArtifact(retainedPath);
        setResultRecords(
            &projectData,
            QStringLiteral("dem_results"),
            QJsonArray{demRecord(
                retainedPath, QFileInfo(retainedPath).absolutePath())});
        QString saveError;
        ASSERT_TRUE(projectData.saveProject(&saveError))
            << saveError.toStdString();

        const QString transactionId = QStringLiteral("crashed-reopen");
        transactionRoot = QDir(managedRoot(projectData)).filePath(
            QStringLiteral(".plascan_cleanup_trash/%1")
                .arg(transactionId));
        ASSERT_TRUE(QDir().mkpath(transactionRoot));
        stagedPath = QDir(transactionRoot).filePath(
            QStringLiteral("000000_retained.tif"));
        ASSERT_TRUE(QDir().rename(retainedPath, stagedPath));
        createRecoveryTransaction(&projectData,
                                  transactionId,
                                  retainedPath,
                                  stagedPath,
                                  QStringLiteral("staging"),
                                  QStringLiteral("planned"));
    }

    ASSERT_FALSE(QFileInfo::exists(retainedPath));
    ASSERT_TRUE(QFileInfo::exists(stagedPath));
    ProjectData reopened;
    xjw::core::project::ProjectResourceCleanupService::
        installAutomaticRecovery(&reopened);
    QString openError;
    ASSERT_TRUE(reopened.openProject(projectPath, &openError))
        << openError.toStdString();

    EXPECT_TRUE(QFileInfo::exists(retainedPath));
    EXPECT_FALSE(QFileInfo::exists(stagedPath));
    EXPECT_FALSE(QFileInfo::exists(transactionRoot));
    const QJsonArray records = reopened.metadata().value(
        QStringLiteral("dem_results")).toArray();
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records.first().toObject().value(
                  QStringLiteral("dem_tif")).toString(),
              retainedPath);
}

TEST(ProjectResourceCleanupTest,
     ReopenRollsBackMetadataCommittedBeforeStagingMarker)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath = temporary.filePath(
        QStringLiteral("metadata_wal_recovery.plascan"));
    QString artifactPath;
    QString transactionRoot;
    {
        ProjectData projectData;
        ASSERT_TRUE(projectData.createProject(
            projectPath, QStringLiteral("metadata_wal_recovery")));
        artifactPath = QDir(managedRoot(projectData)).filePath(
            QStringLiteral(
                "reconstruction/terrain/wal-run/terrain.tif"));
        writeArtifact(artifactPath);
        setResultRecords(
            &projectData,
            QStringLiteral("dem_results"),
            QJsonArray{demRecord(
                artifactPath, QFileInfo(artifactPath).absolutePath())});
        QString saveError;
        ASSERT_TRUE(projectData.saveProject(&saveError))
            << saveError.toStdString();

        const QJsonObject originalMetadata = projectData.metadata();
        QJsonObject updatedMetadata = originalMetadata;
        updatedMetadata[QStringLiteral("dem_results")] = QJsonArray();
        const QString transactionId = QStringLiteral(
            "crashed-after-metadata-commit");
        transactionRoot = QDir(managedRoot(projectData)).filePath(
            QStringLiteral(".plascan_cleanup_trash/%1")
                .arg(transactionId));
        ASSERT_TRUE(QDir().mkpath(transactionRoot));
        const QString stagedPath = QDir(transactionRoot).filePath(
            QStringLiteral("000000_terrain.tif"));
        ASSERT_TRUE(QDir().rename(artifactPath, stagedPath));
        createRecoveryTransaction(
            &projectData,
            transactionId,
            artifactPath,
            stagedPath,
            QStringLiteral("staging"),
            QStringLiteral("staged"),
            false,
            originalMetadata,
            updatedMetadata);

        QString commitError;
        ASSERT_TRUE(projectData.commitResourceCleanupMetadata(
            updatedMetadata, &commitError)) << commitError.toStdString();
        ASSERT_TRUE(projectData.metadata().value(
            QStringLiteral("dem_results")).toArray().isEmpty());
        ASSERT_FALSE(QFileInfo::exists(artifactPath));
    }

    ProjectData reopened;
    xjw::core::project::ProjectResourceCleanupService::
        installAutomaticRecovery(&reopened);
    QString openError;
    ASSERT_TRUE(reopened.openProject(projectPath, &openError))
        << openError.toStdString();

    EXPECT_TRUE(QFileInfo::exists(artifactPath));
    EXPECT_FALSE(QFileInfo::exists(transactionRoot));
    const QJsonArray records = reopened.metadata().value(
        QStringLiteral("dem_results")).toArray();
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records.first().toObject().value(
                  QStringLiteral("dem_tif")).toString(),
              artifactPath);
}

TEST(ProjectResourceCleanupTest,
     RecoveryPreservesThirdMetadataStateWhileRestoringArtifact)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString projectPath = temporary.filePath(
        QStringLiteral("third_state_recovery.plascan"));
    QString artifactPath;
    QString transactionRoot;
    {
        ProjectData projectData;
        ASSERT_TRUE(projectData.createProject(
            projectPath, QStringLiteral("third_state_recovery")));
        artifactPath = QDir(managedRoot(projectData)).filePath(
            QStringLiteral(
                "reconstruction/terrain/third-run/terrain.tif"));
        writeArtifact(artifactPath);
        setResultRecords(
            &projectData,
            QStringLiteral("dem_results"),
            QJsonArray{demRecord(
                artifactPath, QFileInfo(artifactPath).absolutePath())});
        QString saveError;
        ASSERT_TRUE(projectData.saveProject(&saveError))
            << saveError.toStdString();

        const QJsonObject originalMetadata = projectData.metadata();
        QJsonObject updatedMetadata = originalMetadata;
        updatedMetadata[QStringLiteral("dem_results")] = QJsonArray();
        QJsonObject thirdMetadata = updatedMetadata;
        thirdMetadata[QStringLiteral("concurrent_edit")] =
            QStringLiteral("must-survive");
        const QString transactionId = QStringLiteral(
            "crashed-with-third-state");
        transactionRoot = QDir(managedRoot(projectData)).filePath(
            QStringLiteral(".plascan_cleanup_trash/%1")
                .arg(transactionId));
        ASSERT_TRUE(QDir().mkpath(transactionRoot));
        const QString stagedPath = QDir(transactionRoot).filePath(
            QStringLiteral("000000_terrain.tif"));
        ASSERT_TRUE(QDir().rename(artifactPath, stagedPath));
        createRecoveryTransaction(
            &projectData,
            transactionId,
            artifactPath,
            stagedPath,
            QStringLiteral("staging"),
            QStringLiteral("staged"),
            false,
            originalMetadata,
            updatedMetadata);

        QString commitError;
        ASSERT_TRUE(projectData.commitResourceCleanupMetadata(
            thirdMetadata, &commitError)) << commitError.toStdString();
    }

    ProjectData reopened;
    xjw::core::project::ProjectResourceCleanupService::
        installAutomaticRecovery(&reopened);
    QString openError;
    ASSERT_TRUE(reopened.openProject(projectPath, &openError))
        << openError.toStdString();

    EXPECT_TRUE(QFileInfo::exists(artifactPath));
    EXPECT_FALSE(QFileInfo::exists(transactionRoot));
    EXPECT_EQ(reopened.metadata().value(
                  QStringLiteral("concurrent_edit")).toString(),
              QStringLiteral("must-survive"));
    EXPECT_TRUE(reopened.metadata().value(
        QStringLiteral("dem_results")).toArray().isEmpty());
}

TEST(ProjectResourceCleanupTest,
     PreparedCleanupCommitsOnWorkerAndFinalizesOnSessionThread)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("prepared_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(
        projectPath, QStringLiteral("prepared_cleanup")));

    const QString outputDirectory = QDir(managedRoot(projectData)).filePath(
        QStringLiteral("reconstruction/terrain/prepared-run"));
    const QString demPath = QDir(outputDirectory).filePath(
        QStringLiteral("prepared.tif"));
    writeArtifact(demPath);
    setResultRecords(
        &projectData,
        QStringLiteral("dem_results"),
        QJsonArray{demRecord(demPath, outputDirectory)});
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const auto prepared =
        xjw::core::project::ProjectResourceCleanupService::
            prepareGeneratedDataCleanup(
                &projectData,
                QStringLiteral("DEM"),
                QStringList{demPath});
    ASSERT_TRUE(prepared.requiresExecution());
    ASSERT_TRUE(projectData.metadata().value(
        QStringLiteral("dem_results")).toArray().isEmpty());

    xjw::core::project::ResourceCleanupResult workerResult;
    std::thread worker([&]()
    {
        workerResult =
            xjw::core::project::ProjectResourceCleanupService::
                executePreparedCleanup(prepared);
    });
    worker.join();

    ASSERT_TRUE(workerResult.success)
        << workerResult.errorMessage.toStdString();
    EXPECT_FALSE(QFileInfo::exists(demPath));
    EXPECT_TRUE(projectData.metadata().value(
        QStringLiteral("dem_results")).toArray().isEmpty());
    EXPECT_TRUE(
        xjw::core::project::ProjectResourceCleanupService::
            finalizePreparedCleanup(
                &projectData, prepared, workerResult));
    EXPECT_TRUE(projectData.metadata().value(
        QStringLiteral("dem_results")).toArray().isEmpty());
}

TEST(ProjectResourceCleanupTest,
     NewerMetadataGenerationInvalidatesPreparedCleanupWithoutDataLoss)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("stale_prepared_cleanup.plascan"));
    ASSERT_TRUE(projectData.createProject(
        projectPath, QStringLiteral("stale_prepared_cleanup")));

    const QString outputDirectory = QDir(managedRoot(projectData)).filePath(
        QStringLiteral("reconstruction/terrain/stale-run"));
    const QString demPath = QDir(outputDirectory).filePath(
        QStringLiteral("stale.tif"));
    writeArtifact(demPath);
    setResultRecords(
        &projectData,
        QStringLiteral("dem_results"),
        QJsonArray{demRecord(demPath, outputDirectory)});
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const auto prepared =
        xjw::core::project::ProjectResourceCleanupService::
            prepareGeneratedDataCleanup(
                &projectData,
                QStringLiteral("DEM"),
                QStringList{demPath});
    ASSERT_TRUE(prepared.requiresExecution());

    QString blockedError;
    EXPECT_FALSE(projectData.saveProject(&blockedError));
    EXPECT_FALSE(blockedError.isEmpty());
    blockedError.clear();
    EXPECT_FALSE(projectData.commitResourceCleanupMetadata(
        projectData.metadata(), &blockedError));
    EXPECT_FALSE(blockedError.isEmpty());
    EXPECT_FALSE(projectData.saveTemporaryMetadata());
    blockedError.clear();
    EXPECT_FALSE(projectData.openProject(
        projectPath, &blockedError));
    EXPECT_FALSE(blockedError.isEmpty());
    projectData.closeProject();
    EXPECT_TRUE(projectData.hasProject());

    QJsonObject newerMetadata = projectData.metadata();
    newerMetadata[QStringLiteral("concurrent_edit")] = true;
    projectData.updateMetadata(newerMetadata, true);

    const auto workerResult =
        xjw::core::project::ProjectResourceCleanupService::
            executePreparedCleanup(prepared);
    EXPECT_FALSE(workerResult.success);
    EXPECT_FALSE(workerResult.metadataStateCommitted);
    EXPECT_FALSE(
        xjw::core::project::ProjectResourceCleanupService::
            finalizePreparedCleanup(
                &projectData, prepared, workerResult));
    EXPECT_TRUE(QFileInfo::exists(demPath));
    EXPECT_TRUE(projectData.metadata().value(
        QStringLiteral("concurrent_edit")).toBool());
    EXPECT_EQ(projectData.metadata().value(
                  QStringLiteral("dem_results")).toArray().size(),
              1);
    EXPECT_TRUE(QFileInfo::exists(QDir(managedRoot(projectData)).filePath(
        QStringLiteral(".plascan_cleanup_trash"))))
        << "元数据代次冲突时必须保留 WAL，供下次清理或重开恢复";
}

TEST(ProjectResourceCleanupTest,
     FinalizePreservesMetadataAddedAfterWorkerCommit)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("post_commit_edit.plascan"));
    ASSERT_TRUE(projectData.createProject(
        projectPath, QStringLiteral("post_commit_edit")));

    const QString outputDirectory = QDir(managedRoot(projectData)).filePath(
        QStringLiteral("reconstruction/terrain/post-commit-run"));
    const QString demPath = QDir(outputDirectory).filePath(
        QStringLiteral("post-commit.tif"));
    writeArtifact(demPath);
    setResultRecords(
        &projectData,
        QStringLiteral("dem_results"),
        QJsonArray{demRecord(demPath, outputDirectory)});
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const auto prepared =
        xjw::core::project::ProjectResourceCleanupService::
            prepareGeneratedDataCleanup(
                &projectData,
                QStringLiteral("DEM"),
                QStringList{demPath});
    ASSERT_TRUE(prepared.requiresExecution());
    const auto workerResult =
        xjw::core::project::ProjectResourceCleanupService::
            executePreparedCleanup(prepared);
    ASSERT_TRUE(workerResult.success)
        << workerResult.errorMessage.toStdString();

    QJsonObject concurrentMetadata = projectData.metadata();
    concurrentMetadata[QStringLiteral("post_commit_edit")] =
        QStringLiteral("preserve-me");
    projectData.updateMetadata(concurrentMetadata, true);
    projectData.scheduleTemporaryMetadataSave();

    EXPECT_FALSE(
        xjw::core::project::ProjectResourceCleanupService::
            finalizePreparedCleanup(
                &projectData, prepared, workerResult));
    EXPECT_FALSE(QFileInfo::exists(demPath));
    EXPECT_TRUE(projectData.metadata().value(
        QStringLiteral("dem_results")).toArray().isEmpty());
    EXPECT_EQ(projectData.metadata().value(
                  QStringLiteral("post_commit_edit")).toString(),
              QStringLiteral("preserve-me"));
}

TEST(ProjectResourceCleanupTest,
     RecoveryRejectsLinkedTransactionDestination)
{
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData projectData;
    const QString projectPath = temporary.filePath(
        QStringLiteral("linked_recovery.plascan"));
    ASSERT_TRUE(projectData.createProject(
        projectPath, QStringLiteral("linked_recovery")));
    QString saveError;
    ASSERT_TRUE(projectData.saveProject(&saveError))
        << saveError.toStdString();

    const QString root = managedRoot(projectData);
    const QString sourceDirectory = QDir(root).filePath(
        QStringLiteral("reconstruction/terrain/linked-source"));
    const QString externalDirectory = temporary.filePath(
        QStringLiteral("external-linked-transaction"));
    const QString externalArtifact = QDir(externalDirectory).filePath(
        QStringLiteral("must-survive.bin"));
    writeArtifact(externalArtifact);

    const QString transactionId = QStringLiteral("linked-destination");
    const QString transactionRoot = QDir(root).filePath(
        QStringLiteral(".plascan_cleanup_trash/%1")
            .arg(transactionId));
    ASSERT_TRUE(QDir().mkpath(transactionRoot));
    const QString linkedDestination = QDir(transactionRoot).filePath(
        QStringLiteral("000000_linked-source"));
    QString linkError;
    ASSERT_TRUE(createDirectoryLink(externalDirectory,
                                    linkedDestination,
                                    &linkError))
        << linkError.toStdString();
    createRecoveryTransaction(&projectData,
                              transactionId,
                              sourceDirectory,
                              linkedDestination,
                              QStringLiteral("staging"),
                              QStringLiteral("staged"),
                              true);

    const auto result =
        xjw::core::project::ProjectResourceCleanupService::
            cleanupGeneratedData(
                &projectData,
                QStringLiteral("DEM"),
                QStringList{temporary.filePath(
                    QStringLiteral("missing-external.tif"))});

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.errorMessage.isEmpty());
    EXPECT_TRUE(QFileInfo::exists(externalArtifact));
    EXPECT_TRUE(QFileInfo::exists(transactionRoot));
    const QFileInfo linkInfo(linkedDestination);
    EXPECT_TRUE(linkInfo.isSymLink() || linkInfo.isJunction());
}
