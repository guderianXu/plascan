#include "project/services/ProjectResourceService.h"

#include "project/services/ProjectLifecycleService.h"
#include "project/services/ProjectSession.h"
#include "project/services/ProjectResourceCleanupCoordinator.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "project/ProjectSessionModel.h"
#include "project/ProjectIO.h"
#include "project/ProjectWorkspaceStore.h"
#include "project/support/ProjectSurveyControl.h"
#include "placamera_runtime/ProjectCameraStore.h"
#include "io/MarkerCsv.h"

#include <placamera/rpc_camera.h>

#include <gdal_priv.h>

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QFile>
#include <QDirIterator>
#include <QThread>
#include <QTemporaryDir>
#include <QStringList>

#include <gtest/gtest.h>

#include <functional>

namespace
{

    QCoreApplication& qtApplication()
    {
        static QCoreApplication* application = nullptr;
        if (!application)
        {
            int argc = 1;
            auto* argv = new char*[2];
            argv[0] = const_cast<char*>("test_project_resource_service");
            argv[1] = nullptr;
            application = new QCoreApplication(argc, argv);
        }
        return *application;
    }

    bool waitUntil(const std::function<bool()>& predicate, int timeoutMs = 2000)
    {
        QElapsedTimer timer;
        timer.start();
        while (!predicate() && timer.elapsed() < timeoutMs)
        {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
            QThread::msleep(5);
        }
        return predicate();
    }

    class CancelMessages final : public ProjectUiMessageAdapter
    {
    public:
        void information(QWidget*, const QString&, const QString&) override
        {
        }
        void warning(QWidget*, const QString&, const QString&) override
        {
        }
        void critical(QWidget*, const QString&, const QString&) override
        {
        }
        UiAnswer question(QWidget*, const QString&, const QString&, UiAnswer) override
        {
            return UiAnswer::Cancel;
        }
        UiDialogResult getText(QWidget*, const QString&, const QString&, const QString&) override
        {
            return {};
        }
        UiDialogResult getDouble(QWidget*, const QString&, const QString&, double, double, double, int) override
        {
            return {};
        }
        UiDialogResult getItem(QWidget*, const QString&, const QString&, const QStringList&, int) override
        {
            return {};
        }
        UiDialogResult selectOpenFile(QWidget*, const QString&, const QString&, const QString&, QDir::Filters) override
        {
            return {};
        }
        UiDialogResult selectOpenFiles(QWidget*, const QString&, const QString&, const QString&, QDir::Filters) override
        {
            return {};
        }
        UiDialogResult selectDirectory(QWidget*, const QString&, const QString&, QDir::Filters) override
        {
            return {};
        }
        UiDialogResult
        selectSaveFile(QWidget*, const QString&, const QString&, const QString&, QDir::Filters, const QString&) override
        {
            return {};
        }
    };

    QString rpcCoefficients(int activeIndex)
    {
        QStringList values;
        for (int index = 0; index < 20; ++index)
        {
            values.append(index == activeIndex ? QStringLiteral("1") : QStringLiteral("0"));
        }
        return values.join(QLatin1Char(' '));
    }

    bool writeRpcRaster(const QString& path)
    {
        GDALAllRegister();
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (!driver)
        {
            return false;
        }
        GDALDataset* dataset = driver->Create(QFile::encodeName(path).constData(), 32, 24, 1, GDT_Byte, nullptr);
        if (!dataset)
        {
            return false;
        }
        const QMap<QString, QString> metadata{{QStringLiteral("LINE_OFF"), QStringLiteral("12")},
                                              {QStringLiteral("SAMP_OFF"), QStringLiteral("16")},
                                              {QStringLiteral("LAT_OFF"), QStringLiteral("20")},
                                              {QStringLiteral("LONG_OFF"), QStringLiteral("110")},
                                              {QStringLiteral("HEIGHT_OFF"), QStringLiteral("1000")},
                                              {QStringLiteral("LINE_SCALE"), QStringLiteral("12")},
                                              {QStringLiteral("SAMP_SCALE"), QStringLiteral("16")},
                                              {QStringLiteral("LAT_SCALE"), QStringLiteral("0.1")},
                                              {QStringLiteral("LONG_SCALE"), QStringLiteral("0.1")},
                                              {QStringLiteral("HEIGHT_SCALE"), QStringLiteral("1000")},
                                              {QStringLiteral("LINE_NUM_COEFF"), rpcCoefficients(2)},
                                              {QStringLiteral("LINE_DEN_COEFF"), rpcCoefficients(0)},
                                              {QStringLiteral("SAMP_NUM_COEFF"), rpcCoefficients(1)},
                                              {QStringLiteral("SAMP_DEN_COEFF"), rpcCoefficients(0)}};
        bool written = true;
        for (auto it = metadata.constBegin(); it != metadata.constEnd(); ++it)
        {
            written = dataset->SetMetadataItem(it.key().toUtf8().constData(), it.value().toUtf8().constData(), "RPC") ==
                          CE_None &&
                      written;
        }
        GDALClose(dataset);
        return written;
    }

} // namespace

TEST(ProjectResourceServiceTest, CancelledPhotoSelectionDoesNotMutateData)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("photos.plascan")),
                                   QStringLiteral("photos")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    const QJsonObject before = data.metadata();
    const bool dirtyBefore = data.isDirty();

    const auto result = service.addPhoto();

    EXPECT_EQ(result.status, xjw::gui::project::OperationStatus::Cancelled);
    EXPECT_EQ(data.metadata(), before);
    EXPECT_EQ(data.isDirty(), dirtyBefore);
}

TEST(ProjectResourceServiceTest, EmptyExplicitImportIsCancelled)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("asset.plascan")),
                                   QStringLiteral("asset")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    const QJsonObject before = data.metadata();
    const bool dirtyBefore = data.isDirty();

    const auto result = service.importProjectAsset(QString(), false);

    EXPECT_EQ(result.status, xjw::gui::project::OperationStatus::Cancelled);
    EXPECT_EQ(data.metadata(), before);
    EXPECT_EQ(data.isDirty(), dirtyBefore);
}

TEST(ProjectResourceServiceTest, CancelledReferenceAndSurveyImportsDoNotMutateData)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("imports.plascan")),
                                   QStringLiteral("imports")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    const QJsonObject before = data.metadata();
    const bool dirtyBefore = data.isDirty();

    const auto referenceResult = service.importReferenceDataset();
    const auto surveyResult = service.importSurveyControlCsv();

    EXPECT_EQ(referenceResult.status, xjw::gui::project::OperationStatus::Cancelled);
    EXPECT_EQ(surveyResult.status, xjw::gui::project::OperationStatus::Cancelled);
    EXPECT_EQ(data.metadata(), before);
    EXPECT_EQ(data.isDirty(), dirtyBefore);
}

TEST(ProjectResourceServiceTest, InvalidBaPriorImportDoesNotMutateMetadata)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString invalidPath = temporary.filePath(QStringLiteral("prior.csv"));
    QFile invalid(invalidPath);
    ASSERT_TRUE(invalid.open(QIODevice::WriteOnly));
    ASSERT_GT(invalid.write("not-a-supported-prior"), 0);
    invalid.close();

    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("ba_prior.plascan")),
                                   QStringLiteral("ba_prior")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    const QJsonObject before = data.metadata();
    const bool dirtyBefore = data.isDirty();

    const auto result = service.registerReferenceDataset(invalidPath, QString(), QStringLiteral("ba_prior"));

    EXPECT_EQ(result.status, xjw::gui::project::OperationStatus::Success);
    ASSERT_TRUE(waitUntil([&service] { return !service.isBusy(); }));
    EXPECT_EQ(data.metadata(), before);
    EXPECT_EQ(data.isDirty(), dirtyBefore);
}

TEST(ProjectResourceServiceTest, ResourceMutationIsRejectedBySharedLifecycleGate)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString imagePath = temporary.filePath(QStringLiteral("busy.tif"));
    QFile image(imagePath);
    ASSERT_TRUE(image.open(QIODevice::WriteOnly));
    ASSERT_GT(image.write("fake image"), 0);
    image.close();

    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("busy.plascan")), QStringLiteral("busy")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    ASSERT_TRUE(session.tryBeginOperation(QStringLiteral("打开项目")));
    const QJsonObject before = data.metadata();

    const auto result = service.addPhotos({imagePath}, QStringLiteral("busy"));

    EXPECT_EQ(result.status, xjw::gui::project::OperationStatus::Failed);
    EXPECT_EQ(data.metadata(), before);
    session.endOperation(QStringLiteral("打开项目"));
}

TEST(ProjectResourceServiceTest, PackResourceWritesMetadataAfterWorkerCompletes)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString sourcePath = temporary.filePath(QStringLiteral("external.txt"));
    QFile source(sourcePath);
    ASSERT_TRUE(source.open(QIODevice::WriteOnly));
    ASSERT_GT(source.write("pack me"), 0);
    source.close();

    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("pack.plascan")), QStringLiteral("pack")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    bool changed = false;
    QObject::connect(&service, &xjw::gui::project::ProjectResourceService::resourceChanged,
                     [&changed]() { changed = true; });

    const auto result = service.packResource(sourcePath);

    ASSERT_EQ(result.status, xjw::gui::project::OperationStatus::Success);
    ASSERT_TRUE(waitUntil([&changed] { return changed; }));
    EXPECT_FALSE(data.metadata().value(QStringLiteral("packed_resources")).toArray().isEmpty());
    EXPECT_FALSE(service.isBusy());
}

TEST(ProjectResourceServiceTest, StaleSurveyImportDoesNotReplaceSidecarOrMetadata)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData data;
    const QString projectPath = temporary.filePath(QStringLiteral("survey.plascan"));
    ASSERT_TRUE(data.createProject(projectPath, QStringLiteral("survey")));
    const QString sidecarPath = xjw::common::project::ProjectIO::markerSetPath(projectPath);
    const auto oldMarkers = xjw::control_points::parseMarkerCsv(
        QStringLiteral("role,id,x,y,z\ncontrol,OLD,1,2,3\n"));
    ASSERT_TRUE(oldMarkers.ok) << oldMarkers.error.toStdString();
    ASSERT_TRUE(xjw::gui::project::writeSurveyControlMarkerSet(sidecarPath, oldMarkers.markerSet).imported);

    QFile oldSidecar(sidecarPath);
    ASSERT_TRUE(oldSidecar.open(QIODevice::ReadOnly));
    const QByteArray oldBytes = oldSidecar.readAll();
    const QJsonObject oldMetadata = data.coreFilesMeta();

    const QString csvPath = temporary.filePath(QStringLiteral("new-control.csv"));
    QFile csv(csvPath);
    ASSERT_TRUE(csv.open(QIODevice::WriteOnly | QIODevice::Text));
    ASSERT_GT(csv.write("role,id,x,y,z\ncontrol,NEW,4,5,6\n"), 0);
    csv.close();

    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    const auto result = service.importSurveyControlCsv(csvPath);
    ASSERT_EQ(result.status, xjw::gui::project::OperationStatus::Success);

    // Let the worker reach its completion barrier, but do not process the GUI
    // callback yet.  The callback must still be the first code allowed to
    // touch the final sidecar.
    QThread::msleep(100);
    // Force the queued callback stale before it can commit the prepared import.
    session.advanceGeneration();
    ASSERT_TRUE(waitUntil([&service] { return !service.isBusy(); }));

    QFile unchangedSidecar(sidecarPath);
    ASSERT_TRUE(unchangedSidecar.open(QIODevice::ReadOnly));
    EXPECT_EQ(unchangedSidecar.readAll(), oldBytes);
    EXPECT_EQ(data.coreFilesMeta(), oldMetadata);
    const QDir sidecarDirectory(QFileInfo(sidecarPath).absolutePath());
    EXPECT_TRUE(sidecarDirectory.entryList({QStringLiteral("marker_set.json.*")},
                                           QDir::Files | QDir::Hidden).isEmpty());
}

TEST(ProjectResourceServiceTest, FailedPackDoesNotLeavePartiallyStagedResource)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString sourcePath = temporary.filePath(QStringLiteral("partially-readable"));
    ASSERT_TRUE(QDir().mkpath(sourcePath));
    QFile readable(QDir(sourcePath).filePath(QStringLiteral("a-readable.bin")));
    ASSERT_TRUE(readable.open(QIODevice::WriteOnly));
    ASSERT_GT(readable.write("copied first"), 0);
    readable.close();
    const QString blockedPath = QDir(sourcePath).filePath(QStringLiteral("z-blocked.bin"));
    QFile blocked(blockedPath);
    ASSERT_TRUE(blocked.open(QIODevice::WriteOnly));
    ASSERT_GT(blocked.write("must fail"), 0);
    blocked.close();
    ASSERT_TRUE(blocked.setPermissions(QFileDevice::Permissions()));

    ProjectData data;
    const QString projectPath = temporary.filePath(QStringLiteral("pack-failure.plascan"));
    ASSERT_TRUE(data.createProject(projectPath, QStringLiteral("pack-failure")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);

    const auto result = service.packResource(sourcePath);
    ASSERT_EQ(result.status, xjw::gui::project::OperationStatus::Success);
    ASSERT_TRUE(waitUntil([&service] { return !service.isBusy(); }));
    EXPECT_TRUE(data.coreFilesMeta().value(QStringLiteral("packed_resources")).toArray().isEmpty());

    const QString runtimeRoot = ProjectWorkspaceStore(projectPath, data.activeChunkDirectory()).runtimeRoot();
    ASSERT_FALSE(runtimeRoot.isEmpty());
    QDirIterator iterator(runtimeRoot, QDir::AllEntries | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
    while (iterator.hasNext())
    {
        const QString path = QDir::fromNativeSeparators(QDir::cleanPath(iterator.next()));
        EXPECT_FALSE(path.contains(QStringLiteral("/assets/packed/"))) << qPrintable(path);
        EXPECT_FALSE(path.contains(QStringLiteral(".packing-"))) << qPrintable(path);
    }
    ASSERT_TRUE(blocked.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner));
}

TEST(ProjectResourceServiceTest, DuplicatePhotoImportDoesNotMutateSavedMetadata)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString imagePath = temporary.filePath(QStringLiteral("image.tif"));
    QFile image(imagePath);
    ASSERT_TRUE(image.open(QIODevice::WriteOnly));
    ASSERT_GT(image.write("fake image"), 0);
    image.close();

    ProjectData data;
    ASSERT_TRUE(
        data.createProject(temporary.filePath(QStringLiteral("duplicate.plascan")), QStringLiteral("duplicate")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    bool importFinished = false;
    QObject::connect(&service,
                     &xjw::gui::project::ProjectResourceService::imageImportFinished,
                     [&importFinished](bool, const QString&) { importFinished = true; });

    const auto firstResult = service.addPhotos({imagePath}, QStringLiteral("test"));
    ASSERT_EQ(firstResult.status, xjw::gui::project::OperationStatus::Success);
    ASSERT_TRUE(waitUntil([&importFinished] { return importFinished; }));
    QString saveError;
    ASSERT_TRUE(data.saveProject(&saveError)) << qPrintable(saveError);
    const QJsonObject before = data.metadata();

    importFinished = false;
    const auto duplicateResult = service.addPhotos({imagePath}, QStringLiteral("duplicate"));

    EXPECT_EQ(duplicateResult.status, xjw::gui::project::OperationStatus::Success);
    ASSERT_TRUE(waitUntil([&importFinished] { return importFinished; }));
    EXPECT_EQ(data.metadata(), before);
    EXPECT_FALSE(data.isDirty());
    EXPECT_EQ(session.allImages(), QStringList{QDir::cleanPath(imagePath)});
}

TEST(ProjectResourceServiceTest, ImportsRpcRasterAsNativeCamera)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString imagePath = temporary.filePath(QStringLiteral("rpc.tif"));
    ASSERT_TRUE(writeRpcRaster(imagePath));

    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("rpc.plascan")), QStringLiteral("rpc")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    bool finished = false;
    bool succeeded = false;
    QObject::connect(&service,
                     &xjw::gui::project::ProjectResourceService::imageImportFinished,
                     [&finished, &succeeded](bool success, const QString&)
                     {
                         finished = true;
                         succeeded = success;
                     });

    const auto result = service.addPhotos({imagePath}, QStringLiteral("rpc-test"));

    ASSERT_EQ(result.status, xjw::gui::project::OperationStatus::Success);
    ASSERT_TRUE(waitUntil([&finished] { return finished; }));
    ASSERT_TRUE(succeeded);
    const auto loaded = xjw::placamera_runtime::loadProjectCameras(data.coreFilesMeta());
    ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
    ASSERT_EQ(loaded.instances.size(), 1U);
    const auto* camera = dynamic_cast<const placamera::RpcModel*>(loaded.instances.values().front().get());
    ASSERT_NE(camera, nullptr);
    EXPECT_EQ(camera->imageSize().samples, 32);
    EXPECT_EQ(camera->imageSize().lines, 24);
    EXPECT_DOUBLE_EQ(camera->rpcDefinition().parameters().longitudeOffset, 110.0);
    const QJsonObject record = data.coreFilesMeta().value(QStringLiteral("camera_instances")).toArray().first().toObject();
    EXPECT_EQ(record.value(QStringLiteral("state")).toObject().value(QStringLiteral("source")),
              QStringLiteral("rpc_raster"));
}

TEST(ProjectResourceServiceTest, PortableExportRejectsDuplicateRequest)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("portable.plascan")),
                                   QStringLiteral("portable")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    ProjectLifecycleService lifecycle(&data, &session, &messages, nullptr, nullptr);
    xjw::gui::project::ProjectResourceService service(&session, &messages, nullptr, nullptr);
    service.setPortableExportLauncher(
        [&lifecycle](const QString& path, QString* error) { return lifecycle.startPortableExport(path, error); });

    const auto firstResult = service.exportPortableProject(temporary.filePath(QStringLiteral("portable.zip")));
    ASSERT_EQ(firstResult.status, xjw::gui::project::OperationStatus::Success);
    EXPECT_TRUE(service.isPortableExportInProgress());

    const auto duplicateResult = service.exportPortableProject(temporary.filePath(QStringLiteral("portable2.zip")));

    EXPECT_EQ(duplicateResult.status, xjw::gui::project::OperationStatus::Failed);
    ASSERT_TRUE(waitUntil([&service] { return !service.isPortableExportInProgress(); }));
}

TEST(ProjectResourceServiceTest, CancelledCleanupDoesNotMutateData)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("cleanup.plascan")),
                                   QStringLiteral("cleanup")));
    xjw::gui::project::ProjectSession session(&data);
    CancelMessages messages;
    xjw::gui::project::ProjectResourceCleanupCoordinator cleanup(&session, &messages);
    xjw::gui::project::ProjectResourceService service(&session, &messages, &cleanup, nullptr);
    const QJsonObject before = data.metadata();
    const bool dirtyBefore = data.isDirty();

    const auto result = service.deleteGeneratedData(QStringLiteral("3D模型"),
                                                    {QStringLiteral("missing.obj")});

    EXPECT_EQ(result.status, xjw::gui::project::OperationStatus::Cancelled);
    EXPECT_EQ(data.metadata(), before);
    EXPECT_EQ(data.isDirty(), dirtyBefore);
    EXPECT_FALSE(cleanup.isRunning());
}

TEST(ProjectResourceServiceTest, OperationResultReportsSuccessOnlyForSuccessfulStatus)
{
    xjw::gui::project::OperationResult result{
        xjw::gui::project::OperationStatus::Success, QStringLiteral("ok")};
    EXPECT_TRUE(result.succeeded());
    result.status = xjw::gui::project::OperationStatus::Cancelled;
    EXPECT_FALSE(result.succeeded());
}
