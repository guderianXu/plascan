#include "project/services/ProjectSession.h"

#include "ProjectCameraIO.h"
#include "camera/project/CameraProjectRecords.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "project/ProjectIO.h"
#include "project/ProjectSessionModel.h"

#include <QCoreApplication>
#include <QDir>
#include <QEvent>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QThreadPool>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <concepts>
#include <thread>

class ProjectDataPersistenceTestPeer
{
public:
    static bool running(const ProjectData& project)
    {
        return project._persistenceRunning;
    }

    static bool hasPendingRequest(const ProjectData& project)
    {
        return project._fullSavePending || project._archiveSyncPending || project._temporarySavePending;
    }

    static quint64 persistenceGeneration(const ProjectData& project)
    {
        return project._persistenceCommitCoordinator ? project._persistenceCommitCoordinator->currentGeneration() : 0;
    }

    static void settle(ProjectData* project)
    {
        ASSERT_NE(project, nullptr);
        ASSERT_NE(project->_persistencePool, nullptr);
        for (int attempt = 0; attempt < 8; ++attempt)
        {
            project->_persistencePool->waitForDone();
            QCoreApplication::sendPostedEvents(project, QEvent::MetaCall);
            QCoreApplication::processEvents(QEventLoop::AllEvents);
            if (!project->_persistenceRunning && !hasPendingRequest(*project))
            {
                return;
            }
        }
        FAIL() << "project persistence did not settle";
    }
};

namespace
{

    using xjw::gui::project::ProjectSession;
    using xjw::gui::project::ProjectSessionContext;

    template <typename T>
    concept PublishesImageMaskRecords = requires(T & session,
                                                 const ProjectSessionContext& expected,
                                                 const QMap<QString, QJsonObject>& records,
                                                 QStringList* images,
                                                 QString* error)
    {
        session.publishImageMaskRecords(expected, records, images, error);
    };

    template <typename T>
    concept ClearsImageMaskRecords = requires(T & session,
                                              const ProjectSessionContext& expected,
                                              const QStringList& requested,
                                              QStringList* images,
                                              QString* error)
    {
        session.clearImageMaskRecords(expected, requested, images, error);
    };

    static_assert(PublishesImageMaskRecords<ProjectSession>);
    static_assert(ClearsImageMaskRecords<ProjectSession>);

    QJsonObject canonicalFrameCamera(int width, int height, const std::array<double, 3>& center)
    {
        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        camera.setPixelPitch(0.01);
        camera.setIntrinsics(1200.0, 1200.0, width * 0.5, height * 0.5);
        camera.setImageSize(xjw::camera_core::ImageSize{width, height});
        camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, center);
        QJsonObject result = xjw::common::project::serializeFramePinholeNumericState(camera);
        result.insert(QStringLiteral("aligned"), true);
        return result;
    }

    class ScopedCoreApplication final
    {
    public:
        ScopedCoreApplication()
        {
            if (!QCoreApplication::instance())
            {
                _arguments[0] = _applicationName;
                _application = std::make_unique<QCoreApplication>(_argumentCount, _arguments);
            }
        }

    private:
        int _argumentCount = 1;
        char _applicationName[32] = "test_project_session";
        char* _arguments[2] = {nullptr, nullptr};
        std::unique_ptr<QCoreApplication> _application;
    };

    xjw::camera_project::CameraInstanceUpdates
    prepareSessionBundleAdjustUpdate(ProjectData* project, const QString& directory, QString* errorMessage)
    {
        const QString imagePath = QDir(directory).filePath(QStringLiteral("session-stage.jpg"));
        QFile image(imagePath);
        if (!image.open(QIODevice::WriteOnly) || image.write("image") != 5)
        {
            if (errorMessage)
            {
                *errorMessage = QStringLiteral("无法创建 session stage 测试影像");
            }
            return {};
        }
        image.close();
        if (!project->addImages({imagePath}, errorMessage))
        {
            return {};
        }
        const QJsonObject camera = canonicalFrameCamera(1024, 768, {1.0, 2.0, 3.0});
        int updatedCount = 0;
        if (!project->setCameraInstances({{imagePath, camera}}, &updatedCount, errorMessage))
        {
            return {};
        }
        const QJsonObject core = project->coreFilesMeta();
        const QJsonObject imageRecord = core.value(QStringLiteral("images")).toArray().at(0).toObject();
        const QString imageId = imageRecord.value(QStringLiteral("image_uuid")).toString();
        const QJsonObject instance = xjw::camera_project::CameraProjectRecords::instanceForImage(core, imageId);
        const QString worldFrame = core.value(QStringLiteral("camera_definitions"))
                                       .toArray()
                                       .at(0)
                                       .toObject()
                                       .value(QStringLiteral("frame"))
                                       .toString();
        QJsonObject update = xjw::camera_project::CameraProjectRecords::modelParametersForImage(core, imageRecord);
        update.insert(QStringLiteral("world_frame"), worldFrame);
        update.insert(QStringLiteral("solution"), QStringLiteral("session-stage"));
        return {{xjw::camera_core::ImageId(imageId.toStdString()),
                 xjw::camera_core::CameraInstanceId(instance.value(QStringLiteral("id")).toString().toStdString()),
                 xjw::coordinate_system::CoordinateFrameId(worldFrame.toStdString()),
                 update}};
    }

    class ProjectSessionTest : public testing::Test
    {
    protected:
        ProjectData data;
        ProjectSession session{&data};
    };

    TEST_F(ProjectSessionTest, ExposesNamedAtomicMaskMutationPorts)
    {
        EXPECT_TRUE((PublishesImageMaskRecords<ProjectSession>));
        EXPECT_TRUE((ClearsImageMaskRecords<ProjectSession>));
    }

    TEST(ProjectSessionMaskMutationTest, PublishesAndClearsOnlyNamedMaskFieldsWithoutLostUpdates)
    {
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        ProjectData projectData;
        ProjectSession session(&projectData);
        const QString projectPath = temporaryDirectory.filePath(QStringLiteral("mask_ports.plascan"));
        const QString imagePath = temporaryDirectory.filePath(QStringLiteral("image.png"));
        QFile imageFile(imagePath);
        ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
        ASSERT_EQ(imageFile.write("image"), 5);
        imageFile.close();
        ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("mask_ports")));
        ASSERT_TRUE(projectData.addImages({imagePath}));

        QJsonObject metadata = projectData.coreFilesMeta();
        metadata.insert(QStringLiteral("same_generation_marker"), QStringLiteral("preserve"));
        QJsonArray images = metadata.value(QStringLiteral("images")).toArray();
        QJsonObject image = images.at(0).toObject();
        image.insert(QStringLiteral("rating"), 5);
        image.insert(QStringLiteral("mask_model_id"), QStringLiteral("old-ai"));
        images.replace(0, image);
        metadata.insert(QStringLiteral("images"), images);
        projectData.persistMetadata(metadata, false);

        int metadataChangedCount = 0;
        QObject::connect(&projectData,
                         &ProjectData::metadataChanged,
                         [&metadataChangedCount](const QJsonObject&) { ++metadataChangedCount; });
        const auto expected = session.context();
        const QString resolvedImage =
            QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(projectPath, imagePath));
        const QJsonObject record{{QStringLiteral("mask_path"), QStringLiteral("mask.png")},
                                 {QStringLiteral("mask_method"), QStringLiteral("threshold")},
                                 {QStringLiteral("mask_updated_at"), QStringLiteral("2026-09-20T00:00:00Z")}};
        QStringList updatedImages;
        QString errorMessage;

        ASSERT_TRUE(session.publishImageMaskRecords(expected, {{resolvedImage, record}}, &updatedImages, &errorMessage))
            << qPrintable(errorMessage);
        EXPECT_EQ(updatedImages, QStringList{resolvedImage});
        EXPECT_EQ(metadataChangedCount, 1);
        QJsonObject updated = projectData.coreFilesMeta();
        QJsonObject updatedImage = updated.value(QStringLiteral("images")).toArray().at(0).toObject();
        EXPECT_EQ(updated.value(QStringLiteral("same_generation_marker")).toString(), QStringLiteral("preserve"));
        EXPECT_EQ(updatedImage.value(QStringLiteral("rating")).toInt(), 5);
        EXPECT_EQ(updatedImage.value(QStringLiteral("mask_method")).toString(), QStringLiteral("threshold"));
        EXPECT_FALSE(updatedImage.contains(QStringLiteral("mask_model_id")));

        updatedImages = {QStringLiteral("sentinel")};
        ASSERT_TRUE(
            session.publishImageMaskRecords(expected, {{resolvedImage, record}}, &updatedImages, &errorMessage));
        EXPECT_TRUE(updatedImages.isEmpty());
        EXPECT_EQ(metadataChangedCount, 1);

        QStringList clearedImages;
        ASSERT_TRUE(session.clearImageMaskRecords(expected, {resolvedImage}, &clearedImages, &errorMessage));
        EXPECT_EQ(clearedImages, QStringList{resolvedImage});
        EXPECT_EQ(metadataChangedCount, 2);
        updatedImage = projectData.coreFilesMeta().value(QStringLiteral("images")).toArray().at(0).toObject();
        EXPECT_EQ(updatedImage.value(QStringLiteral("rating")).toInt(), 5);
        EXPECT_FALSE(updatedImage.contains(QStringLiteral("mask_path")));
        EXPECT_FALSE(updatedImage.contains(QStringLiteral("mask_method")));

        clearedImages = {QStringLiteral("sentinel")};
        ASSERT_TRUE(session.clearImageMaskRecords(expected, {resolvedImage}, &clearedImages, &errorMessage));
        EXPECT_TRUE(clearedImages.isEmpty());
        EXPECT_EQ(metadataChangedCount, 2);
    }

    TEST(ProjectSessionMaskMutationTest, RejectsStaleUnknownDuplicateAndUnknownFieldBeforeMutation)
    {
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        ProjectData projectData;
        ProjectSession session(&projectData);
        const QString projectPath = temporaryDirectory.filePath(QStringLiteral("mask_reject.plascan"));
        const QString imagePath = temporaryDirectory.filePath(QStringLiteral("image.png"));
        QFile imageFile(imagePath);
        ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
        ASSERT_EQ(imageFile.write("image"), 5);
        imageFile.close();
        ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("mask_reject")));
        ASSERT_TRUE(projectData.addImages({imagePath}));
        const QString resolvedImage =
            QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(projectPath, imagePath));
        const QJsonObject before = projectData.coreFilesMeta();
        const auto current = session.context();
        QStringList changed;
        QString errorMessage;

        EXPECT_FALSE(
            session.publishImageMaskRecords(current,
                                            {{resolvedImage, QJsonObject{{QStringLiteral("not_a_mask_field"), true}}}},
                                            &changed,
                                            &errorMessage));
        EXPECT_TRUE(errorMessage.contains(QStringLiteral("未知字段")));
        EXPECT_EQ(projectData.coreFilesMeta(), before);

        errorMessage.clear();
        EXPECT_FALSE(session.publishImageMaskRecords(
            current,
            {{resolvedImage, QJsonObject{{QStringLiteral("mask_method"), QStringLiteral("threshold")}}},
             {QDir(temporaryDirectory.path()).filePath(QStringLiteral("./image.png")),
              QJsonObject{{QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            &changed,
            &errorMessage));
        EXPECT_TRUE(errorMessage.contains(QStringLiteral("重复影像")));
        EXPECT_EQ(projectData.coreFilesMeta(), before);

        const QString unknown = temporaryDirectory.filePath(QStringLiteral("unknown.png"));
        errorMessage.clear();
        EXPECT_FALSE(session.clearImageMaskRecords(current, {unknown}, &changed, &errorMessage));
        EXPECT_TRUE(errorMessage.contains(QStringLiteral("不属于当前项目")));
        EXPECT_EQ(projectData.coreFilesMeta(), before);

        QJsonObject duplicateProjectMetadata = before;
        QJsonArray duplicateImages = duplicateProjectMetadata.value(QStringLiteral("images")).toArray();
        QJsonObject duplicateImage = duplicateImages.at(0).toObject();
        duplicateImage.insert(QStringLiteral("path"),
                              QDir(temporaryDirectory.path()).filePath(QStringLiteral("./image.png")));
        duplicateImages.push_back(duplicateImage);
        duplicateProjectMetadata.insert(QStringLiteral("images"), duplicateImages);
        projectData.persistMetadata(duplicateProjectMetadata, false);
        errorMessage.clear();
        EXPECT_FALSE(session.publishImageMaskRecords(
            current,
            {{resolvedImage, QJsonObject{{QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            &changed,
            &errorMessage));
        EXPECT_TRUE(errorMessage.contains(QStringLiteral("项目包含重复影像")));
        EXPECT_EQ(projectData.coreFilesMeta(), duplicateProjectMetadata);
        projectData.persistMetadata(before, false);

        session.advanceGeneration();
        errorMessage.clear();
        EXPECT_FALSE(session.publishImageMaskRecords(
            current,
            {{resolvedImage, QJsonObject{{QStringLiteral("mask_method"), QStringLiteral("threshold")}}}},
            &changed,
            &errorMessage));
        EXPECT_TRUE(errorMessage.contains(QStringLiteral("会话已变化")));
        EXPECT_EQ(projectData.coreFilesMeta(), before);
    }

    TEST(ProjectSessionMaskMutationTest, CommittedMaskPortRemainsTruthfulAfterSynchronousGenerationReentry)
    {
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        ProjectData projectData;
        ProjectSession session(&projectData);
        const QString projectPath = temporaryDirectory.filePath(QStringLiteral("mask_reentry.plascan"));
        const QString imagePath = temporaryDirectory.filePath(QStringLiteral("image.png"));
        QFile imageFile(imagePath);
        ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
        ASSERT_EQ(imageFile.write("image"), 5);
        imageFile.close();
        ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("mask_reentry")));
        ASSERT_TRUE(projectData.addImages({imagePath}));
        const QString resolvedImage =
            QDir::cleanPath(xjw::common::project::ProjectIO::resolveProjectResourcePath(projectPath, imagePath));
        const auto expected = session.context();
        int metadataChangedCount = 0;
        QObject::connect(
            &projectData,
            &ProjectData::metadataChanged,
            &session,
            [&session, &metadataChangedCount](const QJsonObject&)
            {
                ++metadataChangedCount;
                session.advanceGeneration();
            },
            Qt::DirectConnection);

        QStringList updatedImages;
        QString errorMessage;
        EXPECT_TRUE(session.publishImageMaskRecords(
            expected,
            {{resolvedImage,
              QJsonObject{{QStringLiteral("mask_path"), QStringLiteral("committed.png")},
                          {QStringLiteral("mask_method"), QStringLiteral("interactive")}}}},
            &updatedImages,
            &errorMessage))
            << qPrintable(errorMessage);
        EXPECT_EQ(metadataChangedCount, 1);
        EXPECT_FALSE(session.isCurrent(expected));
        EXPECT_EQ(updatedImages, QStringList{resolvedImage});
        const QJsonObject image =
            projectData.coreFilesMeta().value(QStringLiteral("images")).toArray().at(0).toObject();
        EXPECT_EQ(image.value(QStringLiteral("mask_path")).toString(), QStringLiteral("committed.png"));
    }

    TEST_F(ProjectSessionTest, NullDataReturnsEmptyReadModels)
    {
        ProjectSession session(nullptr);

        EXPECT_EQ(session.data(), nullptr);
        EXPECT_FALSE(session.hasProject());
        EXPECT_TRUE(session.projectPath().isEmpty());
        EXPECT_TRUE(session.activeChunkId().isEmpty());
        EXPECT_TRUE(session.metadata().isEmpty());
        EXPECT_TRUE(session.coreMetadata().isEmpty());
        EXPECT_TRUE(session.imagesByCategory(QStringLiteral("source")).isEmpty());
        EXPECT_TRUE(session.allImages().isEmpty());
        EXPECT_TRUE(session.matchFile({}, {}).isEmpty());
        EXPECT_TRUE(session.loadUiSettings().isEmpty());
        EXPECT_TRUE(session.intersectionResults().isEmpty());
    }

    TEST_F(ProjectSessionTest, NullDataResetsWriteBackCounts)
    {
        ProjectSession nullSession(nullptr);
        int updatedCount = 3;
        int clearedCount = 4;
        QString errorMessage;

        EXPECT_FALSE(nullSession.replaceCameraInstances({}, {}, &updatedCount, &clearedCount, &errorMessage));
        EXPECT_EQ(updatedCount, 0);
        EXPECT_EQ(clearedCount, 0);
        EXPECT_EQ(errorMessage, QStringLiteral("ProjectData 未初始化"));
    }

    TEST_F(ProjectSessionTest, AdvanceGenerationRejectsOldContextBeforeWriteBack)
    {
        const auto oldContext = session.context();

        session.advanceGeneration();

        EXPECT_FALSE(session.isCurrent(oldContext));
        EXPECT_EQ(session.context().generation, oldContext.generation + 1);
    }

    TEST_F(ProjectSessionTest, GuardedCameraAndIntersectionWritesCommitCurrentAndRejectStaleContexts)
    {
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        ASSERT_TRUE(data.createProject(temporaryDirectory.filePath(QStringLiteral("guarded_records.plascan")),
                                       QStringLiteral("guarded_records")));
        const QString sourceImage = temporaryDirectory.filePath(QStringLiteral("source.png"));
        QFile imageFile(sourceImage);
        ASSERT_TRUE(imageFile.open(QIODevice::WriteOnly));
        ASSERT_EQ(imageFile.write("image"), 5);
        imageFile.close();
        ASSERT_TRUE(data.addImages({sourceImage}));
        const QStringList projectImages = data.getAllImages();
        ASSERT_EQ(projectImages.size(), 1);

        const auto currentContext = session.context();
        QJsonObject currentCamera = canonicalFrameCamera(32, 24, {0.0, 0.0, 1.0});
        currentCamera.insert(QStringLiteral("solution"), QStringLiteral("current"));
        int updatedCount = 0;
        QString errorMessage;

        ASSERT_TRUE(session.setCameraInstances(
            currentContext, {{projectImages.front(), currentCamera}}, &updatedCount, &errorMessage))
            << qPrintable(errorMessage);
        EXPECT_EQ(updatedCount, 1);
        ASSERT_TRUE(session.appendIntersectionResult(
            currentContext, QJsonObject{{QStringLiteral("id"), QStringLiteral("current")}}, &errorMessage))
            << qPrintable(errorMessage);
        ASSERT_EQ(session.intersectionResults().size(), 1);

        session.advanceGeneration();
        updatedCount = 7;
        errorMessage.clear();
        QJsonObject staleCamera = currentCamera;
        staleCamera.insert(QStringLiteral("solution"), QStringLiteral("stale"));
        EXPECT_FALSE(session.setCameraInstances(
            currentContext, {{projectImages.front(), staleCamera}}, &updatedCount, &errorMessage));
        EXPECT_EQ(updatedCount, 0);
        EXPECT_EQ(errorMessage, QStringLiteral("项目会话已变化，已拒绝过期结果写回"));
        EXPECT_FALSE(session.appendIntersectionResult(
            currentContext, QJsonObject{{QStringLiteral("id"), QStringLiteral("stale")}}, &errorMessage));
        EXPECT_EQ(session.intersectionResults().size(), 1);

        const QJsonObject core = session.coreMetadata();
        const QJsonObject image = core.value(QStringLiteral("images")).toArray().at(0).toObject();
        EXPECT_EQ(xjw::camera_project::CameraProjectRecords::modelParametersForImage(core, image)
                      .value(QStringLiteral("solution"))
                      .toString(),
                  QStringLiteral("current"));
        QString closeError;
        EXPECT_TRUE(data.closeProject(&closeError)) << qPrintable(closeError);
    }

    TEST(ProjectSessionGuardedMutationTest, TiePointReplacementCommitsCurrentAndRejectsStaleGeneration)
    {
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        ProjectData projectData;
        ProjectSession session(&projectData);
        ASSERT_TRUE(projectData.createProject(temporaryDirectory.filePath(QStringLiteral("guarded.plascan")),
                                              QStringLiteral("guarded")));

        const QString firstCloudPath = temporaryDirectory.filePath(QStringLiteral("first.ply"));
        QFile firstCloud(firstCloudPath);
        ASSERT_TRUE(firstCloud.open(QIODevice::WriteOnly));
        firstCloud.close();
        const auto expectedContext = session.context();

        const auto committed =
            session.replaceTiePointResult(expectedContext, firstCloudPath, 3, {}, temporaryDirectory.path(), {});
        ASSERT_TRUE(committed.success) << qPrintable(committed.errorMessage);
        const QJsonArray committedRecords =
            projectData.metadata().value(QStringLiteral("aerial_triangulation_results")).toArray();
        ASSERT_EQ(committedRecords.size(), 1);
        const QString committedGeneration =
            committedRecords.at(0).toObject().value(QStringLiteral("reconstruction_generation_id")).toString();
        EXPECT_FALSE(committedGeneration.isEmpty());

        const QString secondCloudPath = temporaryDirectory.filePath(QStringLiteral("second.ply"));
        QFile secondCloud(secondCloudPath);
        ASSERT_TRUE(secondCloud.open(QIODevice::WriteOnly));
        secondCloud.close();
        session.advanceGeneration();

        const auto rejected =
            session.replaceTiePointResult(expectedContext, secondCloudPath, 4, {}, temporaryDirectory.path(), {});
        EXPECT_FALSE(rejected.success);
        EXPECT_EQ(rejected.errorMessage, QStringLiteral("项目会话已变化，已拒绝过期结果写回"));
        const QJsonArray retainedRecords =
            projectData.metadata().value(QStringLiteral("aerial_triangulation_results")).toArray();
        ASSERT_EQ(retainedRecords.size(), 1);
        EXPECT_EQ(retainedRecords.at(0).toObject().value(QStringLiteral("reconstruction_generation_id")).toString(),
                  committedGeneration);
    }

    TEST(ProjectSessionGuardedMutationTest, WorkflowWritePortsCommitCurrentAndRejectStaleGeneration)
    {
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        ProjectData projectData;
        ProjectSession session(&projectData);
        ASSERT_TRUE(projectData.createProject(temporaryDirectory.filePath(QStringLiteral("workflow_ports.plascan")),
                                              QStringLiteral("workflow_ports")));
        const auto expectedContext = session.context();
        QString errorMessage;

        QJsonObject metadata = session.metadata();
        metadata[QStringLiteral("workflow_marker")] = QStringLiteral("current");
        ASSERT_TRUE(session.persistMetadata(expectedContext, metadata, true, &errorMessage))
            << qPrintable(errorMessage);
        ASSERT_TRUE(session.upsertResultRecordByPath(
            expectedContext,
            QStringLiteral("dem_results"),
            QStringLiteral("dem_path"),
            QJsonObject{{QStringLiteral("dem_path"), QStringLiteral("dem-current.tif")}},
            true,
            &errorMessage))
            << qPrintable(errorMessage);
        const QVector<xjw::gui::project::ProjectResultRecordUpsert> batch{
            {QStringLiteral("ortho_results"),
             QStringLiteral("output_path"),
             QJsonObject{{QStringLiteral("output_path"), QStringLiteral("ortho-current.tif")}},
             true},
            {QStringLiteral("report_results"),
             QStringLiteral("path"),
             QJsonObject{{QStringLiteral("path"), QStringLiteral("report-current.json")}},
             true},
        };
        ASSERT_TRUE(session.upsertResultRecordsByPath(expectedContext, batch, &errorMessage))
            << qPrintable(errorMessage);
        ASSERT_TRUE(session.replaceResultRecordWithLatest(
            expectedContext,
            QStringLiteral("model_results"),
            QJsonObject{{QStringLiteral("model_path"), QStringLiteral("model-current.obj")}},
            true,
            &errorMessage))
            << qPrintable(errorMessage);
        const QVector<ProjectImageMatchResultRecord> matches{
            {QStringLiteral("image-current.jpg"),
             QStringLiteral("matches-current.pimatch"),
             {QStringLiteral("neighbor.jpg")},
             {}},
        };
        QStringList appended_images;
        QObject::connect(&session,
                         &ProjectSession::imageMatchResultAppended,
                         [&appended_images](const QString& image) { appended_images.append(image); });
        ASSERT_TRUE(session.appendImageMatchResults(expectedContext, matches, &errorMessage))
            << qPrintable(errorMessage);
        EXPECT_EQ(appended_images, QStringList{QStringLiteral("image-current.jpg")});
        ASSERT_TRUE(session.requestProjectSave(expectedContext, &errorMessage)) << qPrintable(errorMessage);

        const QJsonObject committed = projectData.metadataIncludingResults();
        EXPECT_EQ(committed.value(QStringLiteral("workflow_marker")).toString(), QStringLiteral("current"));
        EXPECT_EQ(committed.value(QStringLiteral("dem_results")).toArray().size(), 1);
        EXPECT_EQ(committed.value(QStringLiteral("ortho_results")).toArray().size(), 1);
        EXPECT_EQ(committed.value(QStringLiteral("report_results")).toArray().size(), 1);
        EXPECT_EQ(committed.value(QStringLiteral("model_results")).toArray().size(), 1);
        EXPECT_EQ(committed.value(QStringLiteral("image_match_results")).toArray().size(), 1);

        session.advanceGeneration();
        QJsonObject staleMetadata = committed;
        staleMetadata[QStringLiteral("workflow_marker")] = QStringLiteral("stale");
        errorMessage.clear();
        EXPECT_FALSE(session.persistMetadata(expectedContext, staleMetadata, true, &errorMessage));
        EXPECT_EQ(errorMessage, QStringLiteral("项目会话已变化，已拒绝过期结果写回"));
        EXPECT_FALSE(
            session.upsertResultRecordByPath(expectedContext,
                                             QStringLiteral("dem_results"),
                                             QStringLiteral("dem_path"),
                                             QJsonObject{{QStringLiteral("dem_path"), QStringLiteral("dem-stale.tif")}},
                                             true,
                                             &errorMessage));
        EXPECT_FALSE(session.upsertResultRecordsByPath(expectedContext, batch, &errorMessage));
        EXPECT_FALSE(session.replaceResultRecordWithLatest(
            expectedContext,
            QStringLiteral("model_results"),
            QJsonObject{{QStringLiteral("model_path"), QStringLiteral("model-stale.obj")}},
            true,
            &errorMessage));
        EXPECT_FALSE(session.appendImageMatchResults(expectedContext, matches, &errorMessage));
        EXPECT_FALSE(session.requestProjectSave(expectedContext, &errorMessage));
        EXPECT_EQ(appended_images, QStringList{QStringLiteral("image-current.jpg")});

        const QJsonObject retained = projectData.metadataIncludingResults();
        EXPECT_EQ(retained.value(QStringLiteral("workflow_marker")).toString(), QStringLiteral("current"));
        EXPECT_EQ(retained.value(QStringLiteral("dem_results")).toArray().size(), 1);
        EXPECT_EQ(retained.value(QStringLiteral("model_results")).toArray().size(), 1);
        EXPECT_EQ(retained.value(QStringLiteral("image_match_results")).toArray().size(), 1);
    }

    TEST(ProjectSessionGuardedMutationTest, PersistMetadataRegistersTemporarySnapshotBeforeSynchronousGenerationReentry)
    {
        ScopedCoreApplication application;
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        const QString projectPath = temporaryDirectory.filePath(QStringLiteral("persist_reentry.plascan"));
        ProjectData projectData;
        ProjectSession session(&projectData);
        ASSERT_TRUE(projectData.createProject(projectPath, QStringLiteral("persist_reentry")));
        ProjectDataPersistenceTestPeer::settle(&projectData);

        QString saveError;
        ASSERT_TRUE(projectData.saveProject(&saveError)) << qPrintable(saveError);
        projectData.clearTemporaryMetadata();
        ASSERT_FALSE(projectData.isDirty());
        ASSERT_FALSE(projectData.hasTemporaryMetadata());

        const ProjectSessionContext expectedContext = session.context();
        int metadataChangedCount = 0;
        bool temporaryPersistenceRegisteredBeforeSignal = false;
        QObject::connect(
            &projectData,
            &ProjectData::metadataChanged,
            &session,
            [&projectData, &session, &metadataChangedCount, &temporaryPersistenceRegisteredBeforeSignal](
                const QJsonObject&)
            {
                ++metadataChangedCount;
                if (metadataChangedCount == 1)
                {
                    temporaryPersistenceRegisteredBeforeSignal =
                        ProjectDataPersistenceTestPeer::running(projectData) ||
                        ProjectDataPersistenceTestPeer::hasPendingRequest(projectData);
                    session.advanceGeneration();
                }
            },
            Qt::DirectConnection);

        QJsonObject metadata = session.metadata();
        metadata[QStringLiteral("persist_reentry_marker")] = QStringLiteral("current");
        QString errorMessage;
        EXPECT_TRUE(session.persistMetadata(expectedContext, metadata, false, &errorMessage))
            << qPrintable(errorMessage);
        EXPECT_TRUE(errorMessage.isEmpty());
        EXPECT_EQ(metadataChangedCount, 1);
        EXPECT_TRUE(temporaryPersistenceRegisteredBeforeSignal);
        EXPECT_FALSE(session.isCurrent(expectedContext));
        EXPECT_FALSE(projectData.isDirty());
        EXPECT_EQ(projectData.coreFilesMeta().value(QStringLiteral("persist_reentry_marker")).toString(),
                  QStringLiteral("current"));

        ProjectDataPersistenceTestPeer::settle(&projectData);
        EXPECT_TRUE(projectData.hasTemporaryMetadata());
        QFile recoveryFile(xjw::common::project::ProjectIO::tempFilesPath(projectPath));
        ASSERT_TRUE(recoveryFile.open(QIODevice::ReadOnly));
        const QJsonObject recoveredMetadata = QJsonDocument::fromJson(recoveryFile.readAll()).object();
        EXPECT_EQ(recoveredMetadata.value(QStringLiteral("persist_reentry_marker")).toString(),
                  QStringLiteral("current"));

        const quint64 persistenceGeneration = ProjectDataPersistenceTestPeer::persistenceGeneration(projectData);
        QJsonObject staleMetadata = metadata;
        staleMetadata[QStringLiteral("persist_reentry_marker")] = QStringLiteral("stale");
        errorMessage.clear();
        EXPECT_FALSE(session.persistMetadata(expectedContext, staleMetadata, false, &errorMessage));
        EXPECT_EQ(errorMessage, QStringLiteral("项目会话已变化，已拒绝过期结果写回"));
        EXPECT_EQ(metadataChangedCount, 1);
        EXPECT_EQ(ProjectDataPersistenceTestPeer::persistenceGeneration(projectData), persistenceGeneration);
        EXPECT_FALSE(ProjectDataPersistenceTestPeer::running(projectData));
        EXPECT_FALSE(ProjectDataPersistenceTestPeer::hasPendingRequest(projectData));
        EXPECT_EQ(projectData.coreFilesMeta().value(QStringLiteral("persist_reentry_marker")).toString(),
                  QStringLiteral("current"));
    }

    TEST(ProjectSessionGuardedMutationTest, BatchResultPortCommitsAtomicallyBeforeSynchronousSessionReentry)
    {
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        ProjectData projectData;
        ProjectSession session(&projectData);
        ASSERT_TRUE(projectData.createProject(temporaryDirectory.filePath(QStringLiteral("batch_reentry.plascan")),
                                              QStringLiteral("batch_reentry")));
        const auto expectedContext = session.context();
        int metadataChangedCount = 0;
        QObject::connect(&projectData,
                         &ProjectData::metadataChanged,
                         &session,
                         [&session, &metadataChangedCount](const QJsonObject&)
                         {
                             ++metadataChangedCount;
                             if (metadataChangedCount == 1)
                             {
                                 session.advanceGeneration();
                             }
                         });

        const QVector<xjw::gui::project::ProjectResultRecordUpsert> records{
            {QStringLiteral("dem_results"),
             QStringLiteral("dem_path"),
             QJsonObject{{QStringLiteral("dem_path"), QStringLiteral("first.tif")}},
             true},
            {QStringLiteral("ortho_results"),
             QStringLiteral("output_path"),
             QJsonObject{{QStringLiteral("output_path"), QStringLiteral("stale-second.tif")}},
             true},
        };
        QString errorMessage;
        EXPECT_TRUE(session.upsertResultRecordsByPath(expectedContext, records, &errorMessage))
            << qPrintable(errorMessage);
        EXPECT_EQ(metadataChangedCount, 1);
        EXPECT_FALSE(session.isCurrent(expectedContext));

        const QJsonObject retained = projectData.metadataIncludingResults();
        EXPECT_EQ(retained.value(QStringLiteral("dem_results")).toArray().size(), 1);
        EXPECT_EQ(retained.value(QStringLiteral("ortho_results")).toArray().size(), 1);

        const QVector<xjw::gui::project::ProjectResultRecordUpsert> staleRecords{
            {QStringLiteral("model_results"),
             QStringLiteral("model_path"),
             QJsonObject{{QStringLiteral("model_path"), QStringLiteral("stale.obj")}},
             true},
            {QStringLiteral("report_results"),
             QStringLiteral("path"),
             QJsonObject{{QStringLiteral("path"), QStringLiteral("stale.json")}},
             true},
        };
        EXPECT_FALSE(session.upsertResultRecordsByPath(expectedContext, staleRecords, &errorMessage));
        EXPECT_EQ(metadataChangedCount, 1);
        const QJsonObject afterStaleAttempt = projectData.metadataIncludingResults();
        EXPECT_TRUE(afterStaleAttempt.value(QStringLiteral("model_results")).toArray().isEmpty());
        EXPECT_TRUE(afterStaleAttempt.value(QStringLiteral("report_results")).toArray().isEmpty());
    }

    TEST(ProjectSessionGuardedMutationTest, RejectsNoProjectPathChunkAndWrongThreadWrites)
    {
        ProjectData emptyData;
        ProjectSession emptySession(&emptyData);
        const auto emptyContext = emptySession.context();
        const QVector<xjw::gui::project::ProjectResultRecordUpsert> records{
            {QStringLiteral("dem_results"),
             QStringLiteral("dem_path"),
             QJsonObject{{QStringLiteral("dem_path"), QStringLiteral("dem.tif")}},
             true},
        };
        const QVector<ProjectImageMatchResultRecord> matches{
            {QStringLiteral("image.jpg"), QStringLiteral("match.pimatch"), {}, {}},
        };
        QString errorMessage;
        EXPECT_FALSE(emptySession.upsertResultRecordsByPath(emptyContext, records, &errorMessage));
        EXPECT_EQ(errorMessage, QStringLiteral("没有打开的项目"));
        EXPECT_FALSE(emptySession.appendImageMatchResults(emptyContext, matches, &errorMessage));
        EXPECT_EQ(errorMessage, QStringLiteral("没有打开的项目"));
        EXPECT_FALSE(emptySession.requestProjectSave(emptyContext, &errorMessage));
        EXPECT_EQ(errorMessage, QStringLiteral("没有打开的项目"));

        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        ProjectData projectData;
        ProjectSession session(&projectData);
        ASSERT_TRUE(projectData.createProject(temporaryDirectory.filePath(QStringLiteral("context_guards.plascan")),
                                              QStringLiteral("context_guards")));
        const auto current = session.context();

        auto wrongPath = current;
        wrongPath.projectPath += QStringLiteral(".other");
        EXPECT_FALSE(session.requestProjectSave(wrongPath, &errorMessage));
        EXPECT_EQ(errorMessage, QStringLiteral("项目会话已变化，已拒绝过期结果写回"));

        auto wrongChunk = current;
        wrongChunk.chunkId += QStringLiteral("-other");
        EXPECT_FALSE(session.requestProjectSave(wrongChunk, &errorMessage));
        EXPECT_EQ(errorMessage, QStringLiteral("项目会话已变化，已拒绝过期结果写回"));

        bool workerAccepted = true;
        QString workerError;
        std::thread worker([&]() { workerAccepted = session.requestProjectSave(current, &workerError); });
        worker.join();
        EXPECT_FALSE(workerAccepted);
        EXPECT_EQ(workerError, QStringLiteral("项目结果写回必须在会话线程执行"));
    }

    TEST(ProjectSessionGuardedMutationTest, ImageMatchPortPropagatesLazyResultLoadFailure)
    {
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        const QString projectPath = temporaryDirectory.filePath(QStringLiteral("match_lazy_failure.plascan"));
        {
            ProjectData seed;
            ASSERT_TRUE(seed.createProject(projectPath, QStringLiteral("match_lazy_failure")));
            QString errorMessage;
            ASSERT_TRUE(seed.saveProject(&errorMessage)) << qPrintable(errorMessage);
            ASSERT_TRUE(seed.closeProject(&errorMessage)) << qPrintable(errorMessage);
        }

        ProjectData projectData;
        QString openError;
        ASSERT_TRUE(projectData.openProject(projectPath, &openError)) << qPrintable(openError);
        const QString temporaryResults = xjw::common::project::ProjectIO::tempResultsPath(projectPath);
        QFile invalidResults(temporaryResults);
        ASSERT_TRUE(QDir().mkpath(QFileInfo(temporaryResults).absolutePath()));
        ASSERT_TRUE(invalidResults.open(QIODevice::WriteOnly));
        invalidResults.write(
            QJsonDocument(QJsonObject{{QStringLiteral("dem_results"),
                                       QJsonArray{QJsonObject{
                                           {QStringLiteral("dem_tif"), QStringLiteral("removed-alias.tif")}}}}})
                .toJson(QJsonDocument::Compact));
        invalidResults.close();

        ProjectSession session(&projectData);
        const QVector<ProjectImageMatchResultRecord> matches{
            {QStringLiteral("image.jpg"), QStringLiteral("match.pimatch"), {}, {}},
        };
        QString writeError;
        EXPECT_FALSE(session.appendImageMatchResults(session.context(), matches, &writeError));
        EXPECT_FALSE(writeError.isEmpty());
        EXPECT_TRUE(projectData.metadata().value(QStringLiteral("image_match_results")).toArray().isEmpty());
    }

    TEST(ProjectSessionGuardedMutationTest, BundleAdjustStageCommitRequiresCurrentContextButStaleRollbackIsSafe)
    {
        QTemporaryDir temporaryDirectory;
        ASSERT_TRUE(temporaryDirectory.isValid());
        ProjectData projectData;
        ProjectSession session(&projectData);
        ASSERT_TRUE(projectData.createProject(temporaryDirectory.filePath(QStringLiteral("guarded_ba_stage.plascan")),
                                              QStringLiteral("guarded_ba_stage")));
        QString errorMessage;
        const auto updates = prepareSessionBundleAdjustUpdate(&projectData, temporaryDirectory.path(), &errorMessage);
        ASSERT_FALSE(updates.empty()) << qPrintable(errorMessage);
        const auto expected = session.context();
        const QJsonObject before = projectData.metadataIncludingResults();
        int metadataChangedCount = 0;
        QObject::connect(&projectData,
                         &ProjectData::metadataChanged,
                         [&metadataChangedCount](const QJsonObject&) { ++metadataChangedCount; });
        ProjectBundleAdjustMetadataStageToken token;

        ASSERT_TRUE(session.stageBundleAdjustMetadata(
            expected, updates, QJsonObject{{QStringLiteral("track_count"), 4}}, &token, &errorMessage))
            << qPrintable(errorMessage);
        EXPECT_EQ(metadataChangedCount, 0);
        session.advanceGeneration();
        const auto rejectedCommit =
            session.resolveBundleAdjustMetadataStage(expected, token, ProjectBundleAdjustMetadataStageDecision::Commit);
        EXPECT_EQ(rejectedCommit.status, ProjectBundleAdjustMetadataStageStatus::Rejected);
        EXPECT_EQ(metadataChangedCount, 0);

        const auto staleRollback = session.resolveBundleAdjustMetadataStage(
            expected, token, ProjectBundleAdjustMetadataStageDecision::Rollback);
        EXPECT_EQ(staleRollback.status, ProjectBundleAdjustMetadataStageStatus::RolledBack)
            << qPrintable(staleRollback.errorMessage);
        EXPECT_EQ(projectData.metadataIncludingResults(), before);
        EXPECT_EQ(metadataChangedCount, 0);
    }

    TEST_F(ProjectSessionTest, LifecycleSignalsAdvanceGenerationAndPublishContext)
    {
        int signalCount = 0;
        int openedCount = 0;
        QString openedPath;
        ProjectSessionContext received;
        QObject::connect(&session,
                         &ProjectSession::sessionChanged,
                         [&signalCount, &received](const ProjectSessionContext& context)
                         {
                             ++signalCount;
                             received = context;
                         });
        QObject::connect(&session,
                         &ProjectSession::projectOpened,
                         [&openedCount, &openedPath](const QString& path)
                         {
                             ++openedCount;
                             openedPath = path;
                         });

        ASSERT_TRUE(QMetaObject::invokeMethod(
            &data, "projectOpened", Qt::DirectConnection, Q_ARG(QString, QStringLiteral("/tmp/example.plascan"))));

        EXPECT_EQ(signalCount, 1);
        EXPECT_EQ(openedCount, 1);
        EXPECT_EQ(openedPath, QStringLiteral("/tmp/example.plascan"));
        EXPECT_EQ(received.generation, session.context().generation);
        EXPECT_TRUE(session.isCurrent(received));
    }

    TEST_F(ProjectSessionTest, CameraWriteBackMethodsDelegateProjectDataErrors)
    {
        int updatedCount = 7;
        int clearedCount = 9;
        QString errorMessage;

        EXPECT_FALSE(session.setCameraInstances({}, &updatedCount, &errorMessage));
        EXPECT_EQ(updatedCount, 0);
        EXPECT_EQ(errorMessage, QStringLiteral("没有打开的项目"));

        updatedCount = 7;
        errorMessage.clear();
        EXPECT_FALSE(session.setCameraInstancesById({}, &updatedCount, &errorMessage));
        EXPECT_EQ(updatedCount, 0);
        EXPECT_EQ(errorMessage, QStringLiteral("没有打开的项目"));

        updatedCount = 7;
        clearedCount = 9;
        errorMessage.clear();
        EXPECT_FALSE(session.replaceCameraInstances({}, {}, &updatedCount, &clearedCount, &errorMessage));
        EXPECT_EQ(updatedCount, 0);
        EXPECT_EQ(clearedCount, 0);
        EXPECT_EQ(errorMessage, QStringLiteral("没有打开的项目"));

        updatedCount = 7;
        clearedCount = 9;
        errorMessage.clear();
        EXPECT_FALSE(session.replaceCameraInstancesById({}, {}, &updatedCount, &clearedCount, &errorMessage));
        EXPECT_EQ(updatedCount, 0);
        EXPECT_EQ(clearedCount, 0);
        EXPECT_EQ(errorMessage, QStringLiteral("没有打开的项目"));
    }

    TEST_F(ProjectSessionTest, IntersectionWriteBackDelegatesProjectDataError)
    {
        QString errorMessage;

        EXPECT_FALSE(session.appendIntersectionResult(QJsonObject{{QStringLiteral("id"), 1}}, &errorMessage));
        EXPECT_EQ(errorMessage, QStringLiteral("没有打开的项目"));
    }

} // namespace
