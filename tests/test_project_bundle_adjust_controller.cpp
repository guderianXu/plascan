#include "project/tasks/ProjectBundleAdjustController.h"

#include "placamera_runtime/ProjectCameraStore.h"
#include "project/ProjectSessionModel.h"
#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"

#include <QApplication>
#include <QElapsedTimer>
#include <QFile>
#include <QTemporaryDir>
#include <QThread>
#include <QtTest/QSignalSpy>
#include <QtTest/QTest>

#include <gtest/gtest.h>
#include <placamera/frame_camera.h>

#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{

    QApplication& qtApplication()
    {
        static QApplication* application = nullptr;
        if (!application)
        {
            qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
            int argc = 1;
            auto* argv = new char*[2];
            argv[0] = const_cast<char*>("test_project_bundle_adjust_controller");
            argv[1] = nullptr;
            application = new QApplication(argc, argv);
        }
        return *application;
    }

    placamera::CameraInstanceSet nativeFrameCameras(const QJsonObject& project_meta)
    {
        const placamera::FrameId frame("project-world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("ba-controller-definition"),
                                                      {1200.0, 1200.0, 512.0, 384.0, 0.01, 1, 1},
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        placamera::CameraInstanceSet cameras;
        for (const QJsonValue& value : project_meta.value(QStringLiteral("images")).toArray())
        {
            const std::string image_id = value.toObject().value(QStringLiteral("image_uuid")).toString().toStdString();
            const auto added =
                cameras.add(std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
                    placamera::CameraInstanceId("ba-controller-" + image_id),
                    placamera::ImageId(image_id),
                    definition,
                    {1024, 768},
                    placamera::Pose::create(frame, {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}))));
            if (!added.ok())
            {
                throw std::runtime_error("failed to create native BA controller camera fixture");
            }
        }
        return cameras;
    }

    struct ControlledExecution
    {
        std::atomic<bool> started{false};
        std::atomic<bool> release{false};
        xjw::gui::project::BundleAdjustExecutionResult result;
    };

    class SessionChangingReviewAdapter final : public ProjectUiMessageAdapter
    {
    public:
        explicit SessionChangingReviewAdapter(xjw::gui::project::ProjectSession* session) : _session(session)
        {
        }

        void information(QWidget*, const QString&, const QString&) override
        {
        }
        void warning(QWidget*, const QString&, const QString&) override
        {
            ++warningCount;
        }
        void critical(QWidget*, const QString&, const QString&) override
        {
        }
        UiAnswer question(QWidget*, const QString&, const QString&, UiAnswer) override
        {
            return UiAnswer::Cancel;
        }
        UiReviewDecision review(QWidget*, const UiReviewDialogRequest& request) override
        {
            ++reviewCount;
            lastRequest = request;
            _session->advanceGeneration();
            return UiReviewDecision::Accept;
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

        int reviewCount = 0;
        int warningCount = 0;
        UiReviewDialogRequest lastRequest;

    private:
        xjw::gui::project::ProjectSession* _session = nullptr;
    };

    class ProjectBundleAdjustControllerTest : public testing::Test
    {
    protected:
        void SetUp() override
        {
            qtApplication();
            ASSERT_TRUE(_tempDir.isValid());
            const QString project_path = _tempDir.filePath(QStringLiteral("bundle_adjust_controller.plascan"));
            ASSERT_TRUE(_projectData.createProject(project_path, QStringLiteral("bundle_adjust_controller")));

            for (const QString& name : {QStringLiteral("a.jpg"), QStringLiteral("b.jpg")})
            {
                const QString path = _tempDir.filePath(name);
                QFile file(path);
                ASSERT_TRUE(file.open(QIODevice::WriteOnly));
                file.write("image");
                file.close();
                _images.append(path);
            }
            ASSERT_TRUE(_projectData.addImages(_images));

            const auto cameras = nativeFrameCameras(_projectData.coreFilesMeta());
            QString camera_error;
            int updated_count = 0;
            ASSERT_TRUE(_projectData.upsertNativeCameraInstances(cameras, {}, &updated_count, &camera_error))
                << qPrintable(camera_error);
            ASSERT_EQ(updated_count, 2);
            _successResult = makeSuccessResult();
        }

        xjw::gui::project::BundleAdjustExecutionResult makeSuccessResult() const
        {
            xjw::gui::project::BundleAdjustExecutionResult result;
            result.serviceResult.success = true;
            result.serviceResult.resultJson = QJsonObject{{QStringLiteral("output_dir"), _tempDir.path()},
                                                          {QStringLiteral("track_count"), 8},
                                                          {QStringLiteral("optimized_count"), 8},
                                                          {QStringLiteral("mean_rms_before"), 1.0},
                                                          {QStringLiteral("mean_rms_after"), 0.5}};

            const auto loaded = xjw::placamera_runtime::loadProjectCameras(_projectData.coreFilesMeta());
            if (loaded.ok())
            {
                result.serviceResult.cameraInstances = loaded.instances;
            }
            return result;
        }

        static xjw::gui::project::ProjectBundleAdjustController::ExecutionRunner
        immediateRunner(const xjw::gui::project::BundleAdjustExecutionResult& result)
        {
            return [result](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
            { return result; };
        }

        static xjw::gui::project::ProjectBundleAdjustController::ExecutionRunner
        controlledRunner(const std::shared_ptr<ControlledExecution>& execution)
        {
            return [execution](const QJsonObject&, const QString&, const QStringList&, int, xjw::gui::BaServiceOptions)
            {
                execution->started.store(true, std::memory_order_release);
                while (!execution->release.load(std::memory_order_acquire))
                {
                    QThread::msleep(1);
                }
                return execution->result;
            };
        }

        QTemporaryDir _tempDir;
        ProjectData _projectData;
        QStringList _images;
        xjw::gui::project::BundleAdjustExecutionResult _successResult;
    };

    TEST_F(ProjectBundleAdjustControllerTest, SessionChangeDiscardsPendingPreview)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        controller.setExecutionRunnerForTesting(immediateRunner(_successResult));

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(controller.hasPendingPreview());

        session.advanceGeneration();

        EXPECT_FALSE(controller.hasPendingPreview());
        EXPECT_FALSE(controller.acceptPreview(nullptr));
    }

    TEST_F(ProjectBundleAdjustControllerTest, SessionChangeDropsStaleCompletion)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        auto execution = std::make_shared<ControlledExecution>();
        execution->result = _successResult;
        controller.setExecutionRunnerForTesting(controlledRunner(execution));
        QSignalSpy preview_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::previewReady);
        QSignalSpy finished_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::finished);

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(execution->started.load(std::memory_order_acquire));
        session.advanceGeneration();
        execution->release.store(true, std::memory_order_release);
        QTRY_VERIFY(!controller.isRunning());

        EXPECT_EQ(preview_spy.count(), 0);
        EXPECT_EQ(finished_spy.count(), 0);
        EXPECT_FALSE(controller.hasPendingPreview());
    }

    TEST_F(ProjectBundleAdjustControllerTest, CancelEmitsOneFinishAndDropsLateResult)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        auto execution = std::make_shared<ControlledExecution>();
        execution->result = _successResult;
        controller.setExecutionRunnerForTesting(controlledRunner(execution));
        QSignalSpy preview_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::previewReady);
        QSignalSpy finished_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::finished);

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(execution->started.load(std::memory_order_acquire));
        EXPECT_TRUE(controller.cancel());
        ASSERT_EQ(finished_spy.count(), 1);
        EXPECT_FALSE(finished_spy.at(0).at(0).toBool());

        execution->release.store(true, std::memory_order_release);
        QTRY_VERIFY(!controller.isRunning());

        EXPECT_EQ(finished_spy.count(), 1);
        EXPECT_EQ(preview_spy.count(), 1);
        EXPECT_TRUE(preview_spy.at(0).at(0).toJsonObject().isEmpty());
        EXPECT_FALSE(controller.hasPendingPreview());
    }

    TEST_F(ProjectBundleAdjustControllerTest, RestartIsRejectedUntilCancelledWorkerExits)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        auto execution = std::make_shared<ControlledExecution>();
        execution->result = _successResult;
        controller.setExecutionRunnerForTesting(controlledRunner(execution));

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(execution->started.load(std::memory_order_acquire));
        ASSERT_TRUE(controller.cancel());
        EXPECT_FALSE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));

        execution->release.store(true, std::memory_order_release);
        QTRY_VERIFY(!controller.isRunning());
        controller.setExecutionRunnerForTesting(immediateRunner(_successResult));
        EXPECT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, true, {}));
        QTRY_VERIFY(!controller.isRunning());
    }

    TEST_F(ProjectBundleAdjustControllerTest, DestructorCancelsAndWaitsForWorker)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        auto execution = std::make_shared<ControlledExecution>();
        execution->result = _successResult;
        auto controller = std::make_unique<xjw::gui::project::ProjectBundleAdjustController>(&session, nullptr);
        controller->setExecutionRunnerForTesting(controlledRunner(execution));
        ASSERT_TRUE(controller->startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(execution->started.load(std::memory_order_acquire));

        std::thread releaser(
            [execution]()
            {
                QThread::msleep(100);
                execution->release.store(true, std::memory_order_release);
            });
        QElapsedTimer timer;
        timer.start();
        controller.reset();
        const qint64 elapsed = timer.elapsed();
        releaser.join();

        EXPECT_GE(elapsed, 75);
    }

    TEST_F(ProjectBundleAdjustControllerTest, SuccessKeepsStablePreviewThenFinishedSignalOrder)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        controller.setExecutionRunnerForTesting(immediateRunner(_successResult));
        std::vector<QString> order;
        QObject::connect(&controller,
                         &xjw::gui::project::ProjectBundleAdjustController::progressChanged,
                         &controller,
                         [&order]() { order.emplace_back(QStringLiteral("progress")); });
        QObject::connect(&controller,
                         &xjw::gui::project::ProjectBundleAdjustController::previewReady,
                         &controller,
                         [&order]() { order.emplace_back(QStringLiteral("preview")); });
        QObject::connect(&controller,
                         &xjw::gui::project::ProjectBundleAdjustController::finished,
                         &controller,
                         [&order]() { order.emplace_back(QStringLiteral("finished")); });
        QSignalSpy finished_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::finished);

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_COMPARE(finished_spy.count(), 1);

        ASSERT_GE(order.size(), 3U);
        EXPECT_EQ(order.at(order.size() - 2), QStringLiteral("preview"));
        EXPECT_EQ(order.back(), QStringLiteral("finished"));
        EXPECT_TRUE(finished_spy.at(0).at(0).toBool());
    }

    TEST_F(ProjectBundleAdjustControllerTest, CancelFromFinalProgressEmitsOnlyOneFailureWithoutPreviewOrReview)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        SessionChangingReviewAdapter messages(&session);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, &messages);
        controller.setExecutionRunnerForTesting(immediateRunner(_successResult));
        QSignalSpy preview_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::previewReady);
        QSignalSpy finished_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::finished);
        bool cancellation_requested = false;
        bool cancellation_accepted = false;
        QObject::connect(
            &controller,
            &xjw::gui::project::ProjectBundleAdjustController::progressChanged,
            &controller,
            [&](const QString&, int percent)
            {
                if (percent == 95 && !cancellation_requested)
                {
                    cancellation_requested = true;
                    cancellation_accepted = controller.cancel();
                }
            },
            Qt::DirectConnection);

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(!controller.isRunning());

        EXPECT_TRUE(cancellation_requested);
        EXPECT_TRUE(cancellation_accepted);
        ASSERT_EQ(finished_spy.count(), 1);
        EXPECT_FALSE(finished_spy.at(0).at(0).toBool());
        ASSERT_EQ(preview_spy.count(), 1);
        EXPECT_TRUE(preview_spy.at(0).at(0).toJsonObject().isEmpty());
        EXPECT_EQ(messages.reviewCount, 0);
        EXPECT_FALSE(controller.hasPendingPreview());
    }

    TEST_F(ProjectBundleAdjustControllerTest, SessionChangeFromFinalProgressDropsPreviewTerminalAndReview)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        SessionChangingReviewAdapter messages(&session);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, &messages);
        controller.setExecutionRunnerForTesting(immediateRunner(_successResult));
        QSignalSpy preview_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::previewReady);
        QSignalSpy finished_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::finished);
        bool session_advanced = false;
        QObject::connect(
            &controller,
            &xjw::gui::project::ProjectBundleAdjustController::progressChanged,
            &controller,
            [&](const QString&, int percent)
            {
                if (percent == 95 && !session_advanced)
                {
                    session_advanced = true;
                    session.advanceGeneration();
                }
            },
            Qt::DirectConnection);

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(!controller.isRunning());

        EXPECT_TRUE(session_advanced);
        EXPECT_EQ(preview_spy.count(), 0);
        EXPECT_EQ(finished_spy.count(), 0);
        EXPECT_EQ(messages.reviewCount, 0);
        EXPECT_FALSE(controller.hasPendingPreview());
    }

    TEST_F(ProjectBundleAdjustControllerTest, AcceptPreviewWritesCamerasAndClearsPreview)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        controller.setExecutionRunnerForTesting(immediateRunner(_successResult));

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(controller.hasPendingPreview());

        QString error;
        EXPECT_TRUE(controller.acceptPreview(&error)) << qPrintable(error);
        EXPECT_FALSE(controller.hasPendingPreview());
        EXPECT_EQ(_projectData.getBundleAdjustResults().size(), 1);
    }

    TEST_F(ProjectBundleAdjustControllerTest, FailedAcceptRetainsPreviewForRetry)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        auto invalid_result = _successResult;
        const auto current = invalid_result.serviceResult.cameraInstances.values().front();
        const auto frame = std::dynamic_pointer_cast<const placamera::FramePinholeModel>(current);
        ASSERT_NE(frame, nullptr);
        auto stale = std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("stale-instance"),
            frame->imageId(),
            std::shared_ptr<const placamera::FramePinholeDefinition>(frame, &frame->pinholeDefinition()),
            frame->imageSize(),
            frame->pose(),
            frame->captureTime()));
        invalid_result.serviceResult.cameraInstances = {};
        ASSERT_TRUE(invalid_result.serviceResult.cameraInstances.add(stale).ok());
        ASSERT_TRUE(invalid_result.serviceResult.cameraInstances.add(
            _successResult.serviceResult.cameraInstances.values().at(1)).ok());
        controller.setExecutionRunnerForTesting(immediateRunner(invalid_result));

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(controller.hasPendingPreview());

        QString first_error;
        EXPECT_FALSE(controller.acceptPreview(&first_error));
        EXPECT_FALSE(first_error.isEmpty());
        EXPECT_TRUE(controller.hasPendingPreview());

        QString retry_error;
        EXPECT_FALSE(controller.acceptPreview(&retry_error));
        EXPECT_FALSE(retry_error.isEmpty());
        EXPECT_TRUE(controller.hasPendingPreview());
    }

    TEST_F(ProjectBundleAdjustControllerTest, DiscardClearsPendingPreview)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        controller.setExecutionRunnerForTesting(immediateRunner(_successResult));

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(controller.hasPendingPreview());
        controller.discardPreview();

        EXPECT_FALSE(controller.hasPendingPreview());
        EXPECT_FALSE(controller.acceptPreview(nullptr));
    }

    TEST_F(ProjectBundleAdjustControllerTest, DryRunEmitsResultWithoutPendingPreview)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        auto dry_result = _successResult;
        dry_result.serviceResult.cameraInstances = {};
        controller.setExecutionRunnerForTesting(immediateRunner(dry_result));
        QSignalSpy preview_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::previewReady);
        QSignalSpy finished_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::finished);

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, true, {}));
        QTRY_COMPARE(finished_spy.count(), 1);

        EXPECT_EQ(preview_spy.count(), 1);
        EXPECT_FALSE(controller.hasPendingPreview());
        EXPECT_TRUE(finished_spy.at(0).at(0).toBool());
    }

    TEST_F(ProjectBundleAdjustControllerTest, NewRunClearsOldPreviewBeforeWorkerCompletes)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, nullptr);
        controller.setExecutionRunnerForTesting(immediateRunner(_successResult));

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_VERIFY(controller.hasPendingPreview());
        QTRY_VERIFY(!controller.isRunning());

        auto execution = std::make_shared<ControlledExecution>();
        execution->result = _successResult;
        controller.setExecutionRunnerForTesting(controlledRunner(execution));
        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));

        EXPECT_FALSE(controller.hasPendingPreview());
        execution->release.store(true, std::memory_order_release);
        QTRY_VERIFY(!controller.isRunning());
    }

    TEST_F(ProjectBundleAdjustControllerTest, SessionChangeInsideReviewReturnsWithoutOldSessionWarning)
    {
        xjw::gui::project::ProjectSession session(&_projectData);
        SessionChangingReviewAdapter messages(&session);
        xjw::gui::project::ProjectBundleAdjustController controller(&session, &messages);
        controller.setExecutionRunnerForTesting(immediateRunner(_successResult));
        QSignalSpy finished_spy(&controller, &xjw::gui::project::ProjectBundleAdjustController::finished);

        ASSERT_TRUE(controller.startAsync(_images, _tempDir.path(), 1, false, {}));
        QTRY_COMPARE(finished_spy.count(), 1);

        EXPECT_EQ(messages.reviewCount, 1);
        EXPECT_EQ(messages.warningCount, 0);
        EXPECT_EQ(messages.lastRequest.objectName, QStringLiteral("bundleAdjustPreviewMessageBox"));
        EXPECT_EQ(messages.lastRequest.acceptObjectName, QStringLiteral("keepBundleAdjustPreviewButton"));
        EXPECT_EQ(messages.lastRequest.discardObjectName, QStringLiteral("discardBundleAdjustPreviewButton"));
        EXPECT_EQ(messages.lastRequest.acceptText, QStringLiteral("保留结果"));
        EXPECT_EQ(messages.lastRequest.discardText, QStringLiteral("丢弃结果"));
        EXPECT_FALSE(messages.lastRequest.text.isEmpty());
        EXPECT_FALSE(messages.lastRequest.detailedText.isEmpty());
        EXPECT_FALSE(controller.hasPendingPreview());
        EXPECT_TRUE(_projectData.getBundleAdjustResults().isEmpty());
    }

} // namespace
