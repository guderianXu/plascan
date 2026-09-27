#include "BrowserDebugBridge.h"
#include "TaskRuntimeService.h"
#include "WorkPanelWidget.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineEdit>
#include <QLocalSocket>
#include <QPushButton>
#include <QTemporaryDir>
#include <QThread>
#include <QTimer>
#include <QWidget>

#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace
{
    constexpr auto DebugToken = "browser-debug-test-token-1234567890";

    bool spinUntil(const std::function<bool()>& condition, int timeoutMs = 3000)
    {
        QElapsedTimer timer;
        timer.start();
        while (!condition() && timer.elapsed() < timeoutMs)
        {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
            QThread::msleep(2);
        }
        return condition();
    }

    QJsonObject
    rpc(const QString& serverName, const QString& token, const QString& method, const QJsonObject& parameters = {})
    {
        QLocalSocket socket;
        socket.connectToServer(serverName);
        EXPECT_TRUE(spinUntil([&socket]() { return socket.state() == QLocalSocket::ConnectedState; }));
        if (socket.state() != QLocalSocket::ConnectedState)
        {
            return {};
        }

        const QJsonObject request{
            {QStringLiteral("id"), 7},
            {QStringLiteral("token"), token},
            {QStringLiteral("method"), method},
            {QStringLiteral("params"), parameters},
        };
        socket.write(QJsonDocument(request).toJson(QJsonDocument::Compact) + '\n');
        socket.flush();
        EXPECT_TRUE(spinUntil([&socket]() { return socket.canReadLine(); }));
        return QJsonDocument::fromJson(socket.readLine()).object();
    }

    bool containsNamedObject(const QJsonValue& value, const QString& objectName)
    {
        if (value.isObject())
        {
            const QJsonObject object = value.toObject();
            if (object.value(QStringLiteral("object_name")).toString() == objectName)
            {
                return true;
            }
            for (auto it = object.constBegin(); it != object.constEnd(); ++it)
            {
                if (containsNamedObject(it.value(), objectName))
                {
                    return true;
                }
            }
        }
        else if (value.isArray())
        {
            for (const QJsonValue& item : value.toArray())
            {
                if (containsNamedObject(item, objectName))
                {
                    return true;
                }
            }
        }
        return false;
    }

    class ReleaseGatedExecutor final : public xjw::task_runtime::ITaskExecutor
    {
    public:
        xjw::task_runtime::TaskExecutionOutcome execute(const xjw::task_runtime::TaskDefinition&,
                                                        xjw::task_runtime::TaskExecutionContext& context) override
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _entered = true;
            }
            _condition.notify_all();

            std::unique_lock<std::mutex> lock(_mutex);
            _condition.wait(lock, [this] { return _released; });
            lock.unlock();
            return {context.control.pollAtSafePoint("release_gate") == xjw::task_runtime::TaskControlDecision::Cancel
                        ? xjw::task_runtime::TaskExecutionStatus::Cancelled
                        : xjw::task_runtime::TaskExecutionStatus::Succeeded};
        }

        bool hasEntered() const
        {
            std::lock_guard<std::mutex> lock(_mutex);
            return _entered;
        }

        void release()
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _released = true;
            }
            _condition.notify_all();
        }

    private:
        mutable std::mutex _mutex;
        std::condition_variable _condition;
        bool _entered = false;
        bool _released = false;
    };

    class ExecutorReleaseGuard final
    {
    public:
        explicit ExecutorReleaseGuard(std::shared_ptr<ReleaseGatedExecutor> executor) : _executor(std::move(executor))
        {
        }

        ~ExecutorReleaseGuard()
        {
            _executor->release();
        }

    private:
        std::shared_ptr<ReleaseGatedExecutor> _executor;
    };

    class BrowserDebugBridgeTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_TRUE(_temporaryDirectory.isValid());
            _serverName = _temporaryDirectory.filePath(QStringLiteral("bridge.sock"));
            qputenv("PLASCAN_BROWSER_TEST", "1");
            qputenv("PLASCAN_BROWSER_DEBUG_SOCKET", _serverName.toUtf8());
            qputenv("PLASCAN_BROWSER_DEBUG_TOKEN", DebugToken);
        }

        void TearDown() override
        {
            qunsetenv("PLASCAN_BROWSER_TEST");
            qunsetenv("PLASCAN_BROWSER_DEBUG_SOCKET");
            qunsetenv("PLASCAN_BROWSER_DEBUG_TOKEN");
        }

        QTemporaryDir _temporaryDirectory;
        QString _serverName;
    };

    TEST(BrowserDebugBridgeSecurityTest, RemainsDisabledWithoutExplicitTestFlag)
    {
        qunsetenv("PLASCAN_BROWSER_TEST");
        qunsetenv("PLASCAN_BROWSER_DEBUG_SOCKET");
        qunsetenv("PLASCAN_BROWSER_DEBUG_TOKEN");
        QWidget root;
        xjw::gui::runtime::BrowserDebugBridge bridge(&root);
        QString error;
        EXPECT_FALSE(bridge.start(&error));
        EXPECT_FALSE(error.isEmpty());
        EXPECT_TRUE(bridge.serverName().isEmpty());
    }

    TEST_F(BrowserDebugBridgeTest, RejectsInvalidTokenAndReturnsStructuredSnapshot)
    {
        QWidget root;
        root.setObjectName(QStringLiteral("browserDebugRoot"));
        root.show();
        xjw::gui::runtime::BrowserDebugBridge bridge(&root);
        QString error;
        ASSERT_TRUE(bridge.start(&error)) << error.toStdString();

        const QJsonObject rejected = rpc(_serverName, QStringLiteral("wrong-token"), QStringLiteral("ping"));
        EXPECT_FALSE(rejected.value(QStringLiteral("ok")).toBool());
        EXPECT_EQ(rejected.value(QStringLiteral("error")).toObject().value(QStringLiteral("code")).toString(),
                  QStringLiteral("unauthorized"));

        const QJsonObject response = rpc(_serverName, QString::fromLatin1(DebugToken), QStringLiteral("snapshot"));
        ASSERT_TRUE(response.value(QStringLiteral("ok")).toBool());
        const QJsonObject snapshot = response.value(QStringLiteral("result")).toObject();
        EXPECT_EQ(snapshot.value(QStringLiteral("schema_version")).toInt(), 1);
        EXPECT_TRUE(snapshot.value(QStringLiteral("project")).isObject());
        EXPECT_TRUE(snapshot.value(QStringLiteral("tasks")).isArray());
        EXPECT_TRUE(snapshot.value(QStringLiteral("windows")).isArray());
    }

    TEST_F(BrowserDebugBridgeTest, InspectsAndOperatesOnlySupportedNamedControls)
    {
        QWidget root;
        root.setObjectName(QStringLiteral("browserDebugRoot"));
        auto* button = new QPushButton(QStringLiteral("Run"), &root);
        button->setObjectName(QStringLiteral("runButton"));
        button->show();
        auto* lineEdit = new QLineEdit(&root);
        lineEdit->setObjectName(QStringLiteral("nameEdit"));
        lineEdit->show();
        root.show();

        bool clicked = false;
        QObject::connect(button, &QPushButton::clicked, [&clicked]() { clicked = true; });
        xjw::gui::runtime::BrowserDebugBridge bridge(&root);
        QString error;
        ASSERT_TRUE(bridge.start(&error)) << error.toStdString();

        const QJsonObject treeResponse = rpc(_serverName, QString::fromLatin1(DebugToken), QStringLiteral("ui_tree"));
        ASSERT_TRUE(treeResponse.value(QStringLiteral("ok")).toBool());
        EXPECT_TRUE(containsNamedObject(treeResponse.value(QStringLiteral("result")), QStringLiteral("runButton")));

        const QJsonObject setResponse = rpc(_serverName,
                                            QString::fromLatin1(DebugToken),
                                            QStringLiteral("interact"),
                                            {{QStringLiteral("object_name"), QStringLiteral("nameEdit")},
                                             {QStringLiteral("operation"), QStringLiteral("set_text")},
                                             {QStringLiteral("value"), QStringLiteral("PlaScan")}});
        EXPECT_TRUE(setResponse.value(QStringLiteral("ok")).toBool());
        EXPECT_EQ(lineEdit->text(), QStringLiteral("PlaScan"));

        const QJsonObject clickResponse = rpc(_serverName,
                                              QString::fromLatin1(DebugToken),
                                              QStringLiteral("interact"),
                                              {{QStringLiteral("object_name"), QStringLiteral("runButton")},
                                               {QStringLiteral("operation"), QStringLiteral("activate")}});
        EXPECT_TRUE(clickResponse.value(QStringLiteral("ok")).toBool());
        EXPECT_TRUE(spinUntil([&clicked]() { return clicked; }));

        const QJsonObject unsupported = rpc(_serverName,
                                            QString::fromLatin1(DebugToken),
                                            QStringLiteral("interact"),
                                            {{QStringLiteral("object_name"), QStringLiteral("nameEdit")},
                                             {QStringLiteral("operation"), QStringLiteral("delete_file")}});
        EXPECT_FALSE(unsupported.value(QStringLiteral("ok")).toBool());
        EXPECT_EQ(unsupported.value(QStringLiteral("error")).toObject().value(QStringLiteral("code")).toString(),
                  QStringLiteral("unsupported_operation"));
    }

    TEST_F(BrowserDebugBridgeTest, CapturesCurrentWindowWithoutWritingAnArbitraryPath)
    {
        QWidget root;
        root.resize(320, 180);
        root.show();
        xjw::gui::runtime::BrowserDebugBridge bridge(&root);
        QString error;
        ASSERT_TRUE(bridge.start(&error)) << error.toStdString();

        const QJsonObject response = rpc(_serverName, QString::fromLatin1(DebugToken), QStringLiteral("screenshot"));
        ASSERT_TRUE(response.value(QStringLiteral("ok")).toBool());
        const QJsonObject screenshot = response.value(QStringLiteral("result")).toObject();
        EXPECT_TRUE(screenshot.value(QStringLiteral("available")).toBool());
        EXPECT_EQ(screenshot.value(QStringLiteral("format")).toString(), QStringLiteral("png"));
        EXPECT_FALSE(screenshot.value(QStringLiteral("data_base64")).toString().isEmpty());
    }

    TEST_F(BrowserDebugBridgeTest, ControlsSchedulerManagedTaskByRunId)
    {
        QWidget root;
        root.setObjectName(QStringLiteral("browserDebugRoot"));
        xjw::gui::runtime::TaskRuntimeService service(&root);
        WorkPanelWidget workPanel(&root);
        QObject::connect(&service,
                         &xjw::gui::runtime::TaskRuntimeService::taskSnapshotsChanged,
                         &workPanel,
                         &WorkPanelWidget::setTaskSnapshots);

        xjw::task_runtime::TaskDefinition definition;
        definition.taskId = "queued-depth";
        definition.kind = "not-registered";
        definition.displayName = "Queued depth";
        definition.capabilities.canPause = true;
        definition.capabilities.canCheckpoint = true;
        const auto submitted = service.submit(definition);
        ASSERT_TRUE(submitted.accepted) << submitted.error;
        workPanel.setTaskSnapshots(service.taskSnapshots());
        root.show();

        xjw::gui::runtime::BrowserDebugBridge bridge(&root);
        QString error;
        ASSERT_TRUE(bridge.start(&error)) << error.toStdString();
        const QJsonObject response =
            rpc(_serverName,
                QString::fromLatin1(DebugToken),
                QStringLiteral("task_command"),
                {{QStringLiteral("action"), QStringLiteral("pause")},
                 {QStringLiteral("run_id"), QString::fromStdString(submitted.runIds.front())}});
        ASSERT_TRUE(response.value(QStringLiteral("ok")).toBool());
        EXPECT_EQ(response.value(QStringLiteral("result"))
                      .toObject()
                      .value(QStringLiteral("task"))
                      .toObject()
                      .value(QStringLiteral("state"))
                      .toString(),
                  QStringLiteral("paused"));
    }

    TEST(TaskRuntimeServiceTest, SessionRolloverIsLatestWinsWithoutBlockingTheGuiEventLoop)
    {
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        const QString first_project = temporary_directory.filePath(QStringLiteral("first.plascan"));
        const QString second_project = temporary_directory.filePath(QStringLiteral("second.plascan"));
        const QString final_project = temporary_directory.filePath(QStringLiteral("final.plascan"));

        xjw::gui::runtime::TaskRuntimeService service;
        service.setProjectSession(first_project, QStringLiteral("chunk-a"), 1);
        ASSERT_TRUE(spinUntil([&service] { return !service.isSessionTransitionInProgress(); }));

        auto executor = std::make_shared<ReleaseGatedExecutor>();
        ExecutorReleaseGuard release_guard(executor);
        service.registerExecutor("gated", executor);
        xjw::task_runtime::TaskDefinition definition;
        definition.taskId = "gated-rollover";
        definition.kind = "gated";
        definition.displayName = "Gated rollover";
        definition.projectKey = QFileInfo(first_project).absoluteFilePath().toStdString();
        definition.chunkId = "chunk-a";
        definition.projectGeneration = 1;
        const auto submitted = service.submit(std::move(definition));
        ASSERT_TRUE(submitted.accepted) << submitted.error;
        ASSERT_TRUE(spinUntil([&executor] { return executor->hasEntered(); }));

        bool heartbeat = false;
        QTimer::singleShot(0,
                           [&]
                           {
                               heartbeat = true;
                               executor->release();
                           });
        std::atomic_bool watchdog_released_task = false;
        std::jthread watchdog(
            [executor, &watchdog_released_task](std::stop_token stop_token)
            {
                for (int attempt = 0; attempt < 400 && !stop_token.stop_requested(); ++attempt)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                if (!stop_token.stop_requested())
                {
                    watchdog_released_task.store(true);
                    executor->release();
                }
            });
        service.setProjectSession(second_project, QStringLiteral("chunk-b"), 2);
        service.setProjectSession(final_project, QStringLiteral("chunk-c"), 3);
        watchdog.request_stop();

        EXPECT_FALSE(watchdog_released_task.load());
        ASSERT_TRUE(spinUntil([&heartbeat] { return heartbeat; }));
        ASSERT_TRUE(spinUntil([&service] { return !service.isSessionTransitionInProgress(); }));
        EXPECT_TRUE(service.journalPath().contains(QStringLiteral("final.files/task_runtime")));

        service.shutdownAsync();
        ASSERT_TRUE(spinUntil([&service] { return service.isShutdownComplete(); }));
    }

    TEST(TaskRuntimeServiceTest, NewGenerationRetiresSchedulerEvenWhenJournalPathIsUnchanged)
    {
        QTemporaryDir temporary_directory;
        ASSERT_TRUE(temporary_directory.isValid());
        const QString project = temporary_directory.filePath(QStringLiteral("same-project.plascan"));

        xjw::gui::runtime::TaskRuntimeService service;
        service.setProjectSession(project, QStringLiteral("chunk-a"), 1);
        ASSERT_TRUE(spinUntil([&service] { return !service.isSessionTransitionInProgress(); }));

        service.setProjectSession(project, QStringLiteral("chunk-a"), 2);
        EXPECT_TRUE(service.isSessionTransitionInProgress());
        ASSERT_TRUE(spinUntil([&service] { return !service.isSessionTransitionInProgress(); }));

        service.setProjectSession(project, QStringLiteral("chunk-a"), 2);
        EXPECT_FALSE(service.isSessionTransitionInProgress());

        service.shutdownAsync();
        ASSERT_TRUE(spinUntil([&service] { return service.isShutdownComplete(); }));
    }
} // namespace

int main(int argc, char** argv)
{
    QApplication application(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
