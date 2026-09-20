#include "project/services/ProjectResourceCleanupCoordinator.h"
#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"

#include "project/ProjectIO.h"
#include "project/ProjectSessionModel.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QPushButton>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QThread>

#include <condition_variable>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace
{

using xjw::core::project::PreparedResourceCleanup;
using xjw::core::project::ProjectResourceCleanupService;
using xjw::core::project::ResourceCleanupResult;
using xjw::gui::project::ProjectResourceCleanupCoordinator;
using xjw::gui::project::ProjectResourceCleanupExecutor;
using xjw::gui::project::ProjectSession;

class FakeProjectUiMessageAdapter final : public ProjectUiMessageAdapter
{
public:
    UiAnswer nextAnswer = UiAnswer::Yes;
    std::vector<QString> informationTexts;
    std::vector<QString> warningTexts;

    void information(QWidget*, const QString&, const QString& text) override
    {
        informationTexts.push_back(text);
    }

    void warning(QWidget*, const QString&, const QString& text) override
    {
        warningTexts.push_back(text);
    }

    void critical(QWidget*, const QString&, const QString&) override
    {
    }

    UiAnswer question(QWidget*, const QString&, const QString&, UiAnswer) override
    {
        return nextAnswer;
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

    UiDialogResult selectOpenFile(QWidget*,
                                  const QString&,
                                  const QString&,
                                  const QString&,
                                  QDir::Filters) override
    {
        return {};
    }

    UiDialogResult selectOpenFiles(QWidget*,
                                   const QString&,
                                   const QString&,
                                   const QString&,
                                   QDir::Filters) override
    {
        return {};
    }

    UiDialogResult selectDirectory(QWidget*,
                                   const QString&,
                                   const QString&,
                                   QDir::Filters) override
    {
        return {};
    }

    UiDialogResult selectSaveFile(QWidget*,
                                  const QString&,
                                  const QString&,
                                  const QString&,
                                  QDir::Filters,
                                  const QString&) override
    {
        return {};
    }
};

class GatedCleanupExecutor final : public ProjectResourceCleanupExecutor
{
public:
    bool pauseExecute = false;
    bool returnFailure = false;
    int finalizeCount = 0;

    PreparedResourceCleanup prepare(ProjectData* projectData,
                                    const QString& section,
                                    const QStringList& resourcePaths) override
    {
        return ProjectResourceCleanupService::prepareGeneratedDataCleanup(
            projectData, section, resourcePaths);
    }

    ResourceCleanupResult execute(const PreparedResourceCleanup& prepared) override
    {
        {
            std::lock_guard lock(_mutex);
            _executeStarted = true;
        }
        _condition.notify_all();

        std::unique_lock lock(_mutex);
        _condition.wait(lock, [this] { return !pauseExecute; });
        lock.unlock();

        if (returnFailure)
        {
            ResourceCleanupResult result = prepared.preparationResult();
            result.success = false;
            result.errorMessage = QStringLiteral("fake execute failure");
            return result;
        }
        return ProjectResourceCleanupService::executePreparedCleanup(prepared);
    }

    bool finalize(ProjectData* projectData,
                  const PreparedResourceCleanup& prepared,
                  const ResourceCleanupResult& result) override
    {
        ++finalizeCount;
        return ProjectResourceCleanupService::finalizePreparedCleanup(projectData, prepared, result);
    }

    bool waitForExecuteStart()
    {
        std::unique_lock lock(_mutex);
        return _condition.wait_for(
            lock, std::chrono::seconds(2), [this] { return _executeStarted; });
    }

    void releaseExecute()
    {
        {
            std::lock_guard lock(_mutex);
            pauseExecute = false;
        }
        _condition.notify_all();
    }

private:
    std::mutex _mutex;
    std::condition_variable _condition;
    bool _executeStarted = false;
};

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

class ProjectResourceCleanupCoordinatorTest : public testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(temporary.isValid());
        ASSERT_TRUE(project.createProject(
            temporary.filePath(QStringLiteral("cleanup.plascan")),
            QStringLiteral("cleanup")));

        root = xjw::common::project::ProjectIO::projectRootFromPlascan(
            project.currentProjectPath());
        artifactPath = QDir(root).filePath(QStringLiteral("reconstruction/model.obj"));
        ASSERT_TRUE(QDir().mkpath(QFileInfo(artifactPath).absolutePath()));
        QFile artifact(artifactPath);
        ASSERT_TRUE(artifact.open(QIODevice::WriteOnly));
        ASSERT_GT(artifact.write("model"), 0);

        QJsonObject metadata = project.metadata();
        metadata[QStringLiteral("model_results")] =
            QJsonArray{QJsonObject{{QStringLiteral("model_ply"), artifactPath}}};
        project.updateMetadata(metadata, false);
    }

    QTemporaryDir temporary;
    ProjectData project;
    ProjectSession session{&project};
    QString root;
    QString artifactPath;
};

TEST_F(ProjectResourceCleanupCoordinatorTest, RejectsSecondRequestAndRestoresBusyState)
{
    auto messages = std::make_unique<FakeProjectUiMessageAdapter>();
    auto executor = std::make_shared<GatedCleanupExecutor>();
    executor->pauseExecute = true;
    ProjectResourceCleanupCoordinator coordinator(
        &session, messages.get(), executor);
    QPushButton requestWidget;

    coordinator.deleteGeneratedData(
        QStringLiteral("3D模型"), {artifactPath}, &requestWidget);
    ASSERT_TRUE(executor->waitForExecuteStart());
    EXPECT_TRUE(coordinator.isRunning());
    EXPECT_FALSE(requestWidget.isEnabled());
    EXPECT_TRUE(coordinator.rejectLifecycleChange(QStringLiteral("关闭项目")));

    coordinator.deleteGeneratedData(
        QStringLiteral("3D模型"), {artifactPath}, &requestWidget);
    executor->releaseExecute();

    ASSERT_TRUE(waitUntil([&coordinator] { return !coordinator.isRunning(); }));
    EXPECT_TRUE(requestWidget.isEnabled());
    EXPECT_EQ(executor->finalizeCount, 1);
    EXPECT_FALSE(QFileInfo::exists(artifactPath));
}

TEST_F(ProjectResourceCleanupCoordinatorTest, PreparationFailureReleasesBusyStateAndPresentsResult)
{
    auto messages = std::make_unique<FakeProjectUiMessageAdapter>();
    auto executor = std::make_shared<GatedCleanupExecutor>();
    ProjectResourceCleanupCoordinator coordinator(
        &session, messages.get(), executor);
    QSignalSpy finishedSpy(
        &coordinator, &ProjectResourceCleanupCoordinator::finished);

    coordinator.deleteGeneratedData(
        QStringLiteral("未知分组"), {artifactPath});

    EXPECT_FALSE(coordinator.isRunning());
    EXPECT_EQ(executor->finalizeCount, 0);
    EXPECT_EQ(finishedSpy.count(), 0);
    ASSERT_EQ(messages->warningTexts.size(), 1U);
    EXPECT_NE(messages->warningTexts.front().indexOf(QStringLiteral("暂不支持")), -1);
}

TEST_F(ProjectResourceCleanupCoordinatorTest, ExecuteFailureStillFinalizesAndRestoresBusyState)
{
    auto messages = std::make_unique<FakeProjectUiMessageAdapter>();
    auto executor = std::make_shared<GatedCleanupExecutor>();
    executor->returnFailure = true;
    ProjectResourceCleanupCoordinator coordinator(
        &session, messages.get(), executor);
    QSignalSpy finishedSpy(
        &coordinator, &ProjectResourceCleanupCoordinator::finished);

    coordinator.deleteGeneratedData(
        QStringLiteral("3D模型"), {artifactPath});

    ASSERT_TRUE(waitUntil([&coordinator] { return !coordinator.isRunning(); }));
    EXPECT_EQ(executor->finalizeCount, 1);
    EXPECT_EQ(finishedSpy.count(), 1);
    EXPECT_TRUE(QFileInfo::exists(artifactPath));
    ASSERT_EQ(messages->warningTexts.size(), 1U);
    EXPECT_NE(messages->warningTexts.front().indexOf(QStringLiteral("fake execute failure")), -1);
}

TEST_F(ProjectResourceCleanupCoordinatorTest, StaleSessionSuppressesUserResultButFinalizesTransaction)
{
    auto messages = std::make_unique<FakeProjectUiMessageAdapter>();
    auto executor = std::make_shared<GatedCleanupExecutor>();
    executor->pauseExecute = true;
    ProjectResourceCleanupCoordinator coordinator(
        &session, messages.get(), executor);

    coordinator.deleteGeneratedData(
        QStringLiteral("3D模型"), {artifactPath});
    ASSERT_TRUE(executor->waitForExecuteStart());
    const std::size_t informationBeforeStale = messages->informationTexts.size();
    session.advanceGeneration();
    executor->releaseExecute();

    ASSERT_TRUE(waitUntil([&coordinator] { return !coordinator.isRunning(); }));
    EXPECT_EQ(executor->finalizeCount, 1);
    EXPECT_EQ(messages->informationTexts.size(), informationBeforeStale);
}

TEST_F(ProjectResourceCleanupCoordinatorTest, DestructorWaitsAndFinalizesTransaction)
{
    auto messages = std::make_unique<FakeProjectUiMessageAdapter>();
    auto executor = std::make_shared<GatedCleanupExecutor>();
    executor->pauseExecute = true;
    auto coordinator = std::make_unique<ProjectResourceCleanupCoordinator>(
        &session, messages.get(), executor);
    int finishedCount = 0;
    QObject::connect(
        coordinator.get(),
        &ProjectResourceCleanupCoordinator::finished,
        [&finishedCount](const QString&) { ++finishedCount; });

    coordinator->deleteGeneratedData(
        QStringLiteral("3D模型"), {artifactPath});
    ASSERT_TRUE(executor->waitForExecuteStart());

    std::thread releaser([executor]
                         {
                             std::this_thread::sleep_for(std::chrono::milliseconds(50));
                             executor->releaseExecute();
                         });
    coordinator.reset();
    releaser.join();

    EXPECT_EQ(executor->finalizeCount, 1);
    EXPECT_EQ(finishedCount, 0);
}

} // namespace

int main(int argc, char** argv)
{
    qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
    QApplication application(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
