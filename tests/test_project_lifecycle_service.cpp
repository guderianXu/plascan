#include "project/services/ProjectLifecycleService.h"

#include "project/services/ProjectResourceCleanupCoordinator.h"
#include "project/services/ProjectSession.h"
#include "project/services/ProjectUiMessageAdapter.h"
#include "project/ProjectSessionModel.h"

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QPointer>
#include <QThread>
#include <QTemporaryDir>

#include <gtest/gtest.h>

#include <functional>
#include <memory>

namespace
{

QCoreApplication& qtApplication()
{
    static QCoreApplication* application = nullptr;
    if (!application)
    {
        int argc = 1;
        auto* argv = new char*[2];
        argv[0] = const_cast<char*>("test_project_lifecycle_service");
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

class FakeMessages final : public ProjectUiMessageAdapter
{
public:
    UiDialogResult saveResult;

    void information(QWidget*, const QString&, const QString&) override {}
    void warning(QWidget*, const QString&, const QString&) override {}
    void critical(QWidget*, const QString&, const QString&) override {}
    UiAnswer question(QWidget*, const QString&, const QString&, UiAnswer) override
    {
        return UiAnswer::No;
    }
    UiDialogResult getText(QWidget*, const QString&, const QString&, const QString&) override { return {}; }
    UiDialogResult getDouble(QWidget*, const QString&, const QString&, double, double, double, int) override
    {
        return {};
    }
    UiDialogResult getItem(QWidget*, const QString&, const QString&, const QStringList&, int) override { return {}; }
    UiDialogResult selectOpenFile(QWidget*, const QString&, const QString&, const QString&, QDir::Filters) override
    {
        return {};
    }
    UiDialogResult selectOpenFiles(QWidget*, const QString&, const QString&, const QString&, QDir::Filters) override
    {
        return {};
    }
    UiDialogResult selectDirectory(QWidget*, const QString&, const QString&, QDir::Filters) override { return {}; }
    UiDialogResult selectSaveFile(QWidget*,
                                  const QString&,
                                  const QString&,
                                  const QString&,
                                  QDir::Filters,
                                  const QString&) override
    {
        return saveResult;
    }
};

class DeferredSessionDrain final
{
public:
    bool defer(std::function<void()> continuation)
    {
        ++requestCount;
        if (pendingContinuation)
        {
            return false;
        }
        pendingContinuation = std::move(continuation);
        return true;
    }

    bool hasPendingContinuation() const
    {
        return static_cast<bool>(pendingContinuation);
    }

    void complete()
    {
        std::function<void()> continuation = std::move(pendingContinuation);
        pendingContinuation = {};
        if (continuation)
        {
            continuation();
        }
    }

    int requestCount = 0;
    std::function<void()> pendingContinuation;
};

} // namespace

TEST(ProjectLifecycleServiceTest, CancelledCreateDoesNotCreateProject)
{
    qtApplication();
    ProjectData data;
    xjw::gui::project::ProjectSession session(&data);
    FakeMessages messages;
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);

    service.createNewProject();

    EXPECT_FALSE(data.hasProject());
    EXPECT_FALSE(service.isOpenInProgress());
}

TEST(ProjectLifecycleServiceTest, SaveWithoutProjectIsAStableNoOp)
{
    qtApplication();
    ProjectData data;
    xjw::gui::project::ProjectSession session(&data);
    FakeMessages messages;
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);

    bool saveStarted = false;
    QObject::connect(&service, &ProjectLifecycleService::saveStarted, [&saveStarted]() { saveStarted = true; });
    service.saveProject();

    EXPECT_FALSE(saveStarted);
    EXPECT_FALSE(data.hasProject());
}

TEST(ProjectLifecycleServiceTest, CreateProjectNormalizesExtensionAndPublishesSignals)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData data;
    xjw::gui::project::ProjectSession session(&data);
    FakeMessages messages;
    messages.saveResult = {true, temporary.filePath(QStringLiteral("created")), {}};
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);

    QString createdPath;
    int openedCount = 0;
    QObject::connect(&service, &ProjectLifecycleService::projectCreated,
                     [&createdPath](const QString& path) { createdPath = path; });
    QObject::connect(&service, &ProjectLifecycleService::projectOpened,
                     [&openedCount](ProjectData*) { ++openedCount; });

    service.createNewProject();

    EXPECT_TRUE(data.hasProject());
    EXPECT_TRUE(data.currentProjectPath().endsWith(QStringLiteral(".plascan")));
    EXPECT_EQ(createdPath, data.currentProjectPath());
    EXPECT_EQ(openedCount, 1);
}

TEST(ProjectLifecycleServiceTest, FailedCreateKeepsPreviousSessionContext)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData data;
    const QString oldPath = temporary.filePath(QStringLiteral("old.plascan"));
    ASSERT_TRUE(data.createProject(oldPath, QStringLiteral("old")));
    xjw::gui::project::ProjectSession session(&data);
    const auto before = session.context();

    const QString blockerPath = temporary.filePath(QStringLiteral("not-a-directory"));
    QFile blocker(blockerPath);
    ASSERT_TRUE(blocker.open(QIODevice::WriteOnly));
    blocker.write("blocker");
    blocker.close();

    FakeMessages messages;
    messages.saveResult = {true, QDir(blockerPath).filePath(QStringLiteral("child")), {}};
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);
    service.createNewProject();

    EXPECT_TRUE(data.hasProject());
    EXPECT_EQ(data.currentProjectPath(), oldPath);
    EXPECT_TRUE(session.isCurrent(before));
    EXPECT_EQ(session.context().generation, before.generation);
}

TEST(ProjectLifecycleServiceTest, FailedOpenKeepsPreviousSessionContext)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData data;
    const QString oldPath = temporary.filePath(QStringLiteral("old-open.plascan"));
    ASSERT_TRUE(data.createProject(oldPath, QStringLiteral("old-open")));
    xjw::gui::project::ProjectSession session(&data);
    const auto before = session.context();

    ProjectData target;
    const QString targetPath = temporary.filePath(QStringLiteral("target-open.plascan"));
    ASSERT_TRUE(target.createProject(targetPath, QStringLiteral("target-open")));
    QString saveError;
    ASSERT_TRUE(target.saveProject(&saveError)) << qPrintable(saveError);
    ASSERT_TRUE(target.closeProject(&saveError)) << qPrintable(saveError);

    // Keep the old project open but make ProjectData reject the snapshot
    // application at the actual switch boundary.
    const auto persistence = data.prepareResourceCleanupPersistence(data.coreFilesMeta());
    ASSERT_TRUE(persistence.isValid());

    FakeMessages messages;
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);
    service.openProjectFromPath(targetPath);
    ASSERT_TRUE(waitUntil([&service] { return !service.isOpenInProgress(); }));

    EXPECT_TRUE(data.hasProject());
    EXPECT_EQ(data.currentProjectPath(), oldPath);
    EXPECT_TRUE(session.isCurrent(before));
    EXPECT_EQ(session.context().generation, before.generation);

    EXPECT_TRUE(data.finalizeResourceCleanupPersistence(persistence, false, false));
}

TEST(ProjectLifecycleServiceTest, ValidSavePublishesCompletion)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());

    ProjectData data;
    const QString projectPath = temporary.filePath(QStringLiteral("save.plascan"));
    ASSERT_TRUE(data.createProject(projectPath, QStringLiteral("save")));
    data.updateMetadata(QJsonObject{{QStringLiteral("task5"), true}});

    xjw::gui::project::ProjectSession session(&data);
    FakeMessages messages;
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);
    bool saveStarted = false;
    bool saveFinished = false;
    QObject::connect(&service, &ProjectLifecycleService::saveStarted,
                     [&saveStarted]() { saveStarted = true; });
    QObject::connect(&service, &ProjectLifecycleService::saveFinished,
                     [&saveFinished](bool success) { saveFinished = success; });

    service.saveProject();

    ASSERT_TRUE(waitUntil([&saveFinished] { return saveFinished; }));
    EXPECT_TRUE(saveStarted);
    EXPECT_FALSE(data.isDirty());
}

TEST(ProjectLifecycleServiceTest, InvalidOpenPathReportsLifecycleError)
{
    qtApplication();
    ProjectData data;
    xjw::gui::project::ProjectSession session(&data);
    FakeMessages messages;
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);

    QString operation;
    QString message;
    QObject::connect(&service,
                     &ProjectLifecycleService::lifecycleError,
                     [&operation, &message](const QString& op, const QString& detail)
                     {
                         operation = op;
                         message = detail;
                     });
    service.openProjectFromPath(QStringLiteral("/definitely/missing/project.plascan"));
    ASSERT_TRUE(waitUntil([&operation] { return !operation.isEmpty(); }));

    EXPECT_EQ(operation, QStringLiteral("打开项目"));
    EXPECT_FALSE(message.isEmpty());
    EXPECT_FALSE(data.hasProject());
}

TEST(ProjectLifecycleServiceTest, DuplicateOpenRequestIsRejectedWhileFirstIsRunning)
{
    qtApplication();
    ProjectData data;
    xjw::gui::project::ProjectSession session(&data);
    FakeMessages messages;
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);
    int startedCount = 0;
    QObject::connect(&service, &ProjectLifecycleService::projectOpenStarted,
                     [&startedCount](const QString&) { ++startedCount; });

    service.openProjectFromPath(QStringLiteral("/definitely/missing/first.plascan"));
    service.openProjectFromPath(QStringLiteral("/definitely/missing/second.plascan"));

    ASSERT_TRUE(waitUntil([&service] { return !service.isOpenInProgress(); }));
    EXPECT_EQ(startedCount, 1);
}

TEST(ProjectLifecycleServiceTest, StaleOpenCallbackDoesNotReportAnErrorOrMutateData)
{
    qtApplication();
    ProjectData data;
    xjw::gui::project::ProjectSession session(&data);
    FakeMessages messages;
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);
    bool finished = false;
    bool success = true;
    QString lifecycleError;
    QObject::connect(&service,
                     &ProjectLifecycleService::projectOpenFinished,
                     [&finished, &success](bool opened, const QString&) {
                         finished = true;
                         success = opened;
                     });
    QObject::connect(&service,
                     &ProjectLifecycleService::lifecycleError,
                     [&lifecycleError](const QString&, const QString& message) { lifecycleError = message; });

    service.openProjectFromPath(QStringLiteral("/definitely/missing/stale.plascan"));
    session.advanceGeneration();

    ASSERT_TRUE(waitUntil([&finished] { return finished; }));
    EXPECT_FALSE(success);
    EXPECT_TRUE(lifecycleError.isEmpty());
    EXPECT_FALSE(data.hasProject());
}

TEST(ProjectLifecycleServiceTest, CloseIsGatedWhilePortableExportIsRunning)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("export.plascan")),
                                   QStringLiteral("export")));

    xjw::gui::project::ProjectSession session(&data);
    FakeMessages messages;
    messages.saveResult = {true, temporary.filePath(QStringLiteral("portable")), {}};
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);

    service.exportPortableProject();
    ASSERT_TRUE(service.isPortableExportInProgress());
    service.closeProject();

    EXPECT_TRUE(data.hasProject());
}

TEST(ProjectLifecycleServiceTest, SwitchChunkWaitsForDrainAndRejectsReentryBeforeSingleCommit)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("deferred-switch.plascan")),
                                   QStringLiteral("deferred-switch")));
    const QString first_chunk = data.activeChunkId();
    QString second_chunk;
    QString error_message;
    ASSERT_TRUE(data.createChunk(QStringLiteral("second"), &second_chunk, &error_message)) << qPrintable(error_message);
    ASSERT_TRUE(data.switchChunk(first_chunk, &error_message)) << qPrintable(error_message);

    xjw::gui::project::ProjectSession session(&data);
    const auto before = session.context();
    FakeMessages messages;
    ProjectLifecycleService service(&data, &session, &messages, nullptr, nullptr);
    DeferredSessionDrain drain;
    service.setSessionDrainHandler([&drain](std::function<void()> continuation)
                                   { return drain.defer(std::move(continuation)); });
    int active_chunk_change_count = 0;
    QObject::connect(&data,
                     &ProjectData::activeChunkChanged,
                     &service,
                     [&active_chunk_change_count]() { ++active_chunk_change_count; });

    service.switchChunk(second_chunk);

    ASSERT_TRUE(drain.hasPendingContinuation());
    EXPECT_EQ(drain.requestCount, 1);
    EXPECT_TRUE(service.isBusy());
    EXPECT_EQ(data.activeChunkId(), first_chunk);
    EXPECT_TRUE(session.isCurrent(before));
    EXPECT_EQ(active_chunk_change_count, 0);

    service.switchChunk(second_chunk);
    EXPECT_EQ(drain.requestCount, 1);
    EXPECT_EQ(data.activeChunkId(), first_chunk);
    EXPECT_EQ(active_chunk_change_count, 0);

    drain.complete();

    EXPECT_FALSE(drain.hasPendingContinuation());
    EXPECT_FALSE(service.isBusy());
    EXPECT_EQ(data.activeChunkId(), second_chunk);
    EXPECT_FALSE(session.isCurrent(before));
    EXPECT_EQ(active_chunk_change_count, 1);

    drain.complete();
    EXPECT_EQ(data.activeChunkId(), second_chunk);
    EXPECT_EQ(active_chunk_change_count, 1);
}

TEST(ProjectLifecycleServiceTest, DestroyedOwnerDropsDeferredSessionContinuation)
{
    qtApplication();
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    ProjectData data;
    ASSERT_TRUE(data.createProject(temporary.filePath(QStringLiteral("destroyed-owner.plascan")),
                                   QStringLiteral("destroyed-owner")));
    const QString first_chunk = data.activeChunkId();
    QString second_chunk;
    QString error_message;
    ASSERT_TRUE(data.createChunk(QStringLiteral("second"), &second_chunk, &error_message)) << qPrintable(error_message);
    ASSERT_TRUE(data.switchChunk(first_chunk, &error_message)) << qPrintable(error_message);

    xjw::gui::project::ProjectSession session(&data);
    const auto before = session.context();
    FakeMessages messages;
    DeferredSessionDrain drain;
    auto service = std::make_unique<ProjectLifecycleService>(&data, &session, &messages, nullptr, nullptr);
    service->setSessionDrainHandler([&drain](std::function<void()> continuation)
                                    { return drain.defer(std::move(continuation)); });
    int active_chunk_change_count = 0;
    QObject::connect(
        &data, &ProjectData::activeChunkChanged, [&active_chunk_change_count]() { ++active_chunk_change_count; });

    service->switchChunk(second_chunk);
    ASSERT_TRUE(drain.hasPendingContinuation());
    ASSERT_TRUE(service->isBusy());
    QPointer<ProjectLifecycleService> guarded_service(service.get());

    service.reset();

    EXPECT_TRUE(guarded_service.isNull());
    EXPECT_FALSE(session.isBusy());
    drain.complete();
    QCoreApplication::processEvents(QEventLoop::AllEvents);
    EXPECT_EQ(data.activeChunkId(), first_chunk);
    EXPECT_TRUE(session.isCurrent(before));
    EXPECT_EQ(active_chunk_change_count, 0);
}
