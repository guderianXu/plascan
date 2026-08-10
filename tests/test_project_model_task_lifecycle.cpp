#include "ProjectModelTaskLifecycle.h"
#include "ModelOutputPolicy.h"

#include <gtest/gtest.h>

#include <QDir>
#include <QFile>
#include <QTemporaryDir>

namespace
{

using xjw::gui::project::ProjectModelRunKind;
using xjw::gui::project::ProjectModelTaskLifecycle;
using xjw::gui::project::ProjectSessionContext;

QString modelRunDirectory(const QString &base, const QString &taskId)
{
    return QDir(base).filePath(
        QStringLiteral("model_runs/%1").arg(taskId));
}

void createOwnedModelRun(const QString &base,
                         const QString &taskId,
                         QString *directory)
{
    ASSERT_NE(directory, nullptr);
    QString actualTaskId;
    QString error;
    ASSERT_TRUE(xjw::mesh::workflow::createModelRunOutputDirectory(
        base, taskId, &actualTaskId, directory, &error))
        << error.toStdString();
    ASSERT_EQ(actualTaskId, taskId);
    QFile marker(QDir(*directory).filePath(QStringLiteral("partial.tmp")));
    ASSERT_TRUE(marker.open(QIODevice::WriteOnly));
    ASSERT_EQ(marker.write("partial"), 7);
}

TEST(ProjectModelTaskLifecycleTest,
     RejectsCallbacksAfterProjectChunkOrGenerationSwitch)
{
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    const ProjectSessionContext session{
        QDir(temp.path()).filePath(QStringLiteral("a.plascan")),
        QStringLiteral("chunk-a"),
        7};
    ProjectModelTaskLifecycle lifecycle;
    const auto task = lifecycle.startTask(
        QStringLiteral("task-a"),
        session,
        ProjectModelRunKind::Model,
        temp.path(),
        modelRunDirectory(temp.path(), QStringLiteral("task-a")));
    ASSERT_TRUE(task);
    EXPECT_TRUE(lifecycle.acceptsCallback(task, session));

    ProjectSessionContext switchedProject = session;
    switchedProject.projectPath = QDir(temp.path()).filePath(
        QStringLiteral("b.plascan"));
    EXPECT_FALSE(lifecycle.acceptsCallback(task, switchedProject));

    ProjectSessionContext switchedChunk = session;
    switchedChunk.chunkId = QStringLiteral("chunk-b");
    EXPECT_FALSE(lifecycle.acceptsCallback(task, switchedChunk));

    ProjectSessionContext nextGeneration = session;
    ++nextGeneration.generation;
    lifecycle.requestCancelActive();
    EXPECT_TRUE(lifecycle.isRunning());
    EXPECT_TRUE(lifecycle.isCancelling());
    EXPECT_FALSE(lifecycle.acceptsCallback(task, nextGeneration));
    EXPECT_TRUE(lifecycle.finishIfActive(task));
    EXPECT_FALSE(lifecycle.isRunning());
}

TEST(ProjectModelTaskLifecycleTest,
     DestructionRequestsCancellationAndCleansOnlyUnpublishedRun)
{
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    const QString taskId = QStringLiteral("destructor-task");
    QString runDirectory;
    createOwnedModelRun(temp.path(), taskId, &runDirectory);

    xjw::gui::project::ProjectModelTaskPtr task;
    std::shared_ptr<std::atomic_bool> cancellationFlag;
    {
        ProjectModelTaskLifecycle lifecycle;
        task = lifecycle.startTask(
            taskId,
            {QDir(temp.path()).filePath(QStringLiteral("project.plascan")),
             QStringLiteral("chunk-a"),
             1},
            ProjectModelRunKind::Model,
            temp.path(),
            runDirectory);
        ASSERT_TRUE(task);
        cancellationFlag = task->cancellationFlag();
        EXPECT_FALSE(cancellationFlag->load(std::memory_order_relaxed));
    }

    EXPECT_TRUE(cancellationFlag->load(std::memory_order_relaxed));
    EXPECT_TRUE(QFileInfo::exists(runDirectory));
    task.reset();
    EXPECT_FALSE(QFileInfo::exists(runDirectory));
}

TEST(ProjectModelTaskLifecycleTest,
     LateCompletionCannotFinishOrWriteBackOverNewTask)
{
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    const ProjectSessionContext session{
        QDir(temp.path()).filePath(QStringLiteral("project.plascan")),
        QStringLiteral("chunk-a"),
        2};
    ProjectModelTaskLifecycle lifecycle;
    const auto oldTask = lifecycle.startTask(
        QStringLiteral("old-task"),
        session,
        ProjectModelRunKind::Model,
        temp.path(),
        modelRunDirectory(temp.path(), QStringLiteral("old-task")));
    ASSERT_TRUE(oldTask);
    ASSERT_TRUE(lifecycle.finishIfActive(oldTask));

    const auto newTask = lifecycle.startTask(
        QStringLiteral("new-task"),
        session,
        ProjectModelRunKind::Model,
        temp.path(),
        modelRunDirectory(temp.path(), QStringLiteral("new-task")));
    ASSERT_TRUE(newTask);

    int metadataWrites = 0;
    if (lifecycle.acceptsCallback(oldTask, session))
    {
        ++metadataWrites;
    }
    EXPECT_FALSE(lifecycle.finishIfActive(oldTask));
    EXPECT_EQ(metadataWrites, 0);
    EXPECT_TRUE(lifecycle.isRunning());
    EXPECT_EQ(lifecycle.activeTask()->taskId(), QStringLiteral("new-task"));

    if (lifecycle.acceptsCallback(newTask, session))
    {
        ++metadataWrites;
    }
    EXPECT_EQ(metadataWrites, 1);
    EXPECT_TRUE(lifecycle.finishIfActive(newTask));
}

TEST(ProjectModelTaskLifecycleTest,
     CleanupRefusesDirectoryThatDoesNotMatchTaskIdentity)
{
    QTemporaryDir temp;
    ASSERT_TRUE(temp.isValid());
    QString protectedDirectory;
    createOwnedModelRun(
        temp.path(), QStringLiteral("other-task"), &protectedDirectory);

    ProjectModelTaskLifecycle lifecycle;
    const auto task = lifecycle.startTask(
        QStringLiteral("owned-task"),
        {QDir(temp.path()).filePath(QStringLiteral("project.plascan")),
         QStringLiteral("chunk-a"),
         3},
        ProjectModelRunKind::Model,
        temp.path(),
        protectedDirectory);
    ASSERT_TRUE(task);
    QString error;
    EXPECT_FALSE(task->cleanupUnpublishedRun(&error));
    EXPECT_FALSE(error.isEmpty());
    EXPECT_TRUE(QFileInfo::exists(protectedDirectory));
    task->markPublished();
    EXPECT_TRUE(lifecycle.finishIfActive(task));
}

} // namespace
