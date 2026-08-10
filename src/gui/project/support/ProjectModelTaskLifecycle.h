#pragma once

#include "ProjectSessionContext.h"

#include <QString>

#include <atomic>
#include <memory>

namespace xjw::gui::project
{

enum class ProjectModelRunKind
{
    Model,
    Texture
};

class ProjectModelTaskContext final
{
public:
    ~ProjectModelTaskContext();

    const QString &taskId() const;
    const ProjectSessionContext &session() const;
    std::shared_ptr<std::atomic_bool> cancellationFlag() const;
    bool isCancellationRequested() const;
    void requestCancellation();

    void markPublished();
    bool isPublished() const;
    bool cleanupUnpublishedRun(QString *errorMessage = nullptr) const;

private:
    friend class ProjectModelTaskLifecycle;

    ProjectModelTaskContext(QString taskId,
                            ProjectSessionContext session,
                            ProjectModelRunKind runKind,
                            QString runBaseDirectory,
                            QString runDirectory);

    QString _taskId;
    ProjectSessionContext _session;
    ProjectModelRunKind _runKind = ProjectModelRunKind::Model;
    QString _runBaseDirectory;
    QString _runDirectory;
    std::shared_ptr<std::atomic_bool> _cancellationFlag;
    std::atomic_bool _published{false};
    mutable std::atomic_bool _cleanupCompleted{false};
};

using ProjectModelTaskPtr = std::shared_ptr<ProjectModelTaskContext>;

class ProjectModelTaskLifecycle final
{
public:
    ProjectModelTaskLifecycle() = default;
    ~ProjectModelTaskLifecycle();
    ProjectModelTaskLifecycle(const ProjectModelTaskLifecycle &) = delete;
    ProjectModelTaskLifecycle &operator=(
        const ProjectModelTaskLifecycle &) = delete;

    ProjectModelTaskPtr startTask(const QString &taskId,
                                  const ProjectSessionContext &session,
                                  ProjectModelRunKind runKind,
                                  const QString &runBaseDirectory,
                                  const QString &runDirectory);

    void requestCancelActive();
    bool acceptsCallback(const ProjectModelTaskPtr &task,
                         const ProjectSessionContext &currentSession) const;
    bool finishIfActive(const ProjectModelTaskPtr &task);
    bool isRunning() const;
    bool isCancelling() const;
    ProjectModelTaskPtr activeTask() const;

private:
    ProjectModelTaskPtr _activeTask;
};

} // namespace xjw::gui::project
