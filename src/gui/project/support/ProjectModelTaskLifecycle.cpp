#include "ProjectModelTaskLifecycle.h"

#include "ModelOutputPolicy.h"

#include <utility>

namespace xjw::gui::project
{

ProjectModelTaskContext::ProjectModelTaskContext(
    QString taskId,
    ProjectSessionContext session,
    ProjectModelRunKind runKind,
    QString runBaseDirectory,
    QString runDirectory)
    : _taskId(std::move(taskId))
    , _session(std::move(session))
    , _runKind(runKind)
    , _runBaseDirectory(std::move(runBaseDirectory))
    , _runDirectory(std::move(runDirectory))
    , _cancellationFlag(std::make_shared<std::atomic_bool>(false))
{
}

ProjectModelTaskContext::~ProjectModelTaskContext()
{
    requestCancellation();
    cleanupUnpublishedRun();
}

const QString &ProjectModelTaskContext::taskId() const
{
    return _taskId;
}

const ProjectSessionContext &ProjectModelTaskContext::session() const
{
    return _session;
}

std::shared_ptr<std::atomic_bool>
ProjectModelTaskContext::cancellationFlag() const
{
    return _cancellationFlag;
}

bool ProjectModelTaskContext::isCancellationRequested() const
{
    return _cancellationFlag
        && _cancellationFlag->load(std::memory_order_relaxed);
}

void ProjectModelTaskContext::requestCancellation()
{
    if (_cancellationFlag)
    {
        _cancellationFlag->store(true, std::memory_order_relaxed);
    }
}

void ProjectModelTaskContext::markPublished()
{
    _published.store(true, std::memory_order_release);
}

bool ProjectModelTaskContext::isPublished() const
{
    return _published.load(std::memory_order_acquire);
}

bool ProjectModelTaskContext::cleanupUnpublishedRun(
    QString *errorMessage) const
{
    if (isPublished()
        || _cleanupCompleted.load(std::memory_order_acquire))
    {
        return true;
    }

    const bool removed = _runKind == ProjectModelRunKind::Texture
        ? xjw::mesh::workflow::removeUnpublishedTextureRunDirectory(
              _runBaseDirectory,
              _taskId,
              _runDirectory,
              errorMessage)
        : xjw::mesh::workflow::removeUnpublishedModelRunDirectory(
              _runBaseDirectory,
              _taskId,
              _runDirectory,
              errorMessage);
    if (removed)
    {
        _cleanupCompleted.store(true, std::memory_order_release);
    }
    return removed;
}

ProjectModelTaskLifecycle::~ProjectModelTaskLifecycle()
{
    requestCancelActive();
}

ProjectModelTaskPtr ProjectModelTaskLifecycle::startTask(
    const QString &taskId,
    const ProjectSessionContext &session,
    ProjectModelRunKind runKind,
    const QString &runBaseDirectory,
    const QString &runDirectory)
{
    if (_activeTask || taskId.trimmed().isEmpty())
    {
        return {};
    }

    _activeTask = ProjectModelTaskPtr(new ProjectModelTaskContext(
        taskId.trimmed(),
        session,
        runKind,
        runBaseDirectory,
        runDirectory));
    return _activeTask;
}

void ProjectModelTaskLifecycle::requestCancelActive()
{
    if (_activeTask)
    {
        _activeTask->requestCancellation();
    }
}

bool ProjectModelTaskLifecycle::acceptsCallback(
    const ProjectModelTaskPtr &task,
    const ProjectSessionContext &currentSession) const
{
    return _activeTask
        && task
        && _activeTask.get() == task.get()
        && _activeTask->taskId() == task->taskId()
        && task->session().matches(currentSession);
}

bool ProjectModelTaskLifecycle::finishIfActive(
    const ProjectModelTaskPtr &task)
{
    if (!_activeTask
        || !task
        || _activeTask.get() != task.get()
        || _activeTask->taskId() != task->taskId())
    {
        return false;
    }
    _activeTask.reset();
    return true;
}

bool ProjectModelTaskLifecycle::isRunning() const
{
    return static_cast<bool>(_activeTask);
}

bool ProjectModelTaskLifecycle::isCancelling() const
{
    return _activeTask && _activeTask->isCancellationRequested();
}

ProjectModelTaskPtr ProjectModelTaskLifecycle::activeTask() const
{
    return _activeTask;
}

} // namespace xjw::gui::project
