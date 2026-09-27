#include "ApplicationShutdownCoordinator.h"

#include <QPointer>

#include <utility>

namespace xjw::gui::main_window
{

    ApplicationShutdownCoordinator::ApplicationShutdownCoordinator(Hooks hooks, QObject* parent)
        : QObject(parent), _hooks(std::move(hooks))
    {
        _pollTimer.setSingleShot(true);
        connect(&_pollTimer, &QTimer::timeout, this, &ApplicationShutdownCoordinator::poll);
    }

    bool ApplicationShutdownCoordinator::requestShutdown()
    {
        if (_shutdownInProgress || _readyToClose)
        {
            return false;
        }

        _shutdownInProgress = true;
        setStage(tr("正在等待项目操作安全结束..."));
        if (_hooks.requestResourceCancellation)
        {
            _hooks.requestResourceCancellation();
        }
        schedulePoll(0);
        return true;
    }

    bool ApplicationShutdownCoordinator::isShutdownInProgress() const noexcept
    {
        return _shutdownInProgress;
    }

    bool ApplicationShutdownCoordinator::isReadyToClose() const noexcept
    {
        return _readyToClose;
    }

    void ApplicationShutdownCoordinator::schedulePoll(int delayMs)
    {
        if (!_shutdownInProgress || _pollTimer.isActive())
        {
            return;
        }
        _pollTimer.start(delayMs);
    }

    void ApplicationShutdownCoordinator::poll()
    {
        if (!_shutdownInProgress)
        {
            return;
        }

        if (_hooks.requestResourceCancellation)
        {
            _hooks.requestResourceCancellation();
        }
        if (!_shutdownGateAcquired)
        {
            if (_hooks.projectOperationsIdle && !_hooks.projectOperationsIdle())
            {
                schedulePoll();
                return;
            }
            if (_hooks.tryAcquireShutdownGate && !_hooks.tryAcquireShutdownGate())
            {
                schedulePoll();
                return;
            }
            _shutdownGateAcquired = true;
        }

        if (!_taskDrainStarted)
        {
            setStage(tr("正在取消并排空后台任务..."));
            if (_hooks.projectTaskDrainInProgress && _hooks.projectTaskDrainInProgress())
            {
                schedulePoll();
                return;
            }

            if (!_hooks.startProjectTaskDrain)
            {
                _taskDrainStarted = true;
                _taskDrainFinished = true;
            }
            else
            {
                _taskDrainStarted = true;
                const QPointer<ApplicationShutdownCoordinator> guarded_this(this);
                const bool started = _hooks.startProjectTaskDrain(
                    [guarded_this]
                    {
                        if (!guarded_this)
                        {
                            return;
                        }
                        guarded_this->_taskDrainFinished = true;
                        guarded_this->schedulePoll(0);
                    });
                if (!started)
                {
                    _taskDrainStarted = false;
                    schedulePoll();
                    return;
                }
            }
        }

        if (!_taskDrainFinished)
        {
            schedulePoll();
            return;
        }

        if (!_runtimeShutdownRequested)
        {
            setStage(tr("正在保存任务状态并关闭调度器..."));
            _runtimeShutdownRequested = true;
            if (_hooks.requestRuntimeShutdown)
            {
                _hooks.requestRuntimeShutdown();
            }
        }
        if (_hooks.runtimeShutdownComplete && !_hooks.runtimeShutdownComplete())
        {
            schedulePoll();
            return;
        }

        _shutdownInProgress = false;
        _readyToClose = true;
        setStage(tr("后台任务已安全结束，正在退出..."));
        emit readyToClose();
    }

    void ApplicationShutdownCoordinator::setStage(const QString& message)
    {
        if (_stage == message)
        {
            return;
        }
        _stage = message;
        emit progressChanged(message);
    }

} // namespace xjw::gui::main_window
