#include "TaskbarProgressController.h"

#include <QEvent>
#include <QWidget>

#include <algorithm>
#include <utility>

#ifdef Q_OS_WIN
#include <QAbstractNativeEventFilter>
#include <QCoreApplication>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shobjidl.h>
#endif

namespace xjw::gui::platform
{
namespace
{
constexpr int kAggregateProgressMaximum = 1000;
}

TaskbarProgressSnapshot aggregateTaskbarProgress(
    const QVector<TaskbarProgressItem> &items)
{
    qint64 scaled_sum = 0;
    const int active_count = static_cast<int>(items.size());
    for (const TaskbarProgressItem &item : items)
    {
        if (item.maximum <= 0)
        {
            return {TaskbarProgressState::Indeterminate, 0, 0};
        }

        const int value = std::clamp(item.value, 0, item.maximum);
        scaled_sum += (static_cast<qint64>(value) * kAggregateProgressMaximum
                       + item.maximum / 2)
            / item.maximum;
    }

    if (active_count == 0)
    {
        return {};
    }

    const int aggregate_value = static_cast<int>(
        (scaled_sum + active_count / 2) / active_count);
    return {TaskbarProgressState::Normal,
            std::clamp(aggregate_value, 0, kAggregateProgressMaximum),
            kAggregateProgressMaximum};
}

#ifdef Q_OS_WIN
struct TaskbarProgressController::NativeData
{
    class EventFilter final : public QAbstractNativeEventFilter
    {
    public:
        EventFilter(TaskbarProgressController *owner, UINT taskbarButtonCreatedMessage)
            : _owner(owner)
            , _taskbarButtonCreatedMessage(taskbarButtonCreatedMessage)
        {
        }

        bool nativeEventFilter(const QByteArray &, void *message, qintptr *) override
        {
            const auto *native_message = static_cast<const MSG *>(message);
            if (!_owner || !native_message
                || native_message->message != _taskbarButtonCreatedMessage)
            {
                return false;
            }

            _owner->handleTaskbarButtonCreated(
                reinterpret_cast<quintptr>(native_message->hwnd));
            return false;
        }

    private:
        TaskbarProgressController *_owner = nullptr;
        UINT _taskbarButtonCreatedMessage = 0;
    };

    ~NativeData()
    {
        releaseTaskbar();
        if (ownsComInitialization)
        {
            CoUninitialize();
        }
    }

    bool initializeCom()
    {
        if (comInitializationAttempted)
        {
            return comAvailable;
        }

        comInitializationAttempted = true;
        const HRESULT result = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        if (result == RPC_E_CHANGED_MODE)
        {
            return false;
        }
        if (FAILED(result))
        {
            return false;
        }

        ownsComInitialization = true;
        comAvailable = true;
        return true;
    }

    bool ensureTaskbar()
    {
        if (taskbar)
        {
            return true;
        }
        if (!initializeCom())
        {
            return false;
        }

        ITaskbarList3 *created_taskbar = nullptr;
        const HRESULT create_result = CoCreateInstance(
            CLSID_TaskbarList,
            nullptr,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&created_taskbar));
        if (FAILED(create_result) || !created_taskbar)
        {
            return false;
        }

        const HRESULT initialize_result = created_taskbar->HrInit();
        if (FAILED(initialize_result))
        {
            created_taskbar->Release();
            return false;
        }

        taskbar = created_taskbar;
        return true;
    }

    void releaseTaskbar()
    {
        if (taskbar)
        {
            taskbar->Release();
            taskbar = nullptr;
        }
    }

    std::unique_ptr<EventFilter> eventFilter;
    ITaskbarList3 *taskbar = nullptr;
    HWND window = nullptr;
    bool taskbarButtonReady = false;
    bool comInitializationAttempted = false;
    bool comAvailable = false;
    bool ownsComInitialization = false;
};
#else
struct TaskbarProgressController::NativeData
{
};
#endif

TaskbarProgressController::TaskbarProgressController(QWidget *window, QObject *parent)
    : QObject(parent)
    , _window(window)
    , _native(std::make_unique<NativeData>())
{
#ifdef Q_OS_WIN
    if (_window)
    {
        _window->installEventFilter(this);
        const UINT taskbar_button_created = RegisterWindowMessageW(L"TaskbarButtonCreated");
        if (taskbar_button_created != 0 && QCoreApplication::instance())
        {
            _native->eventFilter = std::make_unique<NativeData::EventFilter>(
                this, taskbar_button_created);
            QCoreApplication::instance()->installNativeEventFilter(
                _native->eventFilter.get());
        }
    }
#endif
}

TaskbarProgressController::~TaskbarProgressController()
{
#ifdef Q_OS_WIN
    if (_window)
    {
        _window->removeEventFilter(this);
    }

    if (_native->taskbar && _native->window)
    {
        _native->taskbar->SetProgressState(_native->window, TBPF_NOPROGRESS);
    }
    if (_native->eventFilter && QCoreApplication::instance())
    {
        QCoreApplication::instance()->removeNativeEventFilter(
            _native->eventFilter.get());
    }
#endif
}

void TaskbarProgressController::updateTask(const QString &taskId,
                                           int value,
                                           int maximum)
{
    if (taskId.isEmpty())
    {
        return;
    }

    _tasks.insert(taskId, {value, maximum});
    refreshProgress();
}

void TaskbarProgressController::finishTask(const QString &taskId)
{
    if (_tasks.remove(taskId))
    {
        refreshProgress();
    }
}

void TaskbarProgressController::clearTasks()
{
    _tasks.clear();
    refreshProgress();
}

bool TaskbarProgressController::hasTask(const QString &taskId) const
{
    return _tasks.contains(taskId);
}

TaskbarProgressSnapshot TaskbarProgressController::currentProgress() const
{
    return _progress;
}

void TaskbarProgressController::refreshProgress()
{
    QVector<TaskbarProgressItem> items;
    items.reserve(_tasks.size());
    for (const TaskbarProgressItem &task : std::as_const(_tasks))
    {
        items.append(task);
    }
    setProgress(aggregateTaskbarProgress(items));
}

void TaskbarProgressController::setProgress(const TaskbarProgressSnapshot &progress)
{
    TaskbarProgressSnapshot sanitized = progress;
    if (sanitized.state == TaskbarProgressState::Normal)
    {
        sanitized.maximum = std::max(1, sanitized.maximum);
        sanitized.value = std::clamp(sanitized.value, 0, sanitized.maximum);
    }
    else
    {
        sanitized.value = 0;
        sanitized.maximum = 0;
    }

    if (_progress.state == sanitized.state
        && _progress.value == sanitized.value
        && _progress.maximum == sanitized.maximum)
    {
        return;
    }

    _progress = sanitized;
    applyProgress();
}

bool TaskbarProgressController::eventFilter(QObject *watched, QEvent *event)
{
#ifdef Q_OS_WIN
    if (watched == _window && event)
    {
        if (event->type() == QEvent::WinIdChange)
        {
            _native->taskbarButtonReady = false;
            _native->window = reinterpret_cast<HWND>(_window->effectiveWinId());
            _native->releaseTaskbar();
        }
    }
#else
    Q_UNUSED(watched);
    Q_UNUSED(event);
#endif
    return QObject::eventFilter(watched, event);
}

void TaskbarProgressController::applyProgress()
{
#ifdef Q_OS_WIN
    if (!_native->taskbarButtonReady || !_native->window)
    {
        return;
    }
    if (_progress.state == TaskbarProgressState::NoProgress && !_native->taskbar)
    {
        return;
    }
    if (!_native->ensureTaskbar())
    {
        return;
    }

    switch (_progress.state)
    {
    case TaskbarProgressState::NoProgress:
        _native->taskbar->SetProgressState(_native->window, TBPF_NOPROGRESS);
        break;
    case TaskbarProgressState::Indeterminate:
        _native->taskbar->SetProgressState(_native->window, TBPF_INDETERMINATE);
        break;
    case TaskbarProgressState::Normal:
        _native->taskbar->SetProgressState(_native->window, TBPF_NORMAL);
        _native->taskbar->SetProgressValue(
            _native->window,
            static_cast<ULONGLONG>(_progress.value),
            static_cast<ULONGLONG>(_progress.maximum));
        break;
    }
#endif
}

void TaskbarProgressController::handleTaskbarButtonCreated(quintptr windowId)
{
#ifdef Q_OS_WIN
    const quintptr current_window_id = _window
        ? static_cast<quintptr>(_window->effectiveWinId())
        : 0;
    if (current_window_id == 0 || current_window_id != windowId)
    {
        return;
    }

    _native->releaseTaskbar();
    _native->window = reinterpret_cast<HWND>(windowId);
    _native->taskbarButtonReady = true;
    applyProgress();
#else
    Q_UNUSED(windowId);
#endif
}

} // namespace xjw::gui::platform
