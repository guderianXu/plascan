#pragma once

#include <QCoreApplication>
#include <QFuture>
#include <QFutureWatcher>
#include <QMetaObject>
#include <QPointer>
#include <QString>
#include <QtConcurrent/QtConcurrent>

#include <atomic>
#include <exception>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace xjw::gui::tasks
{

class TaskCancellationToken final
{
public:
    bool isCancellationRequested() const
    {
        return _flag && _flag->load(std::memory_order_relaxed);
    }

    std::shared_ptr<std::atomic<bool>> sharedFlag() const
    {
        return _flag;
    }

private:
    friend class TaskCancellationSource;
    explicit TaskCancellationToken(std::shared_ptr<std::atomic<bool>> flag)
        : _flag(std::move(flag))
    {
    }

    std::shared_ptr<std::atomic<bool>> _flag;
};

class TaskCancellationSource final
{
public:
    TaskCancellationToken reset()
    {
        _flag = std::make_shared<std::atomic<bool>>(false);
        return TaskCancellationToken(_flag);
    }

    void requestCancellation() const
    {
        if (_flag)
        {
            _flag->store(true, std::memory_order_relaxed);
        }
    }

private:
    std::shared_ptr<std::atomic<bool>> _flag;
};

template <typename Result>
struct TaskOutcome
{
    std::optional<Result> value;
    QString errorMessage;

    bool succeeded() const
    {
        return errorMessage.isEmpty() && value.has_value();
    }
};

template <>
struct TaskOutcome<void>
{
    QString errorMessage;

    bool succeeded() const
    {
        return errorMessage.isEmpty();
    }
};

template <typename Owner, typename Callback>
void postGuarded(const QPointer<Owner> &owner, Callback &&callback)
{
    if (!owner || !QCoreApplication::instance())
    {
        return;
    }

    const QPointer<Owner> self = owner;
    auto callbackPtr = std::make_shared<std::decay_t<Callback>>(std::forward<Callback>(callback));
    QMetaObject::invokeMethod(QCoreApplication::instance(),
                              [self, callbackPtr]() mutable
                              {
                                  if (!self)
                                  {
                                      return;
                                  }
                                  (*callbackPtr)(self.data());
                              },
                              Qt::QueuedConnection);
}

template <typename Owner, typename Work, typename Finished>
QFuture<void> runGuardedWithOutcome(Owner *owner, Work &&work, Finished &&finished)
{
    if (!owner)
    {
        return QFuture<void>();
    }

    using WorkFn = std::decay_t<Work>;
    using FinishedFn = std::decay_t<Finished>;
    using Result = std::invoke_result_t<WorkFn &>;

    auto workPtr = std::make_shared<WorkFn>(std::forward<Work>(work));
    auto finishedPtr = std::make_shared<FinishedFn>(std::forward<Finished>(finished));
    auto outcome = std::make_shared<TaskOutcome<Result>>();
    auto *watcher = new QFutureWatcher<void>(owner);

    QObject::connect(watcher,
                     &QFutureWatcher<void>::finished,
                     owner,
                     [owner, watcher, finishedPtr, outcome]() mutable
    {
        watcher->deleteLater();
        (*finishedPtr)(owner, std::move(*outcome));
    });

    QFuture<void> future = QtConcurrent::run([workPtr, outcome]() mutable
    {
        try
        {
            if constexpr (std::is_void_v<Result>)
            {
                (*workPtr)();
            }
            else
            {
                outcome->value.emplace((*workPtr)());
            }
        }
        catch (const std::exception &exception)
        {
            outcome->errorMessage = QString::fromUtf8(exception.what());
        }
        catch (...)
        {
            outcome->errorMessage = QStringLiteral("后台任务发生未知异常。");
        }
    });
    watcher->setFuture(future);
    return future;
}

} // namespace xjw::gui::tasks
