#pragma once

#include <QFuture>
#include <QFutureWatcher>
#include <QDebug>
#include <QMetaObject>
#include <QPointer>
#include <QString>
#include <QtConcurrent/QtConcurrent>

#include <exception>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace xjw::gui::tasks
{

class TaskFutureRetainer final : public QObject
{
public:
    explicit TaskFutureRetainer(QObject *parent)
        : QObject(parent)
    {
        connect(&_watcher, &QFutureWatcher<void>::finished, this, [this]()
        {
            _future = QFuture<void>();
            deleteLater();
        });
    }

    void retain(QFuture<void> future)
    {
        _future = std::move(future);
        _watcher.setFuture(_future);
    }

private:
    QFuture<void> _future;
    QFutureWatcher<void> _watcher;
};

inline void retainTaskFuture(QObject *owner, const QFuture<void> &future)
{
    if (!owner || !future.isValid())
    {
        return;
    }

    auto *retainer = new TaskFutureRetainer(owner);
    retainer->retain(future);
}

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
void postGuarded(Owner *owner, Callback &&callback)
{
    QPointer<Owner> self(owner);
    if (!self)
    {
        return;
    }

    auto callbackPtr = std::make_shared<std::decay_t<Callback>>(std::forward<Callback>(callback));
    QMetaObject::invokeMethod(self.data(),
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
    QPointer<Owner> self(owner);
    if (!self)
    {
        return QFuture<void>();
    }

    using WorkFn = std::decay_t<Work>;
    using FinishedFn = std::decay_t<Finished>;
    using Result = std::invoke_result_t<WorkFn &>;

    auto workPtr = std::make_shared<WorkFn>(std::forward<Work>(work));
    auto finishedPtr = std::make_shared<FinishedFn>(std::forward<Finished>(finished));

    QFuture<void> future = QtConcurrent::run([self, workPtr, finishedPtr]() mutable
    {
        auto outcome = std::make_shared<TaskOutcome<Result>>();
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

        if (!self)
        {
            return;
        }

        postGuarded(self.data(),
                    [finishedPtr, outcome](Owner *callbackOwner) mutable
                    {
                        (*finishedPtr)(callbackOwner, std::move(*outcome));
                    });
    });
    retainTaskFuture(self.data(), future);
    return future;
}

// 兼容旧回调签名。新代码应使用 runGuardedWithOutcome，以便将异常反馈到具体任务 UI。
template <typename Owner, typename Work, typename Finished>
QFuture<void> runGuarded(Owner *owner, Work &&work, Finished &&finished)
{
    using WorkFn = std::decay_t<Work>;
    using FinishedFn = std::decay_t<Finished>;
    using Result = std::invoke_result_t<WorkFn &>;
    auto finishedPtr = std::make_shared<FinishedFn>(std::forward<Finished>(finished));

    return runGuardedWithOutcome(
        owner,
        std::forward<Work>(work),
        [finishedPtr](Owner *callbackOwner, TaskOutcome<Result> outcome) mutable
        {
            if (!outcome.succeeded())
            {
                callbackOwner->setProperty("lastAsyncTaskError", outcome.errorMessage);
                qWarning().noquote() << QStringLiteral("后台任务失败: %1").arg(outcome.errorMessage);
                if constexpr (std::is_void_v<Result>)
                {
                    (*finishedPtr)(callbackOwner);
                }
                else if constexpr (std::is_default_constructible_v<Result>)
                {
                    (*finishedPtr)(callbackOwner, Result{});
                }
                return;
            }

            if constexpr (std::is_void_v<Result>)
            {
                (*finishedPtr)(callbackOwner);
            }
            else
            {
                (*finishedPtr)(callbackOwner, std::move(*outcome.value));
            }
        });
}

} // namespace xjw::gui::tasks
