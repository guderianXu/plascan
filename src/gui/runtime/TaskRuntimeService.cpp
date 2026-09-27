#include "TaskRuntimeService.h"

#include "GuiTaskRunner.h"
#include "project/ProjectPackageLayout.h"
#include "TaskJournal.h"

#include <QDir>
#include <QFileInfo>
#include <QMetaObject>
#include <QPointer>
#include <QThread>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <optional>
#include <utility>

namespace xjw::gui::runtime
{

    class TaskRuntimeService::ProjectEpochGuard final : public xjw::task_runtime::IProjectEpochGuard
    {
    public:
        void update(std::string projectKey, std::string chunkId, std::uint64_t generation)
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _projectKey = std::move(projectKey);
            _chunkId = std::move(chunkId);
            _generation = generation;
        }

        bool isCurrent(const xjw::task_runtime::TaskDefinition& definition) const override
        {
            if (definition.projectKey.empty())
            {
                return true;
            }
            std::lock_guard<std::mutex> lock(_mutex);
            return definition.projectKey == _projectKey && definition.chunkId == _chunkId &&
                   definition.projectGeneration == _generation;
        }

    private:
        mutable std::mutex _mutex;
        std::string _projectKey;
        std::string _chunkId;
        std::uint64_t _generation = 0;
    };

    namespace
    {

        QString safeJournalName(const QString& chunkId)
        {
            QString safe = chunkId.trimmed();
            for (QChar& character : safe)
            {
                if (!character.isLetterOrNumber() && character != QLatin1Char('-') && character != QLatin1Char('_'))
                {
                    character = QLatin1Char('_');
                }
            }
            return safe.isEmpty() ? QStringLiteral("default") : safe;
        }

        qint64 elapsedMilliseconds(const xjw::task_runtime::TaskRunSnapshot& snapshot)
        {
            const auto start = snapshot.startedAt.value_or(snapshot.submittedAt);
            const auto end = snapshot.finishedAt.value_or(std::chrono::system_clock::now());
            return std::max<qint64>(0, std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        }

        QString stateText(xjw::task_runtime::TaskState state, const QString& blockedReason)
        {
            using xjw::task_runtime::TaskState;
            switch (state)
            {
            case TaskState::Queued:
                return QObject::tr("等待调度");
            case TaskState::Blocked:
                return blockedReason.isEmpty() ? QObject::tr("等待依赖或资源")
                                               : QObject::tr("阻塞：%1").arg(blockedReason);
            case TaskState::Running:
                return QObject::tr("运行中");
            case TaskState::PauseRequested:
                return QObject::tr("等待安全点暂停");
            case TaskState::Paused:
                return QObject::tr("已暂停");
            case TaskState::CancelRequested:
                return QObject::tr("正在取消");
            case TaskState::Succeeded:
                return QObject::tr("已完成");
            case TaskState::Failed:
                return QObject::tr("失败");
            case TaskState::Cancelled:
                return QObject::tr("已取消");
            case TaskState::Interrupted:
                return QObject::tr("异常中断，可从检查点恢复");
            }
            return QObject::tr("未知状态");
        }

    } // namespace

    TaskRuntimeService::TaskRuntimeService(QObject* parent) : QObject(parent)
    {
        setObjectName(QStringLiteral("TaskRuntimeService"));
        createScheduler();
    }

    TaskRuntimeService::~TaskRuntimeService()
    {
        if (_scheduler)
        {
            _scheduler->unsubscribe(_subscriptionId);
            if (_persistenceFuture.isRunning())
            {
                _persistenceFuture.waitForFinished();
            }
            persistNow();
            _scheduler->shutdown();
        }
    }

    xjw::task_runtime::TaskSubmitResult TaskRuntimeService::submit(xjw::task_runtime::TaskDefinition definition)
    {
        if (!_scheduler)
        {
            return {false, _shutdownRequested ? "runtime_shutdown" : "runtime_transitioning", {}};
        }
        return _scheduler->submit(std::move(definition));
    }

    xjw::task_runtime::TaskSubmitResult
    TaskRuntimeService::submitBatch(std::vector<xjw::task_runtime::TaskDefinition> definitions)
    {
        if (!_scheduler)
        {
            return {false, _shutdownRequested ? "runtime_shutdown" : "runtime_transitioning", {}};
        }
        return _scheduler->submitBatch(std::move(definitions));
    }

    void TaskRuntimeService::registerExecutor(const std::string& kind,
                                              std::shared_ptr<xjw::task_runtime::ITaskExecutor> executor)
    {
        if (kind.empty() || !executor)
        {
            return;
        }
        _executors[kind] = executor;
        if (_scheduler)
        {
            _scheduler->registerExecutor(kind, std::move(executor));
        }
    }

    QJsonArray TaskRuntimeService::taskSnapshots() const
    {
        QJsonArray result;
        if (!_scheduler)
        {
            return result;
        }
        for (const xjw::task_runtime::TaskRunSnapshot& snapshot : _scheduler->snapshots())
        {
            result.append(snapshotToJson(snapshot));
        }
        return result;
    }

    QJsonObject TaskRuntimeService::command(const QString& action,
                                            const QString& runId,
                                            const QString& referenceRunId,
                                            int priority,
                                            qulonglong expectedRevision)
    {
        using xjw::task_runtime::TaskCommandResult;
        if (!_scheduler)
        {
            return {
                {QStringLiteral("accepted"), false},
                {QStringLiteral("error"),
                 _shutdownRequested ? QStringLiteral("runtime_shutdown") : QStringLiteral("runtime_transitioning")}};
        }
        const std::optional<std::uint64_t> revision =
            expectedRevision > 0 ? std::optional<std::uint64_t>(expectedRevision) : std::nullopt;
        const std::string run_id = runId.toStdString();
        TaskCommandResult result;
        if (action == QStringLiteral("pause"))
        {
            result = _scheduler->requestPause(run_id, revision);
        }
        else if (action == QStringLiteral("resume"))
        {
            result = _scheduler->resume(run_id, revision);
        }
        else if (action == QStringLiteral("cancel"))
        {
            result = _scheduler->requestCancel(run_id, revision);
        }
        else if (action == QStringLiteral("set_priority"))
        {
            result = _scheduler->setPriority(run_id, priority, revision);
        }
        else if (action == QStringLiteral("move_before"))
        {
            result = _scheduler->moveBefore(run_id, referenceRunId.toStdString(), revision);
        }
        else if (action == QStringLiteral("move_after"))
        {
            result = _scheduler->moveAfter(run_id, referenceRunId.toStdString(), revision);
        }
        else
        {
            return {{QStringLiteral("accepted"), false},
                    {QStringLiteral("error"), QStringLiteral("unknown_task_command")}};
        }

        QJsonObject response{{QStringLiteral("accepted"), result.accepted},
                             {QStringLiteral("error"), QString::fromStdString(result.error)}};
        if (result.snapshot)
        {
            response.insert(QStringLiteral("task"), snapshotToJson(*result.snapshot));
        }
        return response;
    }

    QString TaskRuntimeService::journalPath() const
    {
        return _journalPath;
    }

    bool TaskRuntimeService::isSessionTransitionInProgress() const noexcept
    {
        return _transitionInProgress;
    }

    bool TaskRuntimeService::isShutdownComplete() const noexcept
    {
        return _shutdownComplete;
    }

    void
    TaskRuntimeService::setProjectSession(const QString& projectPath, const QString& chunkId, qulonglong generation)
    {
        if (_shutdownRequested)
        {
            return;
        }

        const QString normalized_path =
            projectPath.trimmed().isEmpty() ? QString() : QFileInfo(projectPath).absoluteFilePath();
        SessionRequest request;
        request.projectPath = normalized_path;
        request.chunkId = chunkId;
        request.generation = generation;
        request.journalPath =
            normalized_path.isEmpty()
                ? QString()
                : QDir(xjw::common::project::ProjectPackageLayout::dataDirectory(normalized_path))
                      .filePath(QStringLiteral("task_runtime/%1.journal").arg(safeJournalName(chunkId)));

        if (!_transitionInProgress && _scheduler && sameSession(_activeSession, request))
        {
            return;
        }

        if (_transitionInProgress)
        {
            _pendingSession = std::move(request);
            _hasPendingSession = true;
            return;
        }

        startSessionTransition(std::move(request));
    }

    void TaskRuntimeService::shutdownAsync()
    {
        if (_shutdownComplete || _shutdownRequested)
        {
            return;
        }

        _shutdownRequested = true;
        _hasPendingSession = false;
        if (!_transitionInProgress)
        {
            startSessionTransition({});
        }
    }

    void TaskRuntimeService::clearHistory()
    {
        if (!_scheduler)
        {
            return;
        }
        _scheduler->clearTerminalRuns();
        scheduleRefresh();
    }

    void TaskRuntimeService::createScheduler()
    {
        xjw::task_runtime::TaskSchedulerLimits limits;
        limits.cpuSlots = std::clamp(QThread::idealThreadCount(), 1, 4);
        _epochGuard = std::make_shared<ProjectEpochGuard>();
        _scheduler = std::make_unique<xjw::task_runtime::TaskScheduler>(std::move(limits));
        _scheduler->setProjectEpochGuard(_epochGuard);
        for (const auto& [kind, executor] : _executors)
        {
            _scheduler->registerExecutor(kind, executor);
        }
        const QPointer<TaskRuntimeService> guarded_this(this);
        _subscriptionId = _scheduler->subscribe(
            [guarded_this](const xjw::task_runtime::TaskEvent&)
            {
                if (!guarded_this)
                {
                    return;
                }
                QMetaObject::invokeMethod(
                    guarded_this.data(),
                    [guarded_this]
                    {
                        if (guarded_this)
                        {
                            guarded_this->scheduleRefresh();
                        }
                    },
                    Qt::QueuedConnection);
            });
    }

    void TaskRuntimeService::startSessionTransition(SessionRequest request)
    {
        const bool starting_transition = !_transitionInProgress;
        _transitionInProgress = true;
        _transitionRequest = request;
        if (starting_transition)
        {
            emit sessionTransitionStarted();
        }

        TransitionWork work;
        work.request = request;
        work.retiringScheduler = std::move(_scheduler);
        work.retiringJournal = _journalPath;
        _pendingPersistence.reset();
        _journalPath.clear();
        if (work.retiringScheduler)
        {
            work.retiringScheduler->unsubscribe(_subscriptionId);
            work.retiringSnapshots = work.retiringScheduler->snapshots();
            if (_epochGuard)
            {
                _epochGuard->update(
                    request.projectPath.toStdString(), request.chunkId.toStdString(), request.generation);
            }
            work.retiringScheduler->requestShutdown();
        }
        _epochGuard.reset();
        emit taskSnapshotsChanged(QJsonArray{});

        _transitionWork = std::move(work);
        if (!_persistenceInProgress)
        {
            launchSessionTransition();
        }
    }

    void TaskRuntimeService::launchSessionTransition()
    {
        if (!_transitionWork)
        {
            return;
        }

        TransitionWork work = std::move(*_transitionWork);
        _transitionWork.reset();

        _transitionFuture = xjw::gui::tasks::runGuardedWithOutcome(
            this,
            [work = std::move(work)]() mutable
            {
                TransitionResult result;
                result.request = work.request;
                if (!work.retiringJournal.isEmpty())
                {
                    std::string error;
                    if (!xjw::task_runtime::TaskJournal::save(
                            std::filesystem::path(work.retiringJournal.toStdString()), work.retiringSnapshots, &error))
                    {
                        result.saveError = QString::fromStdString(error);
                    }
                }
                if (work.retiringScheduler)
                {
                    work.retiringScheduler->shutdown();
                    work.retiringScheduler.reset();
                }
                if (!work.request.journalPath.isEmpty() && QFileInfo::exists(work.request.journalPath))
                {
                    xjw::task_runtime::TaskJournalLoadResult loaded = xjw::task_runtime::TaskJournal::load(
                        std::filesystem::path(work.request.journalPath.toStdString()));
                    if (loaded.succeeded)
                    {
                        result.loadedSnapshots = std::move(loaded.snapshots);
                    }
                    else
                    {
                        result.loadError = QString::fromStdString(loaded.error);
                    }
                }
                return result;
            },
            [](TaskRuntimeService* self, xjw::gui::tasks::TaskOutcome<TransitionResult> outcome)
            {
                if (outcome.succeeded())
                {
                    self->handleTransitionResult(std::move(*outcome.value), {});
                    return;
                }
                TransitionResult result;
                result.request = self->_transitionRequest;
                self->handleTransitionResult(std::move(result), outcome.errorMessage);
            });
    }

    void TaskRuntimeService::handleTransitionResult(TransitionResult result, const QString& workerError)
    {
        bool succeeded = workerError.isEmpty() && result.saveError.isEmpty() && result.loadError.isEmpty();
        if (!workerError.isEmpty())
        {
            emit journalError(tr("任务运行时切换失败：%1").arg(workerError));
        }
        if (!result.saveError.isEmpty())
        {
            emit journalError(tr("无法保存任务恢复记录：%1").arg(result.saveError));
        }
        if (!result.loadError.isEmpty())
        {
            emit journalError(tr("无法读取任务恢复记录：%1").arg(result.loadError));
        }

        if (_shutdownRequested)
        {
            finishShutdown();
            return;
        }

        SessionRequest effective_request = result.request;
        if (_hasPendingSession)
        {
            SessionRequest pending = std::move(_pendingSession);
            _hasPendingSession = false;
            if (pending.journalPath != result.request.journalPath)
            {
                startSessionTransition(std::move(pending));
                return;
            }
            effective_request = std::move(pending);
        }

        _journalPath = effective_request.journalPath;
        _activeSession = effective_request;
        createScheduler();
        _epochGuard->update(effective_request.projectPath.toStdString(),
                            effective_request.chunkId.toStdString(),
                            effective_request.generation);
        if (workerError.isEmpty() && result.loadError.isEmpty() && !result.loadedSnapshots.empty())
        {
            const xjw::task_runtime::TaskSubmitResult restored = _scheduler->restore(std::move(result.loadedSnapshots));
            if (!restored.accepted)
            {
                succeeded = false;
                emit journalError(tr("无法恢复任务队列：%1").arg(QString::fromStdString(restored.error)));
            }
        }

        _transitionInProgress = false;
        _transitionRequest = {};
        scheduleRefresh();
        emit sessionTransitionFinished(succeeded);
    }

    void TaskRuntimeService::finishShutdown()
    {
        _scheduler.reset();
        _epochGuard.reset();
        _journalPath.clear();
        _activeSession = {};
        _transitionRequest = {};
        _transitionWork.reset();
        _pendingPersistence.reset();
        _transitionInProgress = false;
        _shutdownComplete = true;
        emit taskSnapshotsChanged(QJsonArray{});
        emit sessionTransitionFinished(true);
        emit shutdownFinished();
    }

    void TaskRuntimeService::scheduleRefresh()
    {
        if (_refreshScheduled)
        {
            return;
        }
        _refreshScheduled = true;
        QMetaObject::invokeMethod(
            this,
            [this]
            {
                _refreshScheduled = false;
                refreshAndPersist();
            },
            Qt::QueuedConnection);
    }

    void TaskRuntimeService::refreshAndPersist()
    {
        if (!_scheduler)
        {
            emit taskSnapshotsChanged(QJsonArray{});
            return;
        }

        std::vector<xjw::task_runtime::TaskRunSnapshot> snapshots = _scheduler->snapshots();
        QJsonArray serialized;
        for (const xjw::task_runtime::TaskRunSnapshot& snapshot : snapshots)
        {
            serialized.append(snapshotToJson(snapshot));
        }
        emit taskSnapshotsChanged(serialized);

        if (_journalPath.isEmpty())
        {
            return;
        }
        PersistRequest request{_journalPath, std::move(snapshots)};
        if (_persistenceInProgress)
        {
            _pendingPersistence = std::move(request);
            return;
        }
        startPersistence(std::move(request));
    }

    void TaskRuntimeService::startPersistence(PersistRequest request)
    {
        _persistenceInProgress = true;
        _persistenceFuture = xjw::gui::tasks::runGuardedWithOutcome(
            this,
            [request = std::move(request)]() mutable
            {
                std::string error;
                if (!xjw::task_runtime::TaskJournal::save(
                        std::filesystem::path(request.journalPath.toStdString()), request.snapshots, &error) &&
                    error.empty())
                {
                    error = "unknown_journal_save_error";
                }
                return QString::fromStdString(error);
            },
            [](TaskRuntimeService* self, xjw::gui::tasks::TaskOutcome<QString> outcome)
            {
                self->_persistenceInProgress = false;
                if (!outcome.succeeded())
                {
                    emit self->journalError(self->tr("无法保存任务恢复记录：%1").arg(outcome.errorMessage));
                }
                else if (!outcome.value->isEmpty())
                {
                    emit self->journalError(self->tr("无法保存任务恢复记录：%1").arg(*outcome.value));
                }

                if (self->_transitionInProgress && self->_transitionWork)
                {
                    self->_pendingPersistence.reset();
                    self->launchSessionTransition();
                    return;
                }
                if (self->_shutdownRequested || self->_transitionInProgress)
                {
                    self->_pendingPersistence.reset();
                    return;
                }
                if (self->_pendingPersistence)
                {
                    PersistRequest pending = std::move(*self->_pendingPersistence);
                    self->_pendingPersistence.reset();
                    self->startPersistence(std::move(pending));
                }
            });
    }

    void TaskRuntimeService::persistNow()
    {
        if (!_scheduler || _journalPath.isEmpty())
        {
            return;
        }
        std::string error;
        if (!xjw::task_runtime::TaskJournal::save(
                std::filesystem::path(_journalPath.toStdString()), _scheduler->snapshots(), &error))
        {
            emit journalError(tr("无法保存任务恢复记录：%1").arg(QString::fromStdString(error)));
        }
    }

    bool TaskRuntimeService::sameSession(const SessionRequest& first, const SessionRequest& second)
    {
        return first.projectPath == second.projectPath && first.chunkId == second.chunkId &&
               first.generation == second.generation;
    }

    QJsonObject TaskRuntimeService::snapshotToJson(const xjw::task_runtime::TaskRunSnapshot& snapshot)
    {
        const bool terminal = xjw::task_runtime::isTerminalTaskState(snapshot.state);
        const bool resumable = snapshot.state == xjw::task_runtime::TaskState::Paused ||
                               (snapshot.state == xjw::task_runtime::TaskState::Interrupted &&
                                snapshot.definition.capabilities.canCheckpoint && snapshot.checkpoint.has_value());
        const QString blocked_reason = QString::fromStdString(snapshot.blockedReason);
        QJsonObject result{
            {QStringLiteral("scheduler_managed"), true},
            {QStringLiteral("task_id"), QString::fromStdString(snapshot.definition.taskId)},
            {QStringLiteral("run_id"), QString::fromStdString(snapshot.runId)},
            {QStringLiteral("attempt_id"), static_cast<int>(snapshot.attemptId)},
            {QStringLiteral("name"), QString::fromStdString(snapshot.definition.displayName)},
            {QStringLiteral("state"), QString::fromLatin1(xjw::task_runtime::taskStateName(snapshot.state))},
            {QStringLiteral("status_text"), stateText(snapshot.state, blocked_reason)},
            {QStringLiteral("active"), !terminal},
            {QStringLiteral("cancelling"), snapshot.state == xjw::task_runtime::TaskState::CancelRequested},
            {QStringLiteral("blocked_reason"), blocked_reason},
            {QStringLiteral("priority"), snapshot.definition.priority},
            {QStringLiteral("revision"), static_cast<double>(snapshot.revision)},
            {QStringLiteral("queue_sequence"), static_cast<double>(snapshot.queueSequence)},
            {QStringLiteral("progress_value"), static_cast<double>(snapshot.progress.completedUnits)},
            {QStringLiteral("progress_maximum"), static_cast<double>(snapshot.progress.totalUnits)},
            {QStringLiteral("elapsed_ms"), static_cast<double>(elapsedMilliseconds(snapshot))},
            {QStringLiteral("can_pause"), snapshot.definition.capabilities.canPause && !terminal && !resumable},
            {QStringLiteral("can_resume"), resumable},
            {QStringLiteral("can_cancel"), snapshot.definition.capabilities.canCancel && !terminal},
            {QStringLiteral("can_reorder"),
             snapshot.definition.capabilities.canReorder && (snapshot.state == xjw::task_runtime::TaskState::Queued ||
                                                             snapshot.state == xjw::task_runtime::TaskState::Blocked)}};
        if (snapshot.checkpoint)
        {
            result.insert(
                QStringLiteral("checkpoint"),
                QJsonObject{
                    {QStringLiteral("schema_version"), static_cast<int>(snapshot.checkpoint->schemaVersion)},
                    {QStringLiteral("location"), QString::fromStdString(snapshot.checkpoint->location)},
                    {QStringLiteral("input_signature"), QString::fromStdString(snapshot.checkpoint->inputSignature)},
                    {QStringLiteral("completed_units"), static_cast<double>(snapshot.checkpoint->completedUnits)}});
        }
        if (snapshot.error)
        {
            result.insert(QStringLiteral("error"),
                          QJsonObject{{QStringLiteral("code"), QString::fromStdString(snapshot.error->code)},
                                      {QStringLiteral("message"), QString::fromStdString(snapshot.error->message)},
                                      {QStringLiteral("retryable"), snapshot.error->retryable}});
        }
        return result;
    }

} // namespace xjw::gui::runtime
