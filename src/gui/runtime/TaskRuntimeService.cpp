#include "TaskRuntimeService.h"

#include "project/ProjectPackageLayout.h"
#include "TaskJournal.h"

#include <QDir>
#include <QFileInfo>
#include <QMetaObject>
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
            persistNow();
            _scheduler->shutdown();
        }
    }

    xjw::task_runtime::TaskSubmitResult TaskRuntimeService::submit(xjw::task_runtime::TaskDefinition definition)
    {
        return _scheduler->submit(std::move(definition));
    }

    xjw::task_runtime::TaskSubmitResult
    TaskRuntimeService::submitBatch(std::vector<xjw::task_runtime::TaskDefinition> definitions)
    {
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
        _scheduler->registerExecutor(kind, std::move(executor));
    }

    QJsonArray TaskRuntimeService::taskSnapshots() const
    {
        QJsonArray result;
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

    void
    TaskRuntimeService::setProjectSession(const QString& projectPath, const QString& chunkId, qulonglong generation)
    {
        const QString normalized_path =
            projectPath.trimmed().isEmpty() ? QString() : QFileInfo(projectPath).absoluteFilePath();
        const QString new_journal =
            normalized_path.isEmpty()
                ? QString()
                : QDir(xjw::common::project::ProjectPackageLayout::dataDirectory(normalized_path))
                      .filePath(QStringLiteral("task_runtime/%1.journal").arg(safeJournalName(chunkId)));
        if (_journalPath == new_journal)
        {
            _epochGuard->update(normalized_path.toStdString(), chunkId.toStdString(), generation);
            return;
        }

        if (_scheduler)
        {
            _scheduler->unsubscribe(_subscriptionId);
            persistNow();
            _scheduler->shutdown();
        }
        _journalPath = new_journal;
        createScheduler();
        _epochGuard->update(normalized_path.toStdString(), chunkId.toStdString(), generation);

        if (!_journalPath.isEmpty() && QFileInfo::exists(_journalPath))
        {
            const xjw::task_runtime::TaskJournalLoadResult loaded =
                xjw::task_runtime::TaskJournal::load(std::filesystem::path(_journalPath.toStdString()));
            if (!loaded.succeeded)
            {
                emit journalError(tr("无法读取任务恢复记录：%1").arg(QString::fromStdString(loaded.error)));
            }
            else
            {
                const xjw::task_runtime::TaskSubmitResult restored = _scheduler->restore(loaded.snapshots);
                if (!restored.accepted)
                {
                    emit journalError(tr("无法恢复任务队列：%1").arg(QString::fromStdString(restored.error)));
                }
            }
        }
        scheduleRefresh();
    }

    void TaskRuntimeService::clearHistory()
    {
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
        _subscriptionId = _scheduler->subscribe(
            [this](const xjw::task_runtime::TaskEvent&)
            { QMetaObject::invokeMethod(this, [this] { scheduleRefresh(); }, Qt::QueuedConnection); });
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
        emit taskSnapshotsChanged(taskSnapshots());
        persistNow();
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
