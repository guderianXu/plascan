#pragma once

#include "TaskScheduler.h"

#include <QFuture>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QString>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace xjw::gui::runtime
{

    class TaskRuntimeService final : public QObject
    {
        Q_OBJECT

    public:
        explicit TaskRuntimeService(QObject* parent = nullptr);
        ~TaskRuntimeService() override;

        xjw::task_runtime::TaskSubmitResult submit(xjw::task_runtime::TaskDefinition definition);
        xjw::task_runtime::TaskSubmitResult submitBatch(std::vector<xjw::task_runtime::TaskDefinition> definitions);
        void registerExecutor(const std::string& kind, std::shared_ptr<xjw::task_runtime::ITaskExecutor> executor);

        QJsonArray taskSnapshots() const;
        QJsonObject command(const QString& action,
                            const QString& runId,
                            const QString& referenceRunId = {},
                            int priority = 0,
                            qulonglong expectedRevision = 0);
        QString journalPath() const;
        bool isSessionTransitionInProgress() const noexcept;
        bool isShutdownComplete() const noexcept;

    public slots:
        void setProjectSession(const QString& projectPath, const QString& chunkId, qulonglong generation);
        void shutdownAsync();
        void clearHistory();

    signals:
        void taskSnapshotsChanged(const QJsonArray& snapshots);
        void journalError(const QString& message);
        void sessionTransitionStarted();
        void sessionTransitionFinished(bool success);
        void shutdownFinished();

    private:
        class ProjectEpochGuard;
        struct SessionRequest
        {
            QString projectPath;
            QString chunkId;
            QString journalPath;
            qulonglong generation = 0;
        };

        struct TransitionResult
        {
            SessionRequest request;
            std::vector<xjw::task_runtime::TaskRunSnapshot> loadedSnapshots;
            QString saveError;
            QString loadError;
        };

        struct PersistRequest
        {
            QString journalPath;
            std::vector<xjw::task_runtime::TaskRunSnapshot> snapshots;
        };

        struct TransitionWork
        {
            SessionRequest request;
            QString retiringJournal;
            std::vector<xjw::task_runtime::TaskRunSnapshot> retiringSnapshots;
            std::unique_ptr<xjw::task_runtime::TaskScheduler> retiringScheduler;
        };

        void createScheduler();
        void startSessionTransition(SessionRequest request);
        void launchSessionTransition();
        void handleTransitionResult(TransitionResult result, const QString& workerError);
        void finishShutdown();
        void scheduleRefresh();
        void refreshAndPersist();
        void startPersistence(PersistRequest request);
        void persistNow();
        static bool sameSession(const SessionRequest& first, const SessionRequest& second);
        static QJsonObject snapshotToJson(const xjw::task_runtime::TaskRunSnapshot& snapshot);

        std::unique_ptr<xjw::task_runtime::TaskScheduler> _scheduler;
        std::shared_ptr<ProjectEpochGuard> _epochGuard;
        std::unordered_map<std::string, std::shared_ptr<xjw::task_runtime::ITaskExecutor>> _executors;
        std::uint64_t _subscriptionId = 0;
        QString _journalPath;
        SessionRequest _activeSession;
        SessionRequest _transitionRequest;
        SessionRequest _pendingSession;
        QFuture<void> _transitionFuture;
        QFuture<void> _persistenceFuture;
        std::optional<TransitionWork> _transitionWork;
        std::optional<PersistRequest> _pendingPersistence;
        bool _hasPendingSession = false;
        bool _transitionInProgress = false;
        bool _persistenceInProgress = false;
        bool _shutdownRequested = false;
        bool _shutdownComplete = false;
        bool _refreshScheduled = false;
    };

} // namespace xjw::gui::runtime
