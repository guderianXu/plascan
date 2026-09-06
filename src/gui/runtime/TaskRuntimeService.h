#pragma once

#include "TaskScheduler.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QString>

#include <memory>
#include <unordered_map>

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

    public slots:
        void setProjectSession(const QString& projectPath, const QString& chunkId, qulonglong generation);
        void clearHistory();

    signals:
        void taskSnapshotsChanged(const QJsonArray& snapshots);
        void journalError(const QString& message);

    private:
        class ProjectEpochGuard;

        void createScheduler();
        void scheduleRefresh();
        void refreshAndPersist();
        void persistNow();
        static QJsonObject snapshotToJson(const xjw::task_runtime::TaskRunSnapshot& snapshot);

        std::unique_ptr<xjw::task_runtime::TaskScheduler> _scheduler;
        std::shared_ptr<ProjectEpochGuard> _epochGuard;
        std::unordered_map<std::string, std::shared_ptr<xjw::task_runtime::ITaskExecutor>> _executors;
        std::uint64_t _subscriptionId = 0;
        QString _journalPath;
        bool _refreshScheduled = false;
    };

} // namespace xjw::gui::runtime
