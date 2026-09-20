#pragma once

#include "ProjectModelTaskLifecycle.h"
#include "project/tasks/ProjectTaskContext.h"

#include <QFuture>
#include <QObject>
#include <QJsonObject>
#include <QPointer>
#include <QVector>

class ProjectTaskOrchestratorPointModelTestPeer;
class ProjectUiMessageAdapter;

namespace xjw::gui::project
{
    class ProjectSession;
    class ProjectTaskOrchestrator;
} // namespace xjw::gui::project

class ProjectModelManager : public QObject
{
    Q_OBJECT

public:
    ~ProjectModelManager() override;

signals:
    void meshProgressChanged(const QString& stage, int percent);
    void meshProgressFinished(bool success);

private:
    friend class xjw::gui::project::ProjectTaskOrchestrator;
    friend class ::ProjectTaskOrchestratorPointModelTestPeer;

    explicit ProjectModelManager(xjw::gui::project::ProjectSession* session,
                                 ProjectUiMessageAdapter* messages,
                                 QObject* parent = nullptr);
    bool startMeshReconstructionAsync(const QJsonObject& settings,
                                      const xjw::gui::project::ProjectTaskContext& taskContext);
    bool startTextureMappingAsync(const QJsonObject& settings,
                                  const xjw::gui::project::ProjectTaskContext& taskContext);
    void cancelActiveTask();
    void waitForActiveTask();
    bool isRunning() const;
    bool hasPendingWork() const noexcept;
    bool acceptsTaskCallback(const xjw::gui::project::ProjectModelTaskPtr& task) const;
    void trackFutureForTesting(QFuture<void> future);
    void pruneFinishedFutures();
    void trackFuture(QFuture<void> future);

    QPointer<xjw::gui::project::ProjectSession> _session;
    ProjectUiMessageAdapter* _messages = nullptr;
    QVector<QFuture<void>> _futures;
    xjw::gui::project::ProjectModelTaskLifecycle _taskLifecycle;
};
