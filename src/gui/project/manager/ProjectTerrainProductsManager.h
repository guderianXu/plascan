#pragma once

#include "ProjectTerrainRequests.h"
#include "project/tasks/ProjectTaskContext.h"

#include <QFuture>
#include <QObject>
#include <QJsonObject>
#include <QPointer>
#include <QStringList>
#include <QVector>

class ProjectTaskOrchestratorTerrainTestPeer;
class ProjectUiMessageAdapter;

namespace xjw::gui::project
{
    class ProjectSession;
    class ProjectTaskOrchestrator;
} // namespace xjw::gui::project

class ProjectTerrainProductsManager : public QObject
{
    Q_OBJECT

public:
    ~ProjectTerrainProductsManager() override;

signals:
    // DEM/正射实例可能并发或被拒绝启动，使用唯一 ID 保持全局进度互不串线。
    void backgroundTaskProgressChanged(const QString& taskId, int value, int maximum);
    void backgroundTaskFinished(const QString& taskId);
    void demPipelineProgressChanged(const QString& stage, int percent);
    void demPipelineFinished(bool success, const QString& message);
    void orthoPipelineStarted();
    void orthoPipelineProgressChanged(const QString& stage, int percent);
    void orthoPipelineFinished(bool success, const QString& message, const QJsonObject& result);

private:
    friend class xjw::gui::project::ProjectTaskOrchestrator;
    friend class ::ProjectTaskOrchestratorTerrainTestPeer;

    explicit ProjectTerrainProductsManager(xjw::gui::project::ProjectSession* session,
                                           ProjectUiMessageAdapter* messages,
                                           QObject* parent = nullptr);
    void startDemFromPointCloudAsync(const xjw::gui::project::DemGenerationRequest& request,
                                     const xjw::gui::project::ProjectTaskContext& taskContext);
    void cancelDemGeneration(const xjw::gui::project::ProjectTaskContext& taskContext);
    void startMapProjectAsync(const xjw::gui::project::OrthoGenerationRequest& request,
                              const xjw::gui::project::ProjectTaskContext& taskContext);
    void cancelMapProject(const xjw::gui::project::ProjectTaskContext& taskContext);
    void waitForActiveTask();
    bool hasPendingWork() const noexcept;
    void trackFutureForTesting(QFuture<void> future);

    void startRpcStereoDemAsync(const xjw::gui::project::DemGenerationRequest& request,
                                const xjw::gui::project::ProjectTaskContext& taskContext);
    void startRpcDomAsync(const xjw::gui::project::OrthoGenerationRequest& request,
                          const xjw::gui::project::ProjectTaskContext& taskContext);
    void startSmallBodyGlobalAsync(const xjw::gui::project::DemGenerationRequest& request,
                                   const xjw::gui::project::ProjectTaskContext& taskContext);
    bool demContextMatches(const xjw::gui::project::ProjectTaskContext& taskContext, bool requireCurrent = true) const;
    bool orthoContextMatches(const xjw::gui::project::ProjectTaskContext& taskContext,
                             bool requireCurrent = true) const;
    void clearDemContextIfMatches(const xjw::gui::project::ProjectTaskContext& taskContext);
    void clearOrthoContextIfMatches(const xjw::gui::project::ProjectTaskContext& taskContext);
    void pruneFinishedFutures();
    void trackFuture(QFuture<void> future);

    QPointer<xjw::gui::project::ProjectSession> _session;
    ProjectUiMessageAdapter* _messages = nullptr;
    QVector<QFuture<void>> _futures;
    xjw::gui::project::ProjectTaskContext _demContext;
    xjw::gui::project::ProjectTaskContext _orthoContext;
};
