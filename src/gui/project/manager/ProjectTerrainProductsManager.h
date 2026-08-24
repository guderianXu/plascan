#pragma once

#include "ProjectTerrainRequests.h"

#include <QObject>
#include <QJsonObject>
#include <QStringList>

#include <atomic>
#include <memory>

class QWidget;
class ProjectData;
class ProjectManager;

class ProjectTerrainProductsManager : public QObject
{
    Q_OBJECT

public:
    explicit ProjectTerrainProductsManager(ProjectManager *owner,
                                           ProjectData *projectData,
                                           QWidget *parentWidget,
                                           QObject *parent = nullptr);

    void startDemFromPointCloudAsync(
        const xjw::gui::project::DemGenerationRequest &request);
    void cancelDemGeneration();

    void startMapProjectAsync(
        const xjw::gui::project::OrthoGenerationRequest &request);
    void cancelMapProject();

signals:
    // DEM/正射实例可能并发或被拒绝启动，使用唯一 ID 保持全局进度互不串线。
    void backgroundTaskProgressChanged(const QString &taskId, int value, int maximum);
    void backgroundTaskFinished(const QString &taskId);
    void demPipelineProgressChanged(const QString &stage, int percent);
    void demPipelineFinished(bool success, const QString &message);
    void orthoPipelineStarted();
    void orthoPipelineProgressChanged(const QString &stage, int percent);
    void orthoPipelineFinished(bool success,
                               const QString &message,
                               const QJsonObject &result);

private:
    void startRpcStereoDemAsync(
        const xjw::gui::project::DemGenerationRequest &request);
    void startRpcDomAsync(
        const xjw::gui::project::OrthoGenerationRequest &request);
    void startSmallBodyGlobalAsync(
        const xjw::gui::project::DemGenerationRequest &request);

    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    std::shared_ptr<std::atomic_bool> _demCancelFlag;
    QString _demTaskId;
    QString _demTaskChunkId;
    std::shared_ptr<std::atomic_bool> _orthoCancelFlag;
    QString _orthoTaskChunkId;
};
