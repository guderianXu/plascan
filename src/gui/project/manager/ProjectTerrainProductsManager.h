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

    void startMapProjectAsync(
        const xjw::gui::project::OrthoGenerationRequest &request);
    void cancelMapProject();

signals:
    void demPipelineProgressChanged(const QString &stage, int percent);
    void demPipelineFinished(bool success, const QString &message);
    void orthoPipelineStarted();
    void orthoPipelineProgressChanged(const QString &stage, int percent);
    void orthoPipelineFinished(bool success,
                               const QString &message,
                               const QJsonObject &result);

private:
    bool ensureProjectOpen(const QString &message = QStringLiteral("请先打开项目"),
                           const QString &title = QStringLiteral("提示")) const;

    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    std::shared_ptr<std::atomic_bool> _orthoCancelFlag;
    QString _orthoTaskChunkId;
};
