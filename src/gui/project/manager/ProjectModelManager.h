#pragma once

#include "ProjectModelTaskLifecycle.h"

#include <QObject>
#include <QJsonObject>

class QWidget;
class ProjectData;
class ProjectManager;

class ProjectModelManager : public QObject
{
    Q_OBJECT

public:
    explicit ProjectModelManager(ProjectManager *owner,
                                 ProjectData *projectData,
                                 QWidget *parentWidget,
                                 QObject *parent = nullptr);
    ~ProjectModelManager() override;

    bool startMeshReconstructionAsync(const QJsonObject &settings);
    void startTextureMappingAsync(const QJsonObject &settings);
    void cancelActiveTask();
    bool isRunning() const;
    bool acceptsTaskCallback(
        const xjw::gui::project::ProjectModelTaskPtr &task) const;

signals:
    void meshProgressChanged(const QString &stage, int percent);
    void meshProgressFinished(bool success);

private:
    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    xjw::gui::project::ProjectModelTaskLifecycle _taskLifecycle;
};
