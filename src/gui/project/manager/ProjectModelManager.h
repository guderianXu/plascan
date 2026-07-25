#pragma once

#include <QObject>
#include <QJsonObject>

#include <atomic>
#include <memory>

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

    void startGenerateModelAsync();
    bool startMeshReconstructionAsync(const QJsonObject &settings);
    void startTextureMappingAsync(const QJsonObject &settings);
    void cancelActiveTask();
    bool isRunning() const;

signals:
    void meshProgressChanged(const QString &stage, int percent);
    void meshProgressFinished(bool success);

private:
    bool ensureProjectOpen(const QString &message,
                           const QString &title) const;
    void finalizeModelGenerationSuccess(const QJsonObject &terrainResult,
                                        const QString &sourceCloudPath,
                                        bool sourceIsDense);

    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    bool _isRunning = false;
    std::shared_ptr<std::atomic_bool> _activeCancelFlag;
};
