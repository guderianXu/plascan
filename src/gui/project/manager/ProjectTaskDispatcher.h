#pragma once

#include <QObject>
#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include "ProjectReconstructionManager.h"
#include "ProjectTerrainRequests.h"

class ProjectCameraSetupManager;
class ProjectTerrainProductsManager;

class ProjectTaskDispatcher : public QObject
{
    Q_OBJECT

public:
    explicit ProjectTaskDispatcher(ProjectCameraSetupManager *cameraManager,
                                   ProjectTerrainProductsManager *terrainManager,
                                   ProjectReconstructionManager *reconstructionManager,
                                   QObject *parent = nullptr);

    bool importCameraForImage(const QString &imagePath) const;
    bool importCamerasByFilenameBatch() const;
    bool initializeCamerasFromExifOrDefault(const QJsonObject &settings) const;
    bool initializeCamerasFromIntrinsics(const QJsonObject &settings) const;
    bool initializeCameraPosesWithSFM(const QJsonObject &settings) const;

    void startDemFromPointCloudAsync(
        const xjw::gui::project::DemGenerationRequest &request) const;
    void startMapProjectAsync(const QJsonObject &settings) const;
    void cancelMapProject() const;

    void startReconstructionTask(ProjectReconstructionManager::Task task,
                                 const QJsonObject &settings = QJsonObject()) const;

    QJsonArray getAvailableAtResults() const;
    void cancelModelGeneration() const;

private:
    ProjectCameraSetupManager *_cameraManager = nullptr;
    ProjectTerrainProductsManager *_terrainManager = nullptr;
    ProjectReconstructionManager *_reconstructionManager = nullptr;
};
