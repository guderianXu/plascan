#include "ProjectTaskDispatcher.h"

#include "ProjectCameraSetupManager.h"
#include "ProjectReconstructionManager.h"
#include "ProjectTerrainProductsManager.h"

ProjectTaskDispatcher::ProjectTaskDispatcher(ProjectCameraSetupManager *cameraManager,
                                             ProjectTerrainProductsManager *terrainManager,
                                             ProjectReconstructionManager *reconstructionManager,
                                             QObject *parent)
    : QObject(parent)
    , _cameraManager(cameraManager)
    , _terrainManager(terrainManager)
    , _reconstructionManager(reconstructionManager)
{
}

bool ProjectTaskDispatcher::importCameraForImage(const QString &imagePath) const
{
    return _cameraManager ? _cameraManager->importCameraForImage(imagePath) : false;
}

bool ProjectTaskDispatcher::importCamerasByFilenameBatch() const
{
    return _cameraManager ? _cameraManager->importCamerasByFilenameBatch() : false;
}

bool ProjectTaskDispatcher::initializeCamerasFromExifOrDefault(const QJsonObject &settings) const
{
    return _cameraManager ? _cameraManager->initializeCamerasFromExifOrDefault(settings) : false;
}

bool ProjectTaskDispatcher::initializeCamerasFromIntrinsics(const QJsonObject &settings) const
{
    return _cameraManager ? _cameraManager->initializeCamerasFromIntrinsics(settings) : false;
}

bool ProjectTaskDispatcher::initializeCameraPosesWithSFM(const QJsonObject &settings) const
{
    return _cameraManager ? _cameraManager->initializeCameraPosesWithSFM(settings) : false;
}

void ProjectTaskDispatcher::startDemFromPointCloudAsync(
    const xjw::gui::project::DemGenerationRequest &request) const
{
    if (_terrainManager)
    {
        _terrainManager->startDemFromPointCloudAsync(request);
    }
}

void ProjectTaskDispatcher::startMapProjectAsync(const QJsonObject &settings) const
{
    if (!_terrainManager)
    {
        return;
    }

    _terrainManager->startMapProjectAsync(settings);
}

void ProjectTaskDispatcher::cancelMapProject() const
{
    if (_terrainManager)
    {
        _terrainManager->cancelMapProject();
    }
}

void ProjectTaskDispatcher::startReconstructionTask(ProjectReconstructionManager::Task task,
                                                    const QJsonObject &settings) const
{
    if (!_reconstructionManager)
    {
        return;
    }

    _reconstructionManager->startTask(task, settings);
}

QJsonArray ProjectTaskDispatcher::getAvailableAtResults() const
{
    return _reconstructionManager ? _reconstructionManager->getAvailableAtResults() : QJsonArray();
}

void ProjectTaskDispatcher::cancelModelGeneration() const
{
    if (_reconstructionManager)
    {
        _reconstructionManager->cancelModelGeneration();
    }
}
