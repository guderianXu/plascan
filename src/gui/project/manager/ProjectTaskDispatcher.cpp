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

void ProjectTaskDispatcher::startStereoAndPoint2DemAsync(const QStringList &images,
                                                         const QString &outputDir,
                                                         int threads,
                                                         bool genPointCloud,
                                                         double demResolution,
                                                         const QString &demType,
                                                         const QString &tSrs) const
{
    if (!_terrainManager)
    {
        return;
    }

    _terrainManager->startStereoAndPoint2DemAsync(images,
                                                  outputDir,
                                                  threads,
                                                  genPointCloud,
                                                  demResolution,
                                                  demType,
                                                  tSrs);
}

void ProjectTaskDispatcher::startFullDemPipelineAsync(const QStringList &images,
                                                      const QString &outputDir,
                                                      const QJsonObject &pipelineSettings) const
{
    if (_terrainManager)
        _terrainManager->startFullDemPipelineAsync(images, outputDir, pipelineSettings);
}

void ProjectTaskDispatcher::startDemFromDenseCloudAsync(const QString &denseCloudPath,
                                                        const QString &outputDir,
                                                        double demResolution,
                                                        const QString &demType) const
{
    if (_terrainManager)
        _terrainManager->startDemFromDenseCloudAsync(denseCloudPath, outputDir, demResolution, demType);
}

void ProjectTaskDispatcher::startMapProjectAsync(const QStringList &images,
                                                 const QString &demPath,
                                                 const QString &outputPath,
                                                 double resolution) const
{
    if (!_terrainManager)
    {
        return;
    }

    _terrainManager->startMapProjectAsync(images,
                                          demPath,
                                          outputPath,
                                          resolution);
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

void ProjectTaskDispatcher::cancelMvs() const
{
    if (_reconstructionManager)
    {
        _reconstructionManager->cancelMvs();
    }
}

void ProjectTaskDispatcher::cancelModelGeneration() const
{
    if (_reconstructionManager)
    {
        _reconstructionManager->cancelModelGeneration();
    }
}
