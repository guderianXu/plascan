#include "ProjectReconstructionManager.h"

#include "ProjectDenseReconstructionManager.h"
#include "ProjectModelManager.h"
#include "ProjectSparseReconstructionManager.h"

ProjectReconstructionManager::ProjectReconstructionManager(ProjectManager *owner,
                                                           ProjectData *projectData,
                                                           QWidget *parentWidget,
                                                           QObject *parent)
    : QObject(parent)
    , _sparseManager(new ProjectSparseReconstructionManager(owner, projectData, parentWidget, this))
    , _denseManager(new ProjectDenseReconstructionManager(owner, projectData, parentWidget, this))
    , _modelManager(new ProjectModelManager(owner, projectData, parentWidget, this))
{
    connect(_sparseManager, &ProjectSparseReconstructionManager::atProgressChanged,
            this, &ProjectReconstructionManager::atProgressChanged);
    connect(_sparseManager, &ProjectSparseReconstructionManager::atProgressFinished,
            this, &ProjectReconstructionManager::atProgressFinished);

    connect(_denseManager, &ProjectDenseReconstructionManager::mvsProgressChanged,
            this, &ProjectReconstructionManager::mvsProgressChanged);
    connect(_denseManager, &ProjectDenseReconstructionManager::mvsProgressFinished,
            this, &ProjectReconstructionManager::mvsProgressFinished);

    connect(_modelManager, &ProjectModelManager::meshProgressChanged,
            this, &ProjectReconstructionManager::meshProgressChanged);
    connect(_modelManager, &ProjectModelManager::meshProgressFinished,
            this, &ProjectReconstructionManager::meshProgressFinished);
}

QJsonArray ProjectReconstructionManager::getAvailableAtResults() const
{
    return _sparseManager->getAvailableAtResults();
}

void ProjectReconstructionManager::startTask(Task task, const QJsonObject &settings)
{
    switch (task)
    {
    case Task::GenerateModel:
        _modelManager->startGenerateModelAsync();
        break;
    case Task::MeshReconstruction:
        _modelManager->startMeshReconstructionAsync(settings);
        break;
    case Task::TextureMapping:
        _modelManager->startTextureMappingAsync(settings);
        break;
    case Task::Triangulation:
        _sparseManager->startTriangulationAsync(settings);
        break;
    case Task::SparseOutlierRemoval:
        _sparseManager->startSparseCloudOutlierRemovalAsync(settings);
        break;
    case Task::SparseLocalOptimization:
        _sparseManager->startSparseCloudLocalOptimAsync(settings);
        break;
    case Task::SparseRefine:
        _sparseManager->startSparseCloudRefineAsync(settings);
        break;
    case Task::EstimateDepthMaps:
        _denseManager->startEstimateDepthMapsAsync(settings);
        break;
    case Task::FuseDepthMaps:
        _denseManager->startFuseDepthMapsAsync(settings);
        break;
    case Task::GenerateDenseCloud:
        _denseManager->startGenerateDenseCloudAsync(settings);
        break;
    case Task::RefineDenseCloud:
        _denseManager->startDenseCloudRefineAsync(settings);
        break;
    }
}

void ProjectReconstructionManager::cancelMvs()
{
    _denseManager->cancelMvs();
}
