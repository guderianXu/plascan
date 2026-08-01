#include "ProjectReconstructionManager.h"

#include "ProjectModelManager.h"
#include "ProjectSparseReconstructionManager.h"

ProjectReconstructionManager::ProjectReconstructionManager(ProjectManager *owner,
                                                           ProjectData *projectData,
                                                           QWidget *parentWidget,
                                                           QObject *parent)
    : QObject(parent)
    , _sparseManager(new ProjectSparseReconstructionManager(owner, projectData, parentWidget, this))
    , _modelManager(new ProjectModelManager(owner, projectData, parentWidget, this))
{
    connect(_sparseManager, &ProjectSparseReconstructionManager::atProgressChanged,
            this, &ProjectReconstructionManager::atProgressChanged);
    connect(_sparseManager, &ProjectSparseReconstructionManager::atProgressFinished,
            this, &ProjectReconstructionManager::atProgressFinished);

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
        if (settings.isEmpty())
        {
            _modelManager->startGenerateModelAsync();
        }
        else
        {
            _modelManager->startMeshReconstructionAsync(settings);
        }
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
    }
}

void ProjectReconstructionManager::cancelModelGeneration()
{
    if (_modelManager)
    {
        _modelManager->cancelActiveTask();
    }
}
