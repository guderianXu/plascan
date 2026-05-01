#include "ProjectReconstructionManager.h"

#include "ProjectDenseReconstructionManager.h"
#include "ProjectModelManager.h"
#include "ProjectSparseReconstructionManager.h"

ProjectReconstructionManager::ProjectReconstructionManager(ProjectManager *owner,
                                                           ProjectData *projectData,
                                                           QWidget *parentWidget,
                                                           QObject *parent)
    : QObject(parent)
    , m_sparseManager(new ProjectSparseReconstructionManager(owner, projectData, parentWidget, this))
    , m_denseManager(new ProjectDenseReconstructionManager(owner, projectData, parentWidget, this))
    , m_modelManager(new ProjectModelManager(owner, projectData, parentWidget, this))
{
    connect(m_sparseManager, &ProjectSparseReconstructionManager::atProgressChanged,
            this, &ProjectReconstructionManager::atProgressChanged);
    connect(m_sparseManager, &ProjectSparseReconstructionManager::atProgressFinished,
            this, &ProjectReconstructionManager::atProgressFinished);

    connect(m_denseManager, &ProjectDenseReconstructionManager::mvsProgressChanged,
            this, &ProjectReconstructionManager::mvsProgressChanged);
    connect(m_denseManager, &ProjectDenseReconstructionManager::mvsProgressFinished,
            this, &ProjectReconstructionManager::mvsProgressFinished);

    connect(m_modelManager, &ProjectModelManager::meshProgressChanged,
            this, &ProjectReconstructionManager::meshProgressChanged);
    connect(m_modelManager, &ProjectModelManager::meshProgressFinished,
            this, &ProjectReconstructionManager::meshProgressFinished);
}

QJsonArray ProjectReconstructionManager::getAvailableAtResults() const
{
    return m_sparseManager->getAvailableAtResults();
}

void ProjectReconstructionManager::startTask(Task task, const QJsonObject &settings)
{
    switch (task)
    {
    case Task::GenerateModel:
        m_modelManager->startGenerateModelAsync();
        break;
    case Task::MeshReconstruction:
        m_modelManager->startMeshReconstructionAsync(settings);
        break;
    case Task::TextureMapping:
        m_modelManager->startTextureMappingAsync(settings);
        break;
    case Task::Triangulation:
        m_sparseManager->startTriangulationAsync(settings);
        break;
    case Task::SparseOutlierRemoval:
        m_sparseManager->startSparseCloudOutlierRemovalAsync(settings);
        break;
    case Task::SparseLocalOptimization:
        m_sparseManager->startSparseCloudLocalOptimAsync(settings);
        break;
    case Task::SparseRefine:
        m_sparseManager->startSparseCloudRefineAsync(settings);
        break;
    case Task::EstimateDepthMaps:
        m_denseManager->startEstimateDepthMapsAsync(settings);
        break;
    case Task::FuseDepthMaps:
        m_denseManager->startFuseDepthMapsAsync(settings);
        break;
    case Task::GenerateDenseCloud:
        m_denseManager->startGenerateDenseCloudAsync(settings);
        break;
    case Task::RefineDenseCloud:
        m_denseManager->startDenseCloudRefineAsync(settings);
        break;
    }
}

void ProjectReconstructionManager::cancelMvs()
{
    m_denseManager->cancelMvs();
}
