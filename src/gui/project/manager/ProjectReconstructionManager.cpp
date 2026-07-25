#include "ProjectReconstructionManager.h"

#include "ProjectDenseReconstructionManager.h"
#include "ProjectModelGenerationWorkflow.h"
#include "ProjectModelManager.h"
#include "ProjectSparseReconstructionManager.h"

#include <QDebug>

ProjectReconstructionManager::ProjectReconstructionManager(ProjectManager *owner,
                                                           ProjectData *projectData,
                                                           QWidget *parentWidget,
                                                           QObject *parent)
    : QObject(parent)
    , _sparseManager(new ProjectSparseReconstructionManager(owner, projectData, parentWidget, this))
    , _denseManager(new ProjectDenseReconstructionManager(owner, projectData, parentWidget, this))
    , _modelManager(new ProjectModelManager(owner, projectData, parentWidget, this))
    , _modelWorkflow(new ProjectModelGenerationWorkflow(owner,
                                                        projectData,
                                                        parentWidget,
                                                        _denseManager,
                                                        _modelManager,
                                                        this))
{
    connect(_sparseManager, &ProjectSparseReconstructionManager::atProgressChanged,
            this, &ProjectReconstructionManager::atProgressChanged);
    connect(_sparseManager, &ProjectSparseReconstructionManager::atProgressFinished,
            this, &ProjectReconstructionManager::atProgressFinished);

    connect(_denseManager,
            &ProjectDenseReconstructionManager::mvsProgressChanged,
            this,
            [this](const QString &stage, int percent)
    {
        if (!_modelWorkflow->isPreparingDenseCloud())
        {
            emit mvsProgressChanged(stage, percent);
        }
    });
    connect(_denseManager,
            &ProjectDenseReconstructionManager::mvsProgressFinished,
            this,
            [this](bool success)
    {
        if (!_modelWorkflow->consumeInternalMvsFinished())
        {
            emit mvsProgressFinished(success);
        }
    });
    connect(_denseManager, &ProjectDenseReconstructionManager::denseCloudResultReady,
            this, &ProjectReconstructionManager::denseCloudResultReady);

    connect(_modelWorkflow, &ProjectModelGenerationWorkflow::meshProgressChanged,
            this, &ProjectReconstructionManager::meshProgressChanged);
    connect(_modelWorkflow, &ProjectModelGenerationWorkflow::meshProgressFinished,
            this, &ProjectReconstructionManager::meshProgressFinished);
}

QJsonArray ProjectReconstructionManager::getAvailableAtResults() const
{
    return _sparseManager->getAvailableAtResults();
}

void ProjectReconstructionManager::startTask(Task task, const QJsonObject &settings)
{
    const bool is_mvs_task = task == Task::EstimateDepthMaps ||
        task == Task::FuseDepthMaps ||
        task == Task::GenerateDenseCloud ||
        task == Task::RefineDenseCloud;
    const bool is_model_task = task == Task::GenerateModel ||
        task == Task::MeshReconstruction ||
        task == Task::TextureMapping;
    if (is_mvs_task && (_modelManager->isRunning() || _modelWorkflow->isRunning()))
    {
        qWarning() << "[Reconstruction] 模型任务运行期间不能启动 MVS 任务";
        return;
    }
    if (is_model_task && _denseManager->isMvsRunning())
    {
        qWarning() << "[Reconstruction] MVS 任务运行期间不能启动模型任务";
        return;
    }

    switch (task)
    {
    case Task::GenerateModel:
        if (settings.isEmpty())
        {
            _modelManager->startGenerateModelAsync();
        }
        else
        {
            _modelWorkflow->start(settings);
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

void ProjectReconstructionManager::cancelModelGeneration()
{
    if (_modelWorkflow && _modelWorkflow->isRunning())
    {
        _modelWorkflow->cancel();
        return;
    }
    if (_modelManager)
    {
        _modelManager->cancelActiveTask();
    }
}
