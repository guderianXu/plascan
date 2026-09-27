#include "ProjectManager.h"

#include "project/services/ProjectLifecycleService.h"
#include "project/services/ProjectResourceCleanupCoordinator.h"
#include "project/services/ProjectResourceService.h"
#include "project/services/ProjectServiceContainer.h"
#include "project/services/ProjectSession.h"
#include "project/tasks/ProjectTaskOrchestrator.h"

#include <QWidget>

ProjectManager::ProjectManager(ProjectData* projectData, QWidget* parent)
    : QObject(parent),
      _serviceContainer(std::make_unique<xjw::gui::project::ProjectServiceContainer>(projectData, parent))
{
    auto& project_session = _serviceContainer->session();
    auto& project_lifecycle = _serviceContainer->lifecycle();
    auto& project_resources = _serviceContainer->resources();
    auto& project_cleanup = _serviceContainer->cleanup();
    auto& project_tasks = _serviceContainer->tasks();

    connect(&project_lifecycle, &ProjectLifecycleService::projectCreated, this, &ProjectManager::projectCreated);
    connect(
        &project_lifecycle, &ProjectLifecycleService::projectOpenStarted, this, &ProjectManager::projectOpenStarted);
    connect(&project_lifecycle,
            &ProjectLifecycleService::projectOpenProgressChanged,
            this,
            &ProjectManager::projectOpenProgressChanged);
    connect(
        &project_lifecycle, &ProjectLifecycleService::projectOpenFinished, this, &ProjectManager::projectOpenFinished);
    connect(&project_lifecycle, &ProjectLifecycleService::saveStarted, this, &ProjectManager::saveStarted);
    connect(&project_lifecycle, &ProjectLifecycleService::saveFinished, this, &ProjectManager::saveFinished);

    connect(&project_session, &xjw::gui::project::ProjectSession::projectOpened, this, &ProjectManager::projectOpened);
    connect(&project_session, &xjw::gui::project::ProjectSession::projectSaved, this, &ProjectManager::projectSaved);
    connect(&project_session, &xjw::gui::project::ProjectSession::projectClosed, this, &ProjectManager::projectClosed);
    connect(&project_session,
            &xjw::gui::project::ProjectSession::sessionChanged,
            this,
            [this](const xjw::gui::project::ProjectSessionContext&) { emit projectSessionChanged(); });
    connect(&project_session,
            &xjw::gui::project::ProjectSession::chunkListChanged,
            this,
            &ProjectManager::chunkListChanged);
    connect(&project_session,
            &xjw::gui::project::ProjectSession::metadataChanged,
            this,
            &ProjectManager::projectMetadataChanged);
    connect(&project_session,
            &xjw::gui::project::ProjectSession::dirtyStateChanged,
            this,
            &ProjectManager::metadataDirtyChanged);
    connect(&project_session,
            &xjw::gui::project::ProjectSession::imageMatchResultAppended,
            this,
            &ProjectManager::imageMatchResultAppended);
    connect(
        &project_session, &xjw::gui::project::ProjectSession::matchPairReady, this, &ProjectManager::matchPairReady);

    connect(&project_resources,
            &xjw::gui::project::ProjectResourceService::imageImportProgressChanged,
            this,
            &ProjectManager::imageImportProgressChanged);
    connect(&project_resources,
            &xjw::gui::project::ProjectResourceService::imageImportFinished,
            this,
            &ProjectManager::imageImportFinished);
    connect(&project_resources,
            &xjw::gui::project::ProjectResourceService::surveyControlChanged,
            this,
            &ProjectManager::surveyControlChanged);

    connect(&project_cleanup,
            &xjw::gui::project::ProjectResourceCleanupCoordinator::progressChanged,
            this,
            &ProjectManager::backgroundTaskProgressChanged);
    connect(&project_cleanup,
            &xjw::gui::project::ProjectResourceCleanupCoordinator::finished,
            this,
            &ProjectManager::backgroundTaskFinished);

    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationProgressChanged,
            this,
            &ProjectManager::maskGenerationProgressChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::maskGenerationFinished,
            this,
            &ProjectManager::maskGenerationFinished);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::masksGenerated,
            this,
            &ProjectManager::masksGenerated);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaved,
            this,
            &ProjectManager::interactiveMaskSaved);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::interactiveMaskSaveFailed,
            this,
            &ProjectManager::interactiveMaskSaveFailed);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::projectMetadataUpdated,
            this,
            &ProjectManager::projectMetadataUpdated);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::meshProgressChanged,
            this,
            &ProjectManager::meshProgressChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::meshProgressFinished,
            this,
            &ProjectManager::meshProgressFinished);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::pointCloudProgressChanged,
            this,
            &ProjectManager::pointCloudProgressChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::pointCloudProgressFinished,
            this,
            &ProjectManager::pointCloudProgressFinished);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::pointCloudResultReady,
            this,
            &ProjectManager::pointCloudResultReady);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::sparseProgressChanged,
            this,
            &ProjectManager::atProgressChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::sparseComputeDeviceChanged,
            this,
            &ProjectManager::atComputeDeviceChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::sparseFinished,
            this,
            &ProjectManager::atProgressFinished);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::sparseTiePointResultReady,
            this,
            &ProjectManager::tiePointResultReady);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::bundleAdjustProgressChanged,
            this,
            &ProjectManager::atProgressChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::bundleAdjustFinished,
            this,
            &ProjectManager::atProgressFinished);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::bundleAdjustPreviewReady,
            this,
            &ProjectManager::bundleAdjustPreviewReady);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::cameraProgressChanged,
            this,
            &ProjectManager::atProgressChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::cameraFinished,
            this,
            &ProjectManager::atProgressFinished);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::backgroundTaskProgressChanged,
            this,
            &ProjectManager::backgroundTaskProgressChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::backgroundTaskFinished,
            this,
            &ProjectManager::backgroundTaskFinished);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::demPipelineProgressChanged,
            this,
            &ProjectManager::demPipelineProgressChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::demPipelineFinished,
            this,
            &ProjectManager::demPipelineFinished);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::orthoPipelineStarted,
            this,
            &ProjectManager::orthoPipelineStarted);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::orthoPipelineProgressChanged,
            this,
            &ProjectManager::orthoPipelineProgressChanged);
    connect(&project_tasks,
            &xjw::gui::project::ProjectTaskOrchestrator::orthoPipelineFinished,
            this,
            &ProjectManager::orthoPipelineFinished);
}

ProjectManager::~ProjectManager() = default;

xjw::gui::project::ProjectServiceContainer& ProjectManager::services() const
{
    return *_serviceContainer;
}

xjw::gui::project::ProjectSession& ProjectManager::session() const
{
    return _serviceContainer->session();
}

xjw::gui::project::ProjectTaskOrchestrator& ProjectManager::tasks() const
{
    return _serviceContainer->tasks();
}

xjw::gui::project::ProjectResourceService& ProjectManager::resources() const
{
    return _serviceContainer->resources();
}

xjw::gui::project::ProjectResourceCleanupCoordinator& ProjectManager::cleanup() const
{
    return _serviceContainer->cleanup();
}

ProjectLifecycleService& ProjectManager::lifecycle() const
{
    return _serviceContainer->lifecycle();
}
