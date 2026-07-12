#include "ProjectModelGenerationWorkflow.h"

#include "ProjectData.h"
#include "ProjectDenseReconstructionManager.h"
#include "ProjectManager.h"
#include "ProjectModelManager.h"
#include "ProjectModelWorkflowPolicy.h"

#include <QDir>
#include <QFileInfo>
#include <QMessageBox>

ProjectModelGenerationWorkflow::ProjectModelGenerationWorkflow(
    ProjectManager *owner,
    ProjectData *project_data,
    QWidget *parent_widget,
    ProjectDenseReconstructionManager *dense_manager,
    ProjectModelManager *model_manager,
    QObject *parent)
    : QObject(parent)
    , _owner(owner)
    , _projectData(project_data)
    , _parentWidget(parent_widget)
    , _denseManager(dense_manager)
    , _modelManager(model_manager)
{
    connect(_denseManager,
            &ProjectDenseReconstructionManager::mvsProgressChanged,
            this,
            [this](const QString &stage, int percent)
    {
        if (_stage == Stage::DenseCloud && belongsToActiveProject())
        {
            emit meshProgressChanged(QStringLiteral("准备深度图：%1").arg(stage), percent);
        }
    });
    connect(_denseManager,
            &ProjectDenseReconstructionManager::denseCloudResultReady,
            this,
            [this](const QString &dense_cloud_path, int)
    {
        if (_stage != Stage::DenseCloud || !belongsToActiveProject())
        {
            return;
        }

        const QString result_directory = QDir::cleanPath(QFileInfo(dense_cloud_path).absolutePath());
        if (!_expectedDenseOutputDir.isEmpty() &&
            result_directory.compare(_expectedDenseOutputDir, Qt::CaseInsensitive) != 0)
        {
            return;
        }

        QJsonObject model_settings = _pendingModelSettings;
        const QString depth_directory = result_directory;
        model_settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
        model_settings[QStringLiteral("source_path")] = depth_directory;
        model_settings[QStringLiteral("depthMapSourcePath")] = depth_directory;
        model_settings[QStringLiteral("source_point_cloud_path")] = dense_cloud_path;
        model_settings[QStringLiteral("pipeline_mode")] = true;
        startModelStage(model_settings);
    });
    connect(_denseManager,
            &ProjectDenseReconstructionManager::mvsProgressFinished,
            this,
            [this](bool success)
    {
        if (_stage != Stage::DenseCloud)
        {
            return;
        }
        if (!success || !belongsToActiveProject())
        {
            finish(false);
        }
    });
    connect(_modelManager,
            &ProjectModelManager::meshProgressChanged,
            this,
            &ProjectModelGenerationWorkflow::meshProgressChanged);
    connect(_modelManager,
            &ProjectModelManager::meshProgressFinished,
            this,
            [this](bool success)
    {
        if (_stage == Stage::Model)
        {
            finish(success && belongsToActiveProject());
            return;
        }
        emit meshProgressFinished(success);
    });
}

void ProjectModelGenerationWorkflow::start(const QJsonObject &settings)
{
    if (isRunning())
    {
        QMessageBox::information(_parentWidget,
                                 QStringLiteral("生成模型"),
                                 QStringLiteral("已有模型生成任务正在运行，请等待其完成或先取消。"));
        return;
    }
    if (!_owner || !_projectData || !_denseManager || !_modelManager ||
        _owner->currentProjectPath().isEmpty())
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("生成模型"),
                             QStringLiteral("请先打开一个项目。"));
        return;
    }

    _projectPath = _owner->currentProjectPath();
    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(
        settings,
        _projectData->metadata());

    if (_denseManager->isMvsRunning())
    {
        QMessageBox::information(_parentWidget,
                                 QStringLiteral("生成模型"),
                                 QStringLiteral("已有深度图或密集点云任务正在运行，请等待其完成或先取消。"));
        _projectPath.clear();
        return;
    }

    if (decision.action == xjw::gui::project::ModelWorkflowAction::RunMeshDirectly)
    {
        startModelStage(decision.modelSettings);
        return;
    }

    startDenseStage(decision.denseSettings, decision.modelSettings);
}

bool ProjectModelGenerationWorkflow::isRunning() const
{
    return _stage != Stage::Idle;
}

bool ProjectModelGenerationWorkflow::isPreparingDenseCloud() const
{
    return _stage == Stage::DenseCloud;
}

bool ProjectModelGenerationWorkflow::consumeInternalMvsFinished()
{
    if (!_internalMvsFinishPending)
    {
        return false;
    }
    _internalMvsFinishPending = false;
    return true;
}

void ProjectModelGenerationWorkflow::startModelStage(const QJsonObject &settings)
{
    _stage = Stage::Model;
    emit meshProgressChanged(QStringLiteral("正在生成三维模型..."), 0);
    if (!_modelManager->startMeshReconstructionAsync(settings))
    {
        finish(false);
    }
}

void ProjectModelGenerationWorkflow::startDenseStage(const QJsonObject &dense_settings,
                                                      const QJsonObject &model_settings)
{
    _stage = Stage::DenseCloud;
    _pendingModelSettings = model_settings;
    QJsonObject dense_settings_with_runtime = dense_settings;
    dense_settings_with_runtime[QStringLiteral("transient_for_model")] = true;
    QString output_directory = dense_settings_with_runtime.value(QStringLiteral("output_dir")).toString().trimmed();
    const QString project_directory = QFileInfo(_projectPath).absolutePath();
    if (output_directory.isEmpty())
    {
        output_directory = QDir(project_directory).filePath(QStringLiteral("mvs_output"));
    }
    else if (QFileInfo(output_directory).isRelative())
    {
        output_directory = QDir(project_directory).filePath(output_directory);
    }
    _expectedDenseOutputDir = QDir::cleanPath(QFileInfo(output_directory).absoluteFilePath());
    emit meshProgressChanged(QStringLiteral("正在准备深度图..."), 0);

    const QString action = dense_settings_with_runtime.value(QStringLiteral("workflow_action")).toString();
    if (action == QStringLiteral("fuse_existing_depth_maps"))
    {
        if (_denseManager->startFuseDepthMapsAsync(dense_settings_with_runtime))
        {
            _internalMvsFinishPending = true;
        }
        else
        {
            finish(false);
        }
        return;
    }

    if (_denseManager->startGenerateDenseCloudAsync(dense_settings_with_runtime))
    {
        _internalMvsFinishPending = true;
    }
    else
    {
        finish(false);
    }
}

bool ProjectModelGenerationWorkflow::belongsToActiveProject() const
{
    return _owner && !_projectPath.isEmpty() && _owner->currentProjectPath() == _projectPath;
}

void ProjectModelGenerationWorkflow::finish(bool success)
{
    _stage = Stage::Idle;
    _projectPath.clear();
    _expectedDenseOutputDir.clear();
    _pendingModelSettings = QJsonObject();
    emit meshProgressFinished(success);
}
