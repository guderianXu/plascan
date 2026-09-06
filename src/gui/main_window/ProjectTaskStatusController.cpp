#include "ProjectTaskStatusController.h"

#include "ProjectDashboardWidget.h"
#include "ProjectManager.h"
#include "TaskStatusWidget.h"
#include "TaskbarProgressController.h"
#include "Logger.h"

#include <QDateTime>
#include <QJsonArray>
#include <QJsonObject>
#include <QStatusBar>

#include <algorithm>

ProjectTaskStatusController::ProjectTaskStatusController(ProjectManager* projectManager,
                                                         ProjectDashboardWidget* dashboard,
                                                         QStatusBar* statusBar,
                                                         QWidget* widgetParent,
                                                         QObject* parent)
    : QObject(parent), _projectManager(projectManager), _dashboard(dashboard), _statusBar(statusBar),
      _taskbarProgress(new xjw::gui::platform::TaskbarProgressController(widgetParent, this))
{
    _meshStatus = createStatus(QStringLiteral("meshTaskStatus"), 220, tr("正在取消模型生成..."), widgetParent);
    _pointCloudStatus =
        createStatus(QStringLiteral("pointCloudTaskStatus"), 220, tr("正在取消点云创建..."), widgetParent);
    _aerialTriangulationStatus = createStatus(
        QStringLiteral("aerialTriangulationTaskStatus"), 220, tr("正在取消空三/光束法平差..."), widgetParent);
    _tiePointStatus = createStatus(QStringLiteral("tiePointTaskStatus"), 180, tr("正在取消特征匹配..."), widgetParent);
    _maskStatus = createStatus(QStringLiteral("maskTaskStatus"), 180, tr("正在取消生成蒙版..."), widgetParent);
    _imageImportStatus = createStatus(QStringLiteral("imageImportTaskStatus"), 180, QString(), widgetParent);
    _imageImportStatus->setCancellable(false);
    _photoListStatus = createStatus(QStringLiteral("photoListTaskStatus"), 180, QString(), widgetParent);
    _photoListStatus->setCancellable(false);

    connect(_meshStatus, &TaskStatusWidget::cancelRequested, _projectManager, &ProjectManager::cancelModelGeneration);
    connect(_pointCloudStatus,
            &TaskStatusWidget::cancelRequested,
            _projectManager,
            &ProjectManager::cancelPointCloudGeneration);
    connect(_aerialTriangulationStatus, &TaskStatusWidget::cancelRequested, _projectManager, &ProjectManager::cancelAt);
    connect(_tiePointStatus,
            &TaskStatusWidget::cancelRequested,
            this,
            &ProjectTaskStatusController::tiePointCancelRequested);
    connect(_maskStatus, &TaskStatusWidget::cancelRequested, _projectManager, &ProjectManager::cancelMaskGeneration);

    connect(_projectManager, &ProjectManager::meshProgressChanged, this, &ProjectTaskStatusController::updateMesh);
    connect(_projectManager, &ProjectManager::meshProgressFinished, this, &ProjectTaskStatusController::finishMesh);
    connect(_projectManager,
            &ProjectManager::pointCloudProgressChanged,
            this,
            &ProjectTaskStatusController::updatePointCloud);
    connect(_projectManager,
            &ProjectManager::pointCloudProgressFinished,
            this,
            &ProjectTaskStatusController::finishPointCloud);
    connect(_projectManager,
            &ProjectManager::atProgressChanged,
            this,
            &ProjectTaskStatusController::updateAerialTriangulation);
    connect(_projectManager,
            &ProjectManager::atComputeDeviceChanged,
            this,
            &ProjectTaskStatusController::updateAerialTriangulationDevice);
    connect(_projectManager,
            &ProjectManager::atProgressFinished,
            this,
            &ProjectTaskStatusController::finishAerialTriangulation);
    connect(_projectManager,
            &ProjectManager::maskGenerationProgressChanged,
            this,
            &ProjectTaskStatusController::updateMask);
    connect(_projectManager, &ProjectManager::maskGenerationFinished, this, &ProjectTaskStatusController::finishMask);
    connect(_projectManager,
            &ProjectManager::imageImportProgressChanged,
            this,
            &ProjectTaskStatusController::updateImageImport);
    connect(
        _projectManager, &ProjectManager::imageImportFinished, this, &ProjectTaskStatusController::finishImageImport);
    connect(_projectManager,
            &ProjectManager::backgroundTaskProgressChanged,
            _taskbarProgress,
            &xjw::gui::platform::TaskbarProgressController::updateTask);
    connect(_projectManager,
            &ProjectManager::backgroundTaskFinished,
            _taskbarProgress,
            &xjw::gui::platform::TaskbarProgressController::finishTask);
    connect(_projectManager,
            &ProjectManager::projectOpenStarted,
            this,
            [this](const QString&) { _taskbarProgress->updateTask(QStringLiteral("project_open"), 0, 100); });
    connect(_projectManager,
            &ProjectManager::projectOpenProgressChanged,
            this,
            [this](const QString&, int percent)
            { _taskbarProgress->updateTask(QStringLiteral("project_open"), percent, 100); });
    connect(_projectManager,
            &ProjectManager::projectOpenFinished,
            this,
            [this](bool, const QString&) { _taskbarProgress->finishTask(QStringLiteral("project_open")); });
    connect(
        _projectManager, &ProjectManager::projectSessionChanged, this, &ProjectTaskStatusController::resetTaskProgress);
    connect(_projectManager,
            &ProjectManager::saveStarted,
            this,
            [this]() { _taskbarProgress->updateTask(QStringLiteral("project_save"), 0, 0); });
    connect(_projectManager,
            &ProjectManager::saveFinished,
            this,
            [this](bool) { _taskbarProgress->finishTask(QStringLiteral("project_save")); });
    refreshDashboard();
}
void ProjectTaskStatusController::showTiePointProgress(int total)
{
    beginTaskActivity(QStringLiteral("tie_points"), tr("特征匹配"), tr("准备匹配"));
    _tiePointStatus->begin(tr("特征匹配 0/%1").arg(total), 0, total);
    _taskbarProgress->updateTask(QStringLiteral("tie_points"), 0, total);
    refreshDashboard();
    _statusBar->showMessage(QString());
}

void ProjectTaskStatusController::updateTiePointProgress(int done)
{
    const int total = _tiePointStatus->progressMaximum();
    const int value = std::clamp(done, 0, total);
    _tiePointStatus->updateProgress(tr("特征匹配 %1/%2").arg(value).arg(total), value);
    updateTaskActivity(QStringLiteral("tie_points"), _tiePointStatus->statusText());
    _taskbarProgress->updateTask(QStringLiteral("tie_points"), value, total);
    refreshDashboard();
}

void ProjectTaskStatusController::finishTiePointProgress(bool success)
{
    const bool wasActive = _tiePointStatus->isActive() || _taskbarProgress->hasTask(QStringLiteral("tie_points"));
    const qint64 elapsed_ms = _tiePointStatus->elapsedMilliseconds();
    finishTaskActivity(
        QStringLiteral("tie_points"), success, !success, elapsed_ms, success ? tr("匹配完成") : tr("匹配已取消"));
    _tiePointStatus->finish();
    _taskbarProgress->finishTask(QStringLiteral("tie_points"));
    refreshDashboard();
    if (wasActive)
    {
        _statusBar->showMessage(success ? tr("匹配完成") : tr("匹配已取消"), 4000);
    }
}

void ProjectTaskStatusController::updateMesh(const QString& stage, int percent)
{
    updatePercentTask(_meshStatus, stage, percent, false, QStringLiteral("mesh"));
}

void ProjectTaskStatusController::finishMesh(bool success)
{
    finishTask(_meshStatus, success, tr("网格重建完成"), tr("网格重建失败"), QStringLiteral("mesh"));
}

void ProjectTaskStatusController::updatePointCloud(const QString& stage, int percent)
{
    updatePercentTask(_pointCloudStatus, stage, percent, true, QStringLiteral("point_cloud"));
}

void ProjectTaskStatusController::finishPointCloud(bool success)
{
    finishTask(
        _pointCloudStatus, success, tr("点云创建完成"), tr("点云创建已取消或失败"), QStringLiteral("point_cloud"));
}

void ProjectTaskStatusController::updateAerialTriangulation(const QString& stage, int percent)
{
    updatePercentTask(_aerialTriangulationStatus, stage, percent, true, QStringLiteral("aerial_triangulation"));
}

void ProjectTaskStatusController::updateAerialTriangulationDevice(const QString& displayName)
{
    _aerialTriangulationStatus->setDetailText(
        displayName.trimmed().isEmpty() ? QString() : tr("计算设备：%1").arg(displayName.trimmed()));
}

void ProjectTaskStatusController::finishAerialTriangulation(bool success)
{
    finishTask(_aerialTriangulationStatus,
               success,
               tr("空三/光束法平差完成"),
               tr("空三/光束法平差已取消或失败"),
               QStringLiteral("aerial_triangulation"));
}

void ProjectTaskStatusController::updateMask(const QString& stage, int done, int total)
{
    const int maximum = std::max(1, total);
    const int value = std::clamp(done, 0, maximum);
    const QString baseText = tr("生成蒙版 %1/%2").arg(value).arg(maximum);
    const QString text = stage.trimmed().isEmpty() || stage == tr("生成蒙版")
                             ? baseText
                             : tr("%1 %2/%3").arg(stage).arg(value).arg(maximum);
    if (!_maskStatus->isActive())
    {
        beginTaskActivity(QStringLiteral("mask"), tr("生成蒙版"), text);
        _maskStatus->begin(text, 0, maximum);
    }
    _maskStatus->updateProgress(text, value);
    updateTaskActivity(QStringLiteral("mask"), text);
    _taskbarProgress->updateTask(QStringLiteral("mask"), value, maximum);
    refreshDashboard();
    _statusBar->showMessage(QString());
}

void ProjectTaskStatusController::finishMask(bool success)
{
    const bool wasActive = _maskStatus->isActive() || _taskbarProgress->hasTask(QStringLiteral("mask"));
    const bool cancelled = !success && _maskStatus->isCancelling();
    const qint64 elapsed_ms = _maskStatus->elapsedMilliseconds();
    const QString summary = success ? tr("蒙版生成完成") : (cancelled ? tr("蒙版生成已取消") : tr("蒙版生成失败"));
    finishTaskActivity(QStringLiteral("mask"), success, cancelled, elapsed_ms, summary);
    _maskStatus->finish();
    _taskbarProgress->finishTask(QStringLiteral("mask"));
    refreshDashboard();
    if (wasActive)
    {
        _statusBar->showMessage(success ? tr("蒙版生成完成") : tr("蒙版生成已取消或失败"), 4000);
    }
}

void ProjectTaskStatusController::updateImageLoading(const QString& stage, int done, int total)
{
    updateImageLoadingTask(_photoListStatus, QStringLiteral("photo_list"), stage, done, total);
}

void ProjectTaskStatusController::finishImageLoading(bool success, const QString& message)
{
    finishImageLoadingTask(_photoListStatus, QStringLiteral("photo_list"), success, message);
}

void ProjectTaskStatusController::updateImageImport(const QString& stage, int done, int total)
{
    updateImageLoadingTask(_imageImportStatus, QStringLiteral("image_import"), stage, done, total);
}

void ProjectTaskStatusController::finishImageImport(bool success, const QString& message)
{
    finishImageLoadingTask(_imageImportStatus, QStringLiteral("image_import"), success, message);
}

void ProjectTaskStatusController::updateImageLoadingTask(
    TaskStatusWidget* status, const QString& taskId, const QString& stage, int done, int total)
{
    const int maximum = std::max(0, total);
    const int value = maximum > 0 ? std::clamp(done, 0, maximum) : 0;
    const QString text = maximum > 0 ? tr("%1 %2/%3").arg(stage).arg(value).arg(maximum) : stage;
    if (!status->isActive())
    {
        const QString name = taskId == QStringLiteral("image_import") ? tr("导入影像") : tr("加载照片列表");
        beginTaskActivity(taskId, name, text);
        status->begin(text, 0, maximum);
    }
    else if (status->progressMaximum() != maximum)
    {
        status->begin(text, 0, maximum);
    }
    status->updateProgress(text, value);
    updateTaskActivity(taskId, text);
    _taskbarProgress->updateTask(taskId, value, maximum);
    refreshDashboard();
    _statusBar->showMessage(QString());
}

void ProjectTaskStatusController::finishImageLoadingTask(TaskStatusWidget* status,
                                                         const QString& taskId,
                                                         bool success,
                                                         const QString& message)
{
    const bool wasActive = status->isActive() || _taskbarProgress->hasTask(taskId);
    const qint64 elapsed_ms = status->elapsedMilliseconds();
    const QString fallback = success ? tr("影像加载完成") : tr("影像加载已停止或失败");
    const QString summary = message.trimmed().isEmpty() ? fallback : message;
    finishTaskActivity(taskId, success, false, elapsed_ms, summary);
    _taskbarProgress->finishTask(taskId);
    status->finish();
    refreshDashboard();
    if (wasActive)
    {
        _statusBar->showMessage(summary, 4000);
    }
}

TaskStatusWidget* ProjectTaskStatusController::createStatus(const QString& objectName,
                                                            int labelWidth,
                                                            const QString& cancellingText,
                                                            QWidget* widgetParent)
{
    auto* status = new TaskStatusWidget(widgetParent);
    status->setObjectName(objectName);
    status->setLabelMinimumWidth(labelWidth);
    status->setCancellable(true);
    status->setCancellingText(cancellingText);
    connect(status, &TaskStatusWidget::cancelRequested, this, &ProjectTaskStatusController::refreshDashboard);
    _statusBar->addPermanentWidget(status);
    return status;
}

void ProjectTaskStatusController::updatePercentTask(
    TaskStatusWidget* status, const QString& stage, int percent, bool appendIntermediatePercent, const QString& taskId)
{
    if (percent < 0)
    {
        if (!status->isActive() || status->progressMaximum() != 0)
        {
            const QString name =
                taskId == QStringLiteral("mesh")
                    ? tr("网格重建")
                    : (taskId == QStringLiteral("point_cloud") ? tr("创建点云") : tr("空三/光束法平差"));
            beginTaskActivity(taskId, name, stage);
            status->begin(stage, 0, 0);
        }
        status->updateProgress(stage, 0);
        updateTaskActivity(taskId, stage);
        _taskbarProgress->updateTask(taskId, 0, 0);
        refreshDashboard();
        _statusBar->showMessage(QString());
        return;
    }

    const int value = std::clamp(percent, 0, 100);
    const QString text =
        appendIntermediatePercent && value > 0 && value < 100 ? QStringLiteral("%1 %2%").arg(stage).arg(value) : stage;
    if (!status->isActive() || status->progressMaximum() != 100)
    {
        const QString name = taskId == QStringLiteral("mesh")
                                 ? tr("网格重建")
                                 : (taskId == QStringLiteral("point_cloud") ? tr("创建点云") : tr("空三/光束法平差"));
        beginTaskActivity(taskId, name, text);
        status->begin(text, 0, 100);
    }
    status->updateProgress(text, value);
    updateTaskActivity(taskId, text);
    _taskbarProgress->updateTask(taskId, value, 100);
    refreshDashboard();
    _statusBar->showMessage(QString());
}

void ProjectTaskStatusController::finishTask(TaskStatusWidget* status,
                                             bool success,
                                             const QString& successMessage,
                                             const QString& failureMessage,
                                             const QString& taskId)
{
    const bool wasActive = status->isActive() || _taskbarProgress->hasTask(taskId);
    const bool cancelled = !success && status->isCancelling();
    const qint64 elapsed_ms = status->elapsedMilliseconds();
    const QString summary = success ? successMessage : failureMessage;
    finishTaskActivity(taskId, success, cancelled, elapsed_ms, summary);
    status->finish();
    _taskbarProgress->finishTask(taskId);
    refreshDashboard();
    if (wasActive)
    {
        _statusBar->showMessage(summary, 4000);
    }
}

void ProjectTaskStatusController::resetTaskProgress()
{
    const bool keepModelCancellationVisible = _projectManager && _projectManager->isModelGenerationRunning();
    for (TaskStatusWidget* status : {_meshStatus,
                                     _pointCloudStatus,
                                     _aerialTriangulationStatus,
                                     _tiePointStatus,
                                     _maskStatus,
                                     _imageImportStatus,
                                     _photoListStatus})
    {
        if (status && !(keepModelCancellationVisible && status == _meshStatus))
        {
            status->finish();
        }
    }
    _taskbarProgress->clearTasks();
    _activeTaskRuns.clear();
    _taskHistory = {};
    if (keepModelCancellationVisible && _meshStatus)
    {
        _taskbarProgress->updateTask(
            QStringLiteral("mesh"), _meshStatus->progressValue(), std::max(1, _meshStatus->progressMaximum()));
    }
    refreshDashboard();
}

void ProjectTaskStatusController::refreshDashboard()
{
    if (!_dashboard)
    {
        return;
    }
    QJsonArray tasks;
    const auto append = [this, &tasks](const QString& taskId, const QString& name, const TaskStatusWidget* status)
    {
        if (!status || (!status->isActive() && !status->isCancelling()))
        {
            return;
        }
        QJsonObject record;
        record.insert(QStringLiteral("name"), name);
        record.insert(QStringLiteral("task_id"), taskId);
        record.insert(QStringLiteral("status_text"), status->statusText());
        record.insert(QStringLiteral("active"), status->isActive());
        record.insert(QStringLiteral("cancelling"), status->isCancelling());
        record.insert(QStringLiteral("progress_value"), status->progressValue());
        record.insert(QStringLiteral("progress_maximum"), status->progressMaximum());
        record.insert(QStringLiteral("elapsed_ms"), status->elapsedMilliseconds());
        record.insert(QStringLiteral("scheduler_managed"), false);
        record.insert(QStringLiteral("can_pause"), false);
        record.insert(QStringLiteral("can_resume"), false);
        record.insert(QStringLiteral("can_reorder"), false);
        record.insert(QStringLiteral("can_cancel"), false);
        if (_activeTaskRuns.contains(taskId))
        {
            const QJsonObject activity = _activeTaskRuns.value(taskId);
            record.insert(QStringLiteral("run_id"), activity.value(QStringLiteral("run_id")));
            record.insert(QStringLiteral("state"), QStringLiteral("running"));
            record.insert(QStringLiteral("start_sequence"), activity.value(QStringLiteral("start_sequence")));
        }
        tasks.append(record);
    };
    append(QStringLiteral("mesh"), tr("网格重建"), _meshStatus);
    append(QStringLiteral("point_cloud"), tr("创建点云"), _pointCloudStatus);
    append(QStringLiteral("aerial_triangulation"), tr("空三/光束法平差"), _aerialTriangulationStatus);
    append(QStringLiteral("tie_points"), tr("特征匹配"), _tiePointStatus);
    append(QStringLiteral("mask"), tr("生成蒙版"), _maskStatus);
    append(QStringLiteral("image_import"), tr("导入影像"), _imageImportStatus);
    append(QStringLiteral("photo_list"), tr("加载照片列表"), _photoListStatus);
    for (const QJsonValue& scheduled : _schedulerTaskSnapshots)
    {
        tasks.append(scheduled);
    }
    for (const QJsonValue& history : _taskHistory)
    {
        tasks.append(history);
    }
    _dashboard->setTaskSnapshots(tasks);
}

void ProjectTaskStatusController::setSchedulerTaskSnapshots(const QJsonArray& snapshots)
{
    _schedulerTaskSnapshots = snapshots;
    refreshDashboard();
}

void ProjectTaskStatusController::clearTaskHistory()
{
    _taskHistory = {};
    refreshDashboard();
}

void ProjectTaskStatusController::beginTaskActivity(const QString& taskId, const QString& name, const QString& stage)
{
    if (_activeTaskRuns.contains(taskId))
    {
        updateTaskActivity(taskId, stage);
        return;
    }
    Logger::Context context;
    context.category = "Task";
    context.taskId = taskId.toStdString();
    context.stage = stage.toStdString();
    const std::uint64_t sequence =
        Logger::instance()->logWithContext(Logger::Info, tr("开始：%1").arg(name).toStdString(), context);

    QJsonObject activity;
    activity.insert(QStringLiteral("task_id"), taskId);
    activity.insert(QStringLiteral("run_id"),
                    QStringLiteral("%1-%2").arg(taskId).arg(QDateTime::currentMSecsSinceEpoch()));
    activity.insert(QStringLiteral("name"), name);
    activity.insert(QStringLiteral("stage"), stage);
    activity.insert(QStringLiteral("start_sequence"), static_cast<qint64>(sequence));
    _activeTaskRuns.insert(taskId, activity);
}

void ProjectTaskStatusController::updateTaskActivity(const QString& taskId, const QString& stage)
{
    auto iterator = _activeTaskRuns.find(taskId);
    if (iterator != _activeTaskRuns.end())
    {
        iterator->insert(QStringLiteral("stage"), stage);
    }
}

void ProjectTaskStatusController::finishTaskActivity(
    const QString& taskId, bool success, bool cancelled, qint64 elapsedMs, const QString& summary)
{
    auto iterator = _activeTaskRuns.find(taskId);
    if (iterator == _activeTaskRuns.end())
    {
        return;
    }

    QJsonObject activity = iterator.value();
    _activeTaskRuns.erase(iterator);
    const QString state =
        success ? QStringLiteral("succeeded") : (cancelled ? QStringLiteral("cancelled") : QStringLiteral("failed"));
    Logger::Context context;
    context.category = "Task";
    context.taskId = taskId.toStdString();
    context.stage = state.toStdString();
    const Logger::Level level = success ? Logger::Info : (cancelled ? Logger::Warn : Logger::Error);
    const std::uint64_t sequence = Logger::instance()->logWithContext(level, summary.toStdString(), context);

    activity.insert(QStringLiteral("active"), false);
    activity.insert(QStringLiteral("cancelling"), false);
    activity.insert(QStringLiteral("state"), state);
    activity.insert(QStringLiteral("status_text"), summary);
    activity.insert(QStringLiteral("elapsed_ms"), elapsedMs);
    activity.insert(QStringLiteral("end_sequence"), static_cast<qint64>(sequence));
    activity.insert(QStringLiteral("progress_value"), success ? 1 : 0);
    activity.insert(QStringLiteral("progress_maximum"), 1);
    _taskHistory.prepend(activity);
    while (_taskHistory.size() > 100)
    {
        _taskHistory.removeLast();
    }
}
