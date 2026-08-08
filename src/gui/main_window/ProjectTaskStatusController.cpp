#include "ProjectTaskStatusController.h"

#include "ProjectDashboardWidget.h"
#include "ProjectManager.h"
#include "TaskStatusWidget.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QStatusBar>

#include <algorithm>

ProjectTaskStatusController::ProjectTaskStatusController(ProjectManager *projectManager,
                                                         ProjectDashboardWidget *dashboard,
                                                         QStatusBar *statusBar,
                                                         QWidget *widgetParent,
                                                         QObject *parent)
    : QObject(parent)
    , _projectManager(projectManager)
    , _dashboard(dashboard)
    , _statusBar(statusBar)
{
    _meshStatus = createStatus(220, tr("正在取消模型生成..."), widgetParent);
    _pointCloudStatus = createStatus(220, tr("正在取消点云创建..."), widgetParent);
    _aerialTriangulationStatus = createStatus(220, tr("正在取消空三/光束法平差..."), widgetParent);
    _tiePointStatus = createStatus(180, tr("正在取消特征匹配..."), widgetParent);
    _maskStatus = createStatus(180, tr("正在取消生成蒙版..."), widgetParent);
    _imageLoadingStatus = createStatus(180, QString(), widgetParent);
    _imageLoadingStatus->setCancellable(false);

    connect(_meshStatus, &TaskStatusWidget::cancelRequested,
            _projectManager, &ProjectManager::cancelModelGeneration);
    connect(_pointCloudStatus, &TaskStatusWidget::cancelRequested,
            _projectManager, &ProjectManager::cancelPointCloudGeneration);
    connect(_aerialTriangulationStatus, &TaskStatusWidget::cancelRequested,
            _projectManager, &ProjectManager::cancelAt);
    connect(_tiePointStatus, &TaskStatusWidget::cancelRequested,
            this, &ProjectTaskStatusController::tiePointCancelRequested);
    connect(_maskStatus, &TaskStatusWidget::cancelRequested,
            _projectManager, &ProjectManager::cancelMaskGeneration);

    connect(_projectManager, &ProjectManager::meshProgressChanged,
            this, &ProjectTaskStatusController::updateMesh);
    connect(_projectManager, &ProjectManager::meshProgressFinished,
            this, &ProjectTaskStatusController::finishMesh);
    connect(_projectManager, &ProjectManager::pointCloudProgressChanged,
            this, &ProjectTaskStatusController::updatePointCloud);
    connect(_projectManager, &ProjectManager::pointCloudProgressFinished,
            this, &ProjectTaskStatusController::finishPointCloud);
    connect(_projectManager, &ProjectManager::atProgressChanged,
            this, &ProjectTaskStatusController::updateAerialTriangulation);
    connect(_projectManager, &ProjectManager::atProgressFinished,
            this, &ProjectTaskStatusController::finishAerialTriangulation);
    connect(_projectManager, &ProjectManager::maskGenerationProgressChanged,
            this, &ProjectTaskStatusController::updateMask);
    connect(_projectManager, &ProjectManager::maskGenerationFinished,
            this, &ProjectTaskStatusController::finishMask);
    connect(_projectManager, &ProjectManager::imageImportProgressChanged,
            this, &ProjectTaskStatusController::updateImageLoading);
    connect(_projectManager, &ProjectManager::imageImportFinished,
            this, &ProjectTaskStatusController::finishImageLoading);
    refreshDashboard();
}

void ProjectTaskStatusController::showTiePointProgress(int total)
{
    _tiePointStatus->begin(tr("特征匹配 0/%1").arg(total), 0, total);
    refreshDashboard();
    _statusBar->showMessage(QString());
}

void ProjectTaskStatusController::updateTiePointProgress(int done)
{
    const int total = _tiePointStatus->progressMaximum();
    const int value = std::clamp(done, 0, total);
    _tiePointStatus->updateProgress(tr("特征匹配 %1/%2").arg(value).arg(total), value);
    refreshDashboard();
}

void ProjectTaskStatusController::finishTiePointProgress(bool success)
{
    _tiePointStatus->finish();
    refreshDashboard();
    _statusBar->showMessage(success ? tr("匹配完成") : tr("匹配已取消"), 4000);
}

void ProjectTaskStatusController::updateMesh(const QString &stage, int percent)
{
    updatePercentTask(_meshStatus, stage, percent, false);
}

void ProjectTaskStatusController::finishMesh(bool success)
{
    finishTask(_meshStatus, success, tr("网格重建完成"), tr("网格重建失败"));
}

void ProjectTaskStatusController::updatePointCloud(const QString &stage, int percent)
{
    updatePercentTask(_pointCloudStatus, stage, percent, true);
}

void ProjectTaskStatusController::finishPointCloud(bool success)
{
    finishTask(_pointCloudStatus,
               success,
               tr("点云创建完成"),
               tr("点云创建已取消或失败"));
}

void ProjectTaskStatusController::updateAerialTriangulation(const QString &stage, int percent)
{
    updatePercentTask(_aerialTriangulationStatus, stage, percent, true);
}

void ProjectTaskStatusController::finishAerialTriangulation(bool success)
{
    finishTask(_aerialTriangulationStatus,
               success,
               tr("空三/光束法平差完成"),
               tr("空三/光束法平差已取消或失败"));
}

void ProjectTaskStatusController::updateMask(const QString &stage, int done, int total)
{
    const int maximum = std::max(1, total);
    const int value = std::clamp(done, 0, maximum);
    const QString baseText = tr("生成蒙版 %1/%2").arg(value).arg(maximum);
    const QString text = stage.trimmed().isEmpty() || stage == tr("生成蒙版")
        ? baseText
        : tr("%1 %2/%3").arg(stage).arg(value).arg(maximum);
    if (!_maskStatus->isActive())
    {
        _maskStatus->begin(text, 0, maximum);
    }
    _maskStatus->updateProgress(text, value);
    refreshDashboard();
    _statusBar->showMessage(QString());
}

void ProjectTaskStatusController::finishMask(bool success)
{
    _maskStatus->finish();
    refreshDashboard();
    _statusBar->showMessage(success ? tr("蒙版生成完成") : tr("蒙版生成已取消或失败"), 4000);
}

void ProjectTaskStatusController::updateImageLoading(const QString &stage, int done, int total)
{
    const int maximum = std::max(0, total);
    const int value = maximum > 0 ? std::clamp(done, 0, maximum) : 0;
    const QString text = maximum > 0
        ? tr("%1 %2/%3").arg(stage).arg(value).arg(maximum)
        : stage;
    if (!_imageLoadingStatus->isActive())
    {
        _imageLoadingStatus->begin(text, 0, maximum);
    }
    else if (_imageLoadingStatus->progressMaximum() != maximum)
    {
        _imageLoadingStatus->begin(text, 0, maximum);
    }
    _imageLoadingStatus->updateProgress(text, value);
    refreshDashboard();
    _statusBar->showMessage(QString());
}

void ProjectTaskStatusController::finishImageLoading(bool success, const QString &message)
{
    _imageLoadingStatus->finish();
    refreshDashboard();
    const QString fallback = success ? tr("影像加载完成") : tr("影像加载已停止或失败");
    _statusBar->showMessage(message.trimmed().isEmpty() ? fallback : message, 4000);
}

TaskStatusWidget *ProjectTaskStatusController::createStatus(int labelWidth,
                                                           const QString &cancellingText,
                                                           QWidget *widgetParent)
{
    auto *status = new TaskStatusWidget(widgetParent);
    status->setLabelMinimumWidth(labelWidth);
    status->setCancellable(true);
    status->setCancellingText(cancellingText);
    connect(status, &TaskStatusWidget::cancelRequested,
            this, &ProjectTaskStatusController::refreshDashboard);
    _statusBar->addPermanentWidget(status);
    return status;
}

void ProjectTaskStatusController::updatePercentTask(TaskStatusWidget *status,
                                                    const QString &stage,
                                                    int percent,
                                                    bool appendIntermediatePercent)
{
    const int value = std::clamp(percent, 0, 100);
    const QString text = appendIntermediatePercent && value > 0 && value < 100
        ? QStringLiteral("%1 %2%").arg(stage).arg(value)
        : stage;
    if (!status->isActive())
    {
        status->begin(text, 0, 100);
    }
    status->updateProgress(text, value);
    refreshDashboard();
    _statusBar->showMessage(QString());
}

void ProjectTaskStatusController::finishTask(TaskStatusWidget *status,
                                             bool success,
                                             const QString &successMessage,
                                             const QString &failureMessage)
{
    status->finish();
    refreshDashboard();
    _statusBar->showMessage(success ? successMessage : failureMessage, 4000);
}

void ProjectTaskStatusController::refreshDashboard()
{
    if (!_dashboard)
    {
        return;
    }
    QJsonArray tasks;
    const auto append = [&tasks](const QString &name, const TaskStatusWidget *status)
    {
        if (!status || (!status->isActive() && !status->isCancelling()))
        {
            return;
        }
        QJsonObject record;
        record.insert(QStringLiteral("name"), name);
        record.insert(QStringLiteral("status_text"), status->statusText());
        record.insert(QStringLiteral("active"), status->isActive());
        record.insert(QStringLiteral("cancelling"), status->isCancelling());
        record.insert(QStringLiteral("progress_value"), status->progressValue());
        record.insert(QStringLiteral("progress_maximum"), status->progressMaximum());
        tasks.append(record);
    };
    append(tr("网格重建"), _meshStatus);
    append(tr("创建点云"), _pointCloudStatus);
    append(tr("空三/光束法平差"), _aerialTriangulationStatus);
    append(tr("特征匹配"), _tiePointStatus);
    append(tr("生成蒙版"), _maskStatus);
    append(tr("加载影像"), _imageLoadingStatus);
    _dashboard->setTaskSnapshots(tasks);
}
