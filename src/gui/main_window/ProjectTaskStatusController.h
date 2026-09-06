#pragma once

#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>

class ProjectDashboardWidget;
class ProjectManager;
class QStatusBar;
class TaskStatusWidget;
class QWidget;

namespace xjw::gui::platform
{
    class TaskbarProgressController;
}

class ProjectTaskStatusController final : public QObject
{
    Q_OBJECT

public:
    explicit ProjectTaskStatusController(ProjectManager* projectManager,
                                         ProjectDashboardWidget* dashboard,
                                         QStatusBar* statusBar,
                                         QWidget* widgetParent,
                                         QObject* parent = nullptr);

public slots:
    void showTiePointProgress(int total);
    void updateTiePointProgress(int done);
    void finishTiePointProgress(bool success);
    void updateImageLoading(const QString& stage, int done, int total);
    void finishImageLoading(bool success, const QString& message = QString());
    void clearTaskHistory();
    void setSchedulerTaskSnapshots(const QJsonArray& snapshots);

signals:
    void tiePointCancelRequested();

private slots:
    void updateMesh(const QString& stage, int percent);
    void finishMesh(bool success);
    void updatePointCloud(const QString& stage, int percent);
    void finishPointCloud(bool success);
    void updateAerialTriangulation(const QString& stage, int percent);
    void updateAerialTriangulationDevice(const QString& displayName);
    void finishAerialTriangulation(bool success);
    void updateMask(const QString& stage, int done, int total);
    void finishMask(bool success);
    void updateImageImport(const QString& stage, int done, int total);
    void finishImageImport(bool success, const QString& message);

private:
    TaskStatusWidget*
    createStatus(const QString& objectName, int labelWidth, const QString& cancellingText, QWidget* widgetParent);
    void updatePercentTask(TaskStatusWidget* status,
                           const QString& stage,
                           int percent,
                           bool appendIntermediatePercent,
                           const QString& taskId);
    void finishTask(TaskStatusWidget* status,
                    bool success,
                    const QString& successMessage,
                    const QString& failureMessage,
                    const QString& taskId);
    void
    updateImageLoadingTask(TaskStatusWidget* status, const QString& taskId, const QString& stage, int done, int total);
    void finishImageLoadingTask(TaskStatusWidget* status, const QString& taskId, bool success, const QString& message);
    void resetTaskProgress();
    void refreshDashboard();
    void beginTaskActivity(const QString& taskId, const QString& name, const QString& stage);
    void updateTaskActivity(const QString& taskId, const QString& stage);
    void
    finishTaskActivity(const QString& taskId, bool success, bool cancelled, qint64 elapsedMs, const QString& summary);

    ProjectManager* _projectManager = nullptr;
    ProjectDashboardWidget* _dashboard = nullptr;
    QStatusBar* _statusBar = nullptr;
    xjw::gui::platform::TaskbarProgressController* _taskbarProgress = nullptr;
    TaskStatusWidget* _meshStatus = nullptr;
    TaskStatusWidget* _pointCloudStatus = nullptr;
    TaskStatusWidget* _aerialTriangulationStatus = nullptr;
    TaskStatusWidget* _tiePointStatus = nullptr;
    TaskStatusWidget* _maskStatus = nullptr;
    TaskStatusWidget* _imageImportStatus = nullptr;
    TaskStatusWidget* _photoListStatus = nullptr;
    QHash<QString, QJsonObject> _activeTaskRuns;
    QJsonArray _taskHistory;
    QJsonArray _schedulerTaskSnapshots;
};
