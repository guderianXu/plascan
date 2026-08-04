#pragma once

#include <QObject>

class ProjectDashboardWidget;
class ProjectManager;
class QStatusBar;
class TaskStatusWidget;
class QWidget;

class ProjectTaskStatusController final : public QObject
{
    Q_OBJECT

public:
    explicit ProjectTaskStatusController(ProjectManager *projectManager,
                                         ProjectDashboardWidget *dashboard,
                                         QStatusBar *statusBar,
                                         QWidget *widgetParent,
                                         QObject *parent = nullptr);

public slots:
    void showTiePointProgress(int total);
    void updateTiePointProgress(int done);
    void finishTiePointProgress(bool success);
    void updateImageLoading(const QString &stage, int done, int total);
    void finishImageLoading(bool success, const QString &message = QString());

signals:
    void tiePointCancelRequested();

private slots:
    void updateMesh(const QString &stage, int percent);
    void finishMesh(bool success);
    void updatePointCloud(const QString &stage, int percent);
    void finishPointCloud(bool success);
    void updateAerialTriangulation(const QString &stage, int percent);
    void finishAerialTriangulation(bool success);
    void updateMask(const QString &stage, int done, int total);
    void finishMask(bool success);

private:
    TaskStatusWidget *createStatus(int labelWidth,
                                   const QString &cancellingText,
                                   QWidget *widgetParent);
    void refreshDashboard();

    ProjectManager *_projectManager = nullptr;
    ProjectDashboardWidget *_dashboard = nullptr;
    QStatusBar *_statusBar = nullptr;
    TaskStatusWidget *_meshStatus = nullptr;
    TaskStatusWidget *_pointCloudStatus = nullptr;
    TaskStatusWidget *_aerialTriangulationStatus = nullptr;
    TaskStatusWidget *_tiePointStatus = nullptr;
    TaskStatusWidget *_maskStatus = nullptr;
    TaskStatusWidget *_imageLoadingStatus = nullptr;
};
