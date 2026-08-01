#pragma once

#include "ProjectTerrainRequests.h"

#include <QDialog>
#include <QStringList>

class QLabel;
class QPushButton;
class QListWidget;
class QProgressBar;
class QStackedWidget;
class ProjectManager;

// CreateDemDialog — 从已有点云生成 DEM，不在 GUI 中隐式启动稠密重建。
class CreateDemDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CreateDemDialog(ProjectManager *projectManager, QWidget *parent = nullptr);

    void setAvailableImages(const QStringList &images);
    void setDefaultOutput(const QString &outputDir);

    // 流水线进度更新（由外部调用）
    void onPipelineProgress(const QString &stage, int percent);
    void onPipelineFinished(bool success, const QString &message);

signals:
    void requestRun(const xjw::gui::project::DemGenerationRequest &request);

private slots:
    void onBrowseDenseCloud();
    void onRunClicked();

private:
    void setupUi();
    void setRunning(bool running);
    void refreshRunButton();

    ProjectManager *_projectManager = nullptr;
    QStringList _availableImages;
    bool _running = false;

    // 模式切换
    QPushButton *_autoModeBtn = nullptr;
    QPushButton *_manualModeBtn = nullptr;
    QStackedWidget *_modeStack = nullptr;

    class QLineEdit *_denseEdit = nullptr;

    // 进度区域
    QProgressBar *_progressBar = nullptr;
    QLabel *_stageLabel = nullptr;

    // 按钮
    QPushButton *_runBtn = nullptr;
    QPushButton *_closeBtn = nullptr;
};
