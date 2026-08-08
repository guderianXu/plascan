#pragma once

#include "ProjectTerrainRequests.h"

#include <QDialog>

class QLabel;
class QPushButton;
class QProgressBar;

// CreateDemDialog — 从已有点云生成 DEM，不在 GUI 中隐式启动稠密重建。
class CreateDemDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CreateDemDialog(QWidget *parent = nullptr);

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

    bool _running = false;

    class QLineEdit *_denseEdit = nullptr;

    // 进度区域
    QProgressBar *_progressBar = nullptr;
    QLabel *_stageLabel = nullptr;

    // 按钮
    QPushButton *_runBtn = nullptr;
    QPushButton *_closeBtn = nullptr;
};
