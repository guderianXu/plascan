#pragma once

#include "ProjectTerrainRequests.h"

#include <QDialog>

class QLabel;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QPushButton;
class QProgressBar;
class QStackedWidget;

// CreateDemDialog — 从点云生成局部 DEM，或从体固连三角网生成全球 DEM/DOM。
class CreateDemDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CreateDemDialog(QWidget *parent = nullptr);
    void reject() override;

    // 流水线进度更新（由外部调用）
    void onPipelineProgress(const QString &stage, int percent);
    void onPipelineFinished(bool success, const QString &message);

signals:
    void requestRun(const xjw::gui::project::DemGenerationRequest &request);
    void requestCancel();

private slots:
    void onBrowseDenseCloud();
    void onBrowseSurface();
    void onRunClicked();

private:
    void setupUi();
    void setRunning(bool running);
    void refreshRunButton();
    void refreshModeUi();
    bool isSmallBodyGlobalMode() const;

    bool _running = false;
    bool _runningCancelable = false;
    bool _cancelRequested = false;

    QComboBox *_modeCombo = nullptr;
    QStackedWidget *_optionsStack = nullptr;
    QLabel *_hintLabel = nullptr;

    QLineEdit *_denseEdit = nullptr;
    QPushButton *_browseDenseBtn = nullptr;

    QLineEdit *_surfaceEdit = nullptr;
    QPushButton *_browseSurfaceBtn = nullptr;
    QComboBox *_surfaceUnitCombo = nullptr;
    QLineEdit *_targetNameEdit = nullptr;
    QLineEdit *_bodyFixedFrameEdit = nullptr;
    QDoubleSpinBox *_angularResolutionSpin = nullptr;
    QDoubleSpinBox *_referenceRadiusSpin = nullptr;
    QCheckBox *_automaticCenterCheck = nullptr;
    QDoubleSpinBox *_centerXSpin = nullptr;
    QDoubleSpinBox *_centerYSpin = nullptr;
    QDoubleSpinBox *_centerZSpin = nullptr;
    QDoubleSpinBox *_centralMeridianSpin = nullptr;

    // 进度区域
    QProgressBar *_progressBar = nullptr;
    QLabel *_stageLabel = nullptr;

    // 按钮
    QPushButton *_runBtn = nullptr;
    QPushButton *_closeBtn = nullptr;
};
