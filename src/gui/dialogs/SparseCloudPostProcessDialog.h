#pragma once

#include <QDialog>
#include <QJsonArray>
#include <QJsonObject>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;

class SparseCloudPostProcessDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SparseCloudPostProcessDialog(QWidget *parent = nullptr);

    void setAvailableSparseClouds(const QJsonArray &results);
    void applySettings(const QJsonObject &settings);

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private:
    void setupUi();
    void onAnyChanged();
    void onRun();
    void applyPendingSourceSelection();
    void updateStatsLabel();
    void updateSourceModeUi();
    void updateRunButtonState();
    void browseExternalPly();
    bool usingExternalPly() const;
    QJsonObject collectSettings() const;

    // 输入源
    QComboBox *_sourceModeCombo = nullptr;
    QComboBox *_sourceCombo = nullptr;
    QLineEdit *_externalPathEdit = nullptr;
    QPushButton *_browseExternalButton = nullptr;
    QLabel *_statsLabel = nullptr;
    QJsonArray _availableResults;
    int _pendingSourceIdx = -1;
    bool _programmaticUpdate = false;

    // 点级滤波
    QCheckBox *_reprojCheck = nullptr;
    QDoubleSpinBox *_reprojSpin = nullptr;
    QCheckBox *_trackCheck = nullptr;
    QSpinBox *_trackSpin = nullptr;
    QCheckBox *_angleCheck = nullptr;
    QDoubleSpinBox *_angleSpin = nullptr;
    QCheckBox *_statCheck = nullptr;
    QSpinBox *_statKSpin = nullptr;
    QDoubleSpinBox *_statStdSpin = nullptr;
    QCheckBox *_densityCheck = nullptr;
    QDoubleSpinBox *_densityRadiusSpin = nullptr;
    QSpinBox *_densityMinNbSpin = nullptr;

    // 迭代精修（可选 GroupBox）
    QGroupBox *_refineGroup = nullptr;
    QSpinBox *_iterRoundsSpin = nullptr;
    QCheckBox *_retriangCheck = nullptr;
    QCheckBox *_normalConsCheck = nullptr;
    QSpinBox *_threadsSpin = nullptr;

    // 空间清理（可选 GroupBox）
    QGroupBox *_spatialGroup = nullptr;
    QDoubleSpinBox *_voxelSizeSpin = nullptr;
    QSpinBox *_minVoxelPtsSpin = nullptr;
    QCheckBox *_localReprojCheck = nullptr;
    QDoubleSpinBox *_reprojStdMulSpin = nullptr;
    QDoubleSpinBox *_dedupRadiusSpin = nullptr;

    QPushButton *_runButton = nullptr;
};
