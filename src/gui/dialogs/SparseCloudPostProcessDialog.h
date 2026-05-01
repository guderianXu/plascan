#pragma once

#include <QDialog>
#include <QJsonArray>
#include <QJsonObject>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
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
    QJsonObject collectSettings() const;

    // 输入源
    QComboBox  *m_sourceCombo      = nullptr;
    QLabel     *m_statsLabel       = nullptr;
    QJsonArray  m_availableResults;
    int         m_pendingSourceIdx = -1;

    // 点级滤波
    QCheckBox      *m_reprojCheck      = nullptr;
    QDoubleSpinBox *m_reprojSpin       = nullptr;
    QCheckBox      *m_trackCheck       = nullptr;
    QSpinBox       *m_trackSpin        = nullptr;
    QCheckBox      *m_angleCheck       = nullptr;
    QDoubleSpinBox *m_angleSpin        = nullptr;
    QCheckBox      *m_statCheck        = nullptr;
    QSpinBox       *m_statKSpin        = nullptr;
    QDoubleSpinBox *m_statStdSpin      = nullptr;
    QCheckBox      *m_densityCheck     = nullptr;
    QDoubleSpinBox *m_densityRadiusSpin = nullptr;
    QSpinBox       *m_densityMinNbSpin  = nullptr;

    // 迭代精修（可选 GroupBox）
    QGroupBox  *m_refineGroup     = nullptr;
    QSpinBox   *m_iterRoundsSpin  = nullptr;
    QCheckBox  *m_retriangCheck   = nullptr;
    QCheckBox  *m_normalConsCheck = nullptr;
    QSpinBox   *m_threadsSpin     = nullptr;

    // 空间清理（可选 GroupBox）
    QGroupBox      *m_spatialGroup      = nullptr;
    QDoubleSpinBox *m_voxelSizeSpin     = nullptr;
    QSpinBox       *m_minVoxelPtsSpin   = nullptr;
    QCheckBox      *m_localReprojCheck  = nullptr;
    QDoubleSpinBox *m_reprojStdMulSpin  = nullptr;
    QDoubleSpinBox *m_dedupRadiusSpin   = nullptr;

    QPushButton *m_runButton = nullptr;
};