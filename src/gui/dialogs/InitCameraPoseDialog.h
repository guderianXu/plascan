#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

class QFormLayout;
class QGroupBox;

class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QStackedWidget;
class QLabel;

/**
 * @brief 初始化相机位姿对话框。
 *
 * 支持三种模式：
 *   1. 无相机文件 — 从 EXIF 提取焦距 + 传感器数据库查询
 *   2. 仅有内参 — 导入内参矩阵 + 畸变模型
 *   3. 有完整相机文件 — 导入 .tsai / .yaml / .xml
 */
class InitCameraPoseDialog : public QDialog
{
    Q_OBJECT
public:
    explicit InitCameraPoseDialog(QWidget *parent = nullptr);
    void applySettings(const QJsonObject &settings);
    void setAvailableImages(const QStringList &imagePaths);

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private slots:
    void emitSettingsNow();
    void onRun();
    void onModeChanged(int index);
    void onInitTargetModeChanged(int index);
    void onCameraImportModeChanged(int index);
    void onDistortionModelChanged(int index);

private:
    QJsonObject collectSettings() const;
    void updateStatusText();
    void updateTargetUi();

    // ── 模式选择 ──
    QComboBox      *m_modeCombo          = nullptr;
    QStackedWidget *m_modeStack          = nullptr;
    QLabel         *m_statusLabel        = nullptr;

    // ── 应用范围 ──
    QGroupBox      *m_applyBox              = nullptr;
    QFormLayout    *m_applyForm             = nullptr;
    QComboBox      *m_applyScopeCombo       = nullptr;
    QComboBox      *m_applyTargetImageCombo = nullptr;
    QCheckBox      *m_overwriteExistingCheck = nullptr;
    QLabel         *m_applyHintLabel        = nullptr;
    QComboBox      *m_qualityCombo          = nullptr;
    QSpinBox       *m_threadsSpin           = nullptr;

    // ── 模式 1: 无相机文件 ──
    QCheckBox      *m_exifAutoCheck      = nullptr;  ///< 自动读取 EXIF
    QDoubleSpinBox *m_defaultFocalSpin   = nullptr;  ///< 默认焦距 (mm)
    QDoubleSpinBox *m_sensorWidthSpin    = nullptr;  ///< 传感器宽度 (mm)

    // ── 模式 2: 仅有内参 ──
    QDoubleSpinBox *m_fxSpin             = nullptr;
    QDoubleSpinBox *m_fySpin             = nullptr;
    QDoubleSpinBox *m_cxSpin             = nullptr;
    QDoubleSpinBox *m_cySpin             = nullptr;
    QComboBox      *m_distModelCombo     = nullptr;  ///< 畸变模型
    QFormLayout    *m_intrinsicsForm     = nullptr;
    QDoubleSpinBox *m_k1Spin             = nullptr;
    QDoubleSpinBox *m_k2Spin             = nullptr;
    QDoubleSpinBox *m_p1Spin             = nullptr;
    QDoubleSpinBox *m_p2Spin             = nullptr;

    // ── 模式 3: 完整相机文件 ──
    QComboBox      *m_cameraImportModeCombo = nullptr;
    QComboBox      *m_targetImageCombo      = nullptr;
    QComboBox      *m_cameraFormatCombo  = nullptr;  ///< 文件格式
    QLabel         *m_cameraImportHintLabel = nullptr;
    QFormLayout    *m_cameraImportForm      = nullptr;
};
