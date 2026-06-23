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
    void setAvailableFeatureSuffixes(const QStringList &suffixes);

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
    void onMatchPipelineChanged();

private:
    QJsonObject collectSettings() const;
    QString selectedFeatureSuffix() const;
    QString selectedFeatureAlgorithm() const;
    void refreshFeatureSuffixChoices();
    void updateStatusText();
    void updateTargetUi();

    // ── 模式选择 ──
    QComboBox *_modeCombo = nullptr;
    QStackedWidget *_modeStack = nullptr;
    QLabel *_statusLabel = nullptr;

    // ── 应用范围 ──
    QGroupBox *_applyBox = nullptr;
    QFormLayout *_applyForm = nullptr;
    QComboBox *_applyScopeCombo = nullptr;
    QComboBox *_applyTargetImageCombo = nullptr;
    QCheckBox *_overwriteExistingCheck = nullptr;
    QLabel *_applyHintLabel = nullptr;
    QComboBox *_qualityCombo = nullptr;
    QSpinBox *_threadsSpin = nullptr;
    QComboBox *_matchAlgorithmCombo = nullptr;
    QComboBox *_featureSuffixCombo = nullptr;
    QStringList _projectFeatureSuffixes;

    // ── 模式 1: 无相机文件 ──
    QCheckBox *_exifAutoCheck = nullptr;  ///< 自动读取 EXIF
    QDoubleSpinBox *_defaultFocalSpin = nullptr;  ///< 默认焦距 (mm)
    QDoubleSpinBox *_sensorWidthSpin = nullptr;  ///< 传感器宽度 (mm)

    // ── 模式 2: 仅有内参 ──
    QDoubleSpinBox *_fxSpin = nullptr;
    QDoubleSpinBox *_fySpin = nullptr;
    QDoubleSpinBox *_cxSpin = nullptr;
    QDoubleSpinBox *_cySpin = nullptr;
    QComboBox *_distModelCombo = nullptr;  ///< 畸变模型
    QFormLayout *_intrinsicsForm = nullptr;
    QDoubleSpinBox *_k1Spin = nullptr;
    QDoubleSpinBox *_k2Spin = nullptr;
    QDoubleSpinBox *_p1Spin = nullptr;
    QDoubleSpinBox *_p2Spin = nullptr;

    // ── 模式 3: 完整相机文件 ──
    QComboBox *_cameraImportModeCombo = nullptr;
    QComboBox *_targetImageCombo = nullptr;
    QComboBox *_cameraFormatCombo = nullptr;  ///< 文件格式
    QLabel *_cameraImportHintLabel = nullptr;
    QFormLayout *_cameraImportForm = nullptr;
};
