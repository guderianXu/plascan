// SuperPointDialog.h
// SuperPoint 特征点提取配置对话框
// 提供基础、高级、系统、调试参数的分组配置界面

#pragma once

#include <QDialog>
#include <QJsonObject>

class QListWidget;
class QLineEdit;
class QPushButton;
class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QGroupBox;

namespace Ui { class SuperPointDialog; }

class SuperPointDialog : public QDialog 
{
    Q_OBJECT
public:
    explicit SuperPointDialog(QWidget *parent = nullptr);
    ~SuperPointDialog() override;

signals:
    // 用户点击"运行"时发出
    // 对话框只负责收集参数与输入文件列表，不直接执行推理
    // 上层（ProjectManager）收到后负责异步执行和持久化
    void runRequested(const QJsonObject &config, const QStringList &inputs);
    
    // 当参数被修改时立即发出（用于项目配置持久化）
    void settingsChanged(const QJsonObject &settings);

private slots:
    void onAddFiles();
    void onAddFolder();
    void onRemoveSelected();
    void onClearFiles();
    void onBrowseOutput();
    void onRun();
    void onCancel();
    void onResetDefaults();
    void onAlgorithmChanged(int index);

public slots:
    // 将一组设置应用到对话控件（由上层调用）
    void applySettings(const QJsonObject &settings);
    
    // 将项目中已有的影像路径传入对话框，并在列表中以复选框形式展示
    void setProjectImages(const QStringList &paths);

private:
    void setupUi();
    void setupConnections();
    void updatePreview();
    void emitSettingsNow();
    QJsonObject collectSettings() const;

    // 输入输出控件
    QListWidget*      m_fileList{nullptr};
    QPushButton*      m_addFilesBtn{nullptr};
    QPushButton*      m_addFolderBtn{nullptr};
    QPushButton*      m_removeBtn{nullptr};
    QPushButton*      m_clearBtn{nullptr};
    QLineEdit*        m_outputLine{nullptr};
    QPushButton*      m_browseOutBtn{nullptr};

    // 算法选择
    QComboBox*        m_algorithmCombo{nullptr};

    // 基础参数 (Basic)
    QSpinBox*         m_nmsRadiusSpin{nullptr};
    QDoubleSpinBox*   m_detectionThresholdSpin{nullptr};
    QSpinBox*         m_maxKeypointsSpin{nullptr};
    QSpinBox*         m_removeBordersSpin{nullptr};
    QDoubleSpinBox*   m_grayscaleMinSpin{nullptr};
    QDoubleSpinBox*   m_grayscaleMaxSpin{nullptr};

    // 高级参数 (Advanced) - 折叠组
    QGroupBox*        m_advancedGroup{nullptr};
    QCheckBox*        m_normalizeInputChk{nullptr};
    QSpinBox*         m_descriptorDimSpin{nullptr};
    QSpinBox*         m_gridSizeSpin{nullptr};
    QSpinBox*         m_batchSizeSpin{nullptr};
    // 邻域判断控件
    QSpinBox*         m_neighborhoodRadiusSpin{nullptr};
    QDoubleSpinBox*   m_neighborhoodThresholdSpin{nullptr};

    // 系统参数 (System) - 折叠组
    QGroupBox*        m_systemGroup{nullptr};
    QComboBox*        m_deviceCombo{nullptr};
    QCheckBox*        m_allowFallbackChk{nullptr};

    // 调试参数 (Debug) - 折叠组
    QGroupBox*        m_debugGroup{nullptr};
    QCheckBox*        m_saveCsvChk{nullptr};
    QCheckBox*        m_saveOverlayChk{nullptr};

    // 底部按钮
    QPushButton*      m_runBtn{nullptr};
    QPushButton*      m_cancelBtn{nullptr};
    QPushButton*      m_resetBtn{nullptr};
};
