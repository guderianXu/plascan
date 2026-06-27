// FeatureExtractionDialog.h
// 多算法特征提取配置对话框
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
class QFormLayout;
class QStackedWidget;
class QLabel;

namespace Ui { class FeatureExtractionDialog; }

class FeatureExtractionDialog : public QDialog 
{
    Q_OBJECT
public:
    explicit FeatureExtractionDialog(QWidget *parent = nullptr);
    ~FeatureExtractionDialog() override;

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
    void updateModelPathForCurrentAlgorithm();
    void emitSettingsNow();
    QJsonObject collectSettings() const;

    // 输入输出控件
    QListWidget*      _fileList{nullptr};
    QPushButton*      _addFilesBtn{nullptr};
    QPushButton*      _addFolderBtn{nullptr};
    QPushButton*      _removeBtn{nullptr};
    QPushButton*      _clearBtn{nullptr};
    QLineEdit*        _outputLine{nullptr};
    QPushButton*      _browseOutBtn{nullptr};

    // 算法选择
    QComboBox*        _algorithmCombo{nullptr};
    QFormLayout*      _basicForm{nullptr};
    QWidget*          _cudaRowWidget{nullptr};
    QWidget*          _grayRangeWidget{nullptr};

    // 基础参数 (Basic)
    QSpinBox*         _nmsRadiusSpin{nullptr};
    QDoubleSpinBox*   _detectionThresholdSpin{nullptr};
    QSpinBox*         _maxKeypointsSpin{nullptr};
    QSpinBox*         _removeBordersSpin{nullptr};
    QSpinBox*         _grayscaleMinSpin{nullptr};
    QSpinBox*         _grayscaleMaxSpin{nullptr};

    // 高级参数 (Advanced) - 折叠组
    QGroupBox*        _advancedGroup{nullptr};
    QFormLayout*      _advancedForm{nullptr};
    QLabel*           _advancedHintLabel{nullptr};
    QCheckBox*        _normalizeInputChk{nullptr};
    QSpinBox*         _descriptorDimSpin{nullptr};
    QSpinBox*         _gridSizeSpin{nullptr};
    QSpinBox*         _batchSizeSpin{nullptr};
    // 邻域判断控件
    QSpinBox*         _neighborhoodRadiusSpin{nullptr};
    QDoubleSpinBox*   _neighborhoodThresholdSpin{nullptr};

    // 系统参数 (System) - 折叠组
    QGroupBox*        _systemGroup{nullptr};
    QComboBox*        _deviceCombo{nullptr};
    QCheckBox*        _allowFallbackChk{nullptr};
    QLineEdit*        _pythonPathEdit{nullptr};

    // 调试参数 (Debug) - 折叠组
    QGroupBox*        _debugGroup{nullptr};
    QCheckBox*        _saveCsvChk{nullptr};
    QCheckBox*        _saveOverlayChk{nullptr};

    // 底部按钮
    QPushButton*      _runBtn{nullptr};
    QPushButton*      _cancelBtn{nullptr};
    QPushButton*      _resetBtn{nullptr};

    // ── 多算法参数切换 ──
    QStackedWidget*   _paramStack{nullptr};
    QLineEdit*        _modelPathEdit{nullptr};    // DL 算法模型路径 (共享)
    QCheckBox*        _useCudaChk{nullptr};       // CUDA 开关 (共享)
    QSpinBox*         _cudaDeviceSpin{nullptr};   // CUDA 设备ID (共享)
    // 简单算法页控件
    QSpinBox*         _simpleMaxKpSpin{nullptr};  // SIFT/ORB 最大特征数
    QSpinBox*         _diskMaxKpSpin{nullptr};    // DISK 最大KP
    QDoubleSpinBox*   _diskScoreThreshSpin{nullptr};
    QSpinBox*         _alikedMaxKpSpin{nullptr};
    bool              _applyingSettings{false};
};
