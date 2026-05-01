// =============================================================================
// 文件: ModelGenerationDialog.h
// 模块: GUI / 对话框
// 说明:
//   模型生成参数对话框。
//
//   功能:
//     - 选择重建精度预设（低/中/高）→ 控制视差数、网格分辨率
//     - 启用/禁用 CUDA 加速
//     - 配置输出颜色纹理和法向量
//     - 输出设置收集为 QJsonObject，供 startGenerateModelAsync 使用
// =============================================================================
#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QCheckBox;
class QSpinBox;
class QDoubleSpinBox;
class QLabel;
class QGroupBox;
class QPushButton;

class ModelGenerationDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ModelGenerationDialog(QWidget *parent = nullptr);

    /// 将当前 UI 参数收集为 JSON
    QJsonObject collectSettings() const;

    /// 用已保存的设置初始化 UI（用于持久化记忆）
    void applySettings(const QJsonObject &settings);

signals:
    /// 用户点击"确定"后发出，携带收集好的参数
    void accepted(const QJsonObject &settings);

private slots:
    void onPresetChanged(int index);
    void onOk();
    void onCancel();

private:
    void setupUi();
    void applyPreset(int preset); // 0=低 1=中 2=高

    // ── 控件 ──────────────────────────────────────────────────────────────
    QComboBox   *m_presetCombo      = nullptr; ///< 重建精度预设：低/中/高
    QComboBox   *m_outputFormatCombo = nullptr; ///< 最终输出格式
    QSpinBox    *m_disparitySpin    = nullptr; ///< 视差搜索范围（SGBM numDisparities）
    QSpinBox    *m_gridResSpin      = nullptr; ///< 地形网格分辨率（grid width）
    QSpinBox    *m_meshResSpin      = nullptr; ///< 网格重建体素分辨率
    QSpinBox    *m_meshSmoothIterSpin = nullptr; ///< 网格平滑迭代次数
    QDoubleSpinBox *m_meshSmoothLambdaSpin = nullptr; ///< 网格平滑强度
    QDoubleSpinBox *m_meshPaddingSpin = nullptr; ///< 重建包围盒 padding
    QCheckBox   *m_useCudaCheck     = nullptr; ///< 使用 CUDA 立体匹配加速
    QCheckBox   *m_outputColorCheck = nullptr; ///< 输出颜色（XYZRGB）
    QCheckBox   *m_outputNormalCheck= nullptr; ///< 输出法向量
    QLabel      *m_presetDescLabel  = nullptr; ///< 预设说明文字
    QLabel      *m_runtimeHintLabel = nullptr; ///< 运行时间提示

    QPushButton *m_okBtn     = nullptr;
    QPushButton *m_cancelBtn = nullptr;
};
