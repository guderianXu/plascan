#pragma once

#include <QDialog>
#include <QJsonArray>
#include <QJsonObject>

class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;

/**
 * @brief 深度图估计对话框。
 *
 * PatchMatch 立体匹配参数设置，包括代价函数、迭代次数、
 * 分辨率缩放、CUDA 加速等。
 */
class DepthMapEstimateDialog : public QDialog
{
    Q_OBJECT
public:
    /// 构造深度图估计参数对话框。
    /// @param parent 父窗口。
    explicit DepthMapEstimateDialog(QWidget *parent = nullptr);

    /// 设置可用 AT 结果（用于选择深度估计所使用的稀疏点云来源）。
    /// @param atResults 项目 AT 结果摘要数组（ProjectManager::getAvailableAtResults）。
    void setAvailableAtResults(const QJsonArray &atResults);

    /// 应用持久化保存的深度图估计配置。
    /// @param settings 配置参数 JSON 对象。
    void applySettings(const QJsonObject &settings);

signals:
    /// 用户确认执行深度图估计时发出。
    /// @param settings 当前收集到的配置参数。
    void runRequested(const QJsonObject &settings);

    /// 参数变化时发出，供外部实时持久化。
    /// @param settings 当前收集到的配置参数。
    void settingsChanged(const QJsonObject &settings);

private slots:
    /// 收集当前参数并发出 settingsChanged 信号。
    void emitSettingsNow();

    /// 响应“运行”按钮点击事件。
    void onRun();

    /// 根据预设档位更新推荐参数。
    /// @param index 当前预设索引。
    void onPresetChanged(int index);

private:
    /// 收集对话框当前配置。
    /// @return 配置参数 JSON 对象。
    QJsonObject collectSettings() const;

    QComboBox      *m_presetCombo        = nullptr;
    QComboBox      *m_atResultCombo      = nullptr;  ///< AT 结果选择（提供稀疏点云来源）
    QDoubleSpinBox *m_resScaleSpin       = nullptr;  ///< 分辨率缩放
    QSpinBox       *m_iterationsSpin     = nullptr;  ///< PatchMatch 迭代
    QComboBox      *m_costFuncCombo      = nullptr;  ///< 代价函数
    QComboBox      *m_propagCombo        = nullptr;  ///< 传播方式
    QSpinBox       *m_patchSizeSpin      = nullptr;  ///< 窗口大小
    QSpinBox       *m_minViewsSpin       = nullptr;  ///< 最少视图数
    QDoubleSpinBox *m_depthMinSpin       = nullptr;
    QDoubleSpinBox *m_depthMaxSpin       = nullptr;
    QDoubleSpinBox *m_confidenceSpin     = nullptr;  ///< 置信度阈值
    QCheckBox      *m_normalMapCheck     = nullptr;  ///< 输出法向量图
    QCheckBox      *m_cudaCheck          = nullptr;
    QSpinBox       *m_tileWSpin          = nullptr;  ///< Tile 宽
    QSpinBox       *m_tileHSpin          = nullptr;  ///< Tile 高
    QSpinBox       *m_threadsSpin        = nullptr;
    QLabel         *m_estimateLabel      = nullptr;
    bool            m_applyingPreset     = false;  ///< 预设应用中，禁止反向切换到"自定义"
    int             m_pendingAtIndex     = -1;     ///< applySettings 先到时的待应用 AT 索引
};
