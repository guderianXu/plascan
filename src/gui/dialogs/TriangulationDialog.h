#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QPushButton;
class QLabel;

/**
 * @brief 三角化生成初始稀疏点云对话框。
 *
 * 根据已恢复的相机位姿和匹配点，三角化生成三维点，
 * 支持低质量点过滤和基于焦距/基线的动态阈值建议。
 */
class TriangulationDialog : public QDialog
{
    Q_OBJECT
public:
    /// 构造三角化参数对话框。
    /// @param parent 父窗口。
    explicit TriangulationDialog(QWidget *parent = nullptr);

    /// 应用持久化保存的三角化配置。
    /// @param settings 配置参数 JSON 对象。
    void applySettings(const QJsonObject &settings);

signals:
    /// 用户确认执行三角化时发出。
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

    /// 根据焦距和基线推荐阈值参数。
    void onSuggestThresholds();

private:
    /// 收集对话框当前配置。
    /// @return 配置参数 JSON 对象。
    QJsonObject collectSettings() const;

    QComboBox      *m_presetCombo          = nullptr;
    QDoubleSpinBox *m_minAngleSpin         = nullptr;  ///< 最小交会角
    QDoubleSpinBox *m_reprojThreshSpin     = nullptr;  ///< 重投影阈值
    QSpinBox       *m_minObsSpin           = nullptr;  ///< 最少观测数
    QCheckBox      *m_ignoreTwoViewCheck   = nullptr;
    QDoubleSpinBox *m_depthStabSpin        = nullptr;  ///< 深度稳定性
    QComboBox      *m_filterModeCombo      = nullptr;
    QDoubleSpinBox *m_maxReprojErrSpin     = nullptr;
    QDoubleSpinBox *m_minAngleFiltSpin     = nullptr;
    QSpinBox       *m_minTrackLenSpin      = nullptr;
    QSpinBox       *m_threadsSpin          = nullptr;
    QDoubleSpinBox *m_focalLenSpin         = nullptr;
    QDoubleSpinBox *m_baselineSpin         = nullptr;
    QCheckBox      *m_overwriteResultCheck = nullptr;
    QPushButton    *m_suggestBtn           = nullptr;
    QLabel         *m_suggestLabel         = nullptr;
};
