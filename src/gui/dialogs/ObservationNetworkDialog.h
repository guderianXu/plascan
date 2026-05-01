#pragma once
// =============================================================================
// 文件: ObservationNetworkDialog.h
// 模块: gui/dialogs
// 说明:
//   构建观测网络模型参数配置与可视化对话框。
//
//   界面布局:
//     标签页 1 "参数": 预设档位 + 算法参数 + "构建预览"按钮
//     标签页 2 "网络图": ObservationNetworkView + 统计标签 + 力导向布局按钮
// =============================================================================

#include <QDialog>
#include <QJsonObject>
#include <QVector>
#include <QStringList>

#include "graph/ObservationNetworkBuilder.h"   // xjw::MatchEdge / GpsCoord / ObservationNetworkConfig

class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;
class QTabWidget;
class ObservationNetworkView;

class ObservationNetworkDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ObservationNetworkDialog(QWidget *parent = nullptr);

    /**
     * @brief 注入项目匹配数据（在 exec()/show() 前调用）。
     * @param edges      从 ipmatch_results 解析出的边列表
     * @param imageNames 图像名称列表（与 MatchEdge::idx 对应）
     * @param gps        可选 GPS 坐标（KDTree 算法使用）
     */
    void setMatchEdges(const QVector<xjw::MatchEdge>  &edges,
                       const QStringList              &imageNames,
                       const QVector<xjw::GpsCoord>   &gps = {});

    void applySettings(const QJsonObject &settings);

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private slots:
    void emitSettingsNow();
    void onPresetChanged(int idx);
    void onPreview();
    void onRun();

private:
    QJsonObject collectSettings() const;
    xjw::ObservationNetworkConfig buildConfig() const;

    // ── 标签页 ──
    QTabWidget              *m_tabs       = nullptr;
    ObservationNetworkView  *m_netView    = nullptr;
    QLabel                  *m_statsLabel = nullptr;

    // ── 参数控件 ──
    QComboBox      *m_presetCombo        = nullptr;  ///< 预设档位
    QComboBox      *m_graphAlgoCombo     = nullptr;  ///< 连接图算法
    QSpinBox       *m_maxNeighborsSpin   = nullptr;  ///< 最大邻居数
    QSpinBox       *m_minMatchCountSpin  = nullptr;  ///< 最少匹配数
    QDoubleSpinBox *m_minOverlapSpin     = nullptr;  ///< 最小重叠率
    QComboBox      *m_verifyMethodCombo  = nullptr;  ///< 几何验证方法
    QDoubleSpinBox *m_verifyThreshSpin   = nullptr;  ///< 几何验证阈值
    QCheckBox      *m_pruneWeakCheck     = nullptr;  ///< 剪枝弱连接
    QDoubleSpinBox *m_pruneThreshSpin    = nullptr;  ///< 剪枝阈值
    QSpinBox       *m_threadsSpin        = nullptr;

    // ── 注入数据 ──
    QVector<xjw::MatchEdge>  m_matchEdges;
    QStringList              m_imageNames;
    QVector<xjw::GpsCoord>   m_gpsCoords;
};
