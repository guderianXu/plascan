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
    QTabWidget *_tabs = nullptr;
    ObservationNetworkView *_netView = nullptr;
    QLabel *_statsLabel = nullptr;

    // ── 参数控件 ──
    QComboBox *_presetCombo = nullptr;             ///< 预设档位
    QComboBox *_graphAlgoCombo = nullptr;          ///< 连接图算法
    QSpinBox *_maxNeighborsSpin = nullptr;         ///< 最大邻居数
    QSpinBox *_minMatchCountSpin = nullptr;        ///< 最少匹配数
    QDoubleSpinBox *_minOverlapSpin = nullptr;     ///< 最小重叠率
    QComboBox *_verifyMethodCombo = nullptr;       ///< 几何验证方法
    QDoubleSpinBox *_verifyThreshSpin = nullptr;   ///< 几何验证阈值
    QCheckBox *_pruneWeakCheck = nullptr;          ///< 剪枝弱连接
    QDoubleSpinBox *_pruneThreshSpin = nullptr;    ///< 剪枝阈值
    QSpinBox *_threadsSpin = nullptr;

    // ── 注入数据 ──
    QVector<xjw::MatchEdge> _matchEdges;
    QStringList _imageNames;
    QVector<xjw::GpsCoord> _gpsCoords;
};
