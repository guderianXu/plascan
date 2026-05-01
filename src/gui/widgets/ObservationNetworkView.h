#pragma once
// =============================================================================
// 文件: ObservationNetworkView.h
// 模块: gui/widgets
// 说明:
//   观测网络可视化控件（基于 QGraphicsView）。
//
//   布局策略:
//     1. 初始布局：节点均匀排布在圆上。
//     2. Force-directed 精化（Fruchterman-Reingold，50 次迭代，
//        由 QTimer 驱动以保持 UI 响应）。可手动触发。
//
//   视觉设计:
//     - 节点：圆形；半径 ∝ 度数；颜色按度数着色（蓝→绿→橙）
//     - 边  ：直线；线宽/透明度 ∝ 匹配数（权重）
//     - 悬停：显示节点名称 tooltip
// =============================================================================

#include <QGraphicsView>
#include <QColor>
#include <QPointF>
#include <QVector>
#include <QTimer>

// 引入算法数据结构（仅头文件，无链接依赖，算法库单独链接）
#include "graph/ObservationNetworkBuilder.h"

class QGraphicsScene;

/**
 * @brief 观测网络可视化控件。
 *
 * 典型使用:
 * @code
 *   ObservationNetworkView *view = new ObservationNetworkView(this);
 *   view->setNetwork(network);
 *   view->startForceLayout();  // 可选，触发力导向精化
 * @endcode
 */
class ObservationNetworkView : public QGraphicsView
{
    Q_OBJECT
public:
    explicit ObservationNetworkView(QWidget *parent = nullptr);
    ~ObservationNetworkView() override;

    /// 加载网络数据并刷新显示（圆形初始布局）
    void setNetwork(const xjw::ObservationNetwork &net);

    /// 触发 Fruchterman-Reingold 力导向布局精化（异步，不阻塞 UI）
    void startForceLayout();

    /// 重置到圆形布局
    void resetLayout();

    /// 清空显示
    void clearNetwork();

    /// 导出当前视图为图片
    QImage toImage(QSize size = {800, 600}) const;

signals:
    /// 用户点击节点时发射，携带节点名称和索引
    void nodeClicked(int index, const QString &name);

    /// 力导向布局完成
    void forceLayoutDone();

private slots:
    void onForceStep();

private:
    /**
     * @brief 根据网络规模选择初始布局。
     *
     * 小规模网络使用圆形布局；大规模网络使用按节点度数分层的多环布局，
     * 以减少拥挤并突出高连接节点。
     */
    void chooseInitialLayout();

    /**
     * @brief 按连通分量分区布局整个网络。
     *
     * 不同连通分量会被布置到相互分离的区域中，避免大量节点全部挤在同一块区域。
     */
    void componentAwareLayout();

    /**
     * @brief 按节点度数执行多环布局。
     *
     * 高度节点优先放置在中心环，低度节点向外环分布，适合大量影像时的
     * 总体关系浏览。
     */
    void multiRingLayout();

    /**
     * @brief 为指定连通分量执行局部多环布局。
     *
     * @param componentNodeIndices 连通分量内的节点索引集合
     * @param componentCenter      该分量在场景中的中心点
     * @param componentRadius      该分量可使用的布局半径
     */
    void layoutComponentNodes(const QVector<int> &componentNodeIndices,
                              const QPointF &componentCenter,
                              double componentRadius);

    /**
     * @brief 计算网络的连通分量。
     */
    QVector<QVector<int>> computeConnectedComponents() const;

    /**
     * @brief 对节点执行碰撞松弛，减轻圆点堆叠。
     *
     * @param maxIterations 迭代次数
     * @param padding       节点之间的最小额外间距
     */
    void relaxNodeOverlaps(int maxIterations, double padding);

    /**
     * @brief 重建边、标签和高亮相关的渲染缓存。
     */
    void rebuildRenderCache();

    /**
     * @brief 返回节点显示半径。
     *
     * @param nodeIndex 节点索引
     * @param maxDegree 当前网络最大度数
     * @return 节点半径（场景坐标）
     */
    double nodeRadius(int nodeIndex, int maxDegree) const;

    /**
     * @brief 判断当前是否启用大规模高性能显示模式。
     */
    bool isHighPerformanceMode() const;

    /**
     * @brief 判断当前缩放级别下是否应显示标签。
     */
    bool shouldDrawLabels() const;

    /**
     * @brief 根据场景坐标拾取最近节点。
     *
     * @param scenePos 场景坐标
     * @return 命中的节点索引，未命中时返回 -1
     */
    int pickNodeAt(const QPointF &scenePos) const;

    /**
     * @brief 返回当前视图缩放比例。
     */
    double currentViewScale() const;

    // ── 布局 ──
    void buildScene();
    void applyPositions();
    void circularLayout();
    QRectF networkBounds() const;
    void fitNetworkInView();
    void drawBackground(QPainter *painter, const QRectF &rect) override;
    void drawForeground(QPainter *painter, const QRectF &rect) override;
    QColor nodeColor(int degree, int maxDeg) const;
    QColor edgeColor(double weight) const;

    // ── 场景对象 ──
    QGraphicsScene *m_scene = nullptr;

    // ── 数据 ──
    xjw::ObservationNetwork m_net;
    QVector<QPointF> m_pos;    ///< 当前节点位置（场景坐标）
    QVector<double> m_nodeRadii;
    QVector<int> m_visibleEdgeIndices;
    QVector<int> m_visibleLabelIndices;
    QVector<QVector<int>> m_nodeEdgeAdjacency;
    static constexpr double LAYOUT_AREA_SIZE = 500.0; ///< 虚拟场景边长（用于 FR 算法）

    // ── 力导向 ──
    QTimer *m_forceTimer = nullptr;
    int m_forceIter = 0;
    double m_temp = 0.0;
    static constexpr int MAX_FORCE_LAYOUT_ITERATIONS = 80;
    static constexpr double FORCE_LAYOUT_COOL_RATE = 0.92;
    bool m_autoFitPending = false;
    int m_selectedNodeIndex = -1;

    // ── 交互 ──
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
};
