#pragma once

#include <QWidget>
#include <QTransform>
#include <QRectF>
#include <QPointF>
#include <QVector>

class QGraphicsView;
class QGraphicsScene;
class QGraphicsPixmapItem;
class QGraphicsEllipseItem;

// ImageViewWidget: 单个独立的图像查看视图
// 职责：
// - 显示图像并支持缩放、平移
// - 显示匹配点（作为小圆点）
// - 提供视图变换信号供外部同步
// - 支持鼠标交互（悬停高亮）
class ImageViewWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ImageViewWidget(QWidget *parent = nullptr);
    ~ImageViewWidget() override;

    // 加载图像
    bool loadImage(const QString &imagePath);
    
    // 设置匹配点（场景坐标）
    void setMatchPoints(const QVector<QPointF> &points);
    
    // 清除匹配点
    void clearMatchPoints();
    
    // 缩放控制
    void zoomIn();
    void zoomOut();
    void zoomTo(qreal factor);
    void fitToView();
    void resetZoom();
    
    // 获取当前可见区域（场景坐标）
    QRectF visibleSceneRect() const;
    
    // 获取当前变换矩阵
    QTransform currentTransform() const;
    
    // 设置变换矩阵（用于同步）
    void setTransform(const QTransform &transform);
    
    // 获取视图组件（用于坐标转换）
    QGraphicsView* view() const { return m_view; }
    
    // 高亮指定索引的匹配点
    void highlightPoint(int index);
    void clearHighlight();
    
    // 根据掩码设置匹配点的可见性（与覆盖层可见匹配同步）
    void setMatchVisibilityMask(const QVector<bool> &mask);
    
    // 获取当前图像路径
    QString imagePath() const { return m_imagePath; }

    // 返回当前场景中点图元数量（用于外部同步可见性掩码的大小）
    int matchItemCount() const { return m_pointItems.size(); }

signals:
    // 视图变换变化（缩放、平移）
    void viewTransformChanged(const QTransform &transform);
    
    // 可见区域变化
    void visibleRectChanged(const QRectF &rect);
    
    // 鼠标悬停在匹配点上
    void matchPointHovered(int index, const QPointF &scenePos);
    
    // 匹配点被点击
    void matchPointClicked(int index, const QPointF &scenePos);
    // 视图被右键点击（场景坐标）
    void viewRightClicked(const QPointF &scenePos);

protected:
    // 滚轮缩放
    bool eventFilter(QObject *obj, QEvent *event) override;
    
private slots:
    // 视图滚动/缩放时触发
    void onViewChanged();

private:
    void setupView();
    void updatePointsVisibility();
    qreal calculateZoomFactor() const;
    
    QGraphicsView *m_view;
    QGraphicsScene *m_scene;
    QGraphicsPixmapItem *m_imageItem;
    
    QVector<QPointF> m_matchPoints;
    QVector<QGraphicsEllipseItem*> m_pointItems;
    
    QString m_imagePath;
    int m_highlightedIndex;
    
    // 缩放范围限制
    static constexpr qreal MIN_ZOOM = 0.05;
    static constexpr qreal MAX_ZOOM = 50.0;
};
