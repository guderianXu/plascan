#pragma once

#include <QRhiWidget>
#include <QVector>
#include <QPointF>
#include <QColor>
#include <QLineF>
#include <QTransform>

#include "MatchSpatialIndex.h"
#include "MatchGpuRenderer.h"

class ImageViewWidget;

// MatchLineOverlay: 在两个 ImageViewWidget 之间绘制连接线的覆盖层
// 职责：
// - 根据匹配点数据和两个视图的可见区域，动态绘制连接线
// - 只绘制可见区域内的连接线以提高性能
// - 支持视觉样式配置（宽度、透明度）
// - 支持过滤（只显示内点、限制最大数量）
class MatchLineOverlay : public QRhiWidget
{
    Q_OBJECT

public:
    explicit MatchLineOverlay(QWidget *parent = nullptr);
    ~MatchLineOverlay() override;

    // 设置左右视图的引用
    void setViewWidgets(ImageViewWidget *leftView, ImageViewWidget *rightView);
    
    // 设置匹配数据（场景坐标）
    void setMatches(const QVector<QPointF> &ptsA, 
                    const QVector<QPointF> &ptsB);
    
    // 设置内点标记（可选，来自 bundle_adjust）
    void setInlierMask(const QVector<bool> &inlierMask);
    
    // 显示选项
    void setLineWidth(qreal width);
    void setOpacity(qreal opacity); // 0.0 - 1.0
    void setMaxDisplayCount(int maxCount); // 0 = 无限制
    void setShowOnlyInliers(bool onlyInliers);
    void setShowEndPoints(bool show); // 是否在连线两端画小圆点
    
    // 获取当前设置
    qreal lineWidth() const { return _lineWidth; }
    qreal opacity() const { return _opacity; }
    int maxDisplayCount() const { return _maxDisplayCount; }
    bool showOnlyInliers() const { return _showOnlyInliers; }
    bool showEndPoints() const { return _showEndPoints; }
    // 获取当前可见匹配（由外部组件查询以同步点的可见性）
    QVector<int> visibleMatches() const;
    // 高亮/展示控制：仅显示指定索引（当启用时，覆盖层只绘制这些索引）
    void setHighlightedIndices(const QVector<int> &indices);
    void clearHighlightedIndices();
    void setShowOnlyHighlighted(bool onlyHighlighted);
    bool showOnlyHighlighted() const { return _showOnlyHighlighted; }
    int renderedLineCount() const;

signals:
    void visibleMatchesChanged();
    
public slots:
    // 当视图变化时，外部调用此方法触发重绘
    void updateOverlay();

protected:
    void initialize(QRhiCommandBuffer *commandBuffer) override;
    void render(QRhiCommandBuffer *commandBuffer) override;
    void releaseResources() override;

private:
    // 计算哪些匹配点在可见区域内（返回索引列表）
    QVector<int> getVisibleMatches() const;
    QTransform sceneToOverlayTransform(ImageViewWidget *view) const;
    void rebuildRenderCache() const;
    void invalidateGeometryCache();
    QVector<MatchGpuLineBatch> gpuBatches() const;

private:
    ImageViewWidget *_leftView;
    ImageViewWidget *_rightView;
    
    QVector<QPointF> _ptsA;
    QVector<QPointF> _ptsB;
    QVector<bool> _inlierMask;
    xjw::gui::match_viewer::MatchSpatialIndex _leftSpatialIndex;
    xjw::gui::match_viewer::MatchSpatialIndex _rightSpatialIndex;
    
    // 显示选项
    qreal _lineWidth;
    qreal _opacity;
    int _maxDisplayCount;
    bool _showOnlyInliers;
    bool _showEndPoints;

    // 是否只绘制高亮索引（如果为true，则仅绘制 _highlightIndices 中的索引）
    bool _showOnlyHighlighted;
    QVector<int> _highlightIndices;


private:
    // 缓存：当前可见的匹配索引
    mutable QVector<int> _cachedVisibleMatches;
    mutable bool _visibilityCacheValid;
    mutable bool _renderCacheValid;
    mutable QVector<QLineF> _defaultLines;
    mutable QVector<QLineF> _inlierLines;
    mutable QVector<QLineF> _outlierLines;
    MatchGpuRenderer _gpuRenderer;
    mutable quint64 _gpuGeneration = 1;
    bool _rhiReady = false;
    QString _renderError;
};
