#pragma once

#include <QWidget>
#include <QString>
#include <QVector>
#include <QPointF>
#include <QPointer>

class ImageViewWidget;
class MatchLineOverlay;
class DisparityHeatmapOverlay;
class QTimer;

// DualImageViewer: 管理左右两个 ImageViewWidget 和连接线覆盖层
// 职责：
// - 协调左右视图和覆盖层
// - 实现同步模式（缩放/平移联动）
// - 加载匹配数据并分发到各组件
// - 提供统一的配置接口
class DualImageViewer : public QWidget
{
    Q_OBJECT

public:
    explicit DualImageViewer(QWidget *parent = nullptr);
    ~DualImageViewer() override;

    // 加载匹配对
    bool loadMatchPair(const QString &imgA, const QString &imgB,
                       const QString &matchFile);
    
    // 加载匹配对（直接传入数据）
    void loadMatchPair(const QString &imgA, const QString &imgB,
                       const QVector<QPointF> &ptsA,
                       const QVector<QPointF> &ptsB);
    
    // 同步模式控制
    void setSyncMode(bool enabled);
    bool syncMode() const { return m_syncEnabled; }
    
    // 获取子组件
    ImageViewWidget* leftView() const;
    ImageViewWidget* rightView() const;
    MatchLineOverlay* overlay() const;
    DisparityHeatmapOverlay* disparityOverlay() const;
    void setOverlayMode(int mode); // 0=sparse, 1=dense
    int overlayMode() const { return m_overlayMode; }

    // 快捷操作
    void fitBothViews();
    void resetBothViews();
    // 高亮/筛选匹配线：只显示指定索引集合或清除高亮
    void highlightMatchIndex(int index);
    void highlightMatchIndices(const QVector<int> &indices);
    void clearMatchHighlights();
    // 控制是否显示所有匹配线（默认 true）
    void setShowAllMatches(bool showAll);
    
    // 获取统计信息
    int totalMatchCount() const;
    int visibleMatchCount() const;
    QString leftImagePath() const;
    QString rightImagePath() const;

signals:
    // 加载状态
    void matchDataLoaded(int matchCount);
    void loadFailed(const QString &error);
    
    // 同步模式变化
    void syncModeChanged(bool enabled);

protected:
    void resizeEvent(QResizeEvent *event) override;

private slots:
    // 视图变化时的处理
    void onLeftViewChanged(const QTransform &transform);
    void onRightViewChanged(const QTransform &transform);
    
    // 延迟更新覆盖层（性能优化）
    void scheduleOverlayUpdate();
    void updateOverlayNow();

private:
    void setupLayout();
    void connectSignals();
    void updateOverlayGeometry();
    
    // 解析 .match 文件
    bool parseMatchFile(const QString &matchFile,
                        QVector<QPointF> &ptsA,
                        QVector<QPointF> &ptsB);

private:
    QPointer<ImageViewWidget> m_leftView;
    QPointer<ImageViewWidget> m_rightView;
    QPointer<MatchLineOverlay> m_overlay;
    QPointer<DisparityHeatmapOverlay> m_disparityOverlay;
    int m_overlayMode = 0;
    
    bool m_syncEnabled;
    bool m_syncing; // 防止递归同步
    
    QVector<QPointF> m_matchPtsA;
    QVector<QPointF> m_matchPtsB;
    
    // 延迟更新定时器（性能优化）
    QTimer *m_overlayUpdateTimer;
};
