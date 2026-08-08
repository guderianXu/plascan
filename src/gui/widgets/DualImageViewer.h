#pragma once

#include <QWidget>
#include <QString>
#include <QVector>
#include <QPointF>
#include <QPointer>
#include <QByteArray>

#include <cstdint>
#include <optional>

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

    /**
     * @brief 从逐影像 `.pimatch` 分片加载一个像对。
     *
     * algorithmId 为空时选择几何内点最多的变体；非空时，版本和配置指纹共同
     * 指定唯一变体。该接口使查看器不需要知道特征文件或成对文件命名规则。
     */
    bool loadMatchPair(const QString &imgA, const QString &imgB,
                       const QString &matchFile,
                       const QString &algorithmId = QString(),
                       std::uint32_t algorithmVersion = 0,
                       const QByteArray &configFingerprint = QByteArray());
    
    // 加载匹配对（直接传入数据）
    void loadMatchPair(const QString &imgA, const QString &imgB,
                       const QVector<QPointF> &ptsA,
                       const QVector<QPointF> &ptsB);

    // 标记量测模式：仅显示可靠投影与当前候选，不读取或解析匹配文件。
    void setMarkerMeasurement(const QString &anchorImage,
                              const QString &candidateImage,
                              const QPointF &anchorPixel,
                              const std::optional<QPointF> &candidatePixel);
    
    // 同步模式控制
    void setSyncMode(bool enabled);
    
    // 获取子组件
    ImageViewWidget* leftView() const;
    ImageViewWidget* rightView() const;
    MatchLineOverlay* overlay() const;
    DisparityHeatmapOverlay* disparityOverlay() const;
    void setOverlayMode(int mode); // 0=sparse, 1=dense

    // 快捷操作
    void fitBothViews();
    void resetBothViews();
    void clearViewer();
    // 高亮/筛选匹配线：只显示指定索引集合或清除高亮
    void highlightMatchIndex(int index);
    void clearMatchHighlights();
    // 控制是否显示所有匹配线（默认 true）
    void setShowAllMatches(bool showAll);
    
    QString leftImagePath() const;
    QString rightImagePath() const;

signals:
    // 加载状态
    void matchDataLoaded(int matchCount);
    void matchValidityLoaded(int validCount, int invalidCount);
    void loadFailed(const QString &error);
    
    void markerCandidatePicked(const QPointF &pixel);

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
    
    // 解析一个 `.pimatch` 分片中的目标邻接变体。
    bool parseMatchFile(const QString &matchFile,
                        const QString &imgA,
                        const QString &imgB,
                        const QString &algorithmId,
                        std::uint32_t algorithmVersion,
                        const QByteArray &configFingerprint,
                        QVector<QPointF> &ptsA,
                        QVector<QPointF> &ptsB,
                        QVector<bool> &inlierMask);

private:
    QPointer<ImageViewWidget> _leftView;
    QPointer<ImageViewWidget> _rightView;
    QPointer<MatchLineOverlay> _overlay;
    QPointer<DisparityHeatmapOverlay> _disparityOverlay;
    int _overlayMode = 0;
    
    bool _syncEnabled;
    bool _syncing; // 防止递归同步
    
    QVector<QPointF> _matchPtsA;
    QVector<QPointF> _matchPtsB;
    
    // 延迟更新定时器（性能优化）
    QTimer *_overlayUpdateTimer;
};
