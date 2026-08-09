// =============================================================================
// 文件: DisparityHeatmapOverlay.h
// 功能: 视差图热力图叠加层（用于密集匹配可视化）
// =============================================================================
#pragma once

#include <atomic>
#include <memory>

#include <QHash>
#include <QImage>
#include <QPointer>
#include <QPixmap>
#include <QStringList>
#include <QTransform>
#include <QWidget>
#include <opencv2/core.hpp>

class ImageViewWidget;

class DisparityHeatmapOverlay : public QWidget
{
    Q_OBJECT
public:
    explicit DisparityHeatmapOverlay(QWidget *parent = nullptr);
    ~DisparityHeatmapOverlay() override;

    bool loadDisparity(const QString &filepath);
    bool loadDisparity(const cv::Mat &disparity);
    void clear();

    // 热图像素使用目标影像的场景坐标，覆盖层本身固定在目标 viewport 上。
    void setTargetView(ImageViewWidget *view);
    ImageViewWidget *targetView() const;
    QTransform sceneToOverlayTransform() const;
    QPointF mapSceneToOverlay(const QPointF &scenePoint) const;
    void syncToTargetViewport();

    void setOpacity(float opacity);
    void setDisparityRange(float min, float max);
    void setAutoRange(bool enabled);
    void setColormap(int cvColormap);
    void setShowInvalid(bool show);

    float opacity() const { return _opacity; }
    QImage heatmapImage() const { return _heatmapImage; }

signals:
    void heatmapReady(const QSize &size);
    void loadFailed(const QString &message);

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    void rebuildHeatmap();
    void acceptDisparity(cv::Mat disparity);
    void applyHeatmap(const QString &cacheKey, const QImage &image);
    QString currentCacheKey() const;
    void clearCache();
    void cacheHeatmap(const QString &cacheKey, const QImage &image);
    void cancelLoad();
    void cancelBuild();

    cv::Mat _disparity;
    QImage  _heatmapImage;
    QPixmap _heatmap;
    float   _opacity     = 0.6f;
    float   _dispMin     = 0.0f;
    float   _dispMax     = 256.0f;
    bool    _autoRange   = true;
    int     _colormap    = 2; // COLORMAP_JET
    bool    _showInvalid = false;

    QPointer<ImageViewWidget> _targetView;
    std::shared_ptr<std::atomic_bool> _loadCancellation;
    std::shared_ptr<std::atomic_bool> _buildCancellation;
    quint64 _loadGeneration = 0;
    quint64 _buildGeneration = 0;
    quint64 _sourceRevision = 0;

    QHash<QString, QImage> _heatmapCache;
    QStringList _heatmapCacheOrder;
    qsizetype _heatmapCacheBytes = 0;
    static constexpr qsizetype MaximumCacheBytes = 64 * 1024 * 1024;
};
