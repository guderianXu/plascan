// LayerRenderer 负责把影像、特征点、残差和蒙版渲染到 QGraphicsScene。
#pragma once

#include <atomic>
#include <memory>

#include <QObject>
#include <QCache>
#include <QList>
#include <QVector>
#include <QPointF>
#include <QColor>
#include <QImage>
#include <QPainterPath>
#include <QPixmap>
#include <QString>
#include <QRectF>
#include <QTransform>

#include "FeatureResidualLoader.h"
#include "LayerFeatureLoader.h"

class QGraphicsScene;
class QGraphicsPixmapItem;
class QGraphicsItem;

// forward declaration避免在头文件中包含完整的OpenCV
namespace cv { class KeyPoint; }

class LayerRenderer : public QObject
{
    Q_OBJECT
public:
    explicit LayerRenderer(QGraphicsScene *scene, QObject *parent = nullptr);
    ~LayerRenderer() override;

    bool addImageLayer(const QImage &image, int z = 0);
    static QImage loadImageForDisplay(const QString &path, const QString &plascanPath);

    bool setDepthOverlay(const QImage &overlay,
                         const QImage &intensity_base = QImage(),
                         int z = 10);
    void clearDepthOverlay();
    bool hasDepthOverlay() const noexcept { return _depthOverlayItem != nullptr; }

    // 清除所有已添加的图层
    void clear();

    // 异步添加/清除蒙版轮廓覆盖层，蒙版约定为 255=排除区域，0=保留前景。
    // 返回值仅表示请求是否已接受；磁盘读取、轮廓提取和路径构建不会阻塞 GUI 线程。
    bool addMaskContourLayer(const QString &mask_path, int z = 40);
    void clearMaskLayers();

    // 仅清除兴趣点覆盖层（不移除影像图层）
    void clearFeatureLayers();

    // 可配置的兴趣点显示选项
    struct FeatureDisplayOptions {
        xjw::gui::views::FeaturePointSource pointSource =
            xjw::gui::views::FeaturePointSource::ValidTiePoints;
        bool showPoints = true;
        int pointSize = 1;
        QColor pointColor = QColor(0, 120, 255);
        int opacity = 180; // 0-255
        int maxDisplayCount = 0; // 0=all
        bool showResiduals = false;
        double residualScale = 50.0;
        double minimumResidualPx = 0.0;
        double maximumResidualLengthPx = 80.0;
    };

    void setFeatureDisplayOptions(const FeatureDisplayOptions &opts);

    // 由主线程调用：接收解析好的兴趣点列表并在 scene 上绘制（安全）
    void addFeatureItems(const std::vector<cv::KeyPoint> &keypoints);
    void addFeatureResidualItems(const QVector<xjw::gui::views::FeatureResidualVector> &residuals);
    void clearFeatureResidualLayers();

    // 当前影像在场景坐标系中的边界。调用方只读使用，避免在空白区创建量测。
    QRectF imageBounds() const noexcept { return _imageBounds; }

signals:
    void maskContourLayerReady(const QString &mask_path, bool from_cache);
    void maskContourLayerFailed(const QString &mask_path);

private:
    void clearMaskLayerItems();
    bool installMaskContourLayer(const QPainterPath &path, int z);

    QGraphicsScene *_scene{};
    QList<QGraphicsPixmapItem *> _layers{};
    QList<QGraphicsItem *> _featureItems{};
    QList<QGraphicsItem *> _featureResidualItems{};
    QList<QGraphicsItem *> _maskItems{};
    QGraphicsPixmapItem *_depthOverlayItem{};
    QGraphicsPixmapItem *_baseImageItem{};
    QPixmap _baseImagePixmap;
    QTransform _baseImageTransform;
    QPointF _baseImagePosition;
    bool _intensityBaseActive{false};
    QRectF _imageBounds{};

    // QCache 的 cost 使用路径元素数；设置最小条目成本以同时限制小蒙版的条目数量。
    QCache<QString, QPainterPath> _maskContourCache;
    std::shared_ptr<std::atomic<bool>> _maskLoadCancellation;
    quint64 _maskLoadGeneration{0};
    static constexpr int MaximumMaskContourCacheCost = 2'000'000;
    static constexpr int MinimumMaskContourCacheEntryCost = 125'000;

    // 当前兴趣点显示选项
    FeatureDisplayOptions _featureOpts;
};
