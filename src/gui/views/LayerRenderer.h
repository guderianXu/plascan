// LayerRenderer 负责把影像、特征点、残差和蒙版渲染到 QGraphicsScene。
#pragma once

#include <QObject>
#include <QList>
#include <QVector>
#include <QPointF>
#include <QColor>
#include <QImage>
#include <QPixmap>
#include <QString>
#include <QRectF>
#include <QTransform>

#include "FeatureResidualLoader.h"

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

    bool addImageLayer(const QImage &image, int z = 0);
    static QImage loadImageForDisplay(const QString &path, const QString &plascanPath);

    bool setDepthOverlay(const QImage &overlay,
                         const QImage &intensity_base = QImage(),
                         int z = 10);
    void clearDepthOverlay();
    bool hasDepthOverlay() const noexcept { return _depthOverlayItem != nullptr; }

    // 清除所有已添加的图层
    void clear();

    // 添加/清除蒙版轮廓覆盖层，蒙版约定为 255=排除区域，0=保留前景。
    bool addMaskContourLayer(const QString &maskPath, int z = 40);
    void clearMaskLayers();

    // 仅清除兴趣点覆盖层（不移除影像图层）
    void clearFeatureLayers();

    // 可配置的兴趣点显示选项
    struct FeatureDisplayOptions {
        bool showPoints = true;
        bool showScale = false;
        bool showOrientation = false;
        int pointSize = 1;
        double scaleMultiplier = 1.0;
        QColor pointColor = QColor(0, 120, 255);
        QColor scaleColor = QColor(255,255,0);
        QColor orientColor = QColor(255,0,0);
        int opacity = 180; // 0-255
        QString markerShape = QStringLiteral("cross");
        int maxDisplayCount = 0; // 0=all
        bool showTopScores = true;
        bool useFill = false; // 是否使用实心填充(默认空心)
        bool showResiduals = false;
        double residualScale = 10.0;
        double minimumResidualPx = 0.0;
        double maximumResidualLengthPx = 80.0;
        QColor residualColor = QColor(255, 80, 80);
    };

    void setFeatureDisplayOptions(const FeatureDisplayOptions &opts);

    // 由主线程调用：接收解析好的兴趣点列表并在 scene 上绘制（安全）
    void addFeatureItems(const std::vector<cv::KeyPoint> &keypoints);
    void addFeatureResidualItems(const QVector<xjw::gui::views::FeatureResidualVector> &residuals);
    void clearFeatureResidualLayers();

    // 当前影像在场景坐标系中的边界。调用方只读使用，避免在空白区创建量测。
    QRectF imageBounds() const noexcept { return _imageBounds; }

private:
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

    // 当前兴趣点显示选项
    FeatureDisplayOptions _featureOpts;
};
