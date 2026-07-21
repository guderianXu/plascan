// LayerRenderer 负责把影像、特征点和匹配连线渲染到 QGraphicsScene。
// 后续可继续拆分为影像缓存、特征点覆盖层和匹配线覆盖层三个协作类。
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
class QString;

// forward declaration避免在头文件中包含完整的OpenCV
namespace cv { class KeyPoint; }

class LayerRenderer : public QObject
{
    Q_OBJECT
public:
    explicit LayerRenderer(QGraphicsScene *scene, QObject *parent = nullptr);

    // 注入当前项目的 .plascan 路径（用于把转换缓存写入 <projectRoot>/.plascan_tmp）
    void setCurrentProjectPath(const QString &plascanPath);

    // 从文件路径加载影像并添加为图层，返回是否成功
    bool addImageLayer(const QString &path, int z = 0);
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

    // 为指定影像（path）查找对应的 .vwip 并在 scene 上添加兴趣点覆盖层
    // 返回是否成功添加了任何兴趣点（false 表示未找到 .vwip 或没有点）
    bool addFeatureLayerFromVwip(const QString &imagePath);
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

    // 匹配线显示选项
    struct MatchDisplayOptions {
        bool showLines = true;
        QColor lineColor = QColor(0, 255, 255); // 青色
        int lineWidth = 1;
        int opacity = 180;
        bool showOnlyInliers = false;
        int maxDisplayCount = 0; // 0=all
    };

    void setMatchDisplayOptions(const MatchDisplayOptions &opts);

    // 将两幅影像并排加载为一个拼接图层（用于可视化匹配对）。
    // 返回是否成功，并通过 outA/outB 返回对应的 QGraphicsPixmapItem 指针（可为 nullptr）。
    bool addStitchedImagePair(const QString &pathA,
                              const QString &pathB,
                              QGraphicsPixmapItem **outA = nullptr,
                              QGraphicsPixmapItem **outB = nullptr,
                              int gap = 20);

    // 根据两组点绘制匹配连线，ptsA/ptsB 应按同一索引对应；bOffsetX 为第二幅图在 scene 中的 X 偏移量。
    void addMatchLines(const QVector<QPointF> &ptsA, const QVector<QPointF> &ptsB, qreal bOffsetX = 0.0);
    // 清除匹配连线层（不影响影像与兴趣点层）
    void clearMatchLayers();

    // 由主线程调用：接收解析好的兴趣点列表并在 scene 上绘制（安全）
    void addFeatureItems(const std::vector<cv::KeyPoint> &keypoints);
    void addFeatureResidualItems(const QVector<xjw::gui::views::FeatureResidualVector> &residuals);
    void clearFeatureResidualLayers();

    // 当前单幅或拼接影像在场景坐标系中的边界。调用方只读使用，避免在空白区创建量测。
    QRectF imageBounds() const noexcept { return _imageBounds; }

private:
    QGraphicsScene *_scene{};
    QList<QGraphicsPixmapItem *> _layers{};
    QList<QGraphicsItem *> _featureItems{};
    QList<QGraphicsItem *> _featureResidualItems{};
    QList<QGraphicsItem *> _matchItems{};
    QList<QGraphicsItem *> _maskItems{};
    QGraphicsPixmapItem *_depthOverlayItem{};
    QGraphicsPixmapItem *_baseImageItem{};
    QPixmap _baseImagePixmap;
    QTransform _baseImageTransform;
    QPointF _baseImagePosition;
    bool _intensityBaseActive{false};
    QRectF _imageBounds{};

    // 当前项目的 .plascan 文件路径（可为空：表示未打开项目）
    QString _currentProjectPath;
    // 当前兴趣点显示选项
    FeatureDisplayOptions _featureOpts;
    // 当前匹配线显示选项
    MatchDisplayOptions _matchOpts;
};
