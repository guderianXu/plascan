// 替换为最小化 stub：保留类型与接口以避免删除后造成大量引用出错。
#pragma once

#include <QObject>
#include <QList>
#include <QVector>
#include <QPointF>
#include <QColor>
#include <QString>

class QGraphicsScene;
class QGraphicsPixmapItem;
class QGraphicsItem;
class QString;

// forward declaration避免在头文件中包含完整的OpenCV
namespace cv { class KeyPoint; }

// LayerRenderer: 负责把影像层渲染到 QGraphicsScene 上（最小实现）
// 提供加载影像的简单接口和清理接口

class LayerRenderer : public QObject
{
    Q_OBJECT
public:
    explicit LayerRenderer(QGraphicsScene *scene, QObject *parent = nullptr);

    // 注入当前项目的 .plascan 路径（用于把转换缓存写入 <projectRoot>/.plascan_tmp）
    void setCurrentProjectPath(const QString &plascanPath);

    // 从文件路径加载影像并添加为图层，返回是否成功
    bool addImageLayer(const QString &path, int z = 0);

    // 清除所有已添加的图层
    void clear();

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
        QColor pointColor = QColor(255,200,0);
        QColor scaleColor = QColor(255,255,0);
        QColor orientColor = QColor(255,0,0);
        int opacity = 180; // 0-255
        QString markerShape = QStringLiteral("cross");
        int maxDisplayCount = 0; // 0=all
        bool showTopScores = true;
        bool useFill = false; // 是否使用实心填充(默认空心)
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
    bool addStitchedImagePair(const QString &pathA, const QString &pathB, QGraphicsPixmapItem **outA = nullptr, QGraphicsPixmapItem **outB = nullptr, int gap = 20);

    // 根据两组点绘制匹配连线，ptsA/ptsB 应按同一索引对应；bOffsetX 为第二幅图在 scene 中的 X 偏移量。
    void addMatchLines(const QVector<QPointF> &ptsA, const QVector<QPointF> &ptsB, qreal bOffsetX = 0.0);
    // 清除匹配连线层（不影响影像与兴趣点层）
    void clearMatchLayers();

    // 由主线程调用：接收解析好的兴趣点列表并在 scene 上绘制（安全）
    void addFeatureItems(const std::vector<cv::KeyPoint> &keypoints);

private:
    QGraphicsScene *m_scene{};
    QList<QGraphicsPixmapItem *> m_layers{};
    QList<QGraphicsItem *> m_featureItems{};
    QList<QGraphicsItem *> m_matchItems{};

    // 当前项目的 .plascan 文件路径（可为空：表示未打开项目）
    QString m_currentProjectPath;
    // 当前兴趣点显示选项
    FeatureDisplayOptions m_featureOpts;
    // 当前匹配线显示选项
    MatchDisplayOptions m_matchOpts;
};
