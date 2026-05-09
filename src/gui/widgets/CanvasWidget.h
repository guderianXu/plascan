#pragma once

#include <QGraphicsView>
#include "LayerRenderer.h"
#include <QFutureWatcher>
#include <opencv2/core/types.hpp>  // 完整cv::KeyPoint定义,MOC需要
#include <vector>
#include <map>
#include <QDateTime>

// CanvasWidget: 封装 QGraphicsView 并提供后续扩展点（缩放、图层管理、渲染控制）
// 注：保持代码风格——左花括号独立行；所有注释为中文

class QGraphicsScene;
class LayerRenderer;

class CanvasWidget : public QGraphicsView
{
    Q_OBJECT
public:
    explicit CanvasWidget(QWidget *parent = nullptr);

    // 简单缩放接口：放大、缩小、重置视图（适用于演示和菜单绑定）
    void zoomIn();
    void zoomOut();
    void resetView();

public slots:
    // 应用兴趣点显示设置到内部渲染器
    void applyFeatureDisplayOptions(const LayerRenderer::FeatureDisplayOptions &opts);

public slots:
    // 显示指定影像（会清空已有图层并加载新影像）。
    // 说明：
    // - 这里只做“显示”，不修改项目元数据。
    // - 高位深影像的可视化转换在 LayerRenderer 中处理。
    void showImage(const QString &path);

public slots:
    // 控制是否在视图上叠加显示兴趣点
    void setShowInterestPoints(bool show);

    // 切换当前显示的特征提取器后缀 (.sp/.dsk/.alk 等), 重新加载特征点
    void setActiveFeatureSuffix(const QString &suffix);
    // 获取当前活动的特征文件后缀
    QString activeFeatureSuffix() const { return m_activeFeatureSuffix; }

    // 获取当前影像可用的特征文件后缀列表 (供 UI 构建选择器)
    QStringList availableFeatureSuffixes() const;

    // 在主画布上显示一对匹配影像并绘制匹配连线（基于 .match 二进制文件）
    void showMatchedPair(const QString &imgA, const QString &imgB, const QString &matchFile);

    // 强制重新加载指定影像的兴趣点（忽略缓存），用于在外部重新生成 .sp 后刷新显示
    void reloadInterestPoints(const QString &imagePath);

    // 立即（同步）重新读取并刷新指定影像的兴趣点，用于在外部任务写入 .sp 后强制更新 UI
    void immediateReloadInterestPoints(const QString &imagePath);

    // 返回指定影像当前缓存的兴趣点（以 QVariantMap 列表形式，避免在头文件暴露内部类型）
    QList<QVariantMap> getCachedInterestPointsAsVariant(const QString &imagePath) const;

    // 返回当前显示影像的路径（若未显示则为空）
    QString currentImagePath() const;

signals:
    // emitted when features for an image are loaded (count may be zero)
    void featuresLoaded(const QString &imagePath, int count);
    // emitted when the active display image changes (for UI state persistence)
    void activeImageChanged(const QString &imagePath);

private:
    // 启动异步加载 .sp (SuperPoint) 的通用方法（在主线程调度后台任务）
    void startSpLoadForImage(const QString &imagePath);

protected:
    // 当控件变为可见时（如被 QStackedWidget 切换到前台），重新 fitInView 以修正首次显示尺寸
    void showEvent(QShowEvent *event) override;
    // 鼠标滚轮缩放：更符合影像查看习惯
    void wheelEvent(QWheelEvent *event) override;
    // 鼠标事件：支持左键拖动平移与左键点击缩放
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    LayerRenderer *m_layerRenderer{};
    bool m_showInterestPoints{true};  // 默认启用特征点显示
    QString m_activeFeatureSuffix{QStringLiteral(".sp")};  // 当前选择的特征提取器后缀
    // 当前的兴趣点显示选项（由 UI 通过 applyFeatureDisplayOptions 设置）
    LayerRenderer::FeatureDisplayOptions m_currentFeatureOpts;
    QString m_currentImagePath;
    // 后台读取 .sp 的 watcher（每次启动一个异步任务）
    QFutureWatcher<std::vector<cv::KeyPoint>> *m_spWatcher{nullptr};
    // cache: imagePath -> (lastModified, keypoints)
    std::map<QString, std::pair<QDateTime, std::vector<cv::KeyPoint>>> m_spCache;
    QString m_lastRequestedSpPath;
    QString m_lastRequestedSpSuffix;

    // 缩放限制（避免无限放大/缩小导致精度或性能问题）
    double m_zoomFactor{1.0};
    const double m_zoomStep{1.15};
    const double m_zoomMin{0.05};
    const double m_zoomMax{50.0};

    // 将来可添加：加载影像、设置图层可见性、坐标转换、拾取等接口
    // 平移状态
    bool m_isPanning{false};
    QPoint m_lastPanPoint{};
    // 用于判断单击还是拖动的阈值（像素）
    const int m_panThreshold{4};
};