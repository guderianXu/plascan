#pragma once

#include <QGraphicsView>
#include "LayerRenderer.h"
#include "DepthOverlayData.h"
#include "MaskEditor.h"
#include <QFutureWatcher>
#include <QImage>
#include <QJsonObject>
#include <opencv2/core/types.hpp>  // 完整cv::KeyPoint定义,MOC需要
#include <vector>
#include <map>
#include <QDateTime>

// CanvasWidget: 封装 QGraphicsView 并提供后续扩展点（缩放、图层管理、渲染控制）
// 注：保持代码风格——左花括号独立行；所有注释为中文

class QGraphicsScene;
class LayerRenderer;
namespace xjw::gui::widgets { class DepthOverlayController; }

class CanvasWidget : public QGraphicsView
{
    Q_OBJECT
public:
    explicit CanvasWidget(QWidget *parent = nullptr);

    // 简单缩放接口：放大、缩小、重置视图（适用于演示和菜单绑定）
    void zoomIn();
    void zoomOut();
    void resetView();
    void rotateLeft();
    void rotateRight();
    void setViewRotationDegrees(int degrees);
    int viewRotationDegrees() const { return _viewRotationDegrees; }
    bool hasDisplayImage() const { return _singleImageReady; }
    bool showsInterestPoints() const { return _showInterestPoints; }
    bool showsFeatureResiduals() const { return _currentFeatureOpts.showResiduals; }
    bool showsMaskOverlay() const { return _showMaskOverlay; }
    bool depthOverlayEnabled() const { return _depthOverlayEnabled; }
    bool depthOverlayVisible() const { return _depthOverlayVisible; }
    bool depthIntensityVisible() const { return _depthIntensityVisible; }
    bool featureDiagnosticsSuppressed() const { return _depthInspectionActive; }
    xjw::gui::views::DepthOverlayLevel depthOverlayLevel() const { return _depthOverlayLevel; }
    QRectF imageBounds() const
    {
        return _layerRenderer ? _layerRenderer->imageBounds() : QRectF();
    }

public slots:
    // 应用兴趣点显示设置到内部渲染器
    void applyFeatureDisplayOptions(const LayerRenderer::FeatureDisplayOptions &opts);

public slots:
    // 显示指定影像（会清空已有图层并加载新影像）。
    // 说明：
    // - 这里只做“显示”，不修改项目元数据。
    // - 高位深影像的可视化转换在 LayerRenderer 中处理。
    void showImage(const QString &path);

    // 重新读取当前影像的蒙版并刷新轮廓叠加层。
    void reloadMaskOverlay();

    // 控制是否显示蒙版轮廓叠加层。
    void setShowMaskOverlay(bool show);

    void setProjectMetadata(const QJsonObject &metadata);
    void setDepthOverlayEnabled(bool enabled);
    void setDepthOverlayLevel(xjw::gui::views::DepthOverlayLevel level);
    void setDepthIntensityVisible(bool visible);
    void useRectangleMaskTool();
    void useScissorsMaskTool();
    void useSmartPaintMaskTool();
    void useMagicWandMaskTool();
    void showMaskEditorSettings();
    void resetMaskSelection();
    void undoMaskEdit();
    void redoMaskEdit();
    void confirmInteractiveMaskSaved(const QString &imagePath, quint64 revision);

public slots:
    // 控制是否在视图上叠加显示兴趣点
    void setShowInterestPoints(bool show);
    void setShowFeatureResiduals(bool show);

    // 强制重新加载指定影像的匹配观测，用于重新生成 `.pimatch` 后刷新显示。
    void reloadInterestPoints(const QString &imagePath);

    // 返回当前显示影像的路径（若未显示则为空）
    QString currentImagePath() const;

signals:
    // 当前影像参与匹配的观测加载完成；count 允许为零。
    void featuresLoaded(const QString &imagePath, int count);
    // emitted when the active display image changes (for UI state persistence)
    void activeImageChanged(const QString &imagePath);
    void viewRotationChanged(const QString &imagePath, int degrees);
    void displayImageReadyChanged(bool ready);
    void interestPointsVisibilityChanged(bool visible);
    void featureResidualVisibilityChanged(bool visible);
    void maskOverlayVisibilityChanged(bool visible);
    void depthOverlayAvailabilityChanged(bool available);
    void depthOverlayLevelsAvailabilityChanged(bool finalAvailable,
                                               bool level1Available,
                                               bool level2Available,
                                               bool level3Available,
                                               const QString &finalReason,
                                               const QString &level1Reason,
                                               const QString &level2Reason,
                                               const QString &level3Reason);
    void depthOverlayVisibilityChanged(bool visible);
    void depthOverlayError(const QString &message);
    void featureResidualAvailabilityChanged(bool available);
    // 照片区域被右键时发送原始影像像素坐标；视图旋转和缩放不会改变该坐标。
    void imageContextRequested(const QString &imagePath, const QPointF &originalPixel);
    void interactiveMaskEditRequested(const QString &imagePath,
                                      const QImage &mask,
                                      const QString &method,
                                      quint64 revision);
    void maskSelectionActiveChanged(bool active);

private:
    // 在后台加载当前影像的 `.pimatch` 观测，主线程只负责更新场景。
    void startMatchObservationLoadForImage(const QString &imagePath);
    void startResidualLoadForImage(const QString &imagePath);
    QString matchObservationCacheKey(const QString &imagePath) const;
    void clearFeatureCacheForImage(const QString &imagePath);
    void refreshDepthOverlay();
    void setDepthInspectionActive(bool active);
    void setMaskTool(MaskEditor::Tool tool);
    void reloadEditableMask();
    bool shouldRenderFeatureDiagnostics() const { return !_depthInspectionActive; }

protected:
    // 当控件变为可见时（如被 QStackedWidget 切换到前台），重新 fitInView 以修正首次显示尺寸
    void showEvent(QShowEvent *event) override;
    // 鼠标滚轮缩放：更符合影像查看习惯
    void wheelEvent(QWheelEvent *event) override;
    // 鼠标事件：支持左键拖动平移与左键点击缩放
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void contextMenuEvent(QContextMenuEvent *event) override;

private:
    LayerRenderer *_layerRenderer{};
    xjw::gui::widgets::DepthOverlayController *_depthOverlayController{};
    MaskEditor *_maskEditor{};
    bool _showInterestPoints{true};  // 默认启用特征点显示
    bool _showMaskOverlay{true}; // 默认显示照片蒙版轮廓
    // 当前的兴趣点显示选项（由 UI 通过 applyFeatureDisplayOptions 设置）
    LayerRenderer::FeatureDisplayOptions _currentFeatureOpts;
    QString _currentImagePath;
    QFutureWatcher<QImage> *_imageWatcher{nullptr};
    // 缓存键为影像规范路径，时间戳来自对应 `.pimatch`，而不是源影像。
    std::map<QString, std::pair<QDateTime, std::vector<cv::KeyPoint>>> _matchObservationCache;
    int _featureLoadGeneration{0};
    int _residualLoadGeneration{0};
    int _maskLoadGeneration{0};
    quint64 _maskSavedRevision{};
    int _viewRotationDegrees{0};
    bool _singleImageReady{false};
    bool _depthOverlayEnabled{false};
    bool _depthIntensityVisible{false};
    bool _depthOverlayVisible{false};
    bool _depthInspectionActive{false};
    xjw::gui::views::DepthOverlayLevel _depthOverlayLevel{
        xjw::gui::views::DepthOverlayLevel::Final};

    // 缩放限制（避免无限放大/缩小导致精度或性能问题）
    double _zoomFactor{1.0};
    const double _zoomStep{1.15};
    const double _zoomMin{0.05};
    const double _zoomMax{50.0};

    // 将来可添加：加载影像、设置图层可见性、坐标转换、拾取等接口
    // 平移状态
    bool _isPanning{false};
    QPoint _lastPanPoint{};
    // 用于判断单击还是拖动的阈值（像素）
    const int _panThreshold{4};
};
