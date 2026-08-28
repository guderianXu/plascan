#include "CanvasWidget.h"

#include "LayerFeatureLoader.h"
#include "FeatureResidualLoader.h"
#include "LayerRenderer.h"
#include "DepthOverlayController.h"
#include "project/ProjectIO.h"
#include "GuiTaskRunner.h"
#include "io/PathIO.h"
#include "image/MaskEditorSettingsDialog.h"

#include <QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <QImageReader>
#include <QKeyEvent>
#include <QPointer>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <QGraphicsScene>
#include <QDir>
#include <QFileInfo>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QContextMenuEvent>
#include <QScrollBar>
#include <QFile>
#include <QDataStream>
#include <QPointF>
#include <QVector>
#include "Logger.h"
#include "ImageViewRotationSettings.h"

namespace
{

bool isDepthMapPreviewPath(const QString &path)
{
    const QString fileName = QFileInfo(path).fileName();
    return fileName.startsWith(QLatin1String("depth_"), Qt::CaseInsensitive)
        && fileName.endsWith(QLatin1String(".png"), Qt::CaseInsensitive);
}

} // namespace

// CanvasWidget 实现：创建并持有一个 QGraphicsScene，作为渲染的起点

CanvasWidget::CanvasWidget(QWidget *parent)
    : QGraphicsView(parent)
{
    auto *scene = new QGraphicsScene(this);
    setScene(scene);

    // 渲染器：负责把影像层加入到 scene
    _layerRenderer = new LayerRenderer(scene, this);
    _maskEditor = new MaskEditor(scene, this);
    connect(_maskEditor,
            &MaskEditor::maskChanged,
            this,
            [this](const QImage &mask, const QString &method, quint64 revision)
            {
                if (!_currentImagePath.trimmed().isEmpty())
                {
                    emit interactiveMaskEditRequested(_currentImagePath, mask, method, revision);
                }
            });
    connect(_maskEditor,
            &MaskEditor::selectionActiveChanged,
            this,
            &CanvasWidget::maskSelectionActiveChanged);
    _depthOverlayController = new xjw::gui::widgets::DepthOverlayController(this);
    connect(_depthOverlayController,
            &xjw::gui::widgets::DepthOverlayController::overlayReady,
            this,
            [this](const QString &image_path,
                   const QImage &overlay,
                   const QImage &intensity_base)
            {
                if (!_depthOverlayEnabled
                    || QDir::cleanPath(image_path) != QDir::cleanPath(_currentImagePath)
                    || !_layerRenderer)
                {
                    return;
                }
                _depthOverlayVisible = _layerRenderer->setDepthOverlay(
                    overlay, intensity_base, 10);
                emit depthOverlayVisibilityChanged(_depthOverlayVisible);
            });
    connect(_depthOverlayController,
            &xjw::gui::widgets::DepthOverlayController::overlayFailed,
            this,
            [this](const QString &image_path, const QString &error_message)
            {
                if (QDir::cleanPath(image_path) != QDir::cleanPath(_currentImagePath))
                {
                    return;
                }
                if (_layerRenderer)
                {
                    _layerRenderer->clearDepthOverlay();
                }
                _depthOverlayVisible = false;
                setDepthInspectionActive(false);
                emit depthOverlayVisibilityChanged(false);
                emit depthOverlayError(error_message);
            });

    // 默认设置：平滑缩放，开启抗锯齿（若需要可调整）
    setRenderHint(QPainter::Antialiasing, true);
    setRenderHint(QPainter::SmoothPixmapTransform, true);

    // 缩放体验：以鼠标位置为锚点缩放；拖拽时以鼠标为锚点
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorUnderMouse);
    setFocusPolicy(Qt::StrongFocus);

    // 鼠标/交互将在后续扩展中实现
    // 特征点加载 watcher 在每次请求时创建，并用 generation guard 丢弃过期结果。
}

void CanvasWidget::applyFeatureDisplayOptions(const LayerRenderer::FeatureDisplayOptions &opts)
{
    if (!_layerRenderer) return;
    // 保存当前选项并立即注入渲染器
    _currentFeatureOpts = opts;
    const bool pointsChanged = _showInterestPoints != opts.showPoints;
    _showInterestPoints = opts.showPoints;
    _layerRenderer->setFeatureDisplayOptions(opts);
    // Use the options' showPoints to control visibility: 当 opts.showPoints 为 true 时加载特征点，否则清除
    if (shouldRenderFeatureDiagnostics()
        && opts.showPoints && !_currentImagePath.trimmed().isEmpty()
        && !isDepthMapPreviewPath(_currentImagePath)) {
        _layerRenderer->clearFeatureLayers();
        startFeaturePointLoadForImage(_currentImagePath);
    } else {
        _layerRenderer->clearFeatureLayers();
    }
    _layerRenderer->clearFeatureResidualLayers();
    if (shouldRenderFeatureDiagnostics()
        && opts.showResiduals && !_currentImagePath.trimmed().isEmpty()
        && !isDepthMapPreviewPath(_currentImagePath))
    {
        startResidualLoadForImage(_currentImagePath);
    }
    if (pointsChanged)
    {
        emit interestPointsVisibilityChanged(_showInterestPoints);
    }
    emit featureResidualVisibilityChanged(opts.showResiduals);
}

// 当控件从隐藏变为可见时触发（如 QStackedWidget 切换到本页面）。
// 若上次 fitInView 时控件不可见（视口为 0），此处重新适配以确保图像充满视图。
void CanvasWidget::showEvent(QShowEvent *event)
{
    QGraphicsView::showEvent(event);
    if (scene() && !scene()->sceneRect().isEmpty() && _zoomFactor == 1.0)
    {
        fitInView(scene()->sceneRect(), Qt::KeepAspectRatio);
    }
}

void CanvasWidget::showImage(const QString &path)
{
    if (!scene() || !_layerRenderer)
    {
        return;
    }

    // 切换影像时立即作废上一幅影像仍在后台运行的匹配观测和残差请求。
    // 两类回调各自比较代次，不能共用一个未声明的“像对加载”计数器。
    ++_featureLoadGeneration;
    ++_residualLoadGeneration;
    ++_maskLoadGeneration;
    _maskSavedRevision = 0;
    if (_maskEditor)
    {
        _maskEditor->setImage({}, {});
    }
    _singleImageReady = false;
    emit displayImageReadyChanged(false);

    // 清理旧覆盖层；影像图层在新影像加载完成后替换，避免磁盘解码期间界面空白。
    // 此处保留当前变换，不能提前 resetTransform()：新影像异步加载期间仍显示旧影像，
    // 提前重置会让旧影像瞬间按原始比例放大，随后又被 fitInView() 缩回。
    // NOTE: 不调用 scene()->clear() —— 那会使 QGraphicsScene 删除所有 items，
    // 导致 LayerRenderer 持有的指针变为悬空并在后续删除时造成双重释放。
    _layerRenderer->clearFeatureLayers();
    _layerRenderer->clearFeatureResidualLayers();
    _layerRenderer->clearMaskLayers();
    _layerRenderer->clearDepthOverlay();
    if (_depthOverlayController)
    {
        _depthOverlayController->cancelPending();
    }
    if (_depthOverlayVisible)
    {
        _depthOverlayVisible = false;
        emit depthOverlayVisibilityChanged(false);
    }

    if (path.trimmed().isEmpty())
    {
        _currentImagePath.clear();
        _layerRenderer->clear();
        resetTransform();
        _viewRotationDegrees = 0;
        _zoomFactor = 1.0;
        return;
    }

    // 记录当前影像路径
    _currentImagePath = path;

    const QString pathCopy = path;
    const QString projectPath = property("currentProjectPath").toString();
    auto *watcher = new QFutureWatcher<QImage>(this);
    _imageWatcher = watcher;
    QPointer<CanvasWidget> self(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, watcher, [self, watcher, loadedPath = pathCopy]()
    {
        const QImage image = watcher->result();
        watcher->deleteLater();
        if (!self)
        {
            return;
        }
        if (watcher != self->_imageWatcher)
        {
            return;
        }
        self->_imageWatcher = nullptr;
        if (QDir::cleanPath(loadedPath) != QDir::cleanPath(self->_currentImagePath))
        {
            return;
        }
        if (image.isNull() || !self->_layerRenderer)
        {
            LOG_WARN(QStringLiteral("showImage: failed to load image %1").arg(loadedPath));
            return;
        }

        self->_layerRenderer->clear();
        if (!self->_layerRenderer->addImageLayer(image, 0))
        {
            return;
        }
        if (self->_maskEditor)
        {
            self->_maskEditor->setImage(image, self->_layerRenderer->imageBounds());
            self->_maskEditor->setOverlayVisible(self->_showMaskOverlay);
        }

        // 新影像已经替换旧影像；在同一次事件回调内重置后立即适配，界面不会绘制
        // 原始比例的中间帧，同时避免把上一张影像的旋转带到新影像。
        self->resetTransform();
        self->_viewRotationDegrees = 0;
        self->_zoomFactor = 1.0;
        const bool isDepthMap = isDepthMapPreviewPath(loadedPath);
        if (!isDepthMap)
        {
            self->reloadMaskOverlay();
        }

        self->_singleImageReady = true;
        emit self->displayImageReadyChanged(true);

        // 通知外部当前活跃影像已变更（MainWindow 据此持久化状态）
        emit self->activeImageChanged(loadedPath);
        self->refreshDepthOverlay();

        // 让视图自动适配内容
        self->scene()->setSceneRect(self->scene()->itemsBoundingRect());
        self->fitInView(self->scene()->sceneRect(), Qt::KeepAspectRatio);

        // 重新适配后，重置缩放因子
        self->_zoomFactor = 1.0;

        // 自动加载已参与匹配的关键点（默认启用）。深度图不属于匹配任务输入，
        // 因此不为它查询 `.pimatch` 分片。
        if (!isDepthMap && self->shouldRenderFeatureDiagnostics()) {
            if (self->_showInterestPoints)
            {
                self->startFeaturePointLoadForImage(loadedPath);
            }
            if (self->_currentFeatureOpts.showResiduals)
            {
                self->startResidualLoadForImage(loadedPath);
            }
        }
    });
    QFuture<QImage> future = QtConcurrent::run([pathCopy, projectPath]() {
        return LayerRenderer::loadImageForDisplay(pathCopy, projectPath);
    });
    watcher->setFuture(future);
}

void CanvasWidget::setProjectMetadata(const QJsonObject &metadata)
{
    if (!_depthOverlayController)
    {
        return;
    }
    const bool current_depth_changed = _depthOverlayController->setProjectMetadata(
        metadata, _currentImagePath);
    _depthOverlayController->setProjectPath(property("currentProjectPath").toString());
    if (current_depth_changed)
    {
        refreshDepthOverlay();
    }
}

void CanvasWidget::setDepthOverlayEnabled(bool enabled)
{
    if (_depthOverlayEnabled == enabled)
    {
        return;
    }
    _depthOverlayEnabled = enabled;
    if (!enabled)
    {
        if (_depthOverlayController)
        {
            _depthOverlayController->cancelPending();
        }
        if (_layerRenderer)
        {
            _layerRenderer->clearDepthOverlay();
        }
        _depthOverlayVisible = false;
        setDepthInspectionActive(false);
        emit depthOverlayVisibilityChanged(false);
        return;
    }
    refreshDepthOverlay();
}

void CanvasWidget::setDepthOverlayLevel(xjw::gui::views::DepthOverlayLevel level)
{
    if (_depthOverlayLevel == level)
    {
        return;
    }
    _depthOverlayLevel = level;
    refreshDepthOverlay();
}

void CanvasWidget::setDepthIntensityVisible(bool visible)
{
    if (_depthIntensityVisible == visible)
    {
        return;
    }
    _depthIntensityVisible = visible;
    refreshDepthOverlay();
}

void CanvasWidget::refreshDepthOverlay()
{
    if (!_depthOverlayController || !_layerRenderer)
    {
        return;
    }
    const bool is_display_image = _singleImageReady
        && !_currentImagePath.trimmed().isEmpty()
        && !isDepthMapPreviewPath(_currentImagePath);
    const auto availability_for = [this, is_display_image](
                                      xjw::gui::views::DepthOverlayLevel level)
    {
        return is_display_image
            ? _depthOverlayController->artifactAvailability(_currentImagePath, level)
            : xjw::gui::views::DepthOverlayAvailability{};
    };
    const auto final_status = availability_for(
        xjw::gui::views::DepthOverlayLevel::Final);
    const auto level_1_status = availability_for(
        xjw::gui::views::DepthOverlayLevel::Level1);
    const auto level_2_status = availability_for(
        xjw::gui::views::DepthOverlayLevel::Level2);
    const auto level_3_status = availability_for(
        xjw::gui::views::DepthOverlayLevel::Level3);
    const bool final_available = final_status.available;
    const bool level_1_available = level_1_status.available;
    const bool level_2_available = level_2_status.available;
    const bool level_3_available = level_3_status.available;
    const bool any_available = is_display_image
        && _depthOverlayController->anyArtifactAvailable(_currentImagePath);
    const bool selected_available = is_display_image
        && _depthOverlayController->artifactAvailable(_currentImagePath, _depthOverlayLevel);
    emit depthOverlayAvailabilityChanged(any_available);
    emit depthOverlayLevelsAvailabilityChanged(final_available,
                                               level_1_available,
                                               level_2_available,
                                               level_3_available,
                                               final_status.reason,
                                               level_1_status.reason,
                                               level_2_status.reason,
                                               level_3_status.reason);
    if (!_depthOverlayEnabled
        || !is_display_image
        || !selected_available)
    {
        _depthOverlayController->cancelPending();
        _layerRenderer->clearDepthOverlay();
        _depthOverlayVisible = false;
        setDepthInspectionActive(false);
        emit depthOverlayVisibilityChanged(false);
        return;
    }

    _layerRenderer->clearDepthOverlay();
    _depthOverlayVisible = false;
    setDepthInspectionActive(true);
    xjw::gui::views::DepthOverlayRenderOptions options;
    options.showIntensity = _depthIntensityVisible;
    _depthOverlayController->request(_currentImagePath, _depthOverlayLevel, options);
}

void CanvasWidget::setDepthInspectionActive(bool active)
{
    if (_depthInspectionActive == active || !_layerRenderer)
    {
        return;
    }

    _depthInspectionActive = active;
    ++_featureLoadGeneration;
    ++_residualLoadGeneration;
    _layerRenderer->clearFeatureLayers();
    _layerRenderer->clearFeatureResidualLayers();

    if (active || _currentImagePath.trimmed().isEmpty()
        || isDepthMapPreviewPath(_currentImagePath))
    {
        return;
    }
    if (_showInterestPoints && _currentFeatureOpts.showPoints)
    {
        startFeaturePointLoadForImage(_currentImagePath);
    }
    if (_currentFeatureOpts.showResiduals)
    {
        startResidualLoadForImage(_currentImagePath);
    }
}

void CanvasWidget::setShowInterestPoints(bool show)
{
    if (_showInterestPoints == show && _currentFeatureOpts.showPoints == show)
    {
        return;
    }
    _showInterestPoints = show;
    _currentFeatureOpts.showPoints = show;
    if (_layerRenderer)
    {
        _layerRenderer->setFeatureDisplayOptions(_currentFeatureOpts);
    }
    if (!_layerRenderer) return;

    if (shouldRenderFeatureDiagnostics()
        && _showInterestPoints && !_currentImagePath.trimmed().isEmpty())
    {
        // 异步加载当前影像实际参与匹配的观测。
        _layerRenderer->clearFeatureLayers();
        startFeaturePointLoadForImage(_currentImagePath);
    }
    else
    {
        _layerRenderer->clearFeatureLayers();
    }
    emit interestPointsVisibilityChanged(show);
}

void CanvasWidget::setShowFeatureResiduals(bool show)
{
    if (_currentFeatureOpts.showResiduals == show)
    {
        return;
    }
    _currentFeatureOpts.showResiduals = show;
    if (!_layerRenderer)
    {
        return;
    }
    _layerRenderer->setFeatureDisplayOptions(_currentFeatureOpts);
    _layerRenderer->clearFeatureResidualLayers();
    if (shouldRenderFeatureDiagnostics()
        && show && !_currentImagePath.trimmed().isEmpty()
        && !isDepthMapPreviewPath(_currentImagePath))
    {
        startResidualLoadForImage(_currentImagePath);
    }
    emit featureResidualVisibilityChanged(show);
}

void CanvasWidget::startFeaturePointLoadForImage(const QString &imagePath)
{
    if (imagePath.trimmed().isEmpty()) return;
    if (!shouldRenderFeatureDiagnostics())
    {
        if (_layerRenderer)
        {
            _layerRenderer->clearFeatureLayers();
        }
        return;
    }

    const QString imagePathCopy = imagePath;
    const QString projectPath = property("currentProjectPath").toString();
    const xjw::gui::views::FeaturePointSource source = _currentFeatureOpts.pointSource;
    const int generation = ++_featureLoadGeneration;
    const QString cacheKey = featurePointCacheKey(imagePathCopy, source);
    auto it = _featurePointCache.find(cacheKey);
    if (it != _featurePointCache.end())
    {
        const QDateTime currentModified = QFileInfo(it->second.sourcePath).lastModified();
        if (!it->second.sourcePath.isEmpty() && it->second.sourceModified == currentModified)
        {
            const bool is_current_image =
                QDir::cleanPath(imagePathCopy) == QDir::cleanPath(_currentImagePath);
            if (is_current_image && _layerRenderer)
            {
                _layerRenderer->setFeatureDisplayOptions(_currentFeatureOpts);
                _layerRenderer->clearFeatureLayers();
                if (!it->second.keypoints.empty())
                {
                    _layerRenderer->addFeatureItems(it->second.keypoints);
                }
            }
            const int count = static_cast<int>(it->second.keypoints.size());
            emit featuresLoaded(imagePathCopy, count);
            publishFeaturePointStatus(
                it->second.message, it->second.available, count, source);
            return;
        }
    }
    
    QPointer<CanvasWidget> self(this);
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [imagePathCopy, projectPath, source]()
        {
            xjw::gui::views::FeaturePointLoadResult result =
                xjw::gui::views::loadFeaturePointsForImage(projectPath, imagePathCopy, source);
            LOG_DEBUG(QStringLiteral("加载%1 %2 个: %3")
                          .arg(xjw::gui::views::featurePointSourceDisplayName(source))
                          .arg(static_cast<int>(result.keypoints.size()))
                          .arg(imagePathCopy));
            return result;
        },
        [self, imagePathCopy, source, cacheKey, generation](
            CanvasWidget *widget,
            xjw::gui::tasks::TaskOutcome<xjw::gui::views::FeaturePointLoadResult> outcome) mutable
        {
            if (!self || widget != self.data() || !self->shouldRenderFeatureDiagnostics())
            {
                return;
            }
            if (generation != self->_featureLoadGeneration
                || source != self->_currentFeatureOpts.pointSource)
            {
                return;
            }
            if (!outcome.succeeded())
            {
                LOG_ERROR("%s", qUtf8Printable(outcome.errorMessage));
                emit self->featuresLoaded(imagePathCopy, 0);
                self->publishFeaturePointStatus(outcome.errorMessage, false, 0, source);
                return;
            }
            xjw::gui::views::FeaturePointLoadResult result = std::move(*outcome.value);
            std::vector<cv::KeyPoint> keypoints = std::move(result.keypoints);

            const bool isCurrentImage =
                QDir::cleanPath(imagePathCopy) == QDir::cleanPath(self->_currentImagePath);
            if (!cacheKey.isEmpty() && !result.sourcePath.isEmpty())
            {
                CanvasWidget::FeaturePointCacheEntry entry;
                entry.sourcePath = result.sourcePath;
                entry.sourceModified = QFileInfo(result.sourcePath).lastModified();
                entry.keypoints = keypoints;
                entry.message = result.message;
                entry.available = result.available;
                self->_featurePointCache[cacheKey] = std::move(entry);
            }
            if (isCurrentImage && self->_layerRenderer)
            {
                // 重新应用当前显示设置，确保使用 UI 中的参数。
                self->_layerRenderer->setFeatureDisplayOptions(self->_currentFeatureOpts);
                self->_layerRenderer->clearFeatureLayers();
                if (!keypoints.empty())
                {
                    self->_layerRenderer->addFeatureItems(keypoints);
                }
            }
            // 发出信号以便主窗体更新状态栏 / 面板。
            const int count = static_cast<int>(keypoints.size());
            emit self->featuresLoaded(imagePathCopy, count);
            self->publishFeaturePointStatus(result.message, result.available, count, source);
        });
}

void CanvasWidget::reloadMaskOverlay()
{
    if (!_layerRenderer)
    {
        return;
    }

    _layerRenderer->clearMaskLayers();
    if (_currentImagePath.trimmed().isEmpty())
    {
        return;
    }
    if (isDepthMapPreviewPath(_currentImagePath))
    {
        return;
    }

    reloadEditableMask();
    if (!_showMaskOverlay)
    {
        return;
    }

    const QString projectPath = property("currentProjectPath").toString();
    const QString maskPath = xjw::common::project::ProjectIO::findMaskForImage(projectPath, _currentImagePath);
    if (maskPath.isEmpty())
    {
        return;
    }
    _layerRenderer->addMaskContourLayer(maskPath);
}

void CanvasWidget::reloadEditableMask()
{
    if (!_maskEditor || _currentImagePath.trimmed().isEmpty())
    {
        return;
    }
    const QString imagePath = _currentImagePath;
    const QString projectPath = property("currentProjectPath").toString();
    const QString maskPath = xjw::common::project::ProjectIO::findMaskForImage(projectPath, imagePath);
    const int generation = ++_maskLoadGeneration;
    const quint64 editorRevision = _maskEditor->revision();
    if (editorRevision != _maskSavedRevision)
    {
        return;
    }
    if (maskPath.isEmpty())
    {
        _maskEditor->setMask({});
        return;
    }

    auto *watcher = new QFutureWatcher<QImage>(this);
    QPointer<CanvasWidget> self(this);
    connect(watcher,
            &QFutureWatcher<QImage>::finished,
            watcher,
            [self, watcher, imagePath, generation, editorRevision]()
            {
                const QImage mask = watcher->result();
                watcher->deleteLater();
                if (!self || !self->_maskEditor || generation != self->_maskLoadGeneration
                    || QDir::cleanPath(imagePath) != QDir::cleanPath(self->_currentImagePath)
                    || editorRevision != self->_maskEditor->revision())
                {
                    return;
                }
                self->_maskEditor->setMask(mask);
            });
    watcher->setFuture(QtConcurrent::run([maskPath]()
    {
        QImageReader reader(maskPath);
        reader.setAutoTransform(false);
        return reader.read();
    }));
}

void CanvasWidget::startResidualLoadForImage(const QString &imagePath)
{
    if (imagePath.trimmed().isEmpty())
    {
        return;
    }
    if (!shouldRenderFeatureDiagnostics())
    {
        if (_layerRenderer)
        {
            _layerRenderer->clearFeatureResidualLayers();
        }
        return;
    }

    const QString projectPath = property("currentProjectPath").toString();
    const QString imagePathCopy = imagePath;
    const int generation = ++_residualLoadGeneration;
    QPointer<CanvasWidget> self(this);
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [projectPath, imagePathCopy]()
        {
            return xjw::gui::views::loadValidTiePointDiagnosticsForImage(
                projectPath, imagePathCopy);
        },
        [self, imagePathCopy, generation](
            CanvasWidget *widget,
            xjw::gui::tasks::TaskOutcome<
                xjw::gui::views::ValidTiePointDiagnostics> outcome)
        {
            if (!self || widget != self.data()
                || !self->shouldRenderFeatureDiagnostics()
                || generation != self->_residualLoadGeneration)
            {
                return;
            }
            if (QDir::cleanPath(imagePathCopy) != QDir::cleanPath(self->_currentImagePath)
                || !self->_layerRenderer)
            {
                return;
            }
            if (!outcome.succeeded())
            {
                LOG_ERROR("%s", qUtf8Printable(outcome.errorMessage));
                emit self->featureResidualAvailabilityChanged(false);
                self->publishFeatureResidualStatus(outcome.errorMessage, false, 0);
                return;
            }
            xjw::gui::views::ValidTiePointDiagnostics diagnostics =
                std::move(*outcome.value);
            QVector<xjw::gui::views::FeatureResidualVector> residuals =
                std::move(diagnostics.residuals);

            self->_layerRenderer->setFeatureDisplayOptions(self->_currentFeatureOpts);
            self->_layerRenderer->clearFeatureResidualLayers();
            if (self->_currentFeatureOpts.showResiduals && !residuals.isEmpty())
            {
                self->_layerRenderer->addFeatureResidualItems(residuals);
            }
            const bool available = !residuals.isEmpty();
            const int count = residuals.size();
            emit self->featureResidualAvailabilityChanged(available);
            self->publishFeatureResidualStatus(diagnostics.message, available, count);
        });
}

void CanvasWidget::setShowMaskOverlay(bool show)
{
    if (_showMaskOverlay == show)
    {
        return;
    }
    _showMaskOverlay = show;
    if (_maskEditor)
    {
        _maskEditor->setOverlayVisible(show);
    }
    reloadMaskOverlay();
    emit maskOverlayVisibilityChanged(show);
}

void CanvasWidget::useRectangleMaskTool()
{
    setMaskTool(MaskEditor::Tool::Rectangle);
}

void CanvasWidget::useScissorsMaskTool()
{
    setMaskTool(MaskEditor::Tool::Scissors);
}

void CanvasWidget::useSmartPaintMaskTool()
{
    setMaskTool(MaskEditor::Tool::SmartPaint);
}

void CanvasWidget::useMagicWandMaskTool()
{
    setMaskTool(MaskEditor::Tool::MagicWand);
}

void CanvasWidget::showMaskEditorSettings()
{
    MaskEditorSettingsDialog dialog(this);
    if (dialog.exec() != QDialog::Accepted || !_maskEditor)
    {
        return;
    }
    const MaskEditorSettings settings = dialog.settings();
    saveMaskEditorSettings(settings);
    _maskEditor->setSettings(settings);
}

void CanvasWidget::resetMaskSelection()
{
    if (_maskEditor)
    {
        _maskEditor->resetSelection();
    }
}

void CanvasWidget::undoMaskEdit()
{
    if (_maskEditor)
    {
        _maskEditor->undo();
    }
}

void CanvasWidget::redoMaskEdit()
{
    if (_maskEditor)
    {
        _maskEditor->redo();
    }
}

void CanvasWidget::confirmInteractiveMaskSaved(const QString &imagePath, quint64 revision)
{
    if (!_maskEditor || revision != _maskEditor->revision()
        || QDir::cleanPath(imagePath) != QDir::cleanPath(_currentImagePath) || !_layerRenderer)
    {
        return;
    }
    _maskSavedRevision = revision;
    _layerRenderer->clearMaskLayers();
    if (!_showMaskOverlay)
    {
        return;
    }
    const QString maskPath = xjw::common::project::ProjectIO::findMaskForImage(
        property("currentProjectPath").toString(), _currentImagePath);
    if (!maskPath.isEmpty())
    {
        _layerRenderer->addMaskContourLayer(maskPath);
    }
}

void CanvasWidget::setMaskTool(MaskEditor::Tool tool)
{
    if (!_maskEditor || !_singleImageReady)
    {
        return;
    }
    _maskEditor->setTool(tool);
    if (!_showMaskOverlay)
    {
        _showMaskOverlay = true;
        _maskEditor->setOverlayVisible(true);
        reloadMaskOverlay();
        emit maskOverlayVisibilityChanged(true);
    }
    setCursor(Qt::CrossCursor);
    setFocus(Qt::ShortcutFocusReason);
}

void CanvasWidget::reloadInterestPoints(const QString &imagePath)
{
    if (imagePath.trimmed().isEmpty()) return;

    // 删除该影像的全部提取器缓存，保证后续异步读取不会命中过期条目。
    clearFeatureCacheForImage(imagePath);

    // 仅在用户开启叠加兴趣点或当前图像为目标时刷新渲染
    if (shouldRenderFeatureDiagnostics()
        && _showInterestPoints && !_currentImagePath.trimmed().isEmpty()) {
        // 如果请求的路径不是当前显示的影像，仍尝试加载以更新缓存（但不叠加到当前视图）
        if (QDir::cleanPath(imagePath) == QDir::cleanPath(_currentImagePath)) {
            // 对当前显示影像，直接触发加载并叠加
            _layerRenderer->clearFeatureLayers();
            startFeaturePointLoadForImage(_currentImagePath);
        } else {
            // 对非当前显示影像，只更新缓存：启动后台加载但不要修改当前 scene
            startFeaturePointLoadForImage(imagePath);
        }
    } else {
        // 即使未开启叠加，也更新缓存以便后续打开影像时能立即得到最新数据
        startFeaturePointLoadForImage(imagePath);
    }
}

QString CanvasWidget::featurePointCacheKey(
    const QString &imagePath,
    xjw::gui::views::FeaturePointSource source) const
{
    const QString cleanPath = QDir::cleanPath(imagePath.trimmed());
    if (cleanPath.isEmpty())
    {
        return QString();
    }

    const QFileInfo fileInfo(cleanPath);
    const QString canonicalPath = fileInfo.canonicalFilePath();
    const QString absolutePath = canonicalPath.isEmpty()
        ? QDir::cleanPath(fileInfo.absoluteFilePath())
        : QDir::cleanPath(canonicalPath);
    return absolutePath + QLatin1Char('\n') + xjw::gui::views::featurePointSourceToken(source);
}

void CanvasWidget::clearFeatureCacheForImage(const QString &imagePath)
{
    _featurePointCache.erase(featurePointCacheKey(
        imagePath, xjw::gui::views::FeaturePointSource::ExtractedFeatures));
    _featurePointCache.erase(featurePointCacheKey(
        imagePath, xjw::gui::views::FeaturePointSource::RawMatches));
    _featurePointCache.erase(featurePointCacheKey(
        imagePath, xjw::gui::views::FeaturePointSource::ValidTiePoints));
}

void CanvasWidget::publishFeaturePointStatus(
    const QString &message,
    bool available,
    int count,
    xjw::gui::views::FeaturePointSource source)
{
    _featurePointStatusMessage = message;
    _featurePointStatusAvailable = available;
    _featurePointStatusCount = count;
    _featurePointStatusSource = source;
    emit featurePointStatusChanged(message, available, count);
}

void CanvasWidget::publishFeatureResidualStatus(
    const QString &message,
    bool available,
    int count)
{
    _featureResidualStatusMessage = message;
    _featureResidualStatusAvailable = available;
    _featureResidualStatusCount = count;
    emit featureResidualStatusChanged(message, available, count);
}

QString CanvasWidget::currentImagePath() const
{
    return _currentImagePath;
}

void CanvasWidget::zoomIn()
{
    // 放大 1.2 倍
    const double next = _zoomFactor * _zoomStep;
    if (next > _zoomMax) return;
    _zoomFactor = next;
    scale(_zoomStep, _zoomStep);
}

void CanvasWidget::zoomOut()
{
    // 缩小 1/1.2
    const double next = _zoomFactor / _zoomStep;
    if (next < _zoomMin) return;
    _zoomFactor = next;
    scale(1.0 / _zoomStep, 1.0 / _zoomStep);
}

void CanvasWidget::resetView()
{
    resetTransform();
    rotate(_viewRotationDegrees);
    _zoomFactor = 1.0;
    if (scene()) {
        fitInView(scene()->sceneRect(), Qt::KeepAspectRatio);
    }
}

void CanvasWidget::rotateLeft()
{
    if (!_singleImageReady)
    {
        return;
    }

    setViewRotationDegrees(_viewRotationDegrees - 90);
    emit viewRotationChanged(_currentImagePath, _viewRotationDegrees);
}

void CanvasWidget::rotateRight()
{
    if (!_singleImageReady)
    {
        return;
    }

    setViewRotationDegrees(_viewRotationDegrees + 90);
    emit viewRotationChanged(_currentImagePath, _viewRotationDegrees);
}

void CanvasWidget::setViewRotationDegrees(int degrees)
{
    const int normalized = xjw::gui::config::normalizeImageViewRotationDegrees(degrees);
    if (normalized == _viewRotationDegrees)
    {
        return;
    }

    const int delta = normalized - _viewRotationDegrees;
    _viewRotationDegrees = normalized;
    if (!_singleImageReady)
    {
        return;
    }

    const QPoint viewportCenter = viewport()->rect().center();
    const QPointF sceneCenter = mapToScene(viewportCenter);
    rotate(delta);
    centerOn(sceneCenter);
}

void CanvasWidget::wheelEvent(QWheelEvent *event)
{
    if (!event)
    {
        return;
    }

    // angleDelta().y() > 0 表示向上滚动（放大）
    const int delta = event->angleDelta().y();
    if (delta == 0)
    {
        event->accept();
        return;
    }

    if (delta > 0)
    {
        zoomIn();
    }
    else
    {
        zoomOut();
    }
    event->accept();
}

void CanvasWidget::mousePressEvent(QMouseEvent *event)
{
    if (!event) return;

    if (_maskEditor
        && _maskEditor->mousePress(mapToScene(event->pos()), event->button(), event->modifiers()))
    {
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton)
    {
        _isPanning = false;
        _lastPanPoint = event->pos();
        setCursor(Qt::OpenHandCursor);
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void CanvasWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (!event) return;

    if (_maskEditor && _maskEditor->mouseMove(mapToScene(event->pos()), event->buttons()))
    {
        event->accept();
        return;
    }

    if (event->buttons() & Qt::LeftButton)
    {
        const QPoint delta = event->pos() - _lastPanPoint;
        if (!_isPanning)
        {
            if (std::abs(delta.x()) >= _panThreshold || std::abs(delta.y()) >= _panThreshold)
            {
                _isPanning = true;
                setCursor(Qt::ClosedHandCursor);
            }
        }

        if (_isPanning)
        {
            // 反向滚动以产生拖动手感
            horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
            verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
            _lastPanPoint = event->pos();
            event->accept();
            return;
        }
    }

    QGraphicsView::mouseMoveEvent(event);
}

void CanvasWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (!event) return;

    if (_maskEditor
        && _maskEditor->mouseRelease(mapToScene(event->pos()), event->button(), event->modifiers()))
    {
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton)
    {
        if (_isPanning)
        {
            // 结束平移
            _isPanning = false;
            setCursor(Qt::ArrowCursor);
            event->accept();
            return;
        }

        // 没有发生拖动：仅将光标恢复为箭头，不进行单击缩放（用户已请求禁用点击放大行为）
        setCursor(Qt::ArrowCursor);
        event->accept();
        return;
    }

    QGraphicsView::mouseReleaseEvent(event);
}

void CanvasWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (event && _maskEditor
        && _maskEditor->mouseDoubleClick(mapToScene(event->pos()), event->button(), event->modifiers()))
    {
        event->accept();
        return;
    }
    QGraphicsView::mouseDoubleClickEvent(event);
}

void CanvasWidget::keyPressEvent(QKeyEvent *event)
{
    if (!event)
    {
        return;
    }
    if (event->key() == Qt::Key_Escape && _maskEditor && _maskEditor->selectionActive())
    {
        resetMaskSelection();
        event->accept();
        return;
    }
    if (event->matches(QKeySequence::Undo) && _maskEditor && _maskEditor->canUndo())
    {
        undoMaskEdit();
        event->accept();
        return;
    }
    if (event->matches(QKeySequence::Redo) && _maskEditor && _maskEditor->canRedo())
    {
        redoMaskEdit();
        event->accept();
        return;
    }
    QGraphicsView::keyPressEvent(event);
}

void CanvasWidget::contextMenuEvent(QContextMenuEvent *event)
{
    if (!event || !_singleImageReady || _currentImagePath.trimmed().isEmpty() || !_layerRenderer)
    {
        QGraphicsView::contextMenuEvent(event);
        return;
    }

    const QPointF original_pixel = mapToScene(event->pos());
    if (!_layerRenderer->imageBounds().contains(original_pixel))
    {
        event->ignore();
        return;
    }

    emit imageContextRequested(_currentImagePath, original_pixel);
    event->accept();
}
