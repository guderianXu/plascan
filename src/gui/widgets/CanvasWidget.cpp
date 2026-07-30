#include "CanvasWidget.h"

#include "LayerFeatureLoader.h"
#include "FeatureResidualLoader.h"
#include "LayerRenderer.h"
#include "DepthOverlayController.h"
#include "project/ProjectIO.h"
#include "GuiTaskRunner.h"
#include "io/PathIO.h"
#include "project/AspMatchPointReader.h"

#include <QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
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
#include <QVariant>
#include <QVariantMap>
#include <QFile>
#include <QDataStream>
#include <QPointF>
#include <QVector>
#include <QGraphicsPixmapItem>
#include "Logger.h"
#include "ImageViewRotationSettings.h"

namespace
{

struct MatchPairLoadResult
{
    int generation = 0;
    QString imagePathA;
    QString imagePathB;
    QString matchFilePath;
    QImage imageA;
    QImage imageB;
    xjw::common::project::AspMatchPointResult matches;
};

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
        startSpLoadForImage(_currentImagePath);
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

    ++_matchPairLoadGeneration;
    _singleImageReady = false;
    _viewRotationDegrees = 0;
    _zoomFactor = 1.0;
    resetTransform();
    emit displayImageReadyChanged(false);

    // 清理旧覆盖层；影像图层在新影像加载完成后替换，避免磁盘解码期间界面空白。
    // NOTE: 不调用 scene()->clear() —— 那会使 QGraphicsScene 删除所有 items，
    // 导致 LayerRenderer 持有的指针变为悬空并在后续删除时造成双重释放。
    _layerRenderer->clearFeatureLayers();
    _layerRenderer->clearFeatureResidualLayers();
    // 确保清除上一次的匹配连线层，避免其干扰新的场景布局
    _layerRenderer->clearMatchLayers();
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
        _zoomFactor = 1.0;
        return;
    }

    // 将当前项目路径注入渲染器，用于把非 8-bit 影像转换缓存写入项目 .plascan_tmp。
    // 说明：CanvasWidget 不直接依赖 ProjectManager 头文件，这里通过 QObject 动态属性读取。
    // MainWindow/ProjectManager 会在运行时设置该属性。
    if (_layerRenderer)
    {
        const QVariant v = property("currentProjectPath");
        if (v.isValid())
        {
            _layerRenderer->setCurrentProjectPath(v.toString());
        }
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

        // 自动加载特征点（默认启用）
        // 跳过非项目影像（如深度图 depth_*.png）的特征点加载，避免无意义的 .sp 查找
        if (!isDepthMap && self->shouldRenderFeatureDiagnostics()) {
            if (self->_showInterestPoints)
            {
                self->startSpLoadForImage(loadedPath);
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
    _depthOverlayController->setProjectMetadata(metadata);
    _depthOverlayController->setProjectPath(property("currentProjectPath").toString());
    refreshDepthOverlay();
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
        startSpLoadForImage(_currentImagePath);
    }
    if (_currentFeatureOpts.showResiduals)
    {
        startResidualLoadForImage(_currentImagePath);
    }
}

void CanvasWidget::showMatchedPair(const QString &imgA, const QString &imgB, const QString &matchFile)
{
    if (!scene() || !_layerRenderer)
    {
        return;
    }

    const int generation = ++_matchPairLoadGeneration;
    _singleImageReady = false;
    _viewRotationDegrees = 0;
    _currentImagePath.clear();
    emit displayImageReadyChanged(false);
    resetTransform();

    LOG_DEBUG(QStringLiteral("showMatchedPair: imgA=%1 imgB=%2 match=%3").arg(imgA, imgB, matchFile));

    // clear existing layers
    _layerRenderer->clearFeatureLayers();
    _layerRenderer->clearMatchLayers();
    _layerRenderer->clearMaskLayers();
    _layerRenderer->clear();

    // Ensure renderer knows project path (so caching/convert works same as showImage)
    const QString projectPath = property("currentProjectPath").toString();
    if (!projectPath.isEmpty())
    {
        _layerRenderer->setCurrentProjectPath(projectPath);
        if (_depthOverlayController)
        {
            _depthOverlayController->setProjectPath(projectPath);
        }
    }

    xjw::gui::tasks::runGuarded(
        this,
        [generation, imgA, imgB, matchFile, projectPath]()
        {
            MatchPairLoadResult result;
            result.generation = generation;
            result.imagePathA = imgA;
            result.imagePathB = imgB;
            result.matchFilePath = matchFile;
            result.imageA = LayerRenderer::loadImageForDisplay(imgA, projectPath);
            result.imageB = LayerRenderer::loadImageForDisplay(imgB, projectPath);
            result.matches = xjw::common::project::readAspMatchPoints(matchFile);
            return result;
        },
        [](CanvasWidget *self, MatchPairLoadResult result)
        {
            if (result.generation != self->_matchPairLoadGeneration
                || !self->_layerRenderer
                || !self->scene())
            {
                return;
            }

            if (!result.matches.success)
            {
                LOG_WARN(QStringLiteral("showMatchedPair: %1")
                             .arg(result.matches.errorMessage));
            }

            QGraphicsPixmapItem *itemA = nullptr;
            QGraphicsPixmapItem *itemB = nullptr;
            if (!self->_layerRenderer->addStitchedImagePair(
                    result.imageA,
                    result.imageB,
                    result.imagePathA,
                    result.imagePathB,
                    &itemA,
                    &itemB,
                    20))
            {
                LOG_WARN(QStringLiteral("showMatchedPair: 影像加载失败 %1 <-> %2")
                             .arg(result.imagePathA, result.imagePathB));
                self->showImage(result.imagePathA);
                return;
            }

            const qreal secondImageOffset = itemB ? itemB->pos().x() : 0.0;
            if (!result.matches.leftPoints.isEmpty()
                && !result.matches.rightPoints.isEmpty())
            {
                self->_layerRenderer->addMatchLines(
                    result.matches.leftPoints,
                    result.matches.rightPoints,
                    secondImageOffset);
            }

            const QRectF bounds = self->scene()->itemsBoundingRect();
            if (bounds.isEmpty())
            {
                self->_layerRenderer->clear();
                self->showImage(result.imagePathA);
                return;
            }

            self->scene()->setSceneRect(bounds);
            self->fitInView(bounds, Qt::KeepAspectRatio);
            self->_zoomFactor = 1.0;
        });
}

void CanvasWidget::setActiveFeatureSuffix(const QString &suffix)
{
    if (suffix.isEmpty() || suffix == _activeFeatureSuffix) return;
    _activeFeatureSuffix = suffix;
    // 清除当前影像的缓存, 强制重新加载
    if (!_currentImagePath.isEmpty())
        setShowInterestPoints(_showInterestPoints);
}

QStringList CanvasWidget::availableFeatureSuffixes() const
{
    if (_currentImagePath.isEmpty()) return {};
    const QString projectPath = property("currentProjectPath").toString();
    return xjw::common::project::ProjectIO::availableFeatureSuffixes(projectPath, _currentImagePath);
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
        // 异步加载当前影像的特征文件
        _layerRenderer->clearFeatureLayers();
        startSpLoadForImage(_currentImagePath);
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

void CanvasWidget::startSpLoadForImage(const QString &imagePath)
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
    const QString activeSuffix = _activeFeatureSuffix;
    const QString projectPath = property("currentProjectPath").toString();
    const bool shouldEstimateOrientation = _currentFeatureOpts.showOrientation;
    const int generation = ++_featureLoadGeneration;
    // 检查缓存 (key 含 suffix)
    QFileInfo fiCheck(imagePathCopy);
    const QString cacheKey = featureCacheKey(imagePathCopy, activeSuffix);
    auto it = _spCache.find(cacheKey);
    if (it != _spCache.end()) {
        if (it->second.first == fiCheck.lastModified()) {
            const bool isCurrentImage = QDir::cleanPath(imagePathCopy) == QDir::cleanPath(_currentImagePath);
            if (isCurrentImage && _layerRenderer) {
                _layerRenderer->setFeatureDisplayOptions(_currentFeatureOpts);
                _layerRenderer->clearFeatureLayers();
                if (!it->second.second.empty()) _layerRenderer->addFeatureItems(it->second.second);
            }
            emit featuresLoaded(imagePathCopy, static_cast<int>(it->second.second.size()));
            return;
        }
    }
    
    QPointer<CanvasWidget> self(this);
    xjw::gui::tasks::runGuarded(
        this,
        [imagePathCopy, activeSuffix, projectPath, shouldEstimateOrientation]() -> std::vector<cv::KeyPoint>
        {
            std::vector<cv::KeyPoint> empty;
            // 使用当前选中的后缀查找特征文件。
            const QString spFile = xjw::common::project::ProjectIO::featureOutputPathForImage(
                projectPath, imagePathCopy, activeSuffix);
            LOG_DEBUG(QStringLiteral("startSpLoadForImage: suffix=%1 file=%2")
                .arg(activeSuffix, spFile));

            if (spFile.isEmpty() || !QFile::exists(spFile))
            {
                LOG_DEBUG(QStringLiteral("startSpLoadForImage: no %1 found for %2")
                    .arg(activeSuffix, imagePathCopy));
                return empty;
            }

            // 读取特征文件 (支持所有提取器类型)。
            std::vector<cv::KeyPoint> keypoints = xjw::gui::views::loadFeatureKeypointsFromFile(spFile);
            if (keypoints.empty())
            {
                LOG_WARN(QStringLiteral("startSpLoadForImage: failed to read feature file %1").arg(spFile));
                return empty;
            }

            if (shouldEstimateOrientation)
            {
                // 尝试从影像中估计每个 keypoint 的方向（梯度方向），以便显示方向箭头。
                // 默认不开方向显示时跳过整图 imread/Sobel，避免切换影像时抢占磁盘与 CPU。
                try
                {
                    cv::Mat img = xjw::common::io::readImage(imagePathCopy, cv::IMREAD_GRAYSCALE);
                    if (!img.empty())
                    {
                        cv::Mat gx, gy;
                        cv::Sobel(img, gx, CV_32F, 1, 0, 3);
                        cv::Sobel(img, gy, CV_32F, 0, 1, 3);
                        for (auto &kp : keypoints)
                        {
                            int x = static_cast<int>(std::round(kp.pt.x));
                            int y = static_cast<int>(std::round(kp.pt.y));
                            if (x >= 0 && x < gx.cols && y >= 0 && y < gx.rows)
                            {
                                float vx = gx.at<float>(y, x);
                                float vy = gy.at<float>(y, x);
                                if (std::isfinite(vx) &&
                                    std::isfinite(vy) &&
                                    (std::abs(vx) > 1e-6f || std::abs(vy) > 1e-6f))
                                {
                                    double ang =
                                        std::atan2(static_cast<double>(vy), static_cast<double>(vx)) * 180.0 / M_PI;
                                    if (ang < 0)
                                    {
                                        ang += 360.0;
                                    }
                                    kp.angle = static_cast<float>(ang);
                                }
                                else
                                {
                                    kp.angle = 0.0f;
                                }
                            }
                            else
                            {
                                kp.angle = 0.0f;
                            }
                        }
                    }
                }
                catch (...)
                {
                    // 估计失败则忽略，保持原有角度值。
                }
            }

            LOG_DEBUG(QStringLiteral("startSpLoadForImage: loaded %1 keypoints from %2")
                          .arg(static_cast<int>(keypoints.size()))
                          .arg(spFile));
            return keypoints;
        },
        [self, imagePathCopy, activeSuffix, generation](CanvasWidget *widget,
                                                        std::vector<cv::KeyPoint> kps) mutable
        {
            if (!self || widget != self.data() || !self->shouldRenderFeatureDiagnostics())
            {
                return;
            }
            if (generation != self->_featureLoadGeneration)
            {
                return;
            }

            const bool isCurrentImage = QDir::cleanPath(imagePathCopy) == QDir::cleanPath(self->_currentImagePath);
            if (!imagePathCopy.trimmed().isEmpty())
            {
                QFileInfo fi(imagePathCopy);
                const QString key = self->featureCacheKey(imagePathCopy, activeSuffix);
                self->_spCache[key] = std::make_pair(fi.lastModified(), kps);
            }
            if (isCurrentImage && self->_layerRenderer)
            {
                // 重新应用当前显示设置，确保使用 UI 中的参数。
                self->_layerRenderer->setFeatureDisplayOptions(self->_currentFeatureOpts);
                self->_layerRenderer->clearFeatureLayers();
                if (!kps.empty())
                {
                    self->_layerRenderer->addFeatureItems(kps);
                }
            }
            // 发出信号以便主窗体更新状态栏 / 面板。
            emit self->featuresLoaded(imagePathCopy, static_cast<int>(kps.size()));
        });
}

void CanvasWidget::reloadMaskOverlay()
{
    if (!_layerRenderer)
    {
        return;
    }

    _layerRenderer->clearMaskLayers();
    if (!_showMaskOverlay || _currentImagePath.trimmed().isEmpty())
    {
        return;
    }
    if (isDepthMapPreviewPath(_currentImagePath))
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
    xjw::gui::tasks::runGuarded(
        this,
        [projectPath, imagePathCopy]()
        {
            return xjw::gui::views::loadFeatureResidualsForImage(projectPath, imagePathCopy);
        },
        [self, imagePathCopy, generation](CanvasWidget *widget,
                                          QVector<xjw::gui::views::FeatureResidualVector> residuals)
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

            self->_layerRenderer->setFeatureDisplayOptions(self->_currentFeatureOpts);
            self->_layerRenderer->clearFeatureResidualLayers();
            if (self->_currentFeatureOpts.showResiduals && !residuals.isEmpty())
            {
                self->_layerRenderer->addFeatureResidualItems(residuals);
            }
            emit self->featureResidualAvailabilityChanged(!residuals.isEmpty());
        });
}

void CanvasWidget::setShowMaskOverlay(bool show)
{
    if (_showMaskOverlay == show)
    {
        return;
    }
    _showMaskOverlay = show;
    reloadMaskOverlay();
    emit maskOverlayVisibilityChanged(show);
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
            startSpLoadForImage(_currentImagePath);
        } else {
            // 对非当前显示影像，只更新缓存：启动后台加载但不要修改当前 scene
            startSpLoadForImage(imagePath);
        }
    } else {
        // 即使未开启叠加，也更新缓存以便后续打开影像时能立即得到最新数据
        startSpLoadForImage(imagePath);
    }
}

void CanvasWidget::immediateReloadInterestPoints(const QString &imagePath)
{
    reloadInterestPoints(imagePath);
}

QList<QVariantMap> CanvasWidget::getCachedInterestPointsAsVariant(const QString &imagePath) const
{
    QList<QVariantMap> out;
    if (imagePath.trimmed().isEmpty()) return out;
    const auto it = _spCache.find(featureCacheKey(imagePath, _activeFeatureSuffix));
    if (it == _spCache.end()) return out;
    for (const auto &kp : it->second.second) {
        QVariantMap m;
        m.insert(QStringLiteral("x"), kp.pt.x);
        m.insert(QStringLiteral("y"), kp.pt.y);
        m.insert(QStringLiteral("size"), kp.size);
        m.insert(QStringLiteral("angle"), kp.angle);
        m.insert(QStringLiteral("response"), kp.response); // score
        out.append(m);
    }
    return out;
}

QString CanvasWidget::featureCacheKey(const QString &imagePath, const QString &suffix) const
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
    return absolutePath + QChar(0x1F) + suffix.trimmed().toLower();
}

void CanvasWidget::clearFeatureCacheForImage(const QString &imagePath)
{
    const QString keyPrefix = featureCacheKey(imagePath, QString()).section(QChar(0x1F), 0, 0);
    if (keyPrefix.isEmpty())
    {
        return;
    }

    const QString prefix = keyPrefix + QChar(0x1F);
    for (auto it = _spCache.begin(); it != _spCache.end();)
    {
        if (it->first.startsWith(prefix))
        {
            it = _spCache.erase(it);
        }
        else
        {
            ++it;
        }
    }
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
