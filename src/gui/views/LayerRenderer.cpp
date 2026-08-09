#include "LayerRenderer.h"
#include "LayerImageLoader.h"
#include "LayerOverlayItems.h"
#include "Logger.h"

#include <opencv2/core.hpp>

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QCoreApplication>
#include <QImage>
#include <QMetaObject>
#include <QPixmap>
#include <QTransform>

#include <QGraphicsItem>
#include <QPainterPath>
#include <QPointer>
#include <QThreadPool>
#include <QVector>
#include <QPointF>

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <memory>

namespace
{

bool cancellationRequested(const std::shared_ptr<std::atomic<bool>> &cancellation)
{
    return cancellation && cancellation->load(std::memory_order_relaxed);
}

class MaskContourThreadPool final : public QThreadPool
{
public:
    MaskContourThreadPool()
    {
        setMaxThreadCount(2);
        setExpiryTimeout(30'000);
    }
};

QThreadPool *maskContourThreadPool()
{
    static MaskContourThreadPool pool;
    return &pool;
}

} // namespace

LayerRenderer::LayerRenderer(QGraphicsScene *scene, QObject *parent)
    : QObject(parent)
    , _scene(scene)
{
    _maskContourCache.setMaxCost(MaximumMaskContourCacheCost);
}

LayerRenderer::~LayerRenderer()
{
    if (_maskLoadCancellation)
    {
        _maskLoadCancellation->store(true, std::memory_order_relaxed);
    }
}

void LayerRenderer::setFeatureDisplayOptions(const FeatureDisplayOptions &opts)
{
    _featureOpts = opts;
}

QImage LayerRenderer::loadImageForDisplay(const QString &path, const QString &plascanPath)
{
    return xjw::gui::views::loadImageForDisplay(path, plascanPath);
}

bool LayerRenderer::addImageLayer(const QImage &image, int z)
{
    if (!_scene || image.isNull())
    {
        return false;
    }

    QPixmap pix = QPixmap::fromImage(image);
    auto *item = _scene->addPixmap(pix);
    if (!item)
    {
        return false;
    }
    // Ensure newly added pixmap is positioned at origin and visible
    item->setPos(0, 0);
    item->setVisible(true);
    item->setZValue(z);
    _layers.append(item);
    if (z == 0 && !_baseImageItem)
    {
        _baseImageItem = item;
        _baseImagePixmap = pix;
        _baseImageTransform = item->transform();
        _baseImagePosition = item->pos();
    }
    _imageBounds = _imageBounds.isNull() ? item->sceneBoundingRect() : _imageBounds.united(item->sceneBoundingRect());
    return true;
}

bool LayerRenderer::setDepthOverlay(const QImage &overlay,
                                    const QImage &intensity_base,
                                    int z)
{
    if (!_scene || overlay.isNull())
    {
        return false;
    }

    clearDepthOverlay();
    const QRectF target_bounds = _imageBounds.isEmpty()
        ? QRectF(QPointF(0.0, 0.0), QSizeF(overlay.size()))
        : _imageBounds;
    _depthOverlayItem = _scene->addPixmap(QPixmap::fromImage(overlay));
    if (!_depthOverlayItem)
    {
        return false;
    }
    _depthOverlayItem->setPos(target_bounds.topLeft());
    _depthOverlayItem->setTransform(QTransform::fromScale(
        target_bounds.width() / static_cast<qreal>(overlay.width()),
        target_bounds.height() / static_cast<qreal>(overlay.height())));
    _depthOverlayItem->setZValue(z);

    if (!intensity_base.isNull() && _baseImageItem)
    {
        _baseImageItem->setPixmap(QPixmap::fromImage(intensity_base));
        _baseImageItem->setPos(target_bounds.topLeft());
        _baseImageItem->setTransform(QTransform::fromScale(
            target_bounds.width() / static_cast<qreal>(intensity_base.width()),
            target_bounds.height() / static_cast<qreal>(intensity_base.height())));
        _intensityBaseActive = true;
    }
    return true;
}

void LayerRenderer::clearDepthOverlay()
{
    if (_depthOverlayItem)
    {
        if (_scene)
        {
            _scene->removeItem(_depthOverlayItem);
        }
        delete _depthOverlayItem;
        _depthOverlayItem = nullptr;
    }

    if (_intensityBaseActive && _baseImageItem)
    {
        _baseImageItem->setPixmap(_baseImagePixmap);
        _baseImageItem->setTransform(_baseImageTransform);
        _baseImageItem->setPos(_baseImagePosition);
    }
    _intensityBaseActive = false;
}

bool LayerRenderer::addMaskContourLayer(const QString &mask_path, int z)
{
    clearMaskLayers();
    if (!_scene || mask_path.trimmed().isEmpty())
    {
        return false;
    }

    const auto source = xjw::gui::views::resolveMaskContourSource(mask_path);
    if (!source)
    {
        emit maskContourLayerFailed(mask_path);
        return false;
    }

    const quint64 generation = _maskLoadGeneration;
    if (const QPainterPath *cached_path = _maskContourCache.object(source->cacheKey))
    {
        const bool installed = installMaskContourLayer(*cached_path, z);
        if (installed)
        {
            emit maskContourLayerReady(source->normalizedPath, true);
        }
        else
        {
            emit maskContourLayerFailed(source->normalizedPath);
        }
        return installed;
    }

    auto cancellation = std::make_shared<std::atomic<bool>>(false);
    _maskLoadCancellation = cancellation;
    const QPointer<LayerRenderer> self(this);
    const xjw::gui::views::MaskContourSource source_copy = *source;
    maskContourThreadPool()->start(
        [self, source_copy, cancellation, generation, z]()
        {
            QPainterPath path;
            QString error_message;
            try
            {
                path = xjw::gui::views::extractMaskContoursPath(
                    source_copy.normalizedPath, cancellation);
            }
            catch (const cv::Exception &exception)
            {
                error_message = QStringLiteral("OpenCV: %1")
                    .arg(QString::fromUtf8(exception.what()));
            }
            catch (const std::exception &exception)
            {
                error_message = QString::fromUtf8(exception.what());
            }
            catch (...)
            {
                error_message = QStringLiteral("未知后台异常");
            }
            if (cancellationRequested(cancellation))
            {
                return;
            }

            QCoreApplication *application = QCoreApplication::instance();
            if (!application)
            {
                return;
            }

            QMetaObject::invokeMethod(
                application,
                [self,
                 source_copy,
                 cancellation,
                 generation,
                 z,
                 path = std::move(path),
                 error_message = std::move(error_message)]() mutable
                {
                    if (!self || cancellationRequested(cancellation)
                        || generation != self->_maskLoadGeneration)
                    {
                        return;
                    }

                    const auto current_source = xjw::gui::views::resolveMaskContourSource(
                        source_copy.normalizedPath);
                    if (!current_source || current_source->cacheKey != source_copy.cacheKey)
                    {
                        emit self->maskContourLayerFailed(source_copy.normalizedPath);
                        return;
                    }
                    if (!error_message.isEmpty())
                    {
                        LOG_WARN(QStringLiteral("蒙版轮廓后台提取失败：%1（%2）")
                                     .arg(source_copy.normalizedPath, error_message));
                        emit self->maskContourLayerFailed(source_copy.normalizedPath);
                        return;
                    }
                    if (path.isEmpty())
                    {
                        emit self->maskContourLayerFailed(source_copy.normalizedPath);
                        return;
                    }

                    const int element_count = std::min(
                        path.elementCount(), std::numeric_limits<int>::max());
                    const int cache_cost = std::max(
                        LayerRenderer::MinimumMaskContourCacheEntryCost, element_count);
                    self->_maskContourCache.insert(
                        source_copy.cacheKey, new QPainterPath(path), cache_cost);

                    if (!self->installMaskContourLayer(path, z))
                    {
                        emit self->maskContourLayerFailed(source_copy.normalizedPath);
                        return;
                    }
                    emit self->maskContourLayerReady(source_copy.normalizedPath, false);
                },
                Qt::QueuedConnection);
        });
    return true;
}

bool LayerRenderer::installMaskContourLayer(const QPainterPath &path, int z)
{
    if (!_scene || path.isEmpty())
    {
        return false;
    }

    const QList<QGraphicsItem *> items = xjw::gui::views::createMaskContourOverlayItems(path, z);
    for (QGraphicsItem *item : items)
    {
        _scene->addItem(item);
        _maskItems.append(item);
    }
    return !items.isEmpty();
}

void LayerRenderer::clearFeatureLayers()
{
    // Remove items we explicitly tracked
    for (auto *it: std::as_const(_featureItems))
    {
        if (it && _scene)
        {
            _scene->removeItem(it);
            delete it;
        }
    }
    _featureItems.clear();
}

void LayerRenderer::clearFeatureResidualLayers()
{
    for (auto *item : std::as_const(_featureResidualItems))
    {
        if (item && _scene)
        {
            _scene->removeItem(item);
            delete item;
        }
    }
    _featureResidualItems.clear();
}

void LayerRenderer::clearMaskLayers()
{
    ++_maskLoadGeneration;
    if (_maskLoadCancellation)
    {
        _maskLoadCancellation->store(true, std::memory_order_relaxed);
        _maskLoadCancellation.reset();
    }
    clearMaskLayerItems();
}

void LayerRenderer::clearMaskLayerItems()
{
    for (auto *it : std::as_const(_maskItems))
    {
        if (it && _scene)
        {
            _scene->removeItem(it);
            delete it;
        }
    }
    _maskItems.clear();
}

void LayerRenderer::addFeatureItems(const std::vector<cv::KeyPoint> &keypoints)
{
    if (!_scene) return;
    // Debug incoming keypoints for troubleshooting feature rendering
    LOG_DEBUG(QStringLiteral("addFeatureItems: incoming keypoints=%1").arg(static_cast<int>(keypoints.size())));

    if (keypoints.empty() || !_featureOpts.showPoints)
    {
        return;
    }

    auto *item = xjw::gui::views::createFeatureOverlayItem(keypoints, _featureOpts, _imageBounds);
    _scene->addItem(item);
    _featureItems.append(item);
    const int added = 1;
    LOG_DEBUG(QStringLiteral("addFeatureItems: added items=%1 total_scene_items=%2")
                  .arg(added)
                  .arg(_scene ? _scene->items().size() : 0));
}

void LayerRenderer::addFeatureResidualItems(
    const QVector<xjw::gui::views::FeatureResidualVector> &residuals)
{
    if (!_scene || residuals.isEmpty() || !_featureOpts.showResiduals)
    {
        return;
    }

    auto *item = xjw::gui::views::createFeatureResidualOverlayItem(
        residuals, _featureOpts, _imageBounds);
    _scene->addItem(item);
    _featureResidualItems.append(item);
}

void LayerRenderer::clear()
{
    clearDepthOverlay();
    clearMaskLayers();
    clearFeatureResidualLayers();

    for (auto *it: std::as_const(_layers))
    {
        if (it && _scene)
        {
            _scene->removeItem(it);
            delete it;
        }
    }
    _layers.clear();
    _baseImageItem = nullptr;
    _baseImagePixmap = QPixmap();
    _baseImageTransform = QTransform();
    _baseImagePosition = QPointF();
    _imageBounds = QRectF();
}
