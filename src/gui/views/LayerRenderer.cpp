#include "LayerRenderer.h"
#include "LayerImageLoader.h"
#include "LayerOverlayItems.h"
#include "Logger.h"
#include "MaskGenerator.h"
#include "io/PathIO.h"

#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QTransform>

#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QPainterPath>
#include <QPen>
#include <QVector>
#include <QPointF>

namespace
{

QPen makeMaskContourPen(const QColor &color, qreal width)
{
    QPen pen(color, width, Qt::SolidLine, Qt::SquareCap, Qt::RoundJoin);
    pen.setCosmetic(true);
    return pen;
}

} // namespace

LayerRenderer::LayerRenderer(QGraphicsScene *scene, QObject *parent)
    : QObject(parent)
    , _scene(scene)
{
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

bool LayerRenderer::addMaskContourLayer(const QString &maskPath, int z)
{
    if (!_scene || maskPath.trimmed().isEmpty())
    {
        return false;
    }

    const cv::Mat mask = xjw::common::io::readImage(maskPath, cv::IMREAD_GRAYSCALE);
    if (mask.empty())
    {
        return false;
    }

    const auto contours = xjw::mask::extractMaskContours(mask, true);
    if (contours.empty())
    {
        return false;
    }

    QPainterPath path;
    for (const auto &contour : contours)
    {
        if (contour.size() < 2)
        {
            continue;
        }

        path.moveTo(contour.front().x, contour.front().y);
        for (std::size_t i = 1; i < contour.size(); ++i)
        {
            path.lineTo(contour.at(i).x, contour.at(i).y);
        }
        path.closeSubpath();
    }

    if (path.isEmpty())
    {
        return false;
    }

    auto *halo = new QGraphicsPathItem(path);
    halo->setPen(makeMaskContourPen(QColor(0, 0, 0, 210), 4.0));
    halo->setZValue(z);
    _scene->addItem(halo);
    _maskItems.append(halo);

    auto *outline = new QGraphicsPathItem(path);
    outline->setPen(makeMaskContourPen(QColor(255, 255, 255, 245), 1.6));
    outline->setZValue(z + 0.1);
    _scene->addItem(outline);
    _maskItems.append(outline);
    return true;
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
