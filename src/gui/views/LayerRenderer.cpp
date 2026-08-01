#include "LayerRenderer.h"
#include "LayerFeatureLoader.h"
#include "LayerImageLoader.h"
#include "LayerOverlayItems.h"
#include "LayerStitchedDebug.h"
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

void LayerRenderer::setCurrentProjectPath(const QString &plascanPath)
{
    _currentProjectPath = plascanPath;
}

bool LayerRenderer::addImageLayer(const QString &path, int z)
{
    return addImageLayer(loadImageForDisplay(path, _currentProjectPath), z);
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

bool LayerRenderer::addFeatureLayerFromVwip(const QString &imagePath)
{
    if (!_scene) return false;

    const auto keypoints = xjw::gui::views::loadMatchedKeypointsForImage(_currentProjectPath, imagePath);
    if (keypoints.empty()) return false;
    addFeatureItems(keypoints);
    return true;
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

bool LayerRenderer::addStitchedImagePair(const QString &pathA,
                                         const QString &pathB,
                                         QGraphicsPixmapItem **outA,
                                         QGraphicsPixmapItem **outB,
                                         int gap)
{
    const QImage imageA = loadImageForDisplay(pathA, _currentProjectPath);
    const QImage imageB = loadImageForDisplay(pathB, _currentProjectPath);
    return addStitchedImagePair(imageA, imageB, pathA, pathB, outA, outB, gap);
}

bool LayerRenderer::addStitchedImagePair(const QImage &imageA,
                                         const QImage &imageB,
                                         const QString &sourcePathA,
                                         const QString &sourcePathB,
                                         QGraphicsPixmapItem **outA,
                                         QGraphicsPixmapItem **outB,
                                         int gap)
{
    if (!_scene || imageA.isNull() || imageB.isNull())
    {
        return false;
    }

    LOG_DEBUG(QStringLiteral("addStitchedImagePair: %1 <-> %2")
                  .arg(sourcePathA, sourcePathB));

    // clear existing image layers (we expect caller to manage state)
    // We'll add both images using addImageLayer then reposition the second.
    const int before = _layers.size();
    if (!addImageLayer(imageA, 0)) return false;
    QGraphicsPixmapItem *itemA = nullptr;
    if (!_layers.isEmpty()) itemA = _layers.last();

    if (!addImageLayer(imageB, 0)) {
        // cleanup the first if second failed
        if (itemA) {
            _scene->removeItem(itemA);
            _layers.removeOne(itemA);
            delete itemA;
        }
        return false;
    }
    QGraphicsPixmapItem *itemB = nullptr;
    if (!_layers.isEmpty() && _layers.size() > before + 0) itemB = _layers.last();

    if (!itemA || !itemB) return false;

    LOG_DEBUG(QStringLiteral("addStitchedImagePair: itemA size=%1x%2 itemB size=%3x%4")
                  .arg(itemA->pixmap().width())
                  .arg(itemA->pixmap().height())
                  .arg(itemB->pixmap().width())
                  .arg(itemB->pixmap().height()));

    // position B to the right of A
    qreal bx = itemA->pixmap().width() + gap;
    itemB->setPos(bx, 0);
    _imageBounds = itemA->sceneBoundingRect().united(itemB->sceneBoundingRect());

    if (outA) *outA = itemA;
    if (outB) *outB = itemB;

    xjw::gui::views::recordStitchedImagePairDebug(
        _scene,
        _currentProjectPath,
        sourcePathA,
        sourcePathB,
        itemA,
        itemB,
        gap);
    return true;
}

void LayerRenderer::addMatchLines(const QVector<QPointF> &ptsA, const QVector<QPointF> &ptsB, qreal bOffsetX)
{
    if (!_scene) return;
    if (!_matchOpts.showLines) return; // 如果不显示匹配线,直接返回
    
    clearMatchLayers();

    LOG_DEBUG(QStringLiteral("addMatchLines: ptsA=%1 ptsB=%2 bOffsetX=%3")
                  .arg(ptsA.size())
                  .arg(ptsB.size())
                  .arg(bOffsetX));

    const auto items = xjw::gui::views::createMatchOverlayItems(ptsA, ptsB, _matchOpts, bOffsetX);
    for (QGraphicsItem *item : items)
    {
        if (!item)
        {
            continue;
        }
        _scene->addItem(item);
        _matchItems.append(item);
    }
}

void LayerRenderer::setMatchDisplayOptions(const MatchDisplayOptions &opts)
{
    _matchOpts = opts;
}

void LayerRenderer::clearMatchLayers()
{
    for (auto *it: std::as_const(_matchItems)) {
        if (it && _scene) {
            _scene->removeItem(it);
            delete it;
        }
    }
    _matchItems.clear();
}
