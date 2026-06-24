#include "LayerRenderer.h"
#include "LayerFeatureLoader.h"
#include "LayerImageLoader.h"
#include "LayerOverlayItems.h"
#include "LayerStitchedDebug.h"
#include "Logger.h"

#include <opencv2/core/types.hpp>

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>

#include <QGraphicsItem>
#include <QVector>
#include <QPointF>

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
    _imageBounds = _imageBounds.isNull() ? item->sceneBoundingRect() : _imageBounds.united(item->sceneBoundingRect());
    return true;
}

bool LayerRenderer::addFeatureLayerFromVwip(const QString &imagePath)
{
    if (!_scene) return false;

    const auto keypoints = xjw::gui::views::loadFeatureKeypointsForImage(_currentProjectPath, imagePath);
    if (keypoints.empty()) return false;
    addFeatureItems(keypoints);
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
    LOG_DEBUG(QStringLiteral("addFeatureItems: added items=%1 total_scene_items=%2").arg(added).arg(_scene ? _scene->items().size() : 0));
}

void LayerRenderer::clear()
{
    for (auto *it: std::as_const(_layers))
    {
        if (it && _scene)
        {
            _scene->removeItem(it);
            delete it;
        }
    }
    _layers.clear();
    _imageBounds = QRectF();
}

bool LayerRenderer::addStitchedImagePair(const QString &pathA, const QString &pathB, QGraphicsPixmapItem **outA, QGraphicsPixmapItem **outB, int gap)
{
    if (!_scene) return false;

    LOG_DEBUG(QStringLiteral("addStitchedImagePair: %1 <-> %2").arg(pathA, pathB));

    // clear existing image layers (we expect caller to manage state)
    // We'll add both images using addImageLayer then reposition the second.
    const int before = _layers.size();
    if (!addImageLayer(pathA, 0)) return false;
    QGraphicsPixmapItem *itemA = nullptr;
    if (!_layers.isEmpty()) itemA = _layers.last();

    if (!addImageLayer(pathB, 0)) {
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

    LOG_DEBUG(QStringLiteral("addStitchedImagePair: itemA size=%1x%2 itemB size=%3x%4").arg(itemA->pixmap().width()).arg(itemA->pixmap().height()).arg(itemB->pixmap().width()).arg(itemB->pixmap().height()));

    // position B to the right of A
    qreal bx = itemA->pixmap().width() + gap;
    itemB->setPos(bx, 0);
    _imageBounds = itemA->sceneBoundingRect().united(itemB->sceneBoundingRect());

    if (outA) *outA = itemA;
    if (outB) *outB = itemB;

    xjw::gui::views::recordStitchedImagePairDebug(_scene, _currentProjectPath, pathA, pathB, itemA, itemB, gap);
    return true;
}

void LayerRenderer::addMatchLines(const QVector<QPointF> &ptsA, const QVector<QPointF> &ptsB, qreal bOffsetX)
{
    if (!_scene) return;
    if (!_matchOpts.showLines) return; // 如果不显示匹配线,直接返回
    
    clearMatchLayers();

    LOG_DEBUG(QStringLiteral("addMatchLines: ptsA=%1 ptsB=%2 bOffsetX=%3").arg(ptsA.size()).arg(ptsB.size()).arg(bOffsetX));

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
