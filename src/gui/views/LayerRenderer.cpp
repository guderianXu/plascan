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
    , m_scene(scene)
{
}

void LayerRenderer::setFeatureDisplayOptions(const FeatureDisplayOptions &opts)
{
    // store options for future rendering
    // We'll use these in addFeatureItems when drawing items
    // keep a copy as a member variable
    // Use m_featureOpts (add member below)
    m_featureOpts = opts;
}

void LayerRenderer::setCurrentProjectPath(const QString &plascanPath)
{
    m_currentProjectPath = plascanPath;
}

bool LayerRenderer::addImageLayer(const QString &path, int z)
{
    return addImageLayer(loadImageForDisplay(path, m_currentProjectPath), z);
}

QImage LayerRenderer::loadImageForDisplay(const QString &path, const QString &plascanPath)
{
    return xjw::gui::views::loadImageForDisplay(path, plascanPath);
}

bool LayerRenderer::addImageLayer(const QImage &image, int z)
{
    if (!m_scene || image.isNull())
    {
        return false;
    }

    QPixmap pix = QPixmap::fromImage(image);
    auto *item = m_scene->addPixmap(pix);
    if (!item)
    {
        return false;
    }
    // Ensure newly added pixmap is positioned at origin and visible
    item->setPos(0, 0);
    item->setVisible(true);
    item->setZValue(z);
    m_layers.append(item);
    m_imageBounds = m_imageBounds.isNull() ? item->sceneBoundingRect() : m_imageBounds.united(item->sceneBoundingRect());
    return true;
}

bool LayerRenderer::addFeatureLayerFromVwip(const QString &imagePath)
{
    if (!m_scene) return false;

    const auto keypoints = xjw::gui::views::loadFeatureKeypointsForImage(m_currentProjectPath, imagePath);
    if (keypoints.empty()) return false;
    addFeatureItems(keypoints);
    return true;
}

void LayerRenderer::clearFeatureLayers()
{
    // Remove items we explicitly tracked
    for (auto *it: std::as_const(m_featureItems))
    {
        if (it && m_scene)
        {
            m_scene->removeItem(it);
            delete it;
        }
    }
    m_featureItems.clear();
}

void LayerRenderer::addFeatureItems(const std::vector<cv::KeyPoint> &keypoints)
{
    if (!m_scene) return;
    // Debug incoming keypoints for troubleshooting feature rendering
    LOG_DEBUG(QStringLiteral("addFeatureItems: incoming keypoints=%1").arg(static_cast<int>(keypoints.size())));

    if (keypoints.empty() || !m_featureOpts.showPoints)
    {
        return;
    }

    auto *item = xjw::gui::views::createFeatureOverlayItem(keypoints, m_featureOpts, m_imageBounds);
    m_scene->addItem(item);
    m_featureItems.append(item);
    const int added = 1;
    LOG_DEBUG(QStringLiteral("addFeatureItems: added items=%1 total_scene_items=%2").arg(added).arg(m_scene ? m_scene->items().size() : 0));
}

void LayerRenderer::clear()
{
    for (auto *it: std::as_const(m_layers))
    {
        if (it && m_scene)
        {
            m_scene->removeItem(it);
            delete it;
        }
    }
    m_layers.clear();
    m_imageBounds = QRectF();
}

bool LayerRenderer::addStitchedImagePair(const QString &pathA, const QString &pathB, QGraphicsPixmapItem **outA, QGraphicsPixmapItem **outB, int gap)
{
    if (!m_scene) return false;

    LOG_DEBUG(QStringLiteral("addStitchedImagePair: %1 <-> %2").arg(pathA, pathB));

    // clear existing image layers (we expect caller to manage state)
    // We'll add both images using addImageLayer then reposition the second.
    const int before = m_layers.size();
    if (!addImageLayer(pathA, 0)) return false;
    QGraphicsPixmapItem *itemA = nullptr;
    if (!m_layers.isEmpty()) itemA = m_layers.last();

    if (!addImageLayer(pathB, 0)) {
        // cleanup the first if second failed
        if (itemA) {
            m_scene->removeItem(itemA);
            m_layers.removeOne(itemA);
            delete itemA;
        }
        return false;
    }
    QGraphicsPixmapItem *itemB = nullptr;
    if (!m_layers.isEmpty() && m_layers.size() > before + 0) itemB = m_layers.last();

    if (!itemA || !itemB) return false;

    LOG_DEBUG(QStringLiteral("addStitchedImagePair: itemA size=%1x%2 itemB size=%3x%4").arg(itemA->pixmap().width()).arg(itemA->pixmap().height()).arg(itemB->pixmap().width()).arg(itemB->pixmap().height()));

    // position B to the right of A
    qreal bx = itemA->pixmap().width() + gap;
    itemB->setPos(bx, 0);
    m_imageBounds = itemA->sceneBoundingRect().united(itemB->sceneBoundingRect());

    if (outA) *outA = itemA;
    if (outB) *outB = itemB;

    xjw::gui::views::recordStitchedImagePairDebug(m_scene, m_currentProjectPath, pathA, pathB, itemA, itemB, gap);
    return true;
}

void LayerRenderer::addMatchLines(const QVector<QPointF> &ptsA, const QVector<QPointF> &ptsB, qreal bOffsetX)
{
    if (!m_scene) return;
    if (!m_matchOpts.showLines) return; // 如果不显示匹配线,直接返回
    
    clearMatchLayers();

    LOG_DEBUG(QStringLiteral("addMatchLines: ptsA=%1 ptsB=%2 bOffsetX=%3").arg(ptsA.size()).arg(ptsB.size()).arg(bOffsetX));

    const auto items = xjw::gui::views::createMatchOverlayItems(ptsA, ptsB, m_matchOpts, bOffsetX);
    for (QGraphicsItem *item : items)
    {
        if (!item)
        {
            continue;
        }
        m_scene->addItem(item);
        m_matchItems.append(item);
    }
}

void LayerRenderer::setMatchDisplayOptions(const MatchDisplayOptions &opts)
{
    m_matchOpts = opts;
}

void LayerRenderer::clearMatchLayers()
{
    for (auto *it: std::as_const(m_matchItems)) {
        if (it && m_scene) {
            m_scene->removeItem(it);
            delete it;
        }
    }
    m_matchItems.clear();
}
