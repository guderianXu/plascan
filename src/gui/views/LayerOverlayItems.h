#pragma once

#include "LayerRenderer.h"

#include <QList>
#include <QPointF>
#include <QRectF>
#include <QVector>

#include <vector>

class QGraphicsItem;

namespace xjw::gui::views
{

QGraphicsItem *createFeatureOverlayItem(const std::vector<cv::KeyPoint> &keypoints,
                                        const LayerRenderer::FeatureDisplayOptions &options,
                                        const QRectF &imageBounds);

QGraphicsItem *createFeatureResidualOverlayItem(
    const QVector<FeatureResidualVector> &residuals,
    const LayerRenderer::FeatureDisplayOptions &options,
    const QRectF &imageBounds);

QList<QGraphicsItem *> createMatchOverlayItems(const QVector<QPointF> &ptsA,
                                               const QVector<QPointF> &ptsB,
                                               const LayerRenderer::MatchDisplayOptions &options,
                                               qreal bOffsetX);

} // namespace xjw::gui::views
