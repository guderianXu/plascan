#pragma once

#include "LayerRenderer.h"

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

} // namespace xjw::gui::views
