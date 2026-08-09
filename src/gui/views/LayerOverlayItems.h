#pragma once

#include "LayerRenderer.h"

#include <QRectF>
#include <QPainterPath>
#include <QString>
#include <QVector>

#include <atomic>
#include <memory>
#include <optional>
#include <vector>

class QGraphicsItem;

namespace xjw::gui::views
{

struct MaskContourSource
{
    QString normalizedPath;
    QString cacheKey;
};

std::optional<MaskContourSource> resolveMaskContourSource(const QString &mask_path);

// 该函数只做可重入的数据准备，调用方必须把它放在后台线程。
QPainterPath extractMaskContoursPath(
    const QString &mask_path,
    const std::shared_ptr<std::atomic<bool>> &cancellation);

QList<QGraphicsItem *> createMaskContourOverlayItems(const QPainterPath &path, int z);

QGraphicsItem *createFeatureOverlayItem(const std::vector<cv::KeyPoint> &keypoints,
                                        const LayerRenderer::FeatureDisplayOptions &options,
                                        const QRectF &imageBounds);

QGraphicsItem *createFeatureResidualOverlayItem(
    const QVector<FeatureResidualVector> &residuals,
    const LayerRenderer::FeatureDisplayOptions &options,
    const QRectF &imageBounds);

} // namespace xjw::gui::views
