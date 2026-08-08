#include "LayerOverlayItems.h"

#include <opencv2/core/types.hpp>

#include <QBrush>
#include <QGraphicsItem>
#include <QPainter>
#include <QPen>

#include <algorithm>
#include <cmath>
#include <utility>

namespace
{

constexpr double kPi = 3.14159265358979323846;

class BatchedFeatureOverlayItem : public QGraphicsItem
{
public:
    BatchedFeatureOverlayItem(std::vector<cv::KeyPoint> keypoints,
                              const LayerRenderer::FeatureDisplayOptions &options,
                              const QRectF &imageBounds)
        : _keypoints(std::move(keypoints))
        , _options(options)
        , _bounds(computeBounds(_keypoints, options, imageBounds))
    {
        if (_options.maxDisplayCount > 0)
        {
            if (_options.showTopScores)
            {
                std::sort(_keypoints.begin(), _keypoints.end(),
                          [](const auto &a, const auto &b)
                          {
                              return a.response > b.response;
                          });
            }
            if (static_cast<int>(_keypoints.size()) > _options.maxDisplayCount)
            {
                _keypoints.resize(static_cast<size_t>(_options.maxDisplayCount));
                _bounds = computeBounds(_keypoints, _options, imageBounds);
            }
        }
        setZValue(1000.0);
    }

    QRectF boundingRect() const override
    {
        return _bounds;
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) override
    {
        if (!painter || _keypoints.empty() || !_options.showPoints)
        {
            return;
        }

        painter->setRenderHint(QPainter::Antialiasing, true);

        QColor pointColor = _options.pointColor;
        pointColor.setAlpha(_options.opacity);
        QPen pointPen(pointColor);
        pointPen.setWidthF(1.5);
        pointPen.setCosmetic(true);
        QBrush pointBrush = _options.useFill
                                ? QBrush(pointColor)
                                : QBrush(Qt::NoBrush);

        for (const auto &kp : _keypoints)
        {
            drawKeypoint(painter, kp, pointPen, pointBrush);
        }
    }

private:
    static double markerRadius(const cv::KeyPoint &keypoint,
                               const LayerRenderer::FeatureDisplayOptions &options)
    {
        const double sizeFactor = static_cast<double>(options.pointSize) * options.scaleMultiplier;
        return std::max(1.0, std::min(100.0, static_cast<double>(keypoint.size) * sizeFactor));
    }

    static QRectF computeBounds(const std::vector<cv::KeyPoint> &keypoints,
                                const LayerRenderer::FeatureDisplayOptions &options,
                                const QRectF &imageBounds)
    {
        QRectF bounds = imageBounds;
        for (const auto &kp : keypoints)
        {
            const double r = std::max(markerRadius(kp, options), 8.0);
            const QRectF kpRect(static_cast<qreal>(kp.pt.x - r),
                                static_cast<qreal>(kp.pt.y - r),
                                static_cast<qreal>(r * 2.0),
                                static_cast<qreal>(r * 2.0));
            bounds = bounds.isNull() ? kpRect : bounds.united(kpRect);
        }

        if (bounds.isNull())
        {
            return QRectF();
        }
        return bounds.adjusted(-4.0, -4.0, 4.0, 4.0);
    }

    void drawKeypoint(QPainter *painter,
                      const cv::KeyPoint &kp,
                      const QPen &pointPen,
                      const QBrush &pointBrush) const
    {
        const double r = markerRadius(kp, _options);
        const QPointF center(static_cast<qreal>(kp.pt.x), static_cast<qreal>(kp.pt.y));

        painter->setPen(pointPen);
        painter->setBrush(pointBrush);

        if (_options.markerShape == QLatin1String("circle"))
        {
            painter->drawEllipse(center, r, r);
        }
        else if (_options.markerShape == QLatin1String("square"))
        {
            painter->drawRect(QRectF(center.x() - r, center.y() - r, r * 2.0, r * 2.0));
        }
        else if (_options.markerShape == QLatin1String("cross"))
        {
            QPen crossPen(_options.pointColor);
            crossPen.setWidthF(1.0);
            crossPen.setCosmetic(true);
            painter->setPen(crossPen);
            const double crossRadius = std::max(
                1.0,
                static_cast<double>(_options.pointSize) * _options.scaleMultiplier);
            painter->drawLine(QPointF(center.x() - crossRadius, center.y() - crossRadius),
                              QPointF(center.x() + crossRadius, center.y() + crossRadius));
            painter->drawLine(QPointF(center.x() - crossRadius, center.y() + crossRadius),
                              QPointF(center.x() + crossRadius, center.y() - crossRadius));
        }
        else if (_options.markerShape == QLatin1String("dot"))
        {
            QPen dotPen(_options.pointColor);
            dotPen.setWidthF(0.5);
            dotPen.setCosmetic(true);
            QColor fill = _options.pointColor;
            fill.setAlpha(_options.opacity);
            painter->setPen(dotPen);
            painter->setBrush(QBrush(fill));
            const double dotR = std::min(3.0, r * 0.4);
            painter->drawEllipse(center, dotR, dotR);
        }
        else if (_options.markerShape == QLatin1String("point"))
        {
            QColor fill = _options.pointColor;
            fill.setAlpha(_options.opacity);
            QPen pointPixelPen(fill);
            pointPixelPen.setWidthF(0.0);
            pointPixelPen.setCosmetic(true);
            painter->setPen(pointPixelPen);
            painter->setBrush(QBrush(fill));
            painter->drawRect(QRectF(center.x(), center.y(), 1.0, 1.0));
        }
        else
        {
            painter->drawEllipse(center, r, r);
        }

        if (_options.showScale)
        {
            QPen scalePen(_options.scaleColor);
            scalePen.setWidthF(0.8);
            scalePen.setCosmetic(true);
            painter->setPen(scalePen);
            painter->setBrush(Qt::NoBrush);
            const double scaleRadius = static_cast<double>(kp.size) * _options.scaleMultiplier;
            painter->drawEllipse(center, scaleRadius, scaleRadius);
        }

        if (_options.showOrientation && kp.angle >= 0.0f)
        {
            QPen orientPen(_options.orientColor);
            orientPen.setWidthF(1.5);
            orientPen.setCosmetic(true);
            painter->setPen(orientPen);

            const double orientRad = static_cast<double>(kp.angle) * kPi / 180.0;
            const double arrowLen = r * 1.8;
            const QPointF end(center.x() + arrowLen * std::cos(orientRad),
                              center.y() + arrowLen * std::sin(orientRad));
            painter->drawLine(center, end);

            const double arrowHeadLen = r * 0.6;
            const double angle1 = orientRad + kPi * 0.85;
            const double angle2 = orientRad - kPi * 0.85;
            painter->drawLine(end,
                              QPointF(end.x() + arrowHeadLen * std::cos(angle1),
                                      end.y() + arrowHeadLen * std::sin(angle1)));
            painter->drawLine(end,
                              QPointF(end.x() + arrowHeadLen * std::cos(angle2),
                                      end.y() + arrowHeadLen * std::sin(angle2)));
        }
    }

    std::vector<cv::KeyPoint> _keypoints;
    LayerRenderer::FeatureDisplayOptions _options;
    QRectF _bounds;
};

class BatchedFeatureResidualOverlayItem : public QGraphicsItem
{
public:
    BatchedFeatureResidualOverlayItem(QVector<xjw::gui::views::FeatureResidualVector> residuals,
                                      const LayerRenderer::FeatureDisplayOptions &options,
                                      const QRectF &imageBounds)
        : _residuals(std::move(residuals))
        , _options(options)
        , _bounds(imageBounds)
    {
        if (_options.maxDisplayCount > 0 && _residuals.size() > _options.maxDisplayCount)
        {
            std::sort(_residuals.begin(), _residuals.end(), [](const auto &left, const auto &right)
            {
                return left.magnitudePx > right.magnitudePx;
            });
            _residuals.resize(_options.maxDisplayCount);
        }
        setZValue(1001.0);
    }

    QRectF boundingRect() const override
    {
        return _bounds.adjusted(-4.0, -4.0, 4.0, 4.0);
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) override
    {
        if (!painter || !_options.showResiduals)
        {
            return;
        }

        QColor color = _options.residualColor;
        color.setAlpha(_options.opacity);
        QPen pen(color, 1.0);
        pen.setCosmetic(true);
        painter->setPen(pen);
        painter->setBrush(color);

        for (const auto &residual : _residuals)
        {
            if (residual.magnitudePx < _options.minimumResidualPx || residual.magnitudePx <= 1e-9)
            {
                continue;
            }

            QPointF delta = residual.projected - residual.observed;
            delta *= _options.residualScale;
            const double length = std::hypot(delta.x(), delta.y());
            if (_options.maximumResidualLengthPx > 0.0 && length > _options.maximumResidualLengthPx)
            {
                delta *= _options.maximumResidualLengthPx / length;
            }
            const QPointF end = residual.observed + delta;
            painter->drawLine(residual.observed, end);
            painter->drawEllipse(residual.observed, 1.5, 1.5);
        }
    }

private:
    QVector<xjw::gui::views::FeatureResidualVector> _residuals;
    LayerRenderer::FeatureDisplayOptions _options;
    QRectF _bounds;
};

} // namespace

namespace xjw::gui::views
{

QGraphicsItem *createFeatureOverlayItem(const std::vector<cv::KeyPoint> &keypoints,
                                        const LayerRenderer::FeatureDisplayOptions &options,
                                        const QRectF &imageBounds)
{
    return new BatchedFeatureOverlayItem(keypoints, options, imageBounds);
}

QGraphicsItem *createFeatureResidualOverlayItem(
    const QVector<FeatureResidualVector> &residuals,
    const LayerRenderer::FeatureDisplayOptions &options,
    const QRectF &imageBounds)
{
    return new BatchedFeatureResidualOverlayItem(residuals, options, imageBounds);
}

} // namespace xjw::gui::views
