#include "LayerOverlayItems.h"
#include "MaskGenerator.h"
#include "io/PathIO.h"

#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QBrush>
#include <QDir>
#include <QFileInfo>
#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QPainter>
#include <QPainterPath>
#include <QPen>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>

namespace
{

constexpr double kPi = 3.14159265358979323846;

QColor interpolateColor(const QColor &left, const QColor &right, double amount)
{
    const double t = std::clamp(amount, 0.0, 1.0);
    return QColor(
        qRound(left.red() + (right.red() - left.red()) * t),
        qRound(left.green() + (right.green() - left.green()) * t),
        qRound(left.blue() + (right.blue() - left.blue()) * t));
}

bool cancellationRequested(const std::shared_ptr<std::atomic<bool>> &cancellation)
{
    return cancellation && cancellation->load(std::memory_order_relaxed);
}

QPen makeMaskContourPen(const QColor &color, qreal width)
{
    QPen pen(color, width, Qt::SolidLine, Qt::SquareCap, Qt::RoundJoin);
    pen.setCosmetic(true);
    return pen;
}

class BatchedFeatureOverlayItem : public QGraphicsItem
{
public:
    BatchedFeatureOverlayItem(std::vector<cv::KeyPoint> keypoints,
                              const LayerRenderer::FeatureDisplayOptions &options,
                              const QRectF &imageBounds)
        : _keypoints(std::move(keypoints))
        , _options(options)
        , _bounds(imageBounds.adjusted(-64.0, -64.0, 64.0, 64.0))
    {
        if (_options.maxDisplayCount > 0)
        {
            std::sort(_keypoints.begin(), _keypoints.end(),
                      [](const auto &left, const auto &right)
                      {
                          return left.response > right.response;
                      });
            if (static_cast<int>(_keypoints.size()) > _options.maxDisplayCount)
            {
                _keypoints.resize(static_cast<size_t>(_options.maxDisplayCount));
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
        pointPen.setWidthF(1.0);
        pointPen.setCosmetic(true);

        const QTransform worldTransform = painter->worldTransform();
        const double viewScale = std::max(
            1e-6,
            std::hypot(worldTransform.m11(), worldTransform.m12()));
        const double radius = std::max(2.0, static_cast<double>(_options.pointSize) * 2.0)
            / viewScale;

        for (const auto &kp : _keypoints)
        {
            const QPointF center(static_cast<qreal>(kp.pt.x), static_cast<qreal>(kp.pt.y));
            painter->setPen(pointPen);
            painter->setBrush(Qt::NoBrush);
            painter->drawLine(QPointF(center.x() - radius, center.y() - radius),
                              QPointF(center.x() + radius, center.y() + radius));
            painter->drawLine(QPointF(center.x() - radius, center.y() + radius),
                              QPointF(center.x() + radius, center.y() - radius));
        }
    }

private:
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
        return _bounds.adjusted(-256.0, -256.0, 256.0, 256.0);
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) override
    {
        if (!painter || !_options.showResiduals)
        {
            return;
        }

        painter->setRenderHint(QPainter::Antialiasing, true);

        const QColor outlineColor(20, 24, 28, std::min(220, _options.opacity));
        QColor observationColor(250, 250, 250, _options.opacity);
        QPen haloPen(outlineColor, 3.8, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        QPen observationPen(outlineColor, 1.2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        QPen endpointPen(outlineColor, 1.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        haloPen.setCosmetic(true);
        observationPen.setCosmetic(true);
        endpointPen.setCosmetic(true);

        const QTransform worldTransform = painter->worldTransform();
        const double viewScale = std::max(
            1e-6,
            std::hypot(worldTransform.m11(), worldTransform.m12()));
        const double originRadius = 2.7 / viewScale;
        const double endpointRadius = 2.2 / viewScale;
        const double arrowLength = 7.0 / viewScale;
        constexpr double arrowHalfAngle = kPi / 6.0;

        for (const auto &residual : _residuals)
        {
            if (residual.magnitudePx < _options.minimumResidualPx)
            {
                continue;
            }

            if (residual.magnitudePx > 1e-9)
            {
                QPointF delta = residual.projected - residual.observed;
                delta *= _options.residualScale;
                double length = std::hypot(delta.x(), delta.y());
                if (_options.maximumResidualLengthPx > 0.0
                    && length > _options.maximumResidualLengthPx)
                {
                    delta *= _options.maximumResidualLengthPx / length;
                    length = _options.maximumResidualLengthPx;
                }
                const QPointF end = residual.observed + delta;
                QColor vectorColor = xjw::gui::views::residualMagnitudeColor(
                    residual.magnitudePx);
                vectorColor.setAlpha(_options.opacity);
                QPen vectorPen(vectorColor, 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
                vectorPen.setCosmetic(true);

                painter->setBrush(Qt::NoBrush);
                painter->setPen(haloPen);
                painter->drawLine(residual.observed, end);
                painter->setPen(vectorPen);
                painter->drawLine(residual.observed, end);

                if (length * viewScale >= 10.0)
                {
                    const double angle = std::atan2(delta.y(), delta.x());
                    const QPointF arrowLeft(
                        end.x() + arrowLength * std::cos(angle + kPi - arrowHalfAngle),
                        end.y() + arrowLength * std::sin(angle + kPi - arrowHalfAngle));
                    const QPointF arrowRight(
                        end.x() + arrowLength * std::cos(angle + kPi + arrowHalfAngle),
                        end.y() + arrowLength * std::sin(angle + kPi + arrowHalfAngle));
                    painter->setPen(haloPen);
                    painter->drawLine(end, arrowLeft);
                    painter->drawLine(end, arrowRight);
                    painter->setPen(vectorPen);
                    painter->drawLine(end, arrowLeft);
                    painter->drawLine(end, arrowRight);
                }

                painter->setPen(endpointPen);
                painter->setBrush(vectorColor);
                painter->drawEllipse(end, endpointRadius, endpointRadius);
            }

            painter->setPen(observationPen);
            painter->setBrush(observationColor);
            painter->drawEllipse(residual.observed, originRadius, originRadius);
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

QColor residualMagnitudeColor(double magnitudePx)
{
    struct ColorStop
    {
        double magnitude;
        QColor color;
    };
    static const ColorStop stops[] = {
        {0.0, QColor(25, 118, 255)},
        {0.5, QColor(0, 200, 255)},
        {1.0, QColor(24, 213, 111)},
        {1.5, QColor(255, 210, 63)},
        {2.0, QColor(255, 69, 69)}};

    if (!std::isfinite(magnitudePx) || magnitudePx <= stops[0].magnitude)
    {
        return stops[0].color;
    }
    for (size_t index = 1; index < std::size(stops); ++index)
    {
        if (magnitudePx <= stops[index].magnitude)
        {
            const ColorStop &left = stops[index - 1];
            const ColorStop &right = stops[index];
            return interpolateColor(left.color,
                                    right.color,
                                    (magnitudePx - left.magnitude)
                                        / (right.magnitude - left.magnitude));
        }
    }
    return stops[std::size(stops) - 1].color;
}

std::optional<MaskContourSource> resolveMaskContourSource(const QString &mask_path)
{
    const QFileInfo file_info(mask_path.trimmed());
    if (!file_info.exists() || !file_info.isFile())
    {
        return std::nullopt;
    }
    const QString canonical_path = file_info.canonicalFilePath();
    const QString normalized_path = QDir::cleanPath(
        canonical_path.isEmpty() ? file_info.absoluteFilePath() : canonical_path);
    if (normalized_path.isEmpty())
    {
        return std::nullopt;
    }
    return MaskContourSource{
        normalized_path,
        QStringLiteral("%1\n%2\n%3")
            .arg(normalized_path)
            .arg(file_info.lastModified().toMSecsSinceEpoch())
            .arg(file_info.size())};
}

QPainterPath extractMaskContoursPath(
    const QString &mask_path,
    const std::shared_ptr<std::atomic<bool>> &cancellation)
{
    if (cancellationRequested(cancellation))
    {
        return {};
    }
    const cv::Mat mask = xjw::common::io::readImage(mask_path, cv::IMREAD_GRAYSCALE);
    if (mask.empty() || cancellationRequested(cancellation))
    {
        return {};
    }
    const auto contours = xjw::mask::extractMaskContours(mask, true);
    if (contours.empty() || cancellationRequested(cancellation))
    {
        return {};
    }
    QPainterPath path;
    for (const auto &contour : contours)
    {
        if (cancellationRequested(cancellation))
        {
            return {};
        }
        if (contour.size() < 2)
        {
            continue;
        }
        path.moveTo(contour.front().x, contour.front().y);
        for (std::size_t i = 1; i < contour.size(); ++i)
        {
            if ((i & 0x3ffU) == 0U && cancellationRequested(cancellation))
            {
                return {};
            }
            path.lineTo(contour.at(i).x, contour.at(i).y);
        }
        path.closeSubpath();
    }
    return path;
}

QList<QGraphicsItem *> createMaskContourOverlayItems(const QPainterPath &path, int z)
{
    if (path.isEmpty())
    {
        return {};
    }

    auto *halo = new QGraphicsPathItem(path);
    halo->setPen(makeMaskContourPen(QColor(0, 0, 0, 210), 4.0));
    halo->setZValue(z);

    auto *outline = new QGraphicsPathItem(path);
    outline->setPen(makeMaskContourPen(QColor(255, 255, 255, 245), 1.6));
    outline->setZValue(z + 0.1);
    return {halo, outline};
}

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
