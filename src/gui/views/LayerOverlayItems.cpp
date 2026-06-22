#include "LayerOverlayItems.h"

#include <opencv2/core/types.hpp>

#include <QBrush>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QGraphicsLineItem>
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
        : m_keypoints(std::move(keypoints))
        , m_options(options)
        , m_bounds(computeBounds(m_keypoints, options, imageBounds))
    {
        if (m_options.maxDisplayCount > 0)
        {
            if (m_options.showTopScores)
            {
                std::sort(m_keypoints.begin(), m_keypoints.end(),
                          [](const auto &a, const auto &b)
                          {
                              return a.response > b.response;
                          });
            }
            if (static_cast<int>(m_keypoints.size()) > m_options.maxDisplayCount)
            {
                m_keypoints.resize(static_cast<size_t>(m_options.maxDisplayCount));
                m_bounds = computeBounds(m_keypoints, m_options, imageBounds);
            }
        }
        setZValue(1000.0);
    }

    QRectF boundingRect() const override
    {
        return m_bounds;
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) override
    {
        if (!painter || m_keypoints.empty() || !m_options.showPoints)
        {
            return;
        }

        painter->setRenderHint(QPainter::Antialiasing, true);

        QColor pointColor = m_options.pointColor;
        pointColor.setAlpha(m_options.opacity);
        QPen pointPen(pointColor);
        pointPen.setWidthF(1.5);
        pointPen.setCosmetic(true);
        QBrush pointBrush = m_options.useFill
                                ? QBrush(pointColor)
                                : QBrush(Qt::NoBrush);

        for (const auto &kp : m_keypoints)
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
        const double r = markerRadius(kp, m_options);
        const QPointF center(static_cast<qreal>(kp.pt.x), static_cast<qreal>(kp.pt.y));

        painter->setPen(pointPen);
        painter->setBrush(pointBrush);

        if (m_options.markerShape == QLatin1String("circle"))
        {
            painter->drawEllipse(center, r, r);
        }
        else if (m_options.markerShape == QLatin1String("square"))
        {
            painter->drawRect(QRectF(center.x() - r, center.y() - r, r * 2.0, r * 2.0));
        }
        else if (m_options.markerShape == QLatin1String("cross"))
        {
            QPen crossPen(m_options.pointColor);
            crossPen.setWidthF(1.0);
            crossPen.setCosmetic(true);
            painter->setPen(crossPen);
            const double crossRadius = std::max(
                1.0,
                static_cast<double>(m_options.pointSize) * m_options.scaleMultiplier);
            painter->drawLine(QPointF(center.x() - crossRadius, center.y() - crossRadius),
                              QPointF(center.x() + crossRadius, center.y() + crossRadius));
            painter->drawLine(QPointF(center.x() - crossRadius, center.y() + crossRadius),
                              QPointF(center.x() + crossRadius, center.y() - crossRadius));
        }
        else if (m_options.markerShape == QLatin1String("dot"))
        {
            QPen dotPen(m_options.pointColor);
            dotPen.setWidthF(0.5);
            dotPen.setCosmetic(true);
            QColor fill = m_options.pointColor;
            fill.setAlpha(m_options.opacity);
            painter->setPen(dotPen);
            painter->setBrush(QBrush(fill));
            const double dotR = std::min(3.0, r * 0.4);
            painter->drawEllipse(center, dotR, dotR);
        }
        else if (m_options.markerShape == QLatin1String("point"))
        {
            QColor fill = m_options.pointColor;
            fill.setAlpha(m_options.opacity);
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

        if (m_options.showScale)
        {
            QPen scalePen(m_options.scaleColor);
            scalePen.setWidthF(0.8);
            scalePen.setCosmetic(true);
            painter->setPen(scalePen);
            painter->setBrush(Qt::NoBrush);
            const double scaleRadius = static_cast<double>(kp.size) * m_options.scaleMultiplier;
            painter->drawEllipse(center, scaleRadius, scaleRadius);
        }

        if (m_options.showOrientation && kp.angle >= 0.0f)
        {
            QPen orientPen(m_options.orientColor);
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

    std::vector<cv::KeyPoint> m_keypoints;
    LayerRenderer::FeatureDisplayOptions m_options;
    QRectF m_bounds;
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

QList<QGraphicsItem *> createMatchOverlayItems(const QVector<QPointF> &ptsA,
                                               const QVector<QPointF> &ptsB,
                                               const LayerRenderer::MatchDisplayOptions &options,
                                               qreal bOffsetX)
{
    QList<QGraphicsItem *> items;
    if (!options.showLines)
    {
        return items;
    }

    int n = qMin(ptsA.size(), ptsB.size());
    if (options.maxDisplayCount > 0 && n > options.maxDisplayCount)
    {
        n = options.maxDisplayCount;
    }

    QPen linePen(options.lineColor);
    linePen.setWidthF(static_cast<qreal>(options.lineWidth));
    linePen.setColor(QColor(options.lineColor.red(), options.lineColor.green(),
                            options.lineColor.blue(), options.opacity));

    QPen ptPen(options.lineColor);
    QBrush ptBrush(QColor(options.lineColor.red(), options.lineColor.green(),
                          options.lineColor.blue(), options.opacity));

    items.reserve(n * 3);
    for (int i = 0; i < n; ++i)
    {
        const QPointF a = ptsA.at(i);
        const QPointF b = ptsB.at(i);

        auto *ea = new QGraphicsEllipseItem(a.x() - 3, a.y() - 3, 6, 6);
        ea->setPen(ptPen);
        ea->setBrush(ptBrush);
        ea->setZValue(1001.0);
        items.append(ea);

        auto *eb = new QGraphicsEllipseItem(bOffsetX + b.x() - 3, b.y() - 3, 6, 6);
        eb->setPen(ptPen);
        eb->setBrush(ptBrush);
        eb->setZValue(1001.0);
        items.append(eb);

        auto *ln = new QGraphicsLineItem(a.x(), a.y(), bOffsetX + b.x(), b.y());
        ln->setPen(linePen);
        ln->setZValue(1000.5);
        items.append(ln);
    }

    return items;
}

} // namespace xjw::gui::views
