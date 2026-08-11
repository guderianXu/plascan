#include "MatchPointBatchItem.h"

#include <QPainter>
#include <QPen>

#include <algorithm>

MatchPointBatchItem::MatchPointBatchItem(QGraphicsItem *parent)
    : QGraphicsItem(parent)
{
    setZValue(10.0);
    setFlag(QGraphicsItem::ItemUsesExtendedStyleOption, true);
}

QRectF MatchPointBatchItem::boundingRect() const
{
    return _bounds;
}

void MatchPointBatchItem::paint(QPainter *painter,
                                const QStyleOptionGraphicsItem *,
                                QWidget *)
{
    if (!painter)
    {
        return;
    }

    painter->setRenderHint(QPainter::Antialiasing, true);
    if (_pointPaintingEnabled && !_visiblePoints.isEmpty())
    {
        QPen point_pen(Qt::red, 8.0, Qt::SolidLine, Qt::RoundCap);
        point_pen.setCosmetic(true);
        painter->setPen(point_pen);
        painter->drawPoints(_visiblePoints);
    }

    if (_highlightedIndex >= 0 && _highlightedIndex < _points.size())
    {
        QPen highlight_pen(Qt::yellow, 11.0, Qt::SolidLine, Qt::RoundCap);
        highlight_pen.setCosmetic(true);
        painter->setPen(highlight_pen);
        painter->drawPoint(_points.at(_highlightedIndex));
    }
}

void MatchPointBatchItem::setPoints(const QVector<QPointF> &points)
{
    prepareGeometryChange();
    _points = points;
    _bounds = QRectF();
    if (!_points.isEmpty())
    {
        qreal minimum_x = _points.first().x();
        qreal maximum_x = minimum_x;
        qreal minimum_y = _points.first().y();
        qreal maximum_y = minimum_y;
        for (const QPointF &point : _points)
        {
            minimum_x = std::min(minimum_x, point.x());
            maximum_x = std::max(maximum_x, point.x());
            minimum_y = std::min(minimum_y, point.y());
            maximum_y = std::max(maximum_y, point.y());
        }
        _bounds = QRectF(QPointF(minimum_x, minimum_y),
                         QPointF(maximum_x, maximum_y))
                      .normalized()
                      .adjusted(-8.0, -8.0, 8.0, 8.0);
    }
    _visibleIndices.clear();
    _visiblePoints.clear();
    _highlightedIndex = -1;
    update();
}

void MatchPointBatchItem::setVisibleIndices(const QVector<int> &indices)
{
    _visibleIndices = indices;
    rebuildVisiblePoints();
    update();
}

void MatchPointBatchItem::setHighlightedIndex(int index)
{
    _highlightedIndex = index >= 0 && index < _points.size() ? index : -1;
    update();
}

void MatchPointBatchItem::setPointPaintingEnabled(bool enabled)
{
    _pointPaintingEnabled = enabled;
    update();
}

void MatchPointBatchItem::clear()
{
    setPoints({});
}

int MatchPointBatchItem::pointCount() const
{
    return _points.size();
}

int MatchPointBatchItem::visiblePointCount() const
{
    return _visiblePoints.size();
}

const QVector<int> &MatchPointBatchItem::visibleIndices() const
{
    return _visibleIndices;
}

const QVector<QPointF> &MatchPointBatchItem::points() const
{
    return _points;
}

void MatchPointBatchItem::rebuildVisiblePoints()
{
    _visiblePoints.clear();
    _visiblePoints.reserve(_visibleIndices.size());
    for (int index : _visibleIndices)
    {
        if (index >= 0 && index < _points.size())
        {
            _visiblePoints.append(_points.at(index));
        }
    }
}
