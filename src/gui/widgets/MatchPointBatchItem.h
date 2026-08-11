#pragma once

#include <QGraphicsItem>
#include <QPointF>
#include <QPolygonF>
#include <QVector>

class MatchPointBatchItem final : public QGraphicsItem
{
public:
    explicit MatchPointBatchItem(QGraphicsItem *parent = nullptr);

    QRectF boundingRect() const override;
    void paint(QPainter *painter,
               const QStyleOptionGraphicsItem *option,
               QWidget *widget) override;

    void setPoints(const QVector<QPointF> &points);
    void setVisibleIndices(const QVector<int> &indices);
    void setHighlightedIndex(int index);
    void clear();

    int pointCount() const;
    int visiblePointCount() const;
    const QVector<int> &visibleIndices() const;
    const QVector<QPointF> &points() const;

private:
    void rebuildVisiblePoints();

    QVector<QPointF> _points;
    QVector<int> _visibleIndices;
    QPolygonF _visiblePoints;
    QRectF _bounds;
    int _highlightedIndex = -1;
};
