#pragma once

#include <QHash>
#include <QPointF>
#include <QRectF>
#include <QVector>

#include <functional>

namespace xjw::gui::match_viewer
{

class MatchSpatialIndex
{
public:
    explicit MatchSpatialIndex(qreal cellSize = 256.0);

    void build(const QVector<QPointF> &points);
    void clear();

    int size() const;
    int estimatedCandidateCount(const QRectF &rect) const;
    QVector<int> query(const QRectF &rect,
                       int maximumCount,
                       const std::function<bool(int)> &accept,
                       int *visitedCandidates = nullptr) const;

private:
    static quint64 cellKey(int cellX, int cellY);
    int cellCoordinate(qreal value) const;
    bool clippedCellRange(const QRectF &rect,
                          int *minimumX,
                          int *maximumX,
                          int *minimumY,
                          int *maximumY) const;

    qreal _cellSize = 256.0;
    QHash<quint64, QVector<int>> _cells;
    int _pointCount = 0;
    int _minimumCellX = 0;
    int _maximumCellX = -1;
    int _minimumCellY = 0;
    int _maximumCellY = -1;
};

} // namespace xjw::gui::match_viewer
