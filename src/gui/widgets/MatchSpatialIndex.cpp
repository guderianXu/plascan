#include "MatchSpatialIndex.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw::gui::match_viewer
{

MatchSpatialIndex::MatchSpatialIndex(qreal cellSize)
    : _cellSize(std::max<qreal>(1.0, cellSize))
{
}

void MatchSpatialIndex::build(const QVector<QPointF> &points)
{
    clear();
    _pointCount = points.size();
    for (int index = 0; index < points.size(); ++index)
    {
        const int cell_x = cellCoordinate(points.at(index).x());
        const int cell_y = cellCoordinate(points.at(index).y());
        _cells[cellKey(cell_x, cell_y)].append(index);
        if (_maximumCellX < _minimumCellX)
        {
            _minimumCellX = _maximumCellX = cell_x;
            _minimumCellY = _maximumCellY = cell_y;
        }
        else
        {
            _minimumCellX = std::min(_minimumCellX, cell_x);
            _maximumCellX = std::max(_maximumCellX, cell_x);
            _minimumCellY = std::min(_minimumCellY, cell_y);
            _maximumCellY = std::max(_maximumCellY, cell_y);
        }
    }
}

void MatchSpatialIndex::clear()
{
    _cells.clear();
    _pointCount = 0;
    _minimumCellX = 0;
    _maximumCellX = -1;
    _minimumCellY = 0;
    _maximumCellY = -1;
}

int MatchSpatialIndex::size() const
{
    return _pointCount;
}

int MatchSpatialIndex::estimatedCandidateCount(const QRectF &rect) const
{
    int minimum_x = 0;
    int maximum_x = -1;
    int minimum_y = 0;
    int maximum_y = -1;
    if (!clippedCellRange(rect, &minimum_x, &maximum_x, &minimum_y, &maximum_y))
    {
        return 0;
    }

    qint64 estimate = 0;
    for (int cell_y = minimum_y; cell_y <= maximum_y; ++cell_y)
    {
        for (int cell_x = minimum_x; cell_x <= maximum_x; ++cell_x)
        {
            estimate += _cells.value(cellKey(cell_x, cell_y)).size();
        }
    }
    return static_cast<int>(std::min<qint64>(estimate, std::numeric_limits<int>::max()));
}

QVector<int> MatchSpatialIndex::query(const QRectF &rect,
                                      int maximumCount,
                                      const std::function<bool(int)> &accept,
                                      int *visitedCandidates) const
{
    if (visitedCandidates)
    {
        *visitedCandidates = 0;
    }

    int minimum_x = 0;
    int maximum_x = -1;
    int minimum_y = 0;
    int maximum_y = -1;
    if (!clippedCellRange(rect, &minimum_x, &maximum_x, &minimum_y, &maximum_y))
    {
        return {};
    }

    QVector<const QVector<int> *> buckets;
    for (int cell_y = minimum_y; cell_y <= maximum_y; ++cell_y)
    {
        for (int cell_x = minimum_x; cell_x <= maximum_x; ++cell_x)
        {
            const auto cell = _cells.constFind(cellKey(cell_x, cell_y));
            if (cell != _cells.cend() && !cell->isEmpty())
            {
                buckets.append(&cell.value());
            }
        }
    }

    QVector<int> result;
    if (maximumCount > 0)
    {
        result.reserve(maximumCount);
    }

    int bucket_offset = 0;
    bool has_candidates = true;
    while (has_candidates && (maximumCount <= 0 || result.size() < maximumCount))
    {
        has_candidates = false;
        for (const QVector<int> *bucket : buckets)
        {
            if (!bucket || bucket_offset >= bucket->size())
            {
                continue;
            }
            has_candidates = true;
            const int index = bucket->at(bucket_offset);
            if (visitedCandidates)
            {
                ++(*visitedCandidates);
            }
            if (!accept || accept(index))
            {
                result.append(index);
                if (maximumCount > 0 && result.size() >= maximumCount)
                {
                    break;
                }
            }
        }
        ++bucket_offset;
    }
    return result;
}

quint64 MatchSpatialIndex::cellKey(int cellX, int cellY)
{
    return (static_cast<quint64>(static_cast<quint32>(cellX)) << 32)
        | static_cast<quint32>(cellY);
}

int MatchSpatialIndex::cellCoordinate(qreal value) const
{
    return static_cast<int>(std::floor(value / _cellSize));
}

bool MatchSpatialIndex::clippedCellRange(const QRectF &rect,
                                         int *minimumX,
                                         int *maximumX,
                                         int *minimumY,
                                         int *maximumY) const
{
    if (_cells.isEmpty() || !rect.isValid())
    {
        return false;
    }
    const QRectF normalized = rect.normalized();
    *minimumX = std::max(_minimumCellX, cellCoordinate(normalized.left()));
    *maximumX = std::min(_maximumCellX, cellCoordinate(normalized.right()));
    *minimumY = std::max(_minimumCellY, cellCoordinate(normalized.top()));
    *maximumY = std::min(_maximumCellY, cellCoordinate(normalized.bottom()));
    return *minimumX <= *maximumX && *minimumY <= *maximumY;
}

} // namespace xjw::gui::match_viewer
