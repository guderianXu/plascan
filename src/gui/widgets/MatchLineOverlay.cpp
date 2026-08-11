#include "MatchLineOverlay.h"
#include "ImageViewWidget.h"

#include <QPainter>
#include <QPen>
#include <QGraphicsView>
#include <QWidget>

#include <algorithm>
#include <cmath>

MatchLineOverlay::MatchLineOverlay(QWidget *parent)
    : QWidget(parent)
    , _leftView(nullptr)
    , _rightView(nullptr)
    , _lineColor(Qt::yellow)
    , _lineWidth(1.5)
    , _opacity(0.7)
    , _maxDisplayCount(5000)
    , _showOnlyInliers(false)
    , _showEndPoints(true)
    , _rainbowMode(false)
    , _showOnlyHighlighted(false)
    , _visibilityCacheValid(false)
    , _renderCacheValid(false)
{
    // 设置为透明背景，鼠标事件穿透
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_NoSystemBackground);
    setAttribute(Qt::WA_TranslucentBackground);
    setAutoFillBackground(false);
}

MatchLineOverlay::~MatchLineOverlay()
{
}

void MatchLineOverlay::setHighlightedIndices(const QVector<int> &indices)
{
    _highlightIndices = indices;
    invalidateGeometryCache();
    update();
    emit visibleMatchesChanged();
}

void MatchLineOverlay::clearHighlightedIndices()
{
    _highlightIndices.clear();
    invalidateGeometryCache();
    update();
    emit visibleMatchesChanged();
}

void MatchLineOverlay::setShowOnlyHighlighted(bool onlyHighlighted)
{
    if (_showOnlyHighlighted == onlyHighlighted)
    {
        return;
    }

    _showOnlyHighlighted = onlyHighlighted;
    invalidateGeometryCache();
    update();
    emit visibleMatchesChanged();
}

void MatchLineOverlay::setViewWidgets(ImageViewWidget *leftView, ImageViewWidget *rightView)
{
    _leftView = leftView;
    _rightView = rightView;
    
    invalidateGeometryCache();
}

void MatchLineOverlay::setMatches(const QVector<QPointF> &ptsA, 
                                  const QVector<QPointF> &ptsB)
{
    _ptsA = ptsA;
    _ptsB = ptsB;
    _leftSpatialIndex.build(_ptsA);
    _rightSpatialIndex.build(_ptsB);
    invalidateGeometryCache();
    update();
    emit visibleMatchesChanged();
}

void MatchLineOverlay::setInlierMask(const QVector<bool> &inlierMask)
{
    _inlierMask = inlierMask;
    invalidateGeometryCache();
    update();
    emit visibleMatchesChanged();
}

void MatchLineOverlay::setLineColor(const QColor &color)
{
    _lineColor = color;
    update();
}

void MatchLineOverlay::setLineWidth(qreal width)
{
    _lineWidth = width;
    update();
}

void MatchLineOverlay::setOpacity(qreal opacity)
{
    _opacity = qBound(0.0, opacity, 1.0);
    update();
}

void MatchLineOverlay::setMaxDisplayCount(int maxCount)
{
    _maxDisplayCount = maxCount;
    invalidateGeometryCache();
    update();
    emit visibleMatchesChanged();
}

void MatchLineOverlay::setShowOnlyInliers(bool onlyInliers)
{
    _showOnlyInliers = onlyInliers;
    invalidateGeometryCache();
    update();
    emit visibleMatchesChanged();
}

void MatchLineOverlay::setShowEndPoints(bool show)
{
    _showEndPoints = show;
    update();
}

void MatchLineOverlay::setRainbowMode(bool enabled)
{
    if (_rainbowMode == enabled)
    {
        return;
    }
    _rainbowMode = enabled;
    _renderCacheValid = false;
    update();
}

void MatchLineOverlay::updateOverlay()
{
    invalidateGeometryCache();
    update();
}

void MatchLineOverlay::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    if (!_leftView || !_rightView || _ptsA.isEmpty() || _ptsA.size() != _ptsB.size())
    {
        return;
    }
    if (!_renderCacheValid)
    {
        rebuildRenderCache();
    }

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.setOpacity(_opacity);

    const auto draw_batch = [this, &painter](const QVector<QLineF> &lines,
                                              const QVector<QPointF> &endpoints,
                                              const QColor &color)
    {
        if (lines.isEmpty())
        {
            return;
        }
        QPen line_pen(color, _lineWidth);
        line_pen.setCosmetic(true);
        painter.setPen(line_pen);
        painter.drawLines(lines.constData(), lines.size());
        if (_showEndPoints && !endpoints.isEmpty())
        {
            QPen endpoint_pen(color, 4.0, Qt::SolidLine, Qt::RoundCap);
            endpoint_pen.setCosmetic(true);
            painter.setPen(endpoint_pen);
            painter.drawPoints(endpoints.constData(), endpoints.size());
        }
    };

    if (_rainbowMode)
    {
        for (int batch = 0; batch < RainbowBatchCount; ++batch)
        {
            const qreal hue = static_cast<qreal>(batch) / RainbowBatchCount;
            draw_batch(_rainbowLines.at(batch),
                       _rainbowEndpoints.at(batch),
                       QColor::fromHsvF(hue, 1.0, 1.0));
        }
        return;
    }

    draw_batch(_defaultLines, _defaultEndpoints, _lineColor);
    draw_batch(_inlierLines, _inlierEndpoints, QColor(0, 80, 255));
    draw_batch(_outlierLines, _outlierEndpoints, QColor(255, 0, 0));
}

QVector<int> MatchLineOverlay::getVisibleMatches() const
{
    if (_visibilityCacheValid)
    {
        return _cachedVisibleMatches;
    }
    _cachedVisibleMatches.clear();

    if (!_leftView || !_rightView || _ptsA.size() != _ptsB.size())
    {
        _visibilityCacheValid = true;
        return _cachedVisibleMatches;
    }
    const QRectF left_rect = _leftView->visibleSceneRect();
    const QRectF right_rect = _rightView->visibleSceneRect();
    const auto accepted = [this, &left_rect, &right_rect](int index)
    {
        if (index < 0 || index >= _ptsA.size())
        {
            return false;
        }
        if (_showOnlyInliers && !_inlierMask.isEmpty()
            && (index >= _inlierMask.size() || !_inlierMask.at(index)))
        {
            return false;
        }
        return left_rect.contains(_ptsA.at(index)) && right_rect.contains(_ptsB.at(index));
    };

    if (_showOnlyHighlighted)
    {
        for (int index : _highlightIndices)
        {
            if (accepted(index))
            {
                _cachedVisibleMatches.append(index);
            }
        }
    }
    else
    {
        const int left_estimate = _leftSpatialIndex.estimatedCandidateCount(left_rect);
        const int right_estimate = _rightSpatialIndex.estimatedCandidateCount(right_rect);
        const bool query_left = left_estimate <= right_estimate;
        const xjw::gui::match_viewer::MatchSpatialIndex &index =
            query_left ? _leftSpatialIndex : _rightSpatialIndex;
        _cachedVisibleMatches = index.query(
            query_left ? left_rect : right_rect,
            _maxDisplayCount,
            accepted);
    }

    _visibilityCacheValid = true;
    return _cachedVisibleMatches;
}

QTransform MatchLineOverlay::sceneToOverlayTransform(ImageViewWidget *view) const
{
    if (!view || !view->view() || !view->view()->viewport())
    {
        return {};
    }
    const QTransform viewport_transform = view->view()->viewportTransform();
    const QPoint viewport_origin = view->view()->viewport()->mapTo(this, QPoint(0, 0));
    return QTransform(viewport_transform.m11(),
                      viewport_transform.m12(),
                      viewport_transform.m13(),
                      viewport_transform.m21(),
                      viewport_transform.m22(),
                      viewport_transform.m23(),
                      viewport_transform.m31() + viewport_origin.x(),
                      viewport_transform.m32() + viewport_origin.y(),
                      viewport_transform.m33());
}

QVector<int> MatchLineOverlay::visibleMatches() const
{
    return getVisibleMatches();
}

int MatchLineOverlay::renderedLineCount() const
{
    return getVisibleMatches().size();
}

void MatchLineOverlay::rebuildRenderCache() const
{
    _defaultLines.clear();
    _inlierLines.clear();
    _outlierLines.clear();
    _defaultEndpoints.clear();
    _inlierEndpoints.clear();
    _outlierEndpoints.clear();
    for (QVector<QLineF> &lines : _rainbowLines)
    {
        lines.clear();
    }
    for (QVector<QPointF> &points : _rainbowEndpoints)
    {
        points.clear();
    }

    const QVector<int> visible_matches = getVisibleMatches();
    const QTransform left_transform = sceneToOverlayTransform(_leftView);
    const QTransform right_transform = sceneToOverlayTransform(_rightView);
    const bool has_validity_mask = !_inlierMask.isEmpty();
    for (int position = 0; position < visible_matches.size(); ++position)
    {
        const int index = visible_matches.at(position);
        const QPointF left_point = left_transform.map(_ptsA.at(index));
        const QPointF right_point = right_transform.map(_ptsB.at(index));
        const QLineF line(left_point, right_point);

        QVector<QLineF> *lines = &_defaultLines;
        QVector<QPointF> *endpoints = &_defaultEndpoints;
        if (_rainbowMode)
        {
            const int batch = visible_matches.size() > 1
                ? std::min(RainbowBatchCount - 1,
                           static_cast<int>(position * RainbowBatchCount
                                            / visible_matches.size()))
                : 0;
            lines = &_rainbowLines.at(batch);
            endpoints = &_rainbowEndpoints.at(batch);
        }
        else if (has_validity_mask && index < _inlierMask.size())
        {
            lines = _inlierMask.at(index) ? &_inlierLines : &_outlierLines;
            endpoints = _inlierMask.at(index) ? &_inlierEndpoints : &_outlierEndpoints;
        }
        lines->append(line);
        endpoints->append(left_point);
        endpoints->append(right_point);
    }
    _renderCacheValid = true;
}

void MatchLineOverlay::invalidateGeometryCache()
{
    _visibilityCacheValid = false;
    _renderCacheValid = false;
}
