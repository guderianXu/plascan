#include "MatchLineOverlay.h"
#include "ImageViewWidget.h"

#include <QGraphicsView>
#include <QDebug>

#include <algorithm>
#include <cmath>

MatchLineOverlay::MatchLineOverlay(QWidget *parent)
    : QRhiWidget(parent)
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
#if defined(Q_OS_WIN)
    setApi(QRhiWidget::Api::Direct3D11);
#elif defined(Q_OS_MACOS)
    setApi(QRhiWidget::Api::Metal);
#else
    setApi(QRhiWidget::Api::Vulkan);
#endif
    setSampleCount(1);

    // RHI 覆盖层由窗口合成器置于双图视口上方，透明区域保留下层影像。
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_TranslucentBackground);
    setAttribute(Qt::WA_AlwaysStackOnTop);
    setAttribute(Qt::WA_OpaquePaintEvent, false);
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
    if (_leftView)
    {
        _leftView->setMatchPointsVisible(false);
    }
    if (_rightView)
    {
        _rightView->setMatchPointsVisible(false);
    }
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
    ++_gpuGeneration;
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
    ++_gpuGeneration;
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
    ++_gpuGeneration;
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
    ++_gpuGeneration;
    update();
}

void MatchLineOverlay::updateOverlay()
{
    invalidateGeometryCache();
    update();
}

void MatchLineOverlay::initialize(QRhiCommandBuffer *commandBuffer)
{
    Q_UNUSED(commandBuffer);
    _renderError.clear();
    _rhiReady = _gpuRenderer.initialize(
        rhi(), renderTarget(), sampleCount(), _lineWidth, &_renderError);
    if (!_rhiReady)
    {
        qWarning() << _renderError;
    }
}

void MatchLineOverlay::render(QRhiCommandBuffer *commandBuffer)
{
    if (!_rhiReady || !rhi() || !renderTarget())
    {
        return;
    }
    if (!_renderCacheValid)
    {
        rebuildRenderCache();
    }

    if (!_gpuRenderer.render(rhi(),
                             renderTarget(),
                             commandBuffer,
                             size(),
                             gpuBatches(),
                             _showEndPoints,
                             _opacity,
                             _gpuGeneration,
                             _lineWidth,
                             &_renderError))
    {
        qWarning() << _renderError;
    }
}

void MatchLineOverlay::releaseResources()
{
    _gpuRenderer.release();
    _rhiReady = false;
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
    const QPoint viewport_origin = mapFromGlobal(
        view->view()->viewport()->mapToGlobal(QPoint(0, 0)));
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
    for (QVector<QLineF> &lines : _rainbowLines)
    {
        lines.clear();
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
        if (_rainbowMode)
        {
            const int batch = visible_matches.size() > 1
                ? std::min(RainbowBatchCount - 1,
                           static_cast<int>(position * RainbowBatchCount
                                            / visible_matches.size()))
                : 0;
            lines = &_rainbowLines.at(batch);
        }
        else if (has_validity_mask && index < _inlierMask.size())
        {
            lines = _inlierMask.at(index) ? &_inlierLines : &_outlierLines;
        }
        lines->append(line);
    }
    _renderCacheValid = true;
}

void MatchLineOverlay::invalidateGeometryCache()
{
    _visibilityCacheValid = false;
    _renderCacheValid = false;
    ++_gpuGeneration;
}

QVector<MatchGpuLineBatch> MatchLineOverlay::gpuBatches() const
{
    QVector<MatchGpuLineBatch> batches;
    if (_rainbowMode)
    {
        batches.reserve(RainbowBatchCount);
        for (int batch = 0; batch < RainbowBatchCount; ++batch)
        {
            if (!_rainbowLines.at(batch).isEmpty())
            {
                const qreal hue = static_cast<qreal>(batch) / RainbowBatchCount;
                batches.append({&_rainbowLines.at(batch), QColor::fromHsvF(hue, 1.0, 1.0)});
            }
        }
        return batches;
    }
    if (!_defaultLines.isEmpty())
    {
        batches.append({&_defaultLines, _lineColor});
    }
    if (!_inlierLines.isEmpty())
    {
        batches.append({&_inlierLines, QColor(0, 80, 255)});
    }
    if (!_outlierLines.isEmpty())
    {
        batches.append({&_outlierLines, QColor(255, 0, 0)});
    }
    return batches;
}
