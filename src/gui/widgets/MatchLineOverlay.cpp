#include "MatchLineOverlay.h"
#include "ImageViewWidget.h"

#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QGraphicsView>
#include <QSet>
#include <cmath>

MatchLineOverlay::MatchLineOverlay(QWidget *parent)
    : QWidget(parent)
    , _leftView(nullptr)
    , _rightView(nullptr)
    , _lineColor(Qt::yellow)
    , _lineWidth(1.5)
    , _opacity(0.7)
    , _maxDisplayCount(0) // 默认显示全部
    , _showOnlyInliers(false)
    , _showEndPoints(true)
    , _rainbowMode(false)
    , _showOnlyHighlighted(false)
    , _cacheValid(false)
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
    _cacheValid = false;
    update();
}

void MatchLineOverlay::clearHighlightedIndices()
{
    _highlightIndices.clear();
    _cacheValid = false;
    update();
}

void MatchLineOverlay::setViewWidgets(ImageViewWidget *leftView, ImageViewWidget *rightView)
{
    _leftView = leftView;
    _rightView = rightView;
    
    // 连接视图变化信号
    if (_leftView) {
        connect(_leftView, &ImageViewWidget::viewTransformChanged,
            this, &MatchLineOverlay::updateOverlay, Qt::QueuedConnection);
        connect(_leftView, &ImageViewWidget::visibleRectChanged,
                this, &MatchLineOverlay::updateOverlay);
    }
    
    if (_rightView) {
        connect(_rightView, &ImageViewWidget::viewTransformChanged,
            this, &MatchLineOverlay::updateOverlay, Qt::QueuedConnection);
        connect(_rightView, &ImageViewWidget::visibleRectChanged,
                this, &MatchLineOverlay::updateOverlay);
    }
    
    _cacheValid = false;
}

void MatchLineOverlay::setMatches(const QVector<QPointF> &ptsA, 
                                  const QVector<QPointF> &ptsB)
{
    _ptsA = ptsA;
    _ptsB = ptsB;
    _cacheValid = false;
    update();
}

void MatchLineOverlay::setInlierMask(const QVector<bool> &inlierMask)
{
    _inlierMask = inlierMask;
    _cacheValid = false;
    update();
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
    _cacheValid = false;
    update();
}

void MatchLineOverlay::setShowOnlyInliers(bool onlyInliers)
{
    _showOnlyInliers = onlyInliers;
    _cacheValid = false;
    update();
}

void MatchLineOverlay::setShowEndPoints(bool show)
{
    _showEndPoints = show;
    update();
}

void MatchLineOverlay::updateOverlay()
{
    _cacheValid = false;
    update();
}

void MatchLineOverlay::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    
    if (!_leftView || !_rightView) return;
    if (_ptsA.isEmpty() || _ptsB.isEmpty()) return;
    if (_ptsA.size() != _ptsB.size()) return;
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, false);
    
    // 设置透明度
    painter.setOpacity(_opacity);
    
    // 获取可见的匹配
    QVector<int> visibleMatches;
    if (_showOnlyHighlighted && _highlightIndices.isEmpty()) {
        return;
    }
    if (_showOnlyHighlighted && !_highlightIndices.isEmpty()) {
        // 只显示高亮索引（仍然需要校验索引合法性和可见性）
        for (int idx : _highlightIndices) {
            if (idx >= 0 && idx < _ptsA.size()) {
                // 仅当在可见区域内才绘制
                if (isPointVisible(_ptsA[idx], _leftView) && isPointVisible(_ptsB[idx], _rightView)) {
                    visibleMatches.append(idx);
                }
            }
        }
    } else {
        visibleMatches = getVisibleMatches();
    }

    if (visibleMatches.isEmpty()) return;
    
    // 绘制连接线
    QPen linePen;
    linePen.setWidthF(_lineWidth);
    linePen.setCosmetic(true);

    // 如果启用五彩斑斓模式，则为每条线计算不同颜色
    if (_rainbowMode) {
        int total = visibleMatches.size();
        int i = 0;
        for (int idx : visibleMatches) {
            QPointF ptA = _ptsA[idx];
            QPointF ptB = _ptsB[idx];

            QPointF screenA = sceneToScreen(ptA, _leftView);
            QPointF screenB = sceneToScreen(ptB, _rightView);

            // 计算基于索引的色相（0.0 - 1.0）
            double t = (total > 1) ? (double)i / (double)(total - 1) : 0.0;
            QColor c = QColor::fromHsvF(t, 1.0, 1.0);
            linePen.setColor(c);
            painter.setPen(linePen);
            painter.setBrush(c);
            painter.drawLine(screenA, screenB);

            if (_showEndPoints) {
                painter.drawEllipse(screenA, 2, 2);
                painter.drawEllipse(screenB, 2, 2);
            }
            ++i;
        }
    } else {
        const bool hasValidityMask = !_inlierMask.isEmpty();
        const QColor inlierColor(0, 80, 255);
        const QColor outlierColor(255, 0, 0);

        for (int idx : visibleMatches) {
            QPointF ptA = _ptsA[idx];
            QPointF ptB = _ptsB[idx];

            // 转换为屏幕坐标
            QPointF screenA = sceneToScreen(ptA, _leftView);
            QPointF screenB = sceneToScreen(ptB, _rightView);

            QColor drawColor = _lineColor;
            if (hasValidityMask)
            {
                drawColor = idx < _inlierMask.size() && _inlierMask[idx]
                    ? inlierColor
                    : outlierColor;
            }
            linePen.setColor(drawColor);
            painter.setPen(linePen);

            // 绘制连接线
            painter.drawLine(screenA, screenB);

            // 可选：绘制端点
            if (_showEndPoints) {
                painter.setBrush(drawColor);
                painter.drawEllipse(screenA, 2, 2);
                painter.drawEllipse(screenB, 2, 2);
            }
        }
    }
}

QVector<int> MatchLineOverlay::getVisibleMatches() const
{
    if (_cacheValid) {
        return _cachedVisibleMatches;
    }
    
    _cachedVisibleMatches.clear();
    
    if (!_leftView || !_rightView) return _cachedVisibleMatches;
    if (_showOnlyHighlighted && _highlightIndices.isEmpty()) {
        _cacheValid = true;
        return _cachedVisibleMatches;
    }
    
    QSet<int> highlightedSet;
    if (_showOnlyHighlighted && !_highlightIndices.isEmpty())
    {
        highlightedSet.reserve(_highlightIndices.size());
        for (int idx : _highlightIndices)
        {
            highlightedSet.insert(idx);
        }
    }
    
    // 遍历所有匹配点
    for (int i = 0; i < _ptsA.size(); ++i) {
        // 如果只显示内点，检查内点标记
        if (_showOnlyInliers && !_inlierMask.isEmpty()) {
            if (i >= _inlierMask.size() || !_inlierMask[i]) {
                continue;
            }
        }
        // 如果设置了高亮集合且只显示高亮，则跳过非高亮索引
        if (_showOnlyHighlighted && !_highlightIndices.isEmpty()) {
            if (!highlightedSet.contains(i)) continue;
        }
        
        const bool leftVisible = isPointVisible(_ptsA[i], _leftView);
        const bool rightVisible = isPointVisible(_ptsB[i], _rightView);
        if (!leftVisible || !rightVisible)
        {
            continue;
        }

        _cachedVisibleMatches.append(i);

        // 限制最大数量
        if (_maxDisplayCount > 0 &&
            _cachedVisibleMatches.size() >= _maxDisplayCount) {
            break;
        }
    }
    
    _cacheValid = true;
    return _cachedVisibleMatches;
}

QPointF MatchLineOverlay::sceneToScreen(const QPointF &scenePos, 
                                        ImageViewWidget *view) const
{
    if (!view || !view->view()) return QPointF();

    // 场景坐标 -> 视图坐标
    const QPoint viewPos = view->view()->mapFromScene(scenePos);

    // 视图坐标 -> 全局坐标
    const QPoint globalPos = view->view()->viewport()->mapToGlobal(viewPos);

    // 全局坐标 -> 覆盖层坐标
    const QPoint localPos = this->mapFromGlobal(globalPos);
    return QPointF(localPos);
}

QVector<int> MatchLineOverlay::visibleMatches() const
{
    return getVisibleMatches();
}

bool MatchLineOverlay::isPointVisible(const QPointF &scenePos, 
                                      ImageViewWidget *view) const
{
    if (!view || !view->view()) return false;

    QGraphicsView *graphicsView = view->view();
    const QPoint viewportPoint = graphicsView->mapFromScene(scenePos);
    return graphicsView->viewport()->rect().contains(viewportPoint);
}
