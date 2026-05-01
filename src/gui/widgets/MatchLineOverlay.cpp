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
    , m_leftView(nullptr)
    , m_rightView(nullptr)
    , m_lineColor(Qt::yellow)
    , m_lineWidth(1.5)
    , m_opacity(0.7)
    , m_maxDisplayCount(0) // 默认显示全部
    , m_showOnlyInliers(false)
    , m_showEndPoints(true)
    , m_rainbowMode(false)
    , m_showOnlyHighlighted(false)
    , m_cacheValid(false)
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
    m_highlightIndices = indices;
    m_cacheValid = false;
    update();
}

void MatchLineOverlay::clearHighlightedIndices()
{
    m_highlightIndices.clear();
    m_cacheValid = false;
    update();
}

void MatchLineOverlay::setViewWidgets(ImageViewWidget *leftView, ImageViewWidget *rightView)
{
    m_leftView = leftView;
    m_rightView = rightView;
    
    // 连接视图变化信号
    if (m_leftView) {
        connect(m_leftView, &ImageViewWidget::viewTransformChanged,
            this, &MatchLineOverlay::updateOverlay, Qt::QueuedConnection);
        connect(m_leftView, &ImageViewWidget::visibleRectChanged,
                this, &MatchLineOverlay::updateOverlay);
    }
    
    if (m_rightView) {
        connect(m_rightView, &ImageViewWidget::viewTransformChanged,
            this, &MatchLineOverlay::updateOverlay, Qt::QueuedConnection);
        connect(m_rightView, &ImageViewWidget::visibleRectChanged,
                this, &MatchLineOverlay::updateOverlay);
    }
    
    m_cacheValid = false;
}

void MatchLineOverlay::setMatches(const QVector<QPointF> &ptsA, 
                                  const QVector<QPointF> &ptsB)
{
    m_ptsA = ptsA;
    m_ptsB = ptsB;
    m_cacheValid = false;
    update();
}

void MatchLineOverlay::setInlierMask(const QVector<bool> &inlierMask)
{
    m_inlierMask = inlierMask;
    m_cacheValid = false;
    update();
}

void MatchLineOverlay::setLineColor(const QColor &color)
{
    m_lineColor = color;
    update();
}

void MatchLineOverlay::setLineWidth(qreal width)
{
    m_lineWidth = width;
    update();
}

void MatchLineOverlay::setOpacity(qreal opacity)
{
    m_opacity = qBound(0.0, opacity, 1.0);
    update();
}

void MatchLineOverlay::setMaxDisplayCount(int maxCount)
{
    m_maxDisplayCount = maxCount;
    m_cacheValid = false;
    update();
}

void MatchLineOverlay::setShowOnlyInliers(bool onlyInliers)
{
    m_showOnlyInliers = onlyInliers;
    m_cacheValid = false;
    update();
}

void MatchLineOverlay::setShowEndPoints(bool show)
{
    m_showEndPoints = show;
    update();
}

void MatchLineOverlay::updateOverlay()
{
    m_cacheValid = false;
    update();
}

void MatchLineOverlay::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    
    if (!m_leftView || !m_rightView) return;
    if (m_ptsA.isEmpty() || m_ptsB.isEmpty()) return;
    if (m_ptsA.size() != m_ptsB.size()) return;
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, false);
    
    // 设置透明度
    painter.setOpacity(m_opacity);
    
    // 获取可见的匹配
    QVector<int> visibleMatches;
    if (m_showOnlyHighlighted && m_highlightIndices.isEmpty()) {
        return;
    }
    if (m_showOnlyHighlighted && !m_highlightIndices.isEmpty()) {
        // 只显示高亮索引（仍然需要校验索引合法性和可见性）
        for (int idx : m_highlightIndices) {
            if (idx >= 0 && idx < m_ptsA.size()) {
                // 仅当在可见区域内才绘制
                if (isPointVisible(m_ptsA[idx], m_leftView) && isPointVisible(m_ptsB[idx], m_rightView)) {
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
    linePen.setWidthF(m_lineWidth);
    linePen.setCosmetic(true);

    // 如果启用五彩斑斓模式，则为每条线计算不同颜色
    if (m_rainbowMode) {
        int total = visibleMatches.size();
        int i = 0;
        for (int idx : visibleMatches) {
            QPointF ptA = m_ptsA[idx];
            QPointF ptB = m_ptsB[idx];

            QPointF screenA = sceneToScreen(ptA, m_leftView);
            QPointF screenB = sceneToScreen(ptB, m_rightView);

            // 计算基于索引的色相（0.0 - 1.0）
            double t = (total > 1) ? (double)i / (double)(total - 1) : 0.0;
            QColor c = QColor::fromHsvF(t, 1.0, 1.0);
            linePen.setColor(c);
            painter.setPen(linePen);
            painter.setBrush(c);
            painter.drawLine(screenA, screenB);

            if (m_showEndPoints) {
                painter.drawEllipse(screenA, 2, 2);
                painter.drawEllipse(screenB, 2, 2);
            }
            ++i;
        }
    } else {
        linePen.setColor(m_lineColor);
        painter.setPen(linePen);

        for (int idx : visibleMatches) {
            QPointF ptA = m_ptsA[idx];
            QPointF ptB = m_ptsB[idx];

            // 转换为屏幕坐标
            QPointF screenA = sceneToScreen(ptA, m_leftView);
            QPointF screenB = sceneToScreen(ptB, m_rightView);

            // 绘制连接线
            painter.drawLine(screenA, screenB);

            // 可选：绘制端点
            if (m_showEndPoints) {
                painter.setBrush(m_lineColor);
                painter.drawEllipse(screenA, 2, 2);
                painter.drawEllipse(screenB, 2, 2);
            }
        }
    }
}

QVector<int> MatchLineOverlay::getVisibleMatches() const
{
    if (m_cacheValid) {
        return m_cachedVisibleMatches;
    }
    
    m_cachedVisibleMatches.clear();
    
    if (!m_leftView || !m_rightView) return m_cachedVisibleMatches;
    if (m_showOnlyHighlighted && m_highlightIndices.isEmpty()) {
        m_cacheValid = true;
        return m_cachedVisibleMatches;
    }
    
    QSet<int> highlightedSet;
    if (m_showOnlyHighlighted && !m_highlightIndices.isEmpty())
    {
        highlightedSet.reserve(m_highlightIndices.size());
        for (int idx : m_highlightIndices)
        {
            highlightedSet.insert(idx);
        }
    }
    
    // 遍历所有匹配点
    for (int i = 0; i < m_ptsA.size(); ++i) {
        // 如果只显示内点，检查内点标记
        if (m_showOnlyInliers && !m_inlierMask.isEmpty()) {
            if (i >= m_inlierMask.size() || !m_inlierMask[i]) {
                continue;
            }
        }
        // 如果设置了高亮集合且只显示高亮，则跳过非高亮索引
        if (m_showOnlyHighlighted && !m_highlightIndices.isEmpty()) {
            if (!highlightedSet.contains(i)) continue;
        }
        
        const bool leftVisible = isPointVisible(m_ptsA[i], m_leftView);
        const bool rightVisible = isPointVisible(m_ptsB[i], m_rightView);
        if (!leftVisible || !rightVisible)
        {
            continue;
        }

        m_cachedVisibleMatches.append(i);

        // 限制最大数量
        if (m_maxDisplayCount > 0 &&
            m_cachedVisibleMatches.size() >= m_maxDisplayCount) {
            break;
        }
    }
    
    m_cacheValid = true;
    return m_cachedVisibleMatches;
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
