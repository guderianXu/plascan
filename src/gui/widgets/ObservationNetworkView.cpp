// =============================================================================
// 文件: ObservationNetworkView.cpp
// =============================================================================

#include "ObservationNetworkView.h"

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsSimpleTextItem>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QResizeEvent>
#include <QShowEvent>
#include <QToolTip>
#include <QWheelEvent>
#include <QPainterPath>
#include <QtMath>
#include <cmath>
#include <algorithm>
#include <limits>

static constexpr double NODE_MIN_RADIUS = 4.0;
static constexpr double NODE_MAX_RADIUS = 18.0;
static constexpr double SCENE_SIZE = 520.0;
static constexpr int HIGH_PERFORMANCE_NODE_THRESHOLD = 180;
static constexpr int HIGH_PERFORMANCE_EDGE_THRESHOLD = 1600;
static constexpr int FORCE_LAYOUT_NODE_THRESHOLD = 140;
static constexpr int MAX_VISIBLE_EDGES_LARGE = 2500;
static constexpr int MAX_VISIBLE_EDGES_HUGE = 1400;
static constexpr int MAX_VISIBLE_LABELS_NORMAL = 60;
static constexpr int MAX_VISIBLE_LABELS_LARGE = 18;
static constexpr int MAX_VISIBLE_NEIGHBOR_LABELS = 20;

namespace
{

QLineF trimmedEdgeSegment(const QPointF &startPoint,
                          const QPointF &endPoint,
                          double startRadius,
                          double endRadius)
{
    QLineF centerLine(startPoint, endPoint);
    if (centerLine.length() < 1e-6)
    {
        return centerLine;
    }

    const double startOffset = std::min(startRadius * 0.92, centerLine.length() * 0.45);
    const double endOffset = std::min(endRadius * 0.92, centerLine.length() * 0.45);

    centerLine.setP1(centerLine.pointAt(startOffset / centerLine.length()));
    centerLine.setP2(centerLine.pointAt(1.0 - endOffset / centerLine.length()));
    return centerLine;
}

} // namespace

ObservationNetworkView::ObservationNetworkView(QWidget *parent)
    : QGraphicsView(parent)
{
    _scene = new QGraphicsScene(this);
    setScene(_scene);
    setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform | QPainter::TextAntialiasing);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
    setBackgroundBrush(QColor(255, 255, 255));
    setMouseTracking(true);
    setOptimizationFlag(QGraphicsView::DontSavePainterState, true);
    setOptimizationFlag(QGraphicsView::DontAdjustForAntialiasing, true);
    setCacheMode(QGraphicsView::CacheBackground);

    _forceTimer = new QTimer(this);
    _forceTimer->setInterval(16); // ~60fps
    connect(_forceTimer, &QTimer::timeout, this, &ObservationNetworkView::onForceStep);
}

ObservationNetworkView::~ObservationNetworkView()
{
    _forceTimer->stop();
}

void ObservationNetworkView::clearNetwork()
{
    _forceTimer->stop();
    _scene->clear();
    _pos.clear();
    _nodeRadii.clear();
    _visibleEdgeIndices.clear();
    _visibleLabelIndices.clear();
    _nodeEdgeAdjacency.clear();
    _selectedNodeIndex = -1;
    _net = xjw::ObservationNetwork{};
    viewport()->update();
}

void ObservationNetworkView::setNetwork(const xjw::ObservationNetwork &net)
{
    _forceTimer->stop();
    resetTransform();
    _scene->clear();
    _net = net;
    _autoFitPending = false;
    _selectedNodeIndex = -1;

    if (net.numNodes() == 0)
    {
        _pos.clear();
        _nodeRadii.clear();
        _visibleEdgeIndices.clear();
        _visibleLabelIndices.clear();
        _nodeEdgeAdjacency.clear();
        return;
    }

    _nodeEdgeAdjacency.clear();
    _nodeEdgeAdjacency.resize(net.numNodes());
    for (int edgeIndex = 0; edgeIndex < net.numEdges(); ++edgeIndex)
    {
        const auto &edge = net.edges[edgeIndex];
        if (edge.idx0 >= 0 && edge.idx0 < net.numNodes())
        {
            _nodeEdgeAdjacency[edge.idx0].push_back(edgeIndex);
        }
        if (edge.idx1 >= 0 && edge.idx1 < net.numNodes())
        {
            _nodeEdgeAdjacency[edge.idx1].push_back(edgeIndex);
        }
    }

    chooseInitialLayout();
    buildScene();
    _autoFitPending = true;
    fitNetworkInView();
}

void ObservationNetworkView::startForceLayout()
{
    if (_net.numNodes() <= 1)
    {
        emit forceLayoutDone();
        return;
    }

    if (_net.numNodes() > FORCE_LAYOUT_NODE_THRESHOLD)
    {
        multiRingLayout();
        applyPositions();
        _autoFitPending = true;
        fitNetworkInView();
        emit forceLayoutDone();
        return;
    }

    _forceIter = 0;
    _temp = LAYOUT_AREA_SIZE * 0.15;  // 初始温度
    _forceTimer->start();
}

void ObservationNetworkView::resetLayout()
{
    _forceTimer->stop();
    chooseInitialLayout();
    applyPositions();
    _autoFitPending = true;
    fitNetworkInView();
}

void ObservationNetworkView::chooseInitialLayout()
{
    if (isHighPerformanceMode())
    {
        componentAwareLayout();
    }
    else
    {
        circularLayout();
    }
}

// ---------------------------------------------------------------------------
// 圆形初始布局
// ---------------------------------------------------------------------------
void ObservationNetworkView::circularLayout()
{
    const int n = _net.numNodes();
    _pos.resize(n);
    if (n == 1)
    {
        _pos[0] = {SCENE_SIZE / 2, SCENE_SIZE / 2};
        return;
    }
    const double radius = SCENE_SIZE * 0.42;
    const double centerX = SCENE_SIZE / 2;
    const double centerY = SCENE_SIZE / 2;
    for (int i = 0; i < n; ++i)
    {
        double angle = 2.0 * M_PI * i / n - M_PI / 2.0;
        _pos[i] = {centerX + radius * std::cos(angle), centerY + radius * std::sin(angle)};
    }
}

void ObservationNetworkView::multiRingLayout()
{
    const int nodeCount = _net.numNodes();
    _pos.resize(nodeCount);
    if (nodeCount == 0)
    {
        return;
    }

    QVector<int> orderedNodeIndices(nodeCount);
    for (int nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
    {
        orderedNodeIndices[nodeIndex] = nodeIndex;
    }

    std::sort(orderedNodeIndices.begin(), orderedNodeIndices.end(),
              [this](int leftIndex, int rightIndex)
              {
                  const int leftDegree = leftIndex < (int)_net.degrees.size() ? _net.degrees[leftIndex] : 0;
                  const int rightDegree = rightIndex < (int)_net.degrees.size() ? _net.degrees[rightIndex] : 0;
                  if (leftDegree != rightDegree)
                  {
                      return leftDegree > rightDegree;
                  }
                  return leftIndex < rightIndex;
              });

    const double centerX = SCENE_SIZE / 2.0;
    const double centerY = SCENE_SIZE / 2.0;
    const int ringCount = std::max(2, (int)std::ceil(std::sqrt(nodeCount / 24.0)));
    int cursor = 0;

    for (int ringIndex = 0; ringIndex < ringCount && cursor < nodeCount; ++ringIndex)
    {
        const int remainingCount = nodeCount - cursor;
        const int ringCapacity = std::max(1, (ringIndex == 0) ? 1 : ringIndex * 18);
        const int nodesInRing = std::min(remainingCount, ringCapacity);
        const double t = ringCount > 1 ? (double)ringIndex / (ringCount - 1) : 0.0;
        const double ringRadius = 28.0 + t * (SCENE_SIZE * 0.44);

        if (ringIndex == 0)
        {
            const int nodeIndex = orderedNodeIndices[cursor++];
            _pos[nodeIndex] = QPointF(centerX, centerY);
            continue;
        }

        for (int localIndex = 0; localIndex < nodesInRing; ++localIndex)
        {
            const int nodeIndex = orderedNodeIndices[cursor++];
            const double angle = (2.0 * M_PI * localIndex / nodesInRing) - M_PI / 2.0;
            _pos[nodeIndex] = QPointF(centerX + ringRadius * std::cos(angle),
                                       centerY + ringRadius * std::sin(angle));
        }
    }
}

QVector<QVector<int>> ObservationNetworkView::computeConnectedComponents() const
{
    QVector<QVector<int>> components;
    const int nodeCount = _net.numNodes();
    if (nodeCount <= 0)
    {
        return components;
    }

    QVector<bool> visited(nodeCount, false);
    QVector<int> stack;
    for (int startNodeIndex = 0; startNodeIndex < nodeCount; ++startNodeIndex)
    {
        if (visited[startNodeIndex])
        {
            continue;
        }

        QVector<int> component;
        stack.clear();
        stack.push_back(startNodeIndex);
        visited[startNodeIndex] = true;

        while (!stack.isEmpty())
        {
            const int nodeIndex = stack.back();
            stack.pop_back();
            component.push_back(nodeIndex);

            for (int edgeIndex : _nodeEdgeAdjacency.value(nodeIndex))
            {
                const auto &edge = _net.edges[edgeIndex];
                const int neighborIndex = (edge.idx0 == nodeIndex) ? edge.idx1 : edge.idx0;
                if (neighborIndex >= 0 && neighborIndex < nodeCount && !visited[neighborIndex])
                {
                    visited[neighborIndex] = true;
                    stack.push_back(neighborIndex);
                }
            }
        }

        components.push_back(component);
    }

    std::sort(components.begin(), components.end(),
              [](const QVector<int> &leftComponent, const QVector<int> &rightComponent)
              {
                  return leftComponent.size() > rightComponent.size();
              });
    return components;
}

void ObservationNetworkView::layoutComponentNodes(const QVector<int> &componentNodeIndices,
                                                  const QPointF &componentCenter,
                                                  double componentRadius)
{
    if (componentNodeIndices.isEmpty())
    {
        return;
    }

    if (componentNodeIndices.size() == 1)
    {
        _pos[componentNodeIndices.front()] = componentCenter;
        return;
    }

    QVector<int> orderedNodeIndices = componentNodeIndices;
    std::sort(orderedNodeIndices.begin(), orderedNodeIndices.end(),
              [this](int leftIndex, int rightIndex)
              {
                  const int leftDegree = leftIndex < (int)_net.degrees.size() ? _net.degrees[leftIndex] : 0;
                  const int rightDegree = rightIndex < (int)_net.degrees.size() ? _net.degrees[rightIndex] : 0;
                  if (leftDegree != rightDegree)
                  {
                      return leftDegree > rightDegree;
                  }
                  return leftIndex < rightIndex;
              });

    const int ringCount = std::max(2, (int)std::ceil(std::sqrt(orderedNodeIndices.size() / 18.0)));
    int cursor = 0;
    for (int ringIndex = 0; ringIndex < ringCount && cursor < orderedNodeIndices.size(); ++ringIndex)
    {
        const int remainingCount = orderedNodeIndices.size() - cursor;
        const int ringCapacity = (ringIndex == 0) ? 1 : std::max(8, ringIndex * 20);
        const int nodesInRing = std::min(remainingCount, ringCapacity);
        const double t = ringCount > 1 ? (double)ringIndex / (ringCount - 1) : 0.0;
        const double ringRadius = 16.0 + t * componentRadius;

        if (ringIndex == 0)
        {
            _pos[orderedNodeIndices[cursor++]] = componentCenter;
            continue;
        }

        for (int localIndex = 0; localIndex < nodesInRing; ++localIndex)
        {
            const int nodeIndex = orderedNodeIndices[cursor++];
            const double angle = (2.0 * M_PI * localIndex / nodesInRing) - M_PI / 2.0;
            _pos[nodeIndex] = QPointF(componentCenter.x() + ringRadius * std::cos(angle),
                                       componentCenter.y() + ringRadius * std::sin(angle));
        }
    }
}

void ObservationNetworkView::componentAwareLayout()
{
    const int nodeCount = _net.numNodes();
    _pos.resize(nodeCount);
    if (nodeCount == 0)
    {
        return;
    }

    const QVector<QVector<int>> components = computeConnectedComponents();
    if (components.isEmpty())
    {
        circularLayout();
        return;
    }

    if (components.size() == 1)
    {
        multiRingLayout();
        relaxNodeOverlaps(12, 6.0);
        return;
    }

    double totalComponentWidth = 0.0;
    QVector<double> componentRadii;
    componentRadii.reserve(components.size());
    for (const QVector<int> &componentNodeIndices : components)
    {
        const double componentRadius = std::max(55.0,
                                                std::sqrt((double)componentNodeIndices.size()) * 26.0);
        componentRadii.push_back(componentRadius);
        totalComponentWidth += componentRadius * 2.0;
    }

    const int columnCount = std::max(1, (int)std::ceil(std::sqrt((double)components.size())));
    const int rowCount = std::max(1, (int)std::ceil((double)components.size() / columnCount));
    const double cellWidth = std::max(220.0, totalComponentWidth / columnCount);
    const double cellHeight = std::max(220.0, cellWidth * 0.9);

    for (int componentIndex = 0; componentIndex < components.size(); ++componentIndex)
    {
        const int row = componentIndex / columnCount;
        const int column = componentIndex % columnCount;
        const QPointF componentCenter((column + 0.5) * cellWidth,
                                      (row + 0.5) * cellHeight);
        layoutComponentNodes(components[componentIndex], componentCenter, componentRadii[componentIndex]);
    }

    relaxNodeOverlaps(10, 6.0);
}

void ObservationNetworkView::relaxNodeOverlaps(int maxIterations, double padding)
{
    if (_pos.size() < 2)
    {
        return;
    }

    const QRectF bounds = networkBounds().adjusted(60.0, 60.0, -60.0, -60.0);
    for (int iterationIndex = 0; iterationIndex < maxIterations; ++iterationIndex)
    {
        bool moved = false;
        for (int leftNodeIndex = 0; leftNodeIndex < _pos.size(); ++leftNodeIndex)
        {
            for (int rightNodeIndex = leftNodeIndex + 1; rightNodeIndex < _pos.size(); ++rightNodeIndex)
            {
                QPointF delta = _pos[leftNodeIndex] - _pos[rightNodeIndex];
                double distanceSquared = delta.x() * delta.x() + delta.y() * delta.y();
                if (distanceSquared < 1e-6)
                {
                    delta = QPointF(1.0, 0.0);
                    distanceSquared = 1.0;
                }

                const double distance = std::sqrt(distanceSquared);
                const double minDistance = _nodeRadii.value(leftNodeIndex, NODE_MIN_RADIUS)
                    + _nodeRadii.value(rightNodeIndex, NODE_MIN_RADIUS)
                    + padding;
                if (distance >= minDistance)
                {
                    continue;
                }

                const double pushDistance = (minDistance - distance) * 0.52;
                const QPointF direction = delta / distance;
                _pos[leftNodeIndex] += direction * pushDistance;
                _pos[rightNodeIndex] -= direction * pushDistance;
                moved = true;
            }
        }

        for (int nodeIndex = 0; nodeIndex < _pos.size(); ++nodeIndex)
        {
            _pos[nodeIndex].setX(std::clamp(_pos[nodeIndex].x(), bounds.left(), bounds.right()));
            _pos[nodeIndex].setY(std::clamp(_pos[nodeIndex].y(), bounds.top(), bounds.bottom()));
        }

        if (!moved)
        {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// 构建 QGraphicsScene 元素
// ---------------------------------------------------------------------------
QColor ObservationNetworkView::nodeColor(int degree, int maxDeg) const
{
    if (maxDeg == 0)
    {
        return QColor(70, 130, 180);
    }
    double t = std::min(1.0, (double)degree / maxDeg);
    // 蓝 → 绿 → 橙
    if (t < 0.5)
    {
        int r = (int)(70  + (100 - 70)  * t * 2);
        int g = (int)(130 + (200 - 130) * t * 2);
        int b = (int)(180 + (80  - 180) * t * 2);
        return QColor(r, g, b);
    }
    else
    {
        double s = (t - 0.5) * 2;
        int r = (int)(100 + (230 - 100) * s);
        int g = (int)(200 + (160 - 200) * s);
        int b = (int)(80  + (30  -  80) * s);
        return QColor(r, g, b);
    }
}

QColor ObservationNetworkView::edgeColor(double weight) const
{
    // 白底下使用更深的蓝灰色，保证默认缩放也清晰可见
    const int r = (int)(160 - 80 * weight);
    const int g = (int)(170 - 70 * weight);
    const int b = (int)(185 - 25 * weight);
    return QColor(r, g, b, 210 + (int)(45 * weight));
}

void ObservationNetworkView::buildScene()
{
    _scene->clear();
    const int maxDegree = _net.degrees.empty()
        ? 0
        : *std::max_element(_net.degrees.begin(), _net.degrees.end());

    _nodeRadii.resize(_net.numNodes());
    for (int nodeIndex = 0; nodeIndex < _net.numNodes(); ++nodeIndex)
    {
        _nodeRadii[nodeIndex] = nodeRadius(nodeIndex, maxDegree);
    }

    rebuildRenderCache();
    applyPositions();
}

void ObservationNetworkView::applyPositions()
{
    _scene->setSceneRect(networkBounds());
    viewport()->update();
}

QRectF ObservationNetworkView::networkBounds() const
{
    if (_pos.isEmpty())
    {
        return QRectF(-100.0, -100.0, 200.0, 200.0);
    }

    qreal minX = _pos.front().x();
    qreal minY = _pos.front().y();
    qreal maxX = _pos.front().x();
    qreal maxY = _pos.front().y();

    for (const QPointF &p : _pos)
    {
        minX = std::min(minX, p.x());
        minY = std::min(minY, p.y());
        maxX = std::max(maxX, p.x());
        maxY = std::max(maxY, p.y());
    }

    constexpr qreal padding = 80.0;
    return QRectF(QPointF(minX - padding, minY - padding),
                  QPointF(maxX + padding, maxY + padding)).normalized();
}

double ObservationNetworkView::nodeRadius(int nodeIndex, int maxDegree) const
{
    const int degree = (nodeIndex < (int)_net.degrees.size()) ? _net.degrees[nodeIndex] : 0;
    const double t = (maxDegree > 0) ? (double)degree / maxDegree : 0.0;
    const double minRadius = isHighPerformanceMode() ? NODE_MIN_RADIUS : NODE_MIN_RADIUS + 2.0;
    const double maxRadius = isHighPerformanceMode() ? NODE_MAX_RADIUS - 6.0 : NODE_MAX_RADIUS;
    return minRadius + (maxRadius - minRadius) * t;
}

bool ObservationNetworkView::isHighPerformanceMode() const
{
    return _net.numNodes() >= HIGH_PERFORMANCE_NODE_THRESHOLD ||
           _net.numEdges() >= HIGH_PERFORMANCE_EDGE_THRESHOLD;
}

double ObservationNetworkView::currentViewScale() const
{
    return transform().m11();
}

bool ObservationNetworkView::shouldDrawLabels() const
{
    if (_selectedNodeIndex >= 0)
    {
        return true;
    }

    if (!isHighPerformanceMode())
    {
        return currentViewScale() >= 0.55;
    }

    return currentViewScale() >= 1.2;
}

void ObservationNetworkView::rebuildRenderCache()
{
    _visibleEdgeIndices.clear();
    _visibleLabelIndices.clear();

    QVector<int> allEdgeIndices(_net.numEdges());
    for (int edgeIndex = 0; edgeIndex < _net.numEdges(); ++edgeIndex)
    {
        allEdgeIndices[edgeIndex] = edgeIndex;
    }

    auto edgePriorityLess = [this](int leftEdgeIndex, int rightEdgeIndex)
    {
        const auto &leftEdge = _net.edges[leftEdgeIndex];
        const auto &rightEdge = _net.edges[rightEdgeIndex];
        if (leftEdge.weight != rightEdge.weight)
        {
            return leftEdge.weight > rightEdge.weight;
        }
        return leftEdge.numMatches > rightEdge.numMatches;
    };

    if (_selectedNodeIndex >= 0 && _selectedNodeIndex < _nodeEdgeAdjacency.size())
    {
        _visibleEdgeIndices = _nodeEdgeAdjacency[_selectedNodeIndex];
        std::sort(_visibleEdgeIndices.begin(), _visibleEdgeIndices.end(), edgePriorityLess);
    }
    else if (!isHighPerformanceMode())
    {
        _visibleEdgeIndices = allEdgeIndices;
    }
    else
    {
        std::sort(allEdgeIndices.begin(), allEdgeIndices.end(), edgePriorityLess);
        const int edgeLimit = (_net.numNodes() >= 600) ? MAX_VISIBLE_EDGES_HUGE : MAX_VISIBLE_EDGES_LARGE;
        _visibleEdgeIndices = allEdgeIndices.mid(0, std::min(edgeLimit, static_cast<int>(allEdgeIndices.size())));
    }

    QVector<int> allNodeIndices(_net.numNodes());
    for (int nodeIndex = 0; nodeIndex < _net.numNodes(); ++nodeIndex)
    {
        allNodeIndices[nodeIndex] = nodeIndex;
    }

    std::sort(allNodeIndices.begin(), allNodeIndices.end(),
              [this](int leftNodeIndex, int rightNodeIndex)
              {
                  const int leftDegree = leftNodeIndex < (int)_net.degrees.size() ? _net.degrees[leftNodeIndex] : 0;
                  const int rightDegree = rightNodeIndex < (int)_net.degrees.size() ? _net.degrees[rightNodeIndex] : 0;
                  if (leftDegree != rightDegree)
                  {
                      return leftDegree > rightDegree;
                  }
                  return leftNodeIndex < rightNodeIndex;
              });

    if (!shouldDrawLabels())
    {
        if (_selectedNodeIndex >= 0)
        {
            _visibleLabelIndices.push_back(_selectedNodeIndex);
        }
        return;
    }

    if (_selectedNodeIndex >= 0)
    {
        _visibleLabelIndices.push_back(_selectedNodeIndex);
        QVector<int> adjacentEdgeIndices = _nodeEdgeAdjacency.value(_selectedNodeIndex);
        std::sort(adjacentEdgeIndices.begin(), adjacentEdgeIndices.end(), edgePriorityLess);
        for (int edgeIndex : adjacentEdgeIndices)
        {
            const auto &edge = _net.edges[edgeIndex];
            const int neighborIndex = (edge.idx0 == _selectedNodeIndex) ? edge.idx1 : edge.idx0;
            if (!_visibleLabelIndices.contains(neighborIndex))
            {
                _visibleLabelIndices.push_back(neighborIndex);
            }
            if (_visibleLabelIndices.size() >= MAX_VISIBLE_NEIGHBOR_LABELS + 1)
            {
                break;
            }
        }
        return;
    }

    const int labelLimit = isHighPerformanceMode() ? MAX_VISIBLE_LABELS_LARGE : MAX_VISIBLE_LABELS_NORMAL;
    _visibleLabelIndices = allNodeIndices.mid(0, std::min(labelLimit, static_cast<int>(allNodeIndices.size())));
}

void ObservationNetworkView::fitNetworkInView()
{
    if (!_scene || _net.numNodes() == 0 || viewport()->size().isEmpty())
    {
        return;
    }

    const QRectF rect = _scene->sceneRect();
    if (!rect.isValid() || rect.isEmpty())
    {
        return;
    }

    fitInView(rect, Qt::KeepAspectRatio);
    _autoFitPending = false;
    rebuildRenderCache();
    viewport()->update();
}

void ObservationNetworkView::drawBackground(QPainter *painter, const QRectF &rect)
{
    QGraphicsView::drawBackground(painter, rect);

    if (_net.numNodes() == 0)
    {
        return;
    }

    const QRectF visibleSceneRect = mapToScene(viewport()->rect()).boundingRect();
    const int maxDegree = _net.degrees.empty()
        ? 0
        : *std::max_element(_net.degrees.begin(), _net.degrees.end());

    QVector<bool> highlightedNodes(_net.numNodes(), false);
    if (_selectedNodeIndex >= 0 && _selectedNodeIndex < highlightedNodes.size())
    {
        highlightedNodes[_selectedNodeIndex] = true;
        for (int edgeIndex : _nodeEdgeAdjacency.value(_selectedNodeIndex))
        {
            const auto &edge = _net.edges[edgeIndex];
            highlightedNodes[edge.idx0] = true;
            highlightedNodes[edge.idx1] = true;
        }
    }

    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, !isHighPerformanceMode());

    for (int edgeIndex : _visibleEdgeIndices)
    {
        const auto &edge = _net.edges[edgeIndex];
        if (edge.idx0 < 0 || edge.idx0 >= _pos.size() || edge.idx1 < 0 || edge.idx1 >= _pos.size())
        {
            continue;
        }

        const QPointF startPoint = _pos[edge.idx0];
        const QPointF endPoint = _pos[edge.idx1];
        const double startRadius = edge.idx0 < _nodeRadii.size()
            ? _nodeRadii[edge.idx0]
            : nodeRadius(edge.idx0, maxDegree);
        const double endRadius = edge.idx1 < _nodeRadii.size()
            ? _nodeRadii[edge.idx1]
            : nodeRadius(edge.idx1, maxDegree);
        const QLineF visibleEdgeLine = trimmedEdgeSegment(startPoint, endPoint,
                                                          startRadius, endRadius);

        QRectF edgeBounds(visibleEdgeLine.p1(), visibleEdgeLine.p2());
        edgeBounds = edgeBounds.normalized().adjusted(-8.0, -8.0, 8.0, 8.0);
        if (!edgeBounds.intersects(visibleSceneRect))
        {
            continue;
        }

        QColor lineColor = edgeColor(edge.weight);
        double lineWidth = std::clamp(1.0 + 2.4 * edge.weight, 1.0, 3.8);
        if (_selectedNodeIndex >= 0)
        {
            const bool isHighlighted = edge.idx0 == _selectedNodeIndex || edge.idx1 == _selectedNodeIndex;
            if (isHighlighted)
            {
                lineColor = QColor(220, 88, 34, 240);
                lineWidth = std::clamp(lineWidth + 1.6, 2.0, 5.5);
            }
            else
            {
                lineColor.setAlpha(26);
            }
        }

        QPen pen(lineColor, lineWidth);
        pen.setCosmetic(true);
        pen.setCapStyle(Qt::RoundCap);
        painter->setPen(pen);
        painter->drawLine(visibleEdgeLine);
    }

    painter->setPen(Qt::NoPen);
    for (int nodeIndex = 0; nodeIndex < _pos.size(); ++nodeIndex)
    {
        const QPointF nodePosition = _pos[nodeIndex];
        const double radius = nodeIndex < _nodeRadii.size() ? _nodeRadii[nodeIndex] : nodeRadius(nodeIndex, maxDegree);
        QRectF nodeRect(nodePosition.x() - radius, nodePosition.y() - radius,
                        radius * 2.0, radius * 2.0);
        if (!nodeRect.adjusted(-8.0, -8.0, 8.0, 8.0).intersects(visibleSceneRect))
        {
            continue;
        }

        const int degree = nodeIndex < (int)_net.degrees.size() ? _net.degrees[nodeIndex] : 0;
        QColor fillColor = nodeColor(degree, maxDegree);
        if (_selectedNodeIndex >= 0)
        {
            if (nodeIndex == _selectedNodeIndex)
            {
                fillColor = QColor(235, 115, 40);
            }
            else if (!highlightedNodes[nodeIndex])
            {
                fillColor = QColor(185, 185, 185, 90);
            }
        }

        painter->setBrush(fillColor);
        painter->drawEllipse(nodeRect);

        if (nodeIndex == _selectedNodeIndex)
        {
            QPen highlightPen(QColor(70, 30, 10), 2.0);
            highlightPen.setCosmetic(true);
            painter->setPen(highlightPen);
            painter->setBrush(Qt::NoBrush);
            painter->drawEllipse(nodeRect.adjusted(-2.0, -2.0, 2.0, 2.0));
            painter->setPen(Qt::NoPen);
        }
    }

    painter->restore();
}

void ObservationNetworkView::drawForeground(QPainter *painter, const QRectF &rect)
{
    Q_UNUSED(rect);

    QGraphicsView::drawForeground(painter, rect);

    if (_net.numNodes() == 0)
    {
        return;
    }

    painter->save();
    painter->resetTransform();
    painter->setRenderHint(QPainter::Antialiasing, true);

    if (shouldDrawLabels())
    {
        QFont labelFont = painter->font();
        labelFont.setPointSizeF(isHighPerformanceMode() ? 8.8 : 9.6);
        painter->setFont(labelFont);

        for (int nodeIndex : _visibleLabelIndices)
        {
            if (nodeIndex < 0 || nodeIndex >= _pos.size() || nodeIndex >= (int)_net.nodeNames.size())
            {
                continue;
            }

            const QPoint viewPoint = mapFromScene(_pos[nodeIndex]);
            const QString text = QString::fromStdString(_net.nodeNames[nodeIndex]);
            const QFontMetrics fontMetrics(painter->font());
            const QRect textRect = fontMetrics.boundingRect(text);
            const QRect bubbleRect(viewPoint.x() + 10,
                                   viewPoint.y() - textRect.height() - 10,
                                   textRect.width() + 10,
                                   textRect.height() + 6);

            QColor bubbleColor(255, 255, 255, nodeIndex == _selectedNodeIndex ? 245 : 210);
            painter->setPen(QPen(QColor(190, 190, 190, 180), 1.0));
            painter->setBrush(bubbleColor);
            painter->drawRoundedRect(bubbleRect, 5.0, 5.0);
            painter->setPen(nodeIndex == _selectedNodeIndex ? QColor(25, 25, 25) : QColor(55, 55, 55));
            painter->drawText(bubbleRect.adjusted(5, 0, -5, 0), Qt::AlignVCenter | Qt::AlignLeft, text);
        }
    }

    const int maxDeg = _net.degrees.empty() ? 0
        : *std::max_element(_net.degrees.begin(), _net.degrees.end());

    // 计算度（连接数）分档的整数阈值，便于在图例中显示具体范围
    const int lowMax = (maxDeg > 0) ? std::max(1, (int)std::floor(maxDeg / 3.0)) : 0;
    const int midMax = (maxDeg > 0) ? std::max(lowMax, (int)std::floor((2.0 * maxDeg) / 3.0)) : 0;

    const QRect vp = viewport()->rect();
    // 根据条目数动态计算图例区域大小，避免文本重叠或超出视图
    const int entryCount = 3;
    const qreal legendW = 260.0;
    const qreal legendH = 34.0 + entryCount * 24.0 + 28.0;
    const QRectF legendRect(vp.right() - legendW - 12.0, 16.0, legendW, legendH);
    painter->setPen(QPen(QColor(180, 180, 180), 1.0));
    painter->setBrush(QColor(255, 255, 255, 235));
    painter->drawRoundedRect(legendRect, 10.0, 10.0);

    painter->setPen(QColor(35, 35, 35));
    QFont titleFont = painter->font();
    titleFont.setBold(true);
    titleFont.setPointSizeF(titleFont.pointSizeF() > 0 ? titleFont.pointSizeF() : 10.0);
    painter->setFont(titleFont);
    painter->drawText(QRectF(legendRect.left() + 12.0, legendRect.top() + 8.0,
                             legendRect.width() - 24.0, 20.0),
                      Qt::AlignLeft | Qt::AlignVCenter,
                      tr("颜色说明"));

    QFont bodyFont = painter->font();
    bodyFont.setBold(false);
    bodyFont.setPointSizeF(9.5);
    painter->setFont(bodyFont);

    struct LegendEntry {
        QColor color;
        QString text;
    };

    const QVector<LegendEntry> entries = {
        {nodeColor(0, std::max(1, maxDeg)),
         maxDeg > 0 ? tr("蓝色：连接较少（度 ≤ %1）").arg(lowMax) : tr("蓝色：当前节点颜色")},
        {nodeColor(std::clamp((lowMax + midMax) / 2, 1, std::max(1, maxDeg)), std::max(1, maxDeg)),
         tr("绿色：连接中等（%1 < 度 ≤ %2）").arg(lowMax).arg(midMax)},
        {nodeColor(std::max(1, maxDeg), std::max(1, maxDeg)),
         tr("橙色：连接较多（度 > %1）").arg(midMax)}};

    qreal y = legendRect.top() + 34.0;
    for (const auto &entry : entries)
    {
        painter->setPen(QPen(QColor(90, 90, 90), 1.0));
        painter->setBrush(entry.color);
        painter->drawEllipse(QRectF(legendRect.left() + 14.0, y, 12.0, 12.0));
        painter->setPen(QColor(35, 35, 35));
        painter->drawText(QRectF(legendRect.left() + 34.0, y - 4.0,
                                 legendRect.width() - 46.0, 20.0),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          entry.text);
        y += 24.0;
    }

    painter->setPen(QColor(70, 70, 70));
    painter->drawText(QRectF(legendRect.left() + 12.0, legendRect.bottom() - 40.0,
                             legendRect.width() - 24.0, 20.0),
                      Qt::AlignLeft | Qt::AlignVCenter,
                      isHighPerformanceMode()
                          ? tr("大规模模式：默认仅绘制强连接与关键标签；点击节点可聚焦其关联关系")
                          : tr("圆点旁文字为影像名；颜色按与其它节点的连接数（度）分档显示"));

    // 额外说明：橙色表示当前网络中连接数最多（度 = maxDeg）
    if (maxDeg > 0)
    {
        QString extra = (_selectedNodeIndex >= 0)
            ? tr("当前已选节点：%1，已高亮其直接连接关系").arg(QString::fromStdString(_net.nodeNames[_selectedNodeIndex]))
            : tr("例如：当前最大度 = %1，颜色越偏橙表示连接越多").arg(maxDeg);
        painter->drawText(QRectF(legendRect.left() + 12.0, legendRect.bottom() - 20.0,
                                 legendRect.width() - 24.0, 18.0),
                          Qt::AlignLeft | Qt::AlignVCenter,
                          extra);
    }

    painter->restore();
}

// ---------------------------------------------------------------------------
// Fruchterman-Reingold 力导向布局（每帧一步）
// ---------------------------------------------------------------------------
void ObservationNetworkView::onForceStep()
{
    if (_forceIter >= MAX_FORCE_LAYOUT_ITERATIONS || _net.numNodes() < 2)
    {
        _forceTimer->stop();
        if (_autoFitPending)
        {
            fitNetworkInView();
        }
        emit forceLayoutDone();
        return;
    }

    const int    n  = _net.numNodes();
    const double k  = std::sqrt(LAYOUT_AREA_SIZE * LAYOUT_AREA_SIZE / n);
    const double k2 = k * k;
    QVector<QPointF> disp(n, {0, 0});

    // 斥力：所有节点对
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            QPointF d = _pos[i] - _pos[j];
            double dist2 = d.x()*d.x() + d.y()*d.y();
            if (dist2 < 1e-6)
            {
                d = {1, 0};
                dist2 = 1.0;
            }
            double dist  = std::sqrt(dist2);
            double fr    = k2 / dist;
            QPointF force = d / dist * fr;
            disp[i] += force;
            disp[j] -= force;
        }
    }

    // 引力：边
    for (const auto &e : _net.edges)
    {
        QPointF d = _pos[e.idx0] - _pos[e.idx1];
        double dist = std::sqrt(d.x()*d.x() + d.y()*d.y());
        if (dist < 1e-6) continue;
        double fa = dist * dist / k;
        QPointF force = d / dist * fa;
        disp[e.idx0] -= force;
        disp[e.idx1] += force;
    }

    // 应用位移，限制温度
    const double half = SCENE_SIZE / 2;
    for (int i = 0; i < n; ++i)
    {
        double dLen = std::sqrt(disp[i].x()*disp[i].x() + disp[i].y()*disp[i].y());
        if (dLen > 1e-6)
        {
            double clamp = std::min(dLen, _temp);
            _pos[i] += disp[i] / dLen * clamp;
        }
        // 限制在场景内
        _pos[i].setX(std::clamp(_pos[i].x(), -half, SCENE_SIZE + half));
        _pos[i].setY(std::clamp(_pos[i].y(), -half, SCENE_SIZE + half));
    }

    _temp *= FORCE_LAYOUT_COOL_RATE;
    ++_forceIter;

    applyPositions();
}

// ---------------------------------------------------------------------------
// 交互
// ---------------------------------------------------------------------------
void ObservationNetworkView::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        const int pickedNodeIndex = pickNodeAt(mapToScene(event->pos()));
        if (pickedNodeIndex >= 0)
        {
            _selectedNodeIndex = pickedNodeIndex;
            rebuildRenderCache();
            viewport()->update();
            emit nodeClicked(pickedNodeIndex, QString::fromStdString(_net.nodeNames[pickedNodeIndex]));
            event->accept();
            return;
        }

        if (_selectedNodeIndex >= 0)
        {
            _selectedNodeIndex = -1;
            rebuildRenderCache();
            viewport()->update();
        }
    }

    QGraphicsView::mousePressEvent(event);
}

void ObservationNetworkView::mouseMoveEvent(QMouseEvent *event)
{
    const int pickedNodeIndex = pickNodeAt(mapToScene(event->pos()));
    if (pickedNodeIndex >= 0 && pickedNodeIndex < (int)_net.nodeNames.size())
    {
        const int degree = pickedNodeIndex < (int)_net.degrees.size() ? _net.degrees[pickedNodeIndex] : 0;
        QToolTip::showText(event->globalPos(),
                           tr("%1\n度: %2")
                               .arg(QString::fromStdString(_net.nodeNames[pickedNodeIndex]))
                               .arg(degree),
                           this);
    }
    else
    {
        QToolTip::hideText();
    }

    QGraphicsView::mouseMoveEvent(event);
}

void ObservationNetworkView::wheelEvent(QWheelEvent *event)
{
    const double factor = event->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15;
    _autoFitPending = false;
    scale(factor, factor);
    rebuildRenderCache();
    viewport()->update();
    event->accept();
}

void ObservationNetworkView::showEvent(QShowEvent *event)
{
    QGraphicsView::showEvent(event);
    if (_autoFitPending)
    {
        fitNetworkInView();
    }
}

void ObservationNetworkView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    if (_autoFitPending)
    {
        fitNetworkInView();
    }
}

int ObservationNetworkView::pickNodeAt(const QPointF &scenePos) const
{
    if (_pos.isEmpty())
    {
        return -1;
    }

    const int maxDegree = _net.degrees.empty()
        ? 0
        : *std::max_element(_net.degrees.begin(), _net.degrees.end());

    int bestNodeIndex = -1;
    double bestDistanceSquared = std::numeric_limits<double>::max();
    for (int nodeIndex = 0; nodeIndex < _pos.size(); ++nodeIndex)
    {
        const QPointF delta = scenePos - _pos[nodeIndex];
        const double radius = nodeIndex < _nodeRadii.size() ? _nodeRadii[nodeIndex] : nodeRadius(nodeIndex, maxDegree);
        const double distanceSquared = delta.x() * delta.x() + delta.y() * delta.y();
        if (distanceSquared <= radius * radius && distanceSquared < bestDistanceSquared)
        {
            bestDistanceSquared = distanceSquared;
            bestNodeIndex = nodeIndex;
        }
    }

    return bestNodeIndex;
}

QImage ObservationNetworkView::toImage(QSize size) const
{
    QImage img(size, QImage::Format_ARGB32_Premultiplied);
    img.fill(Qt::transparent);
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing);
    const_cast<ObservationNetworkView *>(this)->render(&p);
    return img;
}
