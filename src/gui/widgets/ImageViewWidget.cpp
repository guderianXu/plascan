#include "ImageViewWidget.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QGraphicsEllipseItem>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QScrollBar>
#include <QImage>
#include <QImageReader>
#include <QPixmap>
#include <QPen>
#include <QBrush>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>
#include <cmath>

ImageViewWidget::ImageViewWidget(QWidget *parent)
    : QWidget(parent)
    , m_view(nullptr)
    , m_scene(nullptr)
    , m_imageItem(nullptr)
    , m_highlightedIndex(-1)
{
    setupView();
}

ImageViewWidget::~ImageViewWidget()
{
    clearMatchPoints();
}

void ImageViewWidget::setupView()
{
    m_scene = new QGraphicsScene(this);
    m_view = new QGraphicsView(m_scene, this);
    
    // 设置视图属性
    m_view->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    m_view->setDragMode(QGraphicsView::ScrollHandDrag);
    m_view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    m_view->setResizeAnchor(QGraphicsView::AnchorViewCenter);
    m_view->setBackgroundBrush(QBrush(Qt::darkGray));
    m_view->setFrameShape(QFrame::NoFrame);
    
    // 安装事件过滤器以捕获滚轮事件
    m_view->viewport()->installEventFilter(this);
    
    // 连接滚动条信号以检测视图变化
    connect(m_view->horizontalScrollBar(), &QScrollBar::valueChanged,
        this, &ImageViewWidget::onViewChanged, Qt::QueuedConnection);
    connect(m_view->verticalScrollBar(), &QScrollBar::valueChanged,
        this, &ImageViewWidget::onViewChanged, Qt::QueuedConnection);
    
    // 布局
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_view);
    setLayout(layout);
}

bool ImageViewWidget::loadImage(const QString &imagePath)
{
    // ── 异步解码：后台线程读取原始分辨率，主线程更新 Scene ────────────────
    m_imagePath = imagePath;

    QFuture<QImage> future = QtConcurrent::run(
        [imagePath]() -> QImage {
            QImageReader r(imagePath);
            r.setAutoTransform(true);
            return r.read();
        });

    // 挂监听器，解码完成后切回主线程更新视图
    auto *watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished,
            this, [this, watcher, imagePath]() {
        watcher->deleteLater();
        if (m_imagePath != imagePath) return; // 图像已被其他请求替换

        const QImage img = watcher->result();
        if (img.isNull()) return;

        if (m_imageItem) {
            m_scene->removeItem(m_imageItem);
            delete m_imageItem;
            m_imageItem = nullptr;
        }
        m_imageItem = m_scene->addPixmap(QPixmap::fromImage(img));
        m_imageItem->setZValue(0);
        m_scene->setSceneRect(m_imageItem->boundingRect());
        QTimer::singleShot(10, this, [this]() { fitToView(); });
    });
    watcher->setFuture(future);

    return true; // 异步，立即返回（真正完成在回调中）
}

void ImageViewWidget::setMatchPoints(const QVector<QPointF> &points)
{
    clearMatchPoints();
    m_matchPoints = points;
    
    // 创建点图元
    QPen pen(Qt::red);
    pen.setWidth(2);
    QBrush brush(Qt::red);
    
    // 使用固定屏幕像素大小的点标记，避免在高缩放下遮挡图像细节。
    const qreal screenPointSize = 8.0; // 屏幕像素大小

    for (int i = 0; i < points.size(); ++i) {
        const QPointF &pt = points[i];
        QGraphicsEllipseItem *item = m_scene->addEllipse(
            -screenPointSize / 2.0, -screenPointSize / 2.0,
            screenPointSize, screenPointSize,
            pen, brush);
        item->setZValue(10); // 确保在图像上方
        item->setData(0, i); // 存储索引
        // 忽略视图变换，使点在屏幕上保持恒定像素大小
        item->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
        item->setPos(pt);
        m_pointItems.append(item);
    }
}

void ImageViewWidget::clearMatchPoints()
{
    for (QGraphicsEllipseItem *item : m_pointItems) {
        m_scene->removeItem(item);
        delete item;
    }
    m_pointItems.clear();
    m_matchPoints.clear();
    m_highlightedIndex = -1;
}

void ImageViewWidget::zoomIn()
{
    qreal factor = currentTransform().m11();
    if (factor < MAX_ZOOM) {
        m_view->scale(1.2, 1.2);
        onViewChanged();
    }
}

void ImageViewWidget::zoomOut()
{
    qreal factor = currentTransform().m11();
    if (factor > MIN_ZOOM) {
        m_view->scale(1.0 / 1.2, 1.0 / 1.2);
        onViewChanged();
    }
}

void ImageViewWidget::zoomTo(qreal factor)
{
    factor = qBound(MIN_ZOOM, factor, MAX_ZOOM);
    QTransform trans;
    trans.scale(factor, factor);
    m_view->setTransform(trans);
    onViewChanged();
}

void ImageViewWidget::fitToView()
{
    if (m_imageItem) {
        m_view->fitInView(m_scene->sceneRect(), Qt::KeepAspectRatio);
        onViewChanged();
    }
}

void ImageViewWidget::resetZoom()
{
    m_view->resetTransform();
    onViewChanged();
}

QRectF ImageViewWidget::visibleSceneRect() const
{
    return m_view->mapToScene(m_view->viewport()->rect()).boundingRect();
}

QTransform ImageViewWidget::currentTransform() const
{
    return m_view->transform();
}

void ImageViewWidget::setTransform(const QTransform &transform)
{
    m_view->setTransform(transform);
    onViewChanged();
}

void ImageViewWidget::highlightPoint(int index)
{
    clearHighlight();
    
    if (index >= 0 && index < m_pointItems.size()) {
        m_highlightedIndex = index;
        QGraphicsEllipseItem *item = m_pointItems[index];
        
        // 高亮显示（放大、改变颜色）
        QPen pen(Qt::yellow);
        pen.setWidth(3);
        item->setPen(pen);
        item->setBrush(QBrush(Qt::yellow));
        item->setZValue(20);
        
    }
}

void ImageViewWidget::clearHighlight()
{
    if (m_highlightedIndex >= 0 && m_highlightedIndex < m_pointItems.size()) {
        QGraphicsEllipseItem *item = m_pointItems[m_highlightedIndex];
        QPen pen(Qt::red);
        pen.setWidth(2);
        item->setPen(pen);
        item->setBrush(QBrush(Qt::red));
        item->setZValue(10);
        
        m_highlightedIndex = -1;
    }
}

void ImageViewWidget::setMatchVisibilityMask(const QVector<bool> &mask)
{
    // mask长度可能小于点数；若mask为空则显示所有点
    if (mask.isEmpty()) {
        for (QGraphicsEllipseItem *item : m_pointItems) item->setVisible(true);
        return;
    }

    for (int i = 0; i < m_pointItems.size(); ++i) {
        QGraphicsEllipseItem *item = m_pointItems[i];
        if (!item) continue;
        bool vis = (i < mask.size()) ? mask[i] : false;
        item->setVisible(vis);
    }
}

bool ImageViewWidget::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == m_view->viewport()) {
        if (event->type() == QEvent::Wheel) {
            QWheelEvent *wheelEvent = static_cast<QWheelEvent*>(event);
        
        // 获取滚轮方向
        qreal delta = wheelEvent->angleDelta().y();
        if (delta == 0) {
            return false;
        }
        
        // 计算缩放因子
        qreal scaleFactor = delta > 0 ? 1.15 : (1.0 / 1.15);
        
        // 获取当前缩放级别
        qreal currentZoom = currentTransform().m11();
        qreal newZoom = currentZoom * scaleFactor;
        
        // 限制缩放范围 (0.1x ~ 20x)
        if (newZoom < 0.1 || newZoom > 20.0) {
            return true;
        }
        
        // 执行缩放（以鼠标位置为中心）
        m_view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        m_view->scale(scaleFactor, scaleFactor);
        onViewChanged();
        
            return true; // 阻止事件继续传播
        }

        if (event->type() == QEvent::MouseButtonPress) {
            QMouseEvent *me = static_cast<QMouseEvent*>(event);
            if (me->button() == Qt::RightButton) {
                const QPointF p = m_view->mapToScene(me->pos());
                emit viewRightClicked(p);
                return true; // 阻止默认上下文菜单
            }
        }

        if (event->type() == QEvent::MouseButtonRelease) {
            QMouseEvent *me = static_cast<QMouseEvent*>(event);
            if (me->button() == Qt::MiddleButton) {
                // 中键单击：找最近匹配点
                const qreal threshold = 14.0; // 屏幕像素
                int closestIdx = -1;
                qreal closestDist = threshold + 1.0;
                for (int i = 0; i < m_pointItems.size(); ++i) {
                    if (!m_pointItems[i] || !m_pointItems[i]->isVisible()) continue;
                    const QPointF screenPt = m_view->mapFromScene(m_pointItems[i]->pos());
                    const QPointF d = screenPt - QPointF(me->pos());
                    const qreal dist = std::sqrt(d.x()*d.x() + d.y()*d.y());
                    if (dist < closestDist) {
                        closestDist = dist;
                        closestIdx = i;
                    }
                }
                if (closestIdx >= 0 && closestIdx < m_matchPoints.size()) {
                    emit matchPointClicked(closestIdx, m_matchPoints[closestIdx]);
                }
                return true;
            }
        }
    }
    return QWidget::eventFilter(obj, event);
}

void ImageViewWidget::onViewChanged()
{
    emit viewTransformChanged(currentTransform());
    emit visibleRectChanged(visibleSceneRect());
    updatePointsVisibility();
}

void ImageViewWidget::updatePointsVisibility()
{
    // 可以根据缩放级别调整点的大小
    qreal zoom = currentTransform().m11();
    qreal pointSize = 6.0 / qMax(1.0, zoom * 0.5);

    for (QGraphicsEllipseItem *item : m_pointItems) {
        if (!item) continue;
        // 跳过设置为忽略变换的点（这些点为固定屏幕像素大小）
        if (item->flags() & QGraphicsItem::ItemIgnoresTransformations) continue;

        QPointF center = item->rect().center();
        item->setRect(center.x() - pointSize/2, center.y() - pointSize/2,
                      pointSize, pointSize);
    }
}

qreal ImageViewWidget::calculateZoomFactor() const
{
    return currentTransform().m11();
}
