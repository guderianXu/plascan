#include "ImageViewWidget.h"

#include "ui_ImageViewWidget.h"
#include "LayerRenderer.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QGraphicsEllipseItem>
#include <QWheelEvent>
#include <QScrollBar>
#include <QImage>
#include <QImageReader>
#include <QPixmap>
#include <QPen>
#include <QBrush>
#include <QPointer>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>
#include <algorithm>
#include <cmath>

ImageViewWidget::ImageViewWidget(QWidget *parent)
    : QWidget(parent)
    , _view(nullptr)
    , _scene(nullptr)
    , _imageItem(nullptr)
    , _highlightedIndex(-1)
{
    setupView();
}

ImageViewWidget::~ImageViewWidget()
{
    clearMatchPoints();
}

void ImageViewWidget::setupView()
{
    Ui::ImageViewWidget ui;
    ui.setupUi(this);

    _scene = new QGraphicsScene(this);
    _view = ui.m_view;
    _view->setScene(_scene);
    
    // 设置视图属性
    _view->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    _view->setDragMode(QGraphicsView::ScrollHandDrag);
    _view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    _view->setResizeAnchor(QGraphicsView::AnchorViewCenter);
    _view->setBackgroundBrush(QBrush(Qt::darkGray));
    _view->setFrameShape(QFrame::NoFrame);
    
    // 安装事件过滤器以捕获滚轮事件
    _view->viewport()->installEventFilter(this);
    
    // 连接滚动条信号以检测视图变化
    connect(_view->horizontalScrollBar(), &QScrollBar::valueChanged,
        this, &ImageViewWidget::onViewChanged, Qt::QueuedConnection);
    connect(_view->verticalScrollBar(), &QScrollBar::valueChanged,
        this, &ImageViewWidget::onViewChanged, Qt::QueuedConnection);
    
}

bool ImageViewWidget::loadImage(const QString &imagePath)
{
    // ── 异步解码：后台线程读取原始分辨率，主线程更新 Scene ────────────────
    _imagePath = imagePath;

    QFuture<QImage> future = QtConcurrent::run(
        [imagePath]() -> QImage {
            return LayerRenderer::loadImageForDisplay(imagePath, QString());
        });

    // 挂监听器，解码完成后切回主线程更新视图
    auto *watcher = new QFutureWatcher<QImage>(this);
    QPointer<ImageViewWidget> self(this);
    connect(watcher, &QFutureWatcher<QImage>::finished,
            watcher, [self, watcher, imagePath]() {
        watcher->deleteLater();
        if (!self) return;
        if (self->_imagePath != imagePath) return; // 图像已被其他请求替换

        const QImage img = watcher->result();
        if (img.isNull()) {
            emit self->imageLoadFailed(imagePath, self->tr("无法加载图像：%1").arg(imagePath));
            return;
        }

        if (self->_imageItem) {
            self->_scene->removeItem(self->_imageItem);
            delete self->_imageItem;
            self->_imageItem = nullptr;
        }
        self->_imageItem = self->_scene->addPixmap(QPixmap::fromImage(img));
        self->_imageItem->setZValue(0);
        self->_scene->setSceneRect(self->_imageItem->boundingRect());
        QTimer::singleShot(10, self.data(), [self]() {
            if (self) {
                self->fitToView();
            }
        });
    });
    watcher->setFuture(future);

    return true; // 异步，立即返回（真正完成在回调中）
}

void ImageViewWidget::setMatchPoints(const QVector<QPointF> &points)
{
    clearMatchPoints();
    _matchPoints = points;
    
    // 创建点图元
    QPen pen(Qt::red);
    pen.setWidth(2);
    QBrush brush(Qt::red);
    
    // 使用固定屏幕像素大小的点标记，避免在高缩放下遮挡图像细节。
    const qreal screenPointSize = 8.0; // 屏幕像素大小

    constexpr int maxInitialPointItems = 20000;
    const int pointCount = static_cast<int>(points.size());
    const int pointItemCount = std::min(pointCount, maxInitialPointItems);

    for (int i = 0; i < pointItemCount; ++i) {
        const QPointF &pt = points[i];
        QGraphicsEllipseItem *item = _scene->addEllipse(
            -screenPointSize / 2.0, -screenPointSize / 2.0,
            screenPointSize, screenPointSize,
            pen, brush);
        item->setZValue(10); // 确保在图像上方
        item->setData(0, i); // 存储索引
        // 忽略视图变换，使点在屏幕上保持恒定像素大小
        item->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
        item->setPos(pt);
        _pointItems.append(item);
    }
}

void ImageViewWidget::clearMatchPoints()
{
    for (QGraphicsEllipseItem *item : _pointItems) {
        _scene->removeItem(item);
        delete item;
    }
    _pointItems.clear();
    _matchPoints.clear();
    _highlightedIndex = -1;
}

void ImageViewWidget::zoomIn()
{
    qreal factor = currentTransform().m11();
    if (factor < MAX_ZOOM) {
        _view->scale(1.2, 1.2);
        onViewChanged();
    }
}

void ImageViewWidget::zoomOut()
{
    qreal factor = currentTransform().m11();
    if (factor > MIN_ZOOM) {
        _view->scale(1.0 / 1.2, 1.0 / 1.2);
        onViewChanged();
    }
}

void ImageViewWidget::zoomTo(qreal factor)
{
    factor = qBound(MIN_ZOOM, factor, MAX_ZOOM);
    QTransform trans;
    trans.scale(factor, factor);
    _view->setTransform(trans);
    onViewChanged();
}

void ImageViewWidget::fitToView()
{
    if (_imageItem) {
        _view->fitInView(_scene->sceneRect(), Qt::KeepAspectRatio);
        onViewChanged();
    }
}

void ImageViewWidget::resetZoom()
{
    _view->resetTransform();
    onViewChanged();
}

QRectF ImageViewWidget::visibleSceneRect() const
{
    return _view->mapToScene(_view->viewport()->rect()).boundingRect();
}

QTransform ImageViewWidget::currentTransform() const
{
    return _view->transform();
}

void ImageViewWidget::setTransform(const QTransform &transform)
{
    _view->setTransform(transform);
    onViewChanged();
}

void ImageViewWidget::highlightPoint(int index)
{
    clearHighlight();
    
    if (index >= 0 && index < _pointItems.size()) {
        _highlightedIndex = index;
        QGraphicsEllipseItem *item = _pointItems[index];
        
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
    if (_highlightedIndex >= 0 && _highlightedIndex < _pointItems.size()) {
        QGraphicsEllipseItem *item = _pointItems[_highlightedIndex];
        QPen pen(Qt::red);
        pen.setWidth(2);
        item->setPen(pen);
        item->setBrush(QBrush(Qt::red));
        item->setZValue(10);
        
        _highlightedIndex = -1;
    }
}

void ImageViewWidget::setMatchVisibilityMask(const QVector<bool> &mask)
{
    // mask长度可能小于点数；若mask为空则显示所有点
    if (mask.isEmpty()) {
        for (QGraphicsEllipseItem *item : _pointItems) item->setVisible(true);
        return;
    }

    for (int i = 0; i < _pointItems.size(); ++i) {
        QGraphicsEllipseItem *item = _pointItems[i];
        if (!item) continue;
        bool vis = (i < mask.size()) ? mask[i] : false;
        item->setVisible(vis);
    }
}

bool ImageViewWidget::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == _view->viewport()) {
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
        _view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        _view->scale(scaleFactor, scaleFactor);
        onViewChanged();
        
            return true; // 阻止事件继续传播
        }

        if (event->type() == QEvent::MouseButtonPress) {
            QMouseEvent *me = static_cast<QMouseEvent*>(event);
            if (me->button() == Qt::RightButton) {
                const QPointF p = _view->mapToScene(me->pos());
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
                for (int i = 0; i < _pointItems.size(); ++i) {
                    if (!_pointItems[i] || !_pointItems[i]->isVisible()) continue;
                    const QPointF screenPt = _view->mapFromScene(_pointItems[i]->pos());
                    const QPointF d = screenPt - QPointF(me->pos());
                    const qreal dist = std::sqrt(d.x()*d.x() + d.y()*d.y());
                    if (dist < closestDist) {
                        closestDist = dist;
                        closestIdx = i;
                    }
                }
                if (closestIdx >= 0 && closestIdx < _matchPoints.size()) {
                    emit matchPointClicked(closestIdx, _matchPoints[closestIdx]);
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

    for (QGraphicsEllipseItem *item : _pointItems) {
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
