#include "MarkerOverlayItems.h"

#include <QApplication>
#include <QFontMetricsF>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QPen>

#include <algorithm>

namespace xjw::gui::markers
{

QColor markerProjectionColor(control_points::ProjectionState state, bool qualityFailure)
{
    if (qualityFailure || state == control_points::ProjectionState::Blocked)
    {
        return QColor(220, 50, 47);
    }
    if (state == control_points::ProjectionState::ManualPinned)
    {
        return QColor(24, 169, 87);
    }
    if (state == control_points::ProjectionState::AutoDetected)
    {
        return QColor(30, 111, 232);
    }
    return QColor(120, 126, 135);
}

MarkerOverlayItem::MarkerOverlayItem(control_points::MarkerId markerId,
                                     QString imageId,
                                     QString label,
                                     control_points::ProjectionState state,
                                     bool qualityFailure,
                                     const QRectF &imageBounds,
                                     QGraphicsItem *parent)
    : QGraphicsObject(parent)
    , _markerId(std::move(markerId))
    , _imageId(std::move(imageId))
    , _label(std::move(label))
    , _state(state)
    , _qualityFailure(qualityFailure)
    , _imageBounds(imageBounds)
{
    const QFontMetricsF metrics(QApplication::font());
    _labelWidth = std::max<qreal>(48.0, metrics.horizontalAdvance(_label) + 14.0);
    setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemIsMovable,
            state != control_points::ProjectionState::Blocked
                && state != control_points::ProjectionState::Disabled);
    setCursor(state == control_points::ProjectionState::Blocked
                  ? Qt::ForbiddenCursor
                  : Qt::CrossCursor);
    setZValue(100.0);
}

QRectF MarkerOverlayItem::boundingRect() const
{
    return QRectF(-8.0, -13.0, _labelWidth + 27.0, 27.0);
}

void MarkerOverlayItem::paint(QPainter *painter,
                              const QStyleOptionGraphicsItem *,
                              QWidget *)
{
    const QColor color = markerProjectionColor(_state, _qualityFailure);
    QPen pen(color, 2.0);
    pen.setCosmetic(true);
    painter->setPen(pen);
    painter->setBrush(Qt::white);
    painter->drawEllipse(QPointF(0.0, 0.0), 4.0, 4.0);
    painter->drawLine(QPointF(-8.0, 0.0), QPointF(-4.0, 0.0));
    painter->drawLine(QPointF(4.0, 0.0), QPointF(8.0, 0.0));
    painter->drawLine(QPointF(0.0, -8.0), QPointF(0.0, -4.0));
    painter->drawLine(QPointF(0.0, 4.0), QPointF(0.0, 8.0));

    const QRectF label_rect(12.0, -10.0, _labelWidth, 20.0);
    painter->setPen(QPen(color, 1.0));
    painter->setBrush(QColor(255, 255, 255, 225));
    painter->drawRoundedRect(label_rect, 3.0, 3.0);
    painter->setPen(QColor(31, 35, 40));
    painter->drawText(label_rect.adjusted(6.0, 0.0, -4.0, 0.0),
                      Qt::AlignVCenter | Qt::AlignLeft,
                      _label);
}

QVariant MarkerOverlayItem::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (change == QGraphicsItem::ItemPositionChange && !_imageBounds.isEmpty())
    {
        const QPointF requested = value.toPointF();
        return QPointF(std::clamp(requested.x(), _imageBounds.left(), _imageBounds.right()),
                       std::clamp(requested.y(), _imageBounds.top(), _imageBounds.bottom()));
    }
    return QGraphicsObject::itemChange(change, value);
}

void MarkerOverlayItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    _dragStart = pos();
    QGraphicsObject::mousePressEvent(event);
}

void MarkerOverlayItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    QGraphicsObject::mouseReleaseEvent(event);
    if (QLineF(_dragStart, pos()).length() >= 0.01)
    {
        emit moveFinished(_markerId, _imageId, pos());
    }
}

} // namespace xjw::gui::markers
