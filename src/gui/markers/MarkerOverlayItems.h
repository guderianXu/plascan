#pragma once

#include "model/MarkerTypes.h"

#include <QGraphicsObject>
#include <QRectF>

namespace xjw::gui::markers
{

QColor markerProjectionColor(control_points::ProjectionState state, bool qualityFailure);

class MarkerOverlayItem final : public QGraphicsObject
{
    Q_OBJECT

public:
    MarkerOverlayItem(control_points::MarkerId markerId,
                      QString imageId,
                      QString label,
                      control_points::ProjectionState state,
                      bool qualityFailure,
                      const QRectF &imageBounds,
                      QGraphicsItem *parent = nullptr);

    QRectF boundingRect() const override;
    void paint(QPainter *painter,
               const QStyleOptionGraphicsItem *option,
               QWidget *widget = nullptr) override;

signals:
    void moveFinished(const QString &markerId, const QString &imageId, const QPointF &pixel);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

private:
    control_points::MarkerId _markerId;
    QString _imageId;
    QString _label;
    control_points::ProjectionState _state = control_points::ProjectionState::Predicted;
    bool _qualityFailure = false;
    QRectF _imageBounds;
    QPointF _dragStart;
    qreal _labelWidth = 60.0;
};

} // namespace xjw::gui::markers
