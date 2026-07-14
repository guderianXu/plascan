#include "MarkerWorkspaceController.h"

#include "CanvasWidget.h"
#include "MarkerOverlayItems.h"

#include <QAction>
#include <QCursor>
#include <QGraphicsScene>
#include <QLineF>
#include <QMenu>

#include <cmath>

namespace xjw::gui::markers
{

void MarkerWorkspaceController::showPhotoContextMenu(const QString &imagePath, const QPointF &pixel)
{
    if (!_canvas || imageIdForPath(imagePath).isEmpty())
    {
        return;
    }

    const QString image_id = imageIdForPath(imagePath);
    const control_points::MarkerId nearby_marker = nearestMarker(image_id, pixel);
    QMenu menu(_canvas);
    QAction *add_action = menu.addAction(QStringLiteral("添加新标记"));
    connect(add_action, &QAction::triggered, this, [this, imagePath, pixel]()
    {
        QString error;
        if (!executePhotoCommand(MarkerPhotoCommand::AddNewMarker, imagePath, pixel, {}, &error))
        {
            emit persistenceError(error);
        }
    });

    QMenu *place_menu = menu.addMenu(QStringLiteral("放置已有标记"));
    for (const control_points::Marker &marker : _repository->markerSet().markers())
    {
        QAction *action = place_menu->addAction(marker.label);
        connect(action, &QAction::triggered, this, [this, imagePath, pixel, markerId = marker.id]()
        {
            QString error;
            if (!executePhotoCommand(
                    MarkerPhotoCommand::PlaceExistingMarker, imagePath, pixel, markerId, &error))
            {
                emit persistenceError(error);
            }
        });
    }
    place_menu->setEnabled(!place_menu->actions().isEmpty());

    if (!nearby_marker.isEmpty())
    {
        const auto &projection = _repository->markerSet().marker(nearby_marker).projection(image_id);
        menu.addSeparator();
        QAction *remove_action = menu.addAction(QStringLiteral("移除当前投影"));
        connect(remove_action, &QAction::triggered, this, [this, imagePath, nearby_marker]()
        {
            QString error;
            if (!executePhotoCommand(
                    MarkerPhotoCommand::RemoveProjection, imagePath, {}, nearby_marker, &error))
            {
                emit persistenceError(error);
            }
        });

        const bool blocked = projection.state == control_points::ProjectionState::Blocked;
        QAction *block_action = menu.addAction(blocked
                                                   ? QStringLiteral("解除屏蔽投影")
                                                   : QStringLiteral("屏蔽投影"));
        connect(block_action, &QAction::triggered, this,
                [this, imagePath, nearby_marker, blocked]()
        {
            QString error;
            const MarkerPhotoCommand command = blocked
                ? MarkerPhotoCommand::UnblockProjection
                : MarkerPhotoCommand::BlockProjection;
            if (!executePhotoCommand(command, imagePath, {}, nearby_marker, &error))
            {
                emit persistenceError(error);
            }
        });

        menu.addSeparator();
        QAction *focus_action = menu.addAction(QStringLiteral("聚焦量测"));
        connect(focus_action, &QAction::triggered, this, [this, imagePath, nearby_marker]()
        {
            executePhotoCommand(
                MarkerPhotoCommand::OpenFocusMeasurement, imagePath, {}, nearby_marker, nullptr);
        });
    }

    menu.exec(QCursor::pos());
}

void MarkerWorkspaceController::refreshOverlays()
{
    clearOverlays();
    if (!_canvas || !_canvas->scene() || !_canvas->hasDisplayImage())
    {
        return;
    }

    const QString image_id = imageIdForPath(_canvas->currentImagePath());
    if (image_id.isEmpty())
    {
        return;
    }

    for (const control_points::Marker &marker : _repository->markerSet().markers())
    {
        for (const control_points::MarkerProjection &projection : marker.projections)
        {
            if (projection.imageId != image_id
                || projection.state == control_points::ProjectionState::Disabled)
            {
                continue;
            }
            const bool quality_failure = std::isfinite(projection.residualPx)
                && projection.residualPx > 3.0;
            auto *item = new MarkerOverlayItem(marker.id,
                                               image_id,
                                               marker.label,
                                               projection.state,
                                               quality_failure,
                                               _canvas->imageBounds());
            item->setPos(projection.xy);
            connect(item, &MarkerOverlayItem::moveFinished,
                    this, &MarkerWorkspaceController::handleOverlayMoveFinished);
            _canvas->scene()->addItem(item);
            _overlayItems.push_back(item);
        }
    }
}

void MarkerWorkspaceController::clearOverlays()
{
    for (const QPointer<MarkerOverlayItem> &item : std::as_const(_overlayItems))
    {
        delete item.data();
    }
    _overlayItems.clear();
}

void MarkerWorkspaceController::handleOverlayMoveFinished(const QString &markerId,
                                                          const QString &imageId,
                                                          const QPointF &pixel)
{
    if (!_canvas || imageIdForPath(_canvas->currentImagePath()) != imageId)
    {
        return;
    }
    QString error;
    if (!executePhotoCommand(MarkerPhotoCommand::PlaceExistingMarker,
                             _canvas->currentImagePath(),
                             pixel,
                             markerId,
                             &error))
    {
        emit persistenceError(error);
        refreshOverlays();
    }
}

control_points::MarkerId MarkerWorkspaceController::nearestMarker(const QString &imageId,
                                                                 const QPointF &pixel) const
{
    if (!_canvas)
    {
        return {};
    }
    const qreal radius = QLineF(_canvas->mapToScene(QPoint(0, 0)),
                                _canvas->mapToScene(QPoint(12, 0))).length();
    qreal best_distance = radius;
    control_points::MarkerId best;
    for (const control_points::Marker &marker : _repository->markerSet().markers())
    {
        for (const control_points::MarkerProjection &projection : marker.projections)
        {
            if (projection.imageId != imageId)
            {
                continue;
            }
            const qreal distance = QLineF(projection.xy, pixel).length();
            if (distance <= best_distance)
            {
                best_distance = distance;
                best = marker.id;
            }
        }
    }
    return best;
}

} // namespace xjw::gui::markers
