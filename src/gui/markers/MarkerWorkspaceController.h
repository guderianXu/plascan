#pragma once

#include "ProjectMarkerRepository.h"
#include "MarkerTaskRunner.h"
#include "detection/DetectionReviewStore.h"

#include <QObject>
#include <QPointer>
#include <QVector>

class CanvasWidget;
class ProjectData;
class QUndoStack;

namespace xjw::gui::markers
{

class MarkerOverlayItem;

enum class MarkerPhotoCommand
{
    AddNewMarker,
    PlaceExistingMarker,
    RemoveProjection,
    BlockProjection,
    UnblockProjection,
    DisableProjection,
    OpenFocusMeasurement
};

class MarkerWorkspaceController final : public QObject
{
    Q_OBJECT

public:
    MarkerWorkspaceController(CanvasWidget *canvas, ProjectData *projectData, QObject *parent = nullptr);

    bool openProject(QString *error = nullptr);
    void closeProject();

    bool executePhotoCommand(MarkerPhotoCommand command,
                             const QString &imagePath,
                             const QPointF &pixel = {},
                             const control_points::MarkerId &markerId = {},
                             QString *error = nullptr);

    const control_points::MarkerSet &markerSet() const noexcept;
    QUndoStack *undoStack() const noexcept;
    bool updateMarkerProperties(
        const control_points::MarkerId &markerId,
        const QString &label,
        control_points::MarkerRole role,
        bool enabled,
        const std::optional<control_points::ReferenceCoordinate> &referenceCoordinate,
        QString *error = nullptr);
    bool applyPredictedProjections(const control_points::MarkerId &markerId,
                                   const QVector<control_points::MarkerProjection> &projections,
                                   QString *error = nullptr);
    quint64 markerRevision() const noexcept;
    bool applyDetectionTaskResult(const MarkerDetectionTaskResult &taskResult,
                                  control_points::DetectionIntegrationResult *integrationResult,
                                  QString *error = nullptr);
    const control_points::DetectionReviewQueue &detectionReviewQueue() const noexcept;
    bool acceptDetectionReview(const QString &entryId,
                               const control_points::MarkerId &markerId = {},
                               QString *error = nullptr);
    bool discardDetectionReview(const QString &entryId, QString *error = nullptr);

signals:
    void focusMeasurementRequested(const QString &markerId, const QString &imagePath);
    void persistenceError(const QString &message);
    void markerSetChanged();
    void detectionReviewChanged(int pendingCount);

private:
    void showPhotoContextMenu(const QString &imagePath, const QPointF &pixel);
    void refreshOverlays();
    void clearOverlays();
    void handleOverlayMoveFinished(const QString &markerId,
                                   const QString &imageId,
                                   const QPointF &pixel);
    bool pushChange(const control_points::MarkerChangeSet &change, QString *error);
    bool saveDetectionReview(const control_points::DetectionReviewQueue &queue,
                             QString *error);
    bool mergeDetectionReview(const control_points::DetectionIntegrationResult &integration,
                              quint64 sourceRevision,
                              QString *error);
    QString imageIdForPath(const QString &imagePath) const;
    QString nextMarkerLabel() const;
    control_points::MarkerId nearestMarker(const QString &imageId, const QPointF &pixel) const;

    QPointer<CanvasWidget> _canvas;
    ProjectData *_projectData = nullptr;
    ProjectMarkerRepository *_repository = nullptr;
    QUndoStack *_undoStack = nullptr;
    QVector<QPointer<MarkerOverlayItem>> _overlayItems;
    control_points::DetectionReviewQueue _detectionReviewQueue;
};

} // namespace xjw::gui::markers
