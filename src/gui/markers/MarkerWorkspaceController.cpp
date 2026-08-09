#include "MarkerWorkspaceController.h"

#include "CanvasWidget.h"
#include "MarkerUndoCommand.h"
#include "ProjectData.h"
#include "project/ProjectIO.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QUndoStack>

namespace xjw::gui::markers
{

namespace
{

QString comparablePath(const QString &path)
{
    const QFileInfo info(path);
    const QString canonical = info.exists() ? info.canonicalFilePath() : info.absoluteFilePath();
#ifdef Q_OS_WIN
    return QDir::cleanPath(canonical).toLower();
#else
    return QDir::cleanPath(canonical);
#endif
}

control_points::ProjectionState restoredState(const control_points::MarkerProjection &projection)
{
    if (projection.source.contains(QLatin1String("manual"), Qt::CaseInsensitive))
    {
        return control_points::ProjectionState::ManualPinned;
    }
    if (projection.source.contains(QLatin1String("auto"), Qt::CaseInsensitive))
    {
        return control_points::ProjectionState::AutoDetected;
    }
    return control_points::ProjectionState::Predicted;
}

} // namespace

MarkerWorkspaceController::MarkerWorkspaceController(CanvasWidget *canvas,
                                                     ProjectData *projectData,
                                                     QObject *parent)
    : QObject(parent)
    , _canvas(canvas)
    , _projectData(projectData)
    , _repository(new ProjectMarkerRepository(projectData, this))
    , _undoStack(new QUndoStack(this))
{
    if (_canvas)
    {
        connect(_canvas, &CanvasWidget::imageContextRequested,
                this, &MarkerWorkspaceController::showPhotoContextMenu);
        connect(_canvas, &CanvasWidget::activeImageChanged,
                this, [this](const QString &) { refreshOverlays(); });
    }
    connect(_repository, &ProjectMarkerRepository::markerSetChanged,
            this, [this](quint64, const QVector<QString> &)
    {
        emit markerSetChanged();
        // Repository 可能在覆盖层信号栈内更新，排队刷新可避免删除正在发信号的 item。
        QMetaObject::invokeMethod(this, [this]() { refreshOverlays(); }, Qt::QueuedConnection);
    });
}

bool MarkerWorkspaceController::openProject(QString *error)
{
    _undoStack->clear();
    if (!_repository->open(error))
    {
        return false;
    }
    const auto loaded = control_points::DetectionReviewStore(
                            xjw::common::project::ProjectIO::markerDetectionReviewPath(
                                _projectData->currentProjectPath()))
                            .load();
    if (!loaded.ok)
    {
        if (error)
        {
            *error = loaded.error;
        }
        return false;
    }
    _detectionReviewQueue = loaded.queue;
    emit detectionReviewChanged(_detectionReviewQueue.entries.size());
    return true;
}

void MarkerWorkspaceController::closeProject()
{
    clearOverlays();
    _undoStack->clear();
    _repository->reset();
    _detectionReviewQueue = control_points::DetectionReviewQueue();
    emit detectionReviewChanged(0);
}

bool MarkerWorkspaceController::executePhotoCommand(MarkerPhotoCommand command,
                                                    const QString &imagePath,
                                                    const QPointF &pixel,
                                                    const control_points::MarkerId &markerId,
                                                    QString *error)
{
    const QString image_id = imageIdForPath(imagePath);
    if (image_id.isEmpty())
    {
        if (error)
        {
            *error = QStringLiteral("当前照片不属于项目或缺少稳定影像 UUID: %1").arg(imagePath);
        }
        return false;
    }

    try
    {
        control_points::MarkerProjection projection;
        projection.imageId = image_id;
        projection.imagePathSnapshot = imagePath;
        projection.xy = pixel;
        projection.state = control_points::ProjectionState::ManualPinned;
        projection.sigmaPx = 0.5;
        projection.confidence = 1.0;
        projection.source = QStringLiteral("manual");

        switch (command)
        {
        case MarkerPhotoCommand::AddNewMarker:
            return pushChange(control_points::MarkerChangeSet::createMarkerWithProjection(
                                  _repository->markerSet(),
                                  nextMarkerLabel(),
                                  control_points::MarkerRole::TieMarker,
                                  projection),
                              error);
        case MarkerPhotoCommand::PlaceExistingMarker:
            return pushChange(control_points::MarkerChangeSet::replaceProjection(
                                  _repository->markerSet(), markerId, image_id, projection),
                              error);
        case MarkerPhotoCommand::RemoveProjection:
            return pushChange(control_points::MarkerChangeSet::removeProjection(
                                  _repository->markerSet(), markerId, image_id),
                              error);
        case MarkerPhotoCommand::BlockProjection:
            return pushChange(control_points::MarkerChangeSet::setProjectionState(
                                  _repository->markerSet(),
                                  markerId,
                                  image_id,
                                  control_points::ProjectionState::Blocked,
                                  QStringLiteral("屏蔽标记投影")),
                              error);
        case MarkerPhotoCommand::UnblockProjection:
        {
            const auto &current = _repository->markerSet().marker(markerId).projection(image_id);
            return pushChange(control_points::MarkerChangeSet::setProjectionState(
                                  _repository->markerSet(),
                                  markerId,
                                  image_id,
                                  restoredState(current),
                                  QStringLiteral("解除屏蔽标记投影")),
                              error);
        }
        case MarkerPhotoCommand::DisableProjection:
            return pushChange(control_points::MarkerChangeSet::setProjectionState(
                                  _repository->markerSet(),
                                  markerId,
                                  image_id,
                                  control_points::ProjectionState::Disabled,
                                  QStringLiteral("禁用标记投影")),
                              error);
        case MarkerPhotoCommand::OpenFocusMeasurement:
            emit focusMeasurementRequested(markerId, imagePath);
            return true;
        }
    }
    catch (const std::exception &exception)
    {
        if (error)
        {
            *error = QString::fromUtf8(exception.what());
        }
        return false;
    }
    return false;
}

const control_points::MarkerSet &MarkerWorkspaceController::markerSet() const noexcept
{
    return _repository->markerSet();
}

bool MarkerWorkspaceController::updateMarkerProperties(
    const control_points::MarkerId &markerId,
    const QString &label,
    control_points::MarkerRole role,
    bool enabled,
    const std::optional<control_points::ReferenceCoordinate> &referenceCoordinate,
    QString *error)
{
    try
    {
        return pushChange(control_points::MarkerChangeSet::updateMarker(
                              _repository->markerSet(),
                              markerId,
                              label,
                              role,
                              enabled,
                              referenceCoordinate),
                          error);
    }
    catch (const std::exception &exception)
    {
        if (error)
        {
            *error = QString::fromUtf8(exception.what());
        }
        return false;
    }
}

bool MarkerWorkspaceController::applyPredictedProjections(
    const control_points::MarkerId &markerId,
    const QVector<control_points::MarkerProjection> &projections,
    QString *error)
{
    if (projections.isEmpty())
    {
        return true;
    }
    try
    {
        return pushChange(control_points::MarkerChangeSet::replaceProjections(
                              _repository->markerSet(),
                              markerId,
                              projections,
                              QStringLiteral("生成标记预测投影")),
                          error);
    }
    catch (const std::exception &exception)
    {
        if (error)
        {
            *error = QString::fromUtf8(exception.what());
        }
        return false;
    }
}

quint64 MarkerWorkspaceController::markerRevision() const noexcept
{
    return _repository ? _repository->revision() : 0;
}

bool MarkerWorkspaceController::pushChange(const control_points::MarkerChangeSet &change, QString *error)
{
    _undoStack->push(new MarkerUndoCommand(_repository, change));
    QString save_error;
    if (!_repository->save(&save_error))
    {
        if (error)
        {
            *error = save_error;
        }
        emit persistenceError(save_error);
        return false;
    }
    return true;
}

QString MarkerWorkspaceController::imageIdForPath(const QString &imagePath) const
{
    if (!_projectData)
    {
        return {};
    }
    const QString expected = comparablePath(imagePath);
    const QJsonArray images = _projectData->coreFilesMeta().value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : images)
    {
        const QJsonObject image = value.toObject();
        if (comparablePath(image.value(QStringLiteral("path")).toString()) == expected)
        {
            return image.value(QStringLiteral("image_uuid")).toString();
        }
    }
    return {};
}

QString MarkerWorkspaceController::nextMarkerLabel() const
{
    int index = 1;
    while (true)
    {
        const QString candidate = QStringLiteral("point %1").arg(index++);
        bool exists = false;
        for (const control_points::Marker &marker : _repository->markerSet().markers())
        {
            if (marker.label == candidate)
            {
                exists = true;
                break;
            }
        }
        if (!exists)
        {
            return candidate;
        }
    }
}

} // namespace xjw::gui::markers
