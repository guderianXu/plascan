#include "MarkerWorkspaceController.h"

#include "ProjectData.h"
#include "project/ProjectIO.h"

#include <QHash>
#include <QJsonArray>
#include <QJsonObject>

#include <algorithm>

namespace xjw::gui::markers
{

bool MarkerWorkspaceController::applyDetectionTaskResult(
    const MarkerDetectionTaskResult &taskResult,
    control_points::DetectionIntegrationResult *integrationResult,
    QString *error)
{
    if (!integrationResult || !_repository)
    {
        if (error)
        {
            *error = QStringLiteral("自动标靶检测结果输出参数无效");
        }
        return false;
    }
    if (taskResult.cancelled)
    {
        if (error)
        {
            *error = QStringLiteral("自动标靶检测已取消，未修改标记点");
        }
        return false;
    }

    QHash<QString, QString> current_signatures;
    if (_projectData)
    {
        const QJsonArray images = _projectData->coreFilesMeta().value(QStringLiteral("images")).toArray();
        for (const QJsonValue &value : images)
        {
            const QJsonObject image = value.toObject();
            QString signature = image.value(QStringLiteral("image_content_signature")).toString();
            if (signature.isEmpty())
            {
                signature = image.value(QStringLiteral("content_signature")).toString();
            }
            current_signatures.insert(image.value(QStringLiteral("image_uuid")).toString(), signature);
        }
    }

    if (taskResult.baseRevision != _repository->revision())
    {
        integrationResult->markerSet = _repository->markerSet();
        integrationResult->pendingReview = taskResult.observations;
        for (const control_points::MarkerDetectionObservation &observation : taskResult.observations)
        {
            integrationResult->conflicts.push_back({
                QStringLiteral("repository_revision_changed"),
                QStringLiteral("检测期间标记点被人工修改，自动结果未写入"),
                observation,
            });
        }
        return mergeDetectionReview(*integrationResult, taskResult.baseRevision, error);
    }

    *integrationResult = control_points::DetectionIntegrator::integrate(
        _repository->markerSet(), taskResult.observations, current_signatures);
    if (integrationResult->appliedDetections > 0)
    {
        QVector<control_points::MarkerId> affected;
        for (const control_points::Marker &marker : integrationResult->markerSet.markers())
        {
            affected.push_back(marker.id);
        }
        if (!pushChange(control_points::MarkerChangeSet::replaceMarkerSet(
                            _repository->markerSet(),
                            integrationResult->markerSet,
                            QStringLiteral("自动检测标靶"),
                            affected),
                        error))
        {
            return false;
        }
    }
    return mergeDetectionReview(*integrationResult, taskResult.baseRevision, error);
}

const control_points::DetectionReviewQueue &
MarkerWorkspaceController::detectionReviewQueue() const noexcept
{
    return _detectionReviewQueue;
}

bool MarkerWorkspaceController::acceptDetectionReview(
    const QString &entryId,
    const control_points::MarkerId &markerId,
    QString *error)
{
    const auto found = std::find_if(
        _detectionReviewQueue.entries.cbegin(),
        _detectionReviewQueue.entries.cend(),
        [&entryId](const control_points::DetectionReviewEntry &entry)
    {
        return entry.id == entryId;
    });
    if (found == _detectionReviewQueue.entries.cend())
    {
        if (error)
        {
            *error = QStringLiteral("待复核检测不存在或已处理: %1").arg(entryId);
        }
        return false;
    }

    const control_points::DetectionReviewEntry entry = *found;
    control_points::MarkerProjection projection;
    projection.imageId = entry.observation.imageId;
    projection.imagePathSnapshot = entry.observation.imagePathSnapshot;
    projection.xy = entry.observation.detection.center;
    projection.state = control_points::ProjectionState::AutoDetected;
    projection.sigmaPx = std::max(0.05, entry.observation.detection.centerSigmaPx);
    projection.confidence = entry.observation.detection.confidence;
    projection.source = QStringLiteral("review-accepted:%1")
                            .arg(entry.observation.detection.source);
    projection.imageContentSignature = entry.observation.imageContentSignature;

    const auto family = entry.observation.detection.family;
    const bool coded = entry.observation.detection.targetId >= 0
        && family != control_points::MarkerTargetFamily::NonCodedCircle
        && family != control_points::MarkerTargetFamily::NonCodedFourQuadrant;
    const QString family_name = control_points::markerTargetFamilyName(family);
    control_points::MarkerId resolved_marker_id = markerId;
    control_points::MarkerId identity_marker_id;
    if (coded)
    {
        for (const control_points::Marker &marker : _repository->markerSet().markers())
        {
            if (marker.targetIdentity.has_value()
                && marker.targetIdentity->family.compare(family_name, Qt::CaseInsensitive) == 0
                && marker.targetIdentity->encodedId == entry.observation.detection.targetId)
            {
                identity_marker_id = marker.id;
                break;
            }
        }
        if (resolved_marker_id.isEmpty())
        {
            resolved_marker_id = identity_marker_id;
        }
    }

    try
    {
        control_points::MarkerSet after = _repository->markerSet();
        if (resolved_marker_id.isEmpty())
        {
            resolved_marker_id = after.addMarker(
                nextMarkerLabel(), control_points::MarkerRole::TieMarker);
            if (coded)
            {
                control_points::TargetIdentity identity;
                identity.family = family_name;
                identity.encodedId = entry.observation.detection.targetId;
                identity.rotationDegrees = entry.observation.detection.rotationDegrees;
                identity.generationSource = entry.observation.detection.source;
                after.setTargetIdentity(resolved_marker_id, identity);
            }
        }
        else if (coded)
        {
            const control_points::Marker &target = after.marker(resolved_marker_id);
            if (target.targetIdentity.has_value())
            {
                if (target.targetIdentity->family.compare(family_name, Qt::CaseInsensitive) != 0
                    || target.targetIdentity->encodedId != entry.observation.detection.targetId)
                {
                    if (error)
                    {
                        *error = QStringLiteral("所选标记已有不同的编码 family/ID");
                    }
                    return false;
                }
            }
            else
            {
                if (!identity_marker_id.isEmpty() && identity_marker_id != resolved_marker_id)
                {
                    if (error)
                    {
                        *error = QStringLiteral("该编码 family/ID 已属于其他标记");
                    }
                    return false;
                }
                control_points::TargetIdentity identity;
                identity.family = family_name;
                identity.encodedId = entry.observation.detection.targetId;
                identity.rotationDegrees = entry.observation.detection.rotationDegrees;
                identity.generationSource = entry.observation.detection.source;
                after.setTargetIdentity(resolved_marker_id, identity);
            }
        }

        after.upsertProjection(resolved_marker_id, projection);
        const control_points::MarkerChangeSet change =
            control_points::MarkerChangeSet::replaceMarkerSet(
                _repository->markerSet(),
                after,
                markerId.isEmpty() && identity_marker_id.isEmpty()
                    ? QStringLiteral("接受检测候选并创建标记")
                    : QStringLiteral("接受检测候选并归入标记"),
                {resolved_marker_id});
        if (!pushChange(change, error))
        {
            return false;
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

    control_points::DetectionReviewQueue updated = _detectionReviewQueue;
    updated.entries.erase(
        std::remove_if(updated.entries.begin(), updated.entries.end(),
                       [&entryId](const control_points::DetectionReviewEntry &item)
        {
            return item.id == entryId;
        }),
        updated.entries.end());
    return saveDetectionReview(updated, error);
}

bool MarkerWorkspaceController::discardDetectionReview(const QString &entryId, QString *error)
{
    control_points::DetectionReviewQueue updated = _detectionReviewQueue;
    const qsizetype old_size = updated.entries.size();
    updated.entries.erase(
        std::remove_if(updated.entries.begin(), updated.entries.end(),
                       [&entryId](const control_points::DetectionReviewEntry &item)
        {
            return item.id == entryId;
        }),
        updated.entries.end());
    if (updated.entries.size() == old_size)
    {
        if (error)
        {
            *error = QStringLiteral("待复核检测不存在或已处理: %1").arg(entryId);
        }
        return false;
    }
    return saveDetectionReview(updated, error);
}

bool MarkerWorkspaceController::saveDetectionReview(
    const control_points::DetectionReviewQueue &queue,
    QString *error)
{
    if (!_projectData || !_projectData->hasProject())
    {
        if (error)
        {
            *error = QStringLiteral("没有打开的工程，无法保存检测复核队列");
        }
        return false;
    }
    const auto saved = control_points::DetectionReviewStore(
                           xjw::common::project::ProjectIO::markerDetectionReviewPath(
                               _projectData->currentProjectPath()))
                           .save(queue);
    if (!saved.ok)
    {
        if (error)
        {
            *error = saved.error;
        }
        emit persistenceError(saved.error);
        return false;
    }
    _detectionReviewQueue = saved.queue;
    emit detectionReviewChanged(_detectionReviewQueue.entries.size());
    return true;
}

bool MarkerWorkspaceController::mergeDetectionReview(
    const control_points::DetectionIntegrationResult &integration,
    quint64 sourceRevision,
    QString *error)
{
    if (integration.pendingReview.isEmpty() && integration.conflicts.isEmpty())
    {
        return true;
    }

    control_points::DetectionReviewQueue updated = _detectionReviewQueue;
    updated.sourceRevision = sourceRevision;
    QHash<QString, qsizetype> index_by_id;
    for (qsizetype index = 0; index < updated.entries.size(); ++index)
    {
        index_by_id.insert(updated.entries.at(index).id, index);
    }

    for (const control_points::DetectionConflict &conflict : integration.conflicts)
    {
        const QString key = control_points::detectionReviewEntryId(
            conflict.observation, conflict.reason);
        control_points::DetectionReviewEntry entry;
        entry.id = key;
        entry.reason = conflict.reason;
        entry.message = conflict.message;
        entry.observation = conflict.observation;
        const auto existing = index_by_id.constFind(key);
        if (existing == index_by_id.cend())
        {
            index_by_id.insert(key, updated.entries.size());
            updated.entries.push_back(std::move(entry));
        }
        else
        {
            updated.entries[existing.value()] = std::move(entry);
        }
    }
    return saveDetectionReview(updated, error);
}

} // namespace xjw::gui::markers
