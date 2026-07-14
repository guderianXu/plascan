#include "DetectionIntegrator.h"

#include <QLineF>
#include <QSet>

#include <algorithm>

namespace xjw::control_points
{
namespace
{

QString observationKey(const MarkerDetectionObservation &observation)
{
    return QStringLiteral("%1:%2:%3")
        .arg(markerTargetFamilyName(observation.detection.family))
        .arg(observation.detection.targetId)
        .arg(observation.imageId);
}

bool isCoded(const MarkerDetectionObservation &observation)
{
    return observation.detection.targetId >= 0
        && observation.detection.family != MarkerTargetFamily::NonCodedCircle
        && observation.detection.family != MarkerTargetFamily::NonCodedFourQuadrant;
}

MarkerId findMarkerByIdentity(const MarkerSet &set, const QString &family, int encodedId)
{
    for (const Marker &marker : set.markers())
    {
        if (marker.targetIdentity.has_value()
            && marker.targetIdentity->family == family
            && marker.targetIdentity->encodedId == encodedId)
        {
            return marker.id;
        }
    }
    return {};
}

QString uniqueTargetLabel(const MarkerSet &set, const QString &family, int encodedId)
{
    const QString plain = QStringLiteral("target %1").arg(encodedId);
    const auto label_exists = [&set](const QString &label)
    {
        return std::any_of(set.markers().cbegin(), set.markers().cend(), [&label](const Marker &marker)
        {
            return marker.label == label;
        });
    };
    if (!label_exists(plain))
    {
        return plain;
    }

    const QString qualified = QStringLiteral("target %1 (%2)").arg(encodedId).arg(family);
    if (!label_exists(qualified))
    {
        return qualified;
    }
    int suffix = 2;
    while (label_exists(QStringLiteral("%1 #%2").arg(qualified).arg(suffix)))
    {
        ++suffix;
    }
    return QStringLiteral("%1 #%2").arg(qualified).arg(suffix);
}

const MarkerProjection *projectionForImage(const Marker &marker, const QString &imageId)
{
    const auto found = std::find_if(marker.projections.cbegin(), marker.projections.cend(),
                                    [&imageId](const MarkerProjection &projection)
    {
        return projection.imageId == imageId;
    });
    return found == marker.projections.cend() ? nullptr : &*found;
}

void addConflict(DetectionIntegrationResult *result,
                 const QString &reason,
                 const QString &message,
                 const MarkerDetectionObservation &observation)
{
    result->conflicts.push_back({reason, message, observation});
    result->pendingReview.push_back(observation);
}

} // namespace

DetectionIntegrationResult DetectionIntegrator::integrate(
    const MarkerSet &base,
    const QVector<MarkerDetectionObservation> &observations,
    const QHash<QString, QString> &currentImageSignatures,
    double manualConflictThresholdPx)
{
    DetectionIntegrationResult result;
    result.markerSet = base;

    QHash<QString, MarkerDetectionObservation> best_by_key;
    for (const MarkerDetectionObservation &observation : observations)
    {
        if (!isCoded(observation))
        {
            addConflict(&result,
                        QStringLiteral("unassociated_non_coded"),
                        QStringLiteral("非编码标靶需要相机几何后才能跨影像归并"),
                        observation);
            continue;
        }

        const auto current_signature = currentImageSignatures.constFind(observation.imageId);
        if (current_signature != currentImageSignatures.cend()
            && !observation.imageContentSignature.isEmpty()
            && observation.imageContentSignature != current_signature.value())
        {
            addConflict(&result,
                        QStringLiteral("stale_image"),
                        QStringLiteral("检测后影像内容已变化，结果不能自动合并"),
                        observation);
            continue;
        }

        const QString key = observationKey(observation);
        auto existing = best_by_key.find(key);
        if (existing == best_by_key.end())
        {
            best_by_key.insert(key, observation);
            continue;
        }

        MarkerDetectionObservation loser = observation;
        if (observation.detection.confidence > existing->detection.confidence)
        {
            loser = *existing;
            *existing = observation;
        }
        addConflict(&result,
                    QStringLiteral("duplicate_coded_id"),
                    QStringLiteral("同一影像检测到重复 family/ID，保留高置信候选"),
                    loser);
    }

    QVector<QString> keys = best_by_key.keys();
    std::sort(keys.begin(), keys.end());
    for (const QString &key : keys)
    {
        const MarkerDetectionObservation observation = best_by_key.value(key);
        const QString family = markerTargetFamilyName(observation.detection.family);
        MarkerId marker_id = findMarkerByIdentity(result.markerSet,
                                                  family,
                                                  observation.detection.targetId);
        if (marker_id.isEmpty())
        {
            marker_id = result.markerSet.addMarker(
                uniqueTargetLabel(result.markerSet, family, observation.detection.targetId),
                MarkerRole::TieMarker);
            TargetIdentity identity;
            identity.family = family;
            identity.encodedId = observation.detection.targetId;
            identity.rotationDegrees = observation.detection.rotationDegrees;
            identity.generationSource = observation.detection.source;
            result.markerSet.setTargetIdentity(marker_id, identity);
            ++result.createdMarkers;
        }

        const Marker &marker = result.markerSet.marker(marker_id);
        const MarkerProjection *existing_projection = projectionForImage(marker, observation.imageId);
        if (existing_projection != nullptr)
        {
            if (existing_projection->state == ProjectionState::Blocked
                || existing_projection->state == ProjectionState::Disabled)
            {
                addConflict(&result,
                            QStringLiteral("blocked_projection"),
                            QStringLiteral("用户已屏蔽或禁用该影像上的投影"),
                            observation);
                continue;
            }
            if (existing_projection->state == ProjectionState::ManualPinned)
            {
                if (QLineF(existing_projection->xy, observation.detection.center).length()
                    > manualConflictThresholdPx)
                {
                    addConflict(&result,
                                QStringLiteral("manual_projection_mismatch"),
                                QStringLiteral("自动检测与人工投影位置不一致，保留人工结果"),
                                observation);
                }
                continue;
            }
        }

        MarkerProjection projection;
        projection.imageId = observation.imageId;
        projection.imagePathSnapshot = observation.imagePathSnapshot;
        projection.xy = observation.detection.center;
        projection.state = ProjectionState::AutoDetected;
        projection.sigmaPx = std::max(0.05, observation.detection.centerSigmaPx);
        projection.confidence = observation.detection.confidence;
        projection.source = observation.detection.source;
        projection.imageContentSignature = observation.imageContentSignature;
        result.markerSet.upsertProjection(marker_id, projection);
        ++result.appliedDetections;
    }
    return result;
}

} // namespace xjw::control_points
