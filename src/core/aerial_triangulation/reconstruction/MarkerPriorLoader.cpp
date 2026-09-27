#include "reconstruction/MarkerPriorLoader.h"

#include "io/MarkerSetStore.h"
#include "project/ProjectMetadata.h"
#include "reference/CoordinateReference.h"

#include <QFileInfo>
#include <QHash>

#include <algorithm>

namespace xjw::aerial_triangulation
{
    namespace
    {

        control_points::PriorObservationState priorState(control_points::ProjectionState state)
        {
            switch (state)
            {
            case control_points::ProjectionState::ManualPinned:
                return control_points::PriorObservationState::ManualPinned;
            case control_points::ProjectionState::AutoDetected:
                return control_points::PriorObservationState::AutoDetected;
            case control_points::ProjectionState::Blocked:
                return control_points::PriorObservationState::Blocked;
            case control_points::ProjectionState::Disabled:
                return control_points::PriorObservationState::Disabled;
            case control_points::ProjectionState::Predicted:
                return control_points::PriorObservationState::Predicted;
            }
            return control_points::PriorObservationState::Predicted;
        }

    } // namespace

    MarkerPriorLoadResult MarkerPriorLoader::load(const QString& path,
                                                  const QJsonObject& projectMeta,
                                                  const QMap<QString, ImageId>& imageIdByCanonicalId,
                                                  const placoordinate::CoordinateContext* coordinateContext)
    {
        MarkerPriorLoadResult result;
        if (path.trimmed().isEmpty() || !QFileInfo::exists(path))
        {
            return result;
        }

        const control_points::MarkerSetIoResult loaded = control_points::MarkerSetStore(path).load();
        if (!loaded.ok)
        {
            result.ok = false;
            result.errorMessage = QStringLiteral("无法读取空三人工标记 sidecar: %1").arg(loaded.error);
            return result;
        }

        QHash<QString, QString> pathByUuid;
        QHash<QString, QString> signatureByImageId;
        const QMap<QString, QJsonObject> imageMetaByPath =
            xjw::common::project::projectImageMetaByPath(projectMeta, true);
        for (auto it = imageMetaByPath.cbegin(); it != imageMetaByPath.cend(); ++it)
        {
            const QString normalizedPath = xjw::common::project::normalizePath(it.key());
            const QString imageUuid = it.value().value(QStringLiteral("image_uuid")).toString().trimmed();
            if (!imageUuid.isEmpty())
            {
                pathByUuid.insert(imageUuid, normalizedPath);
            }
            QString signature = it.value().value(QStringLiteral("image_content_signature")).toString();
            if (signature.isEmpty())
            {
                signature = it.value().value(QStringLiteral("content_signature")).toString();
            }
            if (!signature.isEmpty())
            {
                signatureByImageId.insert(imageUuid, signature);
            }
        }

        QMap<control_points::MarkerId, control_points::MetricReferenceCoordinateResult> metricReferenceByMarker;
        std::string commonReferenceKey;
        for (const control_points::Marker& marker : loaded.markerSet.markers())
        {
            if (!marker.enabled || marker.role == control_points::MarkerRole::TieMarker ||
                !marker.referenceCoordinate.has_value())
            {
                continue;
            }

            const control_points::MetricReferenceCoordinateResult metric_reference =
                control_points::resolveMetricReferenceCoordinate(*marker.referenceCoordinate, coordinateContext);
            if (!metric_reference.ok)
            {
                result.ok = false;
                result.errorMessage = QStringLiteral("标记 %1（%2）的参考坐标尚未归一化到 solver frame: %3")
                                          .arg(marker.label, marker.id, QString::fromStdString(metric_reference.error));
                return result;
            }
            if (commonReferenceKey.empty())
            {
                commonReferenceKey = metric_reference.referenceKey;
            }
            else if (metric_reference.referenceKey != commonReferenceKey)
            {
                result.ok = false;
                result.errorMessage =
                    QStringLiteral("标记 %1（%2）使用了不同的 CRS 或垂直基准；同一次空三只能使用一种归一化参考")
                        .arg(marker.label, marker.id);
                return result;
            }
            metricReferenceByMarker.insert(marker.id, metric_reference);
        }

        result.tracks.reserve(static_cast<std::size_t>(loaded.markerSet.markers().size()));
        for (const control_points::Marker& marker : loaded.markerSet.markers())
        {
            if (!marker.enabled)
            {
                continue;
            }

            control_points::PriorTrack track;
            track.markerId = marker.id.toStdString();
            track.confidence = 1.0;
            track.role = marker.role;
            const auto metricReference = metricReferenceByMarker.constFind(marker.id);
            if (metricReference != metricReferenceByMarker.constEnd())
            {
                track.hasReference = true;
                track.referenceUsable = true;
                track.referencePoint = metricReference->pointMetres;
                track.referenceSigma = metricReference->sigmaMetres;
            }

            track.observations.reserve(static_cast<std::size_t>(marker.projections.size()));
            for (const control_points::MarkerProjection& projection : marker.projections)
            {
                const QString canonicalImageId = projection.imageId.trimmed();
                if (canonicalImageId.isEmpty())
                {
                    result.ok = false;
                    result.errorMessage =
                        QStringLiteral("标记 %1 的投影缺少 canonical image_uuid；image_path_snapshot 不能用于绑定相机")
                            .arg(marker.id);
                    return result;
                }
                if (imageIdByCanonicalId.isEmpty())
                {
                    result.ok = false;
                    result.errorMessage =
                        QStringLiteral("标记 %1 需要调用方提供按 canonical image_uuid 建立的相机绑定").arg(marker.id);
                    return result;
                }
                if (!pathByUuid.isEmpty() && !pathByUuid.contains(canonicalImageId))
                {
                    result.ok = false;
                    result.errorMessage = QStringLiteral("标记 %1 引用了工程中不存在的 canonical image_uuid: %2")
                                              .arg(marker.id, canonicalImageId);
                    return result;
                }
                const auto imageIt = imageIdByCanonicalId.constFind(canonicalImageId);
                if (imageIt == imageIdByCanonicalId.constEnd())
                {
                    // 合法但未选入本次空三的影像不参与当前先验；不能用路径快照把它
                    // 重新映射到另一台相机。
                    if (pathByUuid.isEmpty())
                    {
                        result.ok = false;
                        result.errorMessage = QStringLiteral("标记 %1 引用了未绑定的 canonical image_uuid: %2")
                                                  .arg(marker.id, canonicalImageId);
                        return result;
                    }
                    continue;
                }
                const ImageId imageId = imageIt.value();

                control_points::PriorObservation observation;
                observation.imageId = imageId;
                observation.x = projection.xy.x();
                observation.y = projection.xy.y();
                observation.state = priorState(projection.state);
                observation.confidence = projection.state == control_points::ProjectionState::ManualPinned
                                             ? 1.0
                                             : std::clamp(projection.confidence, 0.0, 1.0);
                const QString currentSignature = signatureByImageId.value(canonicalImageId);
                observation.stale = !projection.imageContentSignature.isEmpty() && !currentSignature.isEmpty() &&
                                    projection.imageContentSignature != currentSignature;
                track.observations.push_back(observation);
            }
            if (!track.observations.empty())
            {
                result.tracks.push_back(std::move(track));
            }
        }

        result.scaleBars.reserve(static_cast<std::size_t>(loaded.markerSet.scaleBars().size()));
        for (const control_points::ScaleBar& scaleBar : loaded.markerSet.scaleBars())
        {
            control_points::PriorScaleBar prior;
            prior.scaleBarId = scaleBar.id.toStdString();
            prior.firstMarkerId = scaleBar.firstMarkerId.toStdString();
            prior.secondMarkerId = scaleBar.secondMarkerId.toStdString();
            prior.role = scaleBar.role;
            prior.enabled = scaleBar.enabled;
            prior.measuredDistance = scaleBar.measuredDistance;
            prior.sigma = scaleBar.sigma;
            result.scaleBars.push_back(std::move(prior));
        }
        return result;
    }

} // namespace xjw::aerial_triangulation
