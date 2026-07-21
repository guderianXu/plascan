#include "reconstruction/MarkerPriorLoader.h"

#include "io/MarkerSetStore.h"
#include "project/ProjectCommonUtils.h"
#include "project/ProjectMetadata.h"

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

ImageId imageIdForPath(const QString &path,
                       const QMap<QString, ImageId> &imageIdByPath)
{
    const QString normalized = xjw::common::project::normalizePath(path);
    const auto direct = imageIdByPath.constFind(normalized);
    if (direct != imageIdByPath.cend())
    {
        return direct.value();
    }
    for (auto it = imageIdByPath.cbegin(); it != imageIdByPath.cend(); ++it)
    {
        if (xjw::common::project::pathTokenMatchesImage(normalized, it.key()))
        {
            return it.value();
        }
    }
    return kInvalidImageId;
}

QString resolvedProjectionPath(const control_points::MarkerProjection &projection,
                               const QHash<QString, QString> &pathByUuid,
                               const QJsonObject &projectMeta)
{
    QString path = pathByUuid.value(projection.imageId);
    if (path.isEmpty())
    {
        path = projection.imagePathSnapshot.trimmed();
    }
    if (path.isEmpty())
    {
        return QString();
    }
    const QString projectPath =
        xjw::common::project::resolveProjectImagePathFromToken(path, projectMeta);
    return xjw::common::project::normalizePath(projectPath.isEmpty() ? path : projectPath);
}

} // namespace

MarkerPriorLoadResult MarkerPriorLoader::load(
    const QString &path,
    const QJsonObject &projectMeta,
    const QMap<QString, ImageId> &imageIdByPath)
{
    MarkerPriorLoadResult result;
    if (path.trimmed().isEmpty() || !QFileInfo::exists(path))
    {
        return result;
    }

    const control_points::MarkerSetIoResult loaded =
        control_points::MarkerSetStore(path).load();
    if (!loaded.ok)
    {
        result.ok = false;
        result.errorMessage = QStringLiteral("无法读取空三人工标记 sidecar: %1")
                                  .arg(loaded.error);
        return result;
    }

    QHash<QString, QString> pathByUuid;
    QHash<QString, QString> signatureByPath;
    const QMap<QString, QJsonObject> imageMetaByPath =
        xjw::common::project::projectImageMetaByPath(projectMeta, true);
    for (auto it = imageMetaByPath.cbegin(); it != imageMetaByPath.cend(); ++it)
    {
        const QString normalizedPath = xjw::common::project::normalizePath(it.key());
        const QString imageUuid =
            it.value().value(QStringLiteral("image_uuid")).toString().trimmed();
        if (!imageUuid.isEmpty())
        {
            pathByUuid.insert(imageUuid, normalizedPath);
        }
        QString signature =
            it.value().value(QStringLiteral("image_content_signature")).toString();
        if (signature.isEmpty())
        {
            signature = it.value().value(QStringLiteral("content_signature")).toString();
        }
        if (!signature.isEmpty())
        {
            signatureByPath.insert(normalizedPath, signature);
        }
    }

    result.tracks.reserve(static_cast<std::size_t>(loaded.markerSet.markers().size()));
    for (const control_points::Marker &marker : loaded.markerSet.markers())
    {
        if (!marker.enabled)
        {
            continue;
        }

        control_points::PriorTrack track;
        track.markerId = marker.id.toStdString();
        track.confidence = 1.0;
        track.role = marker.role;
        if (marker.referenceCoordinate.has_value())
        {
            const control_points::ReferenceCoordinate &reference = *marker.referenceCoordinate;
            track.hasReference = true;
            track.referenceUsable = reference.referenceUsable;
            track.referencePoint = {{reference.x, reference.y, reference.z}};
            track.referenceSigma = {{reference.sigmaX, reference.sigmaY, reference.sigmaZ}};
        }

        track.observations.reserve(static_cast<std::size_t>(marker.projections.size()));
        for (const control_points::MarkerProjection &projection : marker.projections)
        {
            const QString imagePath =
                resolvedProjectionPath(projection, pathByUuid, projectMeta);
            const ImageId imageId = imageIdForPath(imagePath, imageIdByPath);
            if (imageId == kInvalidImageId)
            {
                continue;
            }

            control_points::PriorObservation observation;
            observation.imageId = imageId;
            observation.x = projection.xy.x();
            observation.y = projection.xy.y();
            observation.state = priorState(projection.state);
            observation.confidence =
                projection.state == control_points::ProjectionState::ManualPinned
                    ? 1.0
                    : std::clamp(projection.confidence, 0.0, 1.0);
            const QString currentSignature = signatureByPath.value(imagePath);
            observation.stale = !projection.imageContentSignature.isEmpty() &&
                !currentSignature.isEmpty() &&
                projection.imageContentSignature != currentSignature;
            track.observations.push_back(observation);
        }
        if (!track.observations.empty())
        {
            result.tracks.push_back(std::move(track));
        }
    }

    result.scaleBars.reserve(static_cast<std::size_t>(loaded.markerSet.scaleBars().size()));
    for (const control_points::ScaleBar &scaleBar : loaded.markerSet.scaleBars())
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
