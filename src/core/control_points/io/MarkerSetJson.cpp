#include "MarkerSetJson.h"

#include "model/MarkerSetValidator.h"

#include <QJsonArray>

#include <cmath>

namespace xjw::control_points
{

namespace
{

QString markerRoleName(MarkerRole role)
{
    switch (role)
    {
    case MarkerRole::TieMarker: return QStringLiteral("tie_marker");
    case MarkerRole::ControlPoint: return QStringLiteral("control_point");
    case MarkerRole::CheckPoint: return QStringLiteral("check_point");
    }
    return {};
}

bool parseMarkerRole(const QString &name, MarkerRole *role)
{
    if (name == QLatin1String("tie_marker")) *role = MarkerRole::TieMarker;
    else if (name == QLatin1String("control_point")) *role = MarkerRole::ControlPoint;
    else if (name == QLatin1String("check_point")) *role = MarkerRole::CheckPoint;
    else return false;
    return true;
}

QString projectionStateName(ProjectionState state)
{
    switch (state)
    {
    case ProjectionState::ManualPinned: return QStringLiteral("manual_pinned");
    case ProjectionState::AutoDetected: return QStringLiteral("auto_detected");
    case ProjectionState::Predicted: return QStringLiteral("predicted");
    case ProjectionState::Blocked: return QStringLiteral("blocked");
    case ProjectionState::Disabled: return QStringLiteral("disabled");
    }
    return {};
}

bool parseProjectionState(const QString &name, ProjectionState *state)
{
    if (name == QLatin1String("manual_pinned")) *state = ProjectionState::ManualPinned;
    else if (name == QLatin1String("auto_detected")) *state = ProjectionState::AutoDetected;
    else if (name == QLatin1String("predicted")) *state = ProjectionState::Predicted;
    else if (name == QLatin1String("blocked")) *state = ProjectionState::Blocked;
    else if (name == QLatin1String("disabled")) *state = ProjectionState::Disabled;
    else return false;
    return true;
}

QString scaleBarRoleName(ScaleBarRole role)
{
    return role == ScaleBarRole::Control ? QStringLiteral("control") : QStringLiteral("check");
}

bool parseScaleBarRole(const QString &name, ScaleBarRole *role)
{
    if (name == QLatin1String("control")) *role = ScaleBarRole::Control;
    else if (name == QLatin1String("check")) *role = ScaleBarRole::Check;
    else return false;
    return true;
}

QJsonValue finiteOrNull(double value)
{
    return std::isfinite(value) ? QJsonValue(value) : QJsonValue(QJsonValue::Null);
}

double numberOrNan(const QJsonValue &value)
{
    return value.isDouble() ? value.toDouble() : std::numeric_limits<double>::quiet_NaN();
}

} // namespace

QJsonObject MarkerSetJson::encode(const MarkerSet &markerSet)
{
    QJsonArray markers;
    for (const Marker &marker : markerSet._markers)
    {
        QJsonArray projections;
        for (const MarkerProjection &projection : marker.projections)
        {
            projections.append(QJsonObject{
                {QStringLiteral("image_id"), projection.imageId},
                {QStringLiteral("image_path_snapshot"), projection.imagePathSnapshot},
                {QStringLiteral("x"), projection.xy.x()},
                {QStringLiteral("y"), projection.xy.y()},
                {QStringLiteral("state"), projectionStateName(projection.state)},
                {QStringLiteral("sigma_px"), projection.sigmaPx},
                {QStringLiteral("confidence"), projection.confidence},
                {QStringLiteral("residual_px"), finiteOrNull(projection.residualPx)},
                {QStringLiteral("source"), projection.source},
                {QStringLiteral("image_content_signature"), projection.imageContentSignature}
            });
        }

        QJsonValue reference = QJsonValue(QJsonValue::Null);
        if (marker.referenceCoordinate)
        {
            const ReferenceCoordinate &coordinate = *marker.referenceCoordinate;
            reference = QJsonObject{
                {QStringLiteral("x"), coordinate.x},
                {QStringLiteral("y"), coordinate.y},
                {QStringLiteral("z"), coordinate.z},
                {QStringLiteral("sigma_x"), coordinate.sigmaX},
                {QStringLiteral("sigma_y"), coordinate.sigmaY},
                {QStringLiteral("sigma_z"), coordinate.sigmaZ},
                {QStringLiteral("source_crs"), coordinate.sourceCrs},
                {QStringLiteral("axis_order"), coordinate.axisOrder},
                {QStringLiteral("vertical_datum"), coordinate.verticalDatum},
                {QStringLiteral("vertical_unit"), coordinate.verticalUnit},
                {QStringLiteral("reference_usable"), coordinate.referenceUsable},
                {QStringLiteral("reference_error"), coordinate.referenceError}
            };
        }

        QJsonValue target_identity = QJsonValue(QJsonValue::Null);
        if (marker.targetIdentity)
        {
            const TargetIdentity &identity = *marker.targetIdentity;
            target_identity = QJsonObject{
                {QStringLiteral("family"), identity.family},
                {QStringLiteral("encoded_id"), identity.encodedId},
                {QStringLiteral("rotation_degrees"), identity.rotationDegrees},
                {QStringLiteral("generation_source"), identity.generationSource}
            };
        }

        markers.append(QJsonObject{
            {QStringLiteral("id"), marker.id},
            {QStringLiteral("label"), marker.label},
            {QStringLiteral("role"), markerRoleName(marker.role)},
            {QStringLiteral("enabled"), marker.enabled},
            {QStringLiteral("reference_coordinate"), reference},
            {QStringLiteral("target_identity"), target_identity},
            {QStringLiteral("projections"), projections}
        });
    }

    QJsonArray scale_bars;
    for (const ScaleBar &scaleBar : markerSet._scaleBars)
    {
        scale_bars.append(QJsonObject{
            {QStringLiteral("id"), scaleBar.id},
            {QStringLiteral("label"), scaleBar.label},
            {QStringLiteral("first_marker_id"), scaleBar.firstMarkerId},
            {QStringLiteral("second_marker_id"), scaleBar.secondMarkerId},
            {QStringLiteral("role"), scaleBarRoleName(scaleBar.role)},
            {QStringLiteral("enabled"), scaleBar.enabled},
            {QStringLiteral("measured_distance"), scaleBar.measuredDistance},
            {QStringLiteral("sigma"), scaleBar.sigma},
            {QStringLiteral("estimated_distance"), finiteOrNull(scaleBar.estimatedDistance)},
            {QStringLiteral("residual"), finiteOrNull(scaleBar.residual)}
        });
    }

    return QJsonObject{
        {QStringLiteral("schema_version"), markerSet._schemaVersion},
        {QStringLiteral("project_image_revision"), markerSet._projectImageRevision},
        {QStringLiteral("created_at"), markerSet._createdAt.toString(Qt::ISODateWithMs)},
        {QStringLiteral("updated_at"), markerSet._updatedAt.toString(Qt::ISODateWithMs)},
        {QStringLiteral("markers"), markers},
        {QStringLiteral("scale_bars"), scale_bars}
    };
}

bool MarkerSetJson::decode(const QJsonObject &object, MarkerSet *markerSet, QString *error)
{
    if (!markerSet)
    {
        if (error) *error = QStringLiteral("MarkerSet 输出指针为空");
        return false;
    }
    if (object.value(QStringLiteral("schema_version")).toInt(-1) != 1)
    {
        if (error) *error = QStringLiteral("不支持的标记文件 schema_version");
        return false;
    }

    MarkerSet decoded;
    decoded._schemaVersion = 1;
    decoded._projectImageRevision = object.value(QStringLiteral("project_image_revision")).toString();
    decoded._createdAt = QDateTime::fromString(object.value(QStringLiteral("created_at")).toString(),
                                               Qt::ISODateWithMs);
    decoded._updatedAt = QDateTime::fromString(object.value(QStringLiteral("updated_at")).toString(),
                                               Qt::ISODateWithMs);
    if (!decoded._createdAt.isValid() || !decoded._updatedAt.isValid())
    {
        if (error) *error = QStringLiteral("标记文件时间戳无效");
        return false;
    }

    for (const QJsonValue &markerValue : object.value(QStringLiteral("markers")).toArray())
    {
        const QJsonObject markerObject = markerValue.toObject();
        Marker marker;
        marker.id = markerObject.value(QStringLiteral("id")).toString();
        marker.label = markerObject.value(QStringLiteral("label")).toString();
        marker.enabled = markerObject.value(QStringLiteral("enabled")).toBool(true);
        if (!parseMarkerRole(markerObject.value(QStringLiteral("role")).toString(), &marker.role))
        {
            if (error) *error = QStringLiteral("标记角色无效: %1").arg(marker.label);
            return false;
        }

        const QJsonValue referenceValue = markerObject.value(QStringLiteral("reference_coordinate"));
        if (referenceValue.isObject())
        {
            const QJsonObject referenceObject = referenceValue.toObject();
            ReferenceCoordinate coordinate;
            coordinate.x = referenceObject.value(QStringLiteral("x")).toDouble();
            coordinate.y = referenceObject.value(QStringLiteral("y")).toDouble();
            coordinate.z = referenceObject.value(QStringLiteral("z")).toDouble();
            coordinate.sigmaX = referenceObject.value(QStringLiteral("sigma_x")).toDouble(1.0);
            coordinate.sigmaY = referenceObject.value(QStringLiteral("sigma_y")).toDouble(1.0);
            coordinate.sigmaZ = referenceObject.value(QStringLiteral("sigma_z")).toDouble(1.0);
            coordinate.sourceCrs = referenceObject.value(QStringLiteral("source_crs")).toString();
            coordinate.axisOrder = referenceObject.value(QStringLiteral("axis_order"))
                                       .toString(QStringLiteral("traditional_gis"));
            coordinate.verticalDatum =
                referenceObject.value(QStringLiteral("vertical_datum")).toString();
            coordinate.verticalUnit =
                referenceObject.value(QStringLiteral("vertical_unit")).toString();
            coordinate.referenceUsable =
                referenceObject.value(QStringLiteral("reference_usable")).toBool(false);
            coordinate.referenceError =
                referenceObject.value(QStringLiteral("reference_error")).toString();
            marker.referenceCoordinate = coordinate;
        }

        const QJsonValue targetIdentityValue = markerObject.value(QStringLiteral("target_identity"));
        if (targetIdentityValue.isObject())
        {
            const QJsonObject identityObject = targetIdentityValue.toObject();
            TargetIdentity identity;
            identity.family = identityObject.value(QStringLiteral("family")).toString();
            identity.encodedId = identityObject.value(QStringLiteral("encoded_id")).toInt(-1);
            identity.rotationDegrees =
                identityObject.value(QStringLiteral("rotation_degrees")).toDouble();
            identity.generationSource =
                identityObject.value(QStringLiteral("generation_source")).toString();
            marker.targetIdentity = identity;
        }

        for (const QJsonValue &projectionValue : markerObject.value(QStringLiteral("projections")).toArray())
        {
            const QJsonObject projectionObject = projectionValue.toObject();
            MarkerProjection projection;
            projection.imageId = projectionObject.value(QStringLiteral("image_id")).toString();
            projection.imagePathSnapshot = projectionObject.value(QStringLiteral("image_path_snapshot")).toString();
            projection.xy = QPointF(projectionObject.value(QStringLiteral("x")).toDouble(),
                                    projectionObject.value(QStringLiteral("y")).toDouble());
            if (!parseProjectionState(projectionObject.value(QStringLiteral("state")).toString(),
                                      &projection.state))
            {
                if (error) *error = QStringLiteral("标记投影状态无效: %1").arg(projection.imageId);
                return false;
            }
            projection.sigmaPx = projectionObject.value(QStringLiteral("sigma_px")).toDouble(1.0);
            projection.confidence = projectionObject.value(QStringLiteral("confidence")).toDouble();
            projection.residualPx = numberOrNan(projectionObject.value(QStringLiteral("residual_px")));
            projection.source = projectionObject.value(QStringLiteral("source")).toString();
            projection.imageContentSignature =
                projectionObject.value(QStringLiteral("image_content_signature")).toString();
            marker.projections.push_back(projection);
        }
        decoded._markers.push_back(marker);
    }

    for (const QJsonValue &scaleBarValue : object.value(QStringLiteral("scale_bars")).toArray())
    {
        const QJsonObject scaleBarObject = scaleBarValue.toObject();
        ScaleBar scale_bar;
        scale_bar.id = scaleBarObject.value(QStringLiteral("id")).toString();
        scale_bar.label = scaleBarObject.value(QStringLiteral("label")).toString();
        scale_bar.firstMarkerId = scaleBarObject.value(QStringLiteral("first_marker_id")).toString();
        scale_bar.secondMarkerId = scaleBarObject.value(QStringLiteral("second_marker_id")).toString();
        scale_bar.enabled = scaleBarObject.value(QStringLiteral("enabled")).toBool(true);
        scale_bar.measuredDistance = scaleBarObject.value(QStringLiteral("measured_distance")).toDouble();
        scale_bar.sigma = scaleBarObject.value(QStringLiteral("sigma")).toDouble(1.0);
        scale_bar.estimatedDistance = numberOrNan(scaleBarObject.value(QStringLiteral("estimated_distance")));
        scale_bar.residual = numberOrNan(scaleBarObject.value(QStringLiteral("residual")));
        if (!parseScaleBarRole(scaleBarObject.value(QStringLiteral("role")).toString(), &scale_bar.role))
        {
            if (error) *error = QStringLiteral("比例尺角色无效: %1").arg(scale_bar.label);
            return false;
        }
        decoded._scaleBars.push_back(scale_bar);
    }

    const auto issues = MarkerSetValidator::validate(decoded);
    if (!issues.isEmpty())
    {
        if (error) *error = issues.front().message;
        return false;
    }

    *markerSet = decoded;
    return true;
}

} // namespace xjw::control_points
