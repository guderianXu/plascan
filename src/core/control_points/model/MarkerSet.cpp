#include "MarkerSet.h"

#include "reference/CoordinateReference.h"

#include <QUuid>

#include <algorithm>
#include <cmath>

namespace xjw::control_points
{

namespace
{

bool equalDouble(double left, double right)
{
    return (std::isnan(left) && std::isnan(right)) || left == right;
}

auto findMarker(QVector<Marker> &markers, const MarkerId &id)
{
    return std::find_if(markers.begin(), markers.end(), [&id](const Marker &marker)
    {
        return marker.id == id;
    });
}

auto findMarker(const QVector<Marker> &markers, const MarkerId &id)
{
    return std::find_if(markers.cbegin(), markers.cend(), [&id](const Marker &marker)
    {
        return marker.id == id;
    });
}

void validateProjection(const MarkerProjection &projection)
{
    if (projection.imageId.trimmed().isEmpty())
    {
        throw MarkerModelError(QStringLiteral("标记投影缺少影像 UUID"));
    }
    if (!std::isfinite(projection.xy.x()) || !std::isfinite(projection.xy.y()))
    {
        throw MarkerModelError(QStringLiteral("标记投影包含非有限像素坐标"));
    }
    if (!std::isfinite(projection.sigmaPx) || projection.sigmaPx <= 0.0)
    {
        throw MarkerModelError(QStringLiteral("标记投影精度必须为正有限值"));
    }
}

} // namespace

bool operator==(const MarkerProjection &left, const MarkerProjection &right)
{
    return left.imageId == right.imageId
        && left.imagePathSnapshot == right.imagePathSnapshot
        && left.xy == right.xy
        && left.state == right.state
        && equalDouble(left.sigmaPx, right.sigmaPx)
        && equalDouble(left.confidence, right.confidence)
        && equalDouble(left.residualPx, right.residualPx)
        && left.source == right.source
        && left.imageContentSignature == right.imageContentSignature;
}

bool operator==(const ReferenceCoordinate &left, const ReferenceCoordinate &right)
{
    return equalDouble(left.x, right.x)
        && equalDouble(left.y, right.y)
        && equalDouble(left.z, right.z)
        && equalDouble(left.sigmaX, right.sigmaX)
        && equalDouble(left.sigmaY, right.sigmaY)
        && equalDouble(left.sigmaZ, right.sigmaZ)
        && left.sourceCrs == right.sourceCrs
        && left.axisOrder == right.axisOrder
        && left.verticalDatum == right.verticalDatum
        && left.verticalUnit == right.verticalUnit;
}

bool operator==(const TargetIdentity &left, const TargetIdentity &right)
{
    return left.family == right.family
        && left.encodedId == right.encodedId
        && equalDouble(left.rotationDegrees, right.rotationDegrees)
        && left.generationSource == right.generationSource;
}

bool operator==(const Marker &left, const Marker &right)
{
    return left.id == right.id
        && left.label == right.label
        && left.role == right.role
        && left.enabled == right.enabled
        && left.referenceCoordinate == right.referenceCoordinate
        && left.targetIdentity == right.targetIdentity
        && left.projections == right.projections;
}

bool operator==(const ScaleBar &left, const ScaleBar &right)
{
    return left.id == right.id
        && left.label == right.label
        && left.firstMarkerId == right.firstMarkerId
        && left.secondMarkerId == right.secondMarkerId
        && left.role == right.role
        && left.enabled == right.enabled
        && equalDouble(left.measuredDistance, right.measuredDistance)
        && equalDouble(left.sigma, right.sigma)
        && equalDouble(left.estimatedDistance, right.estimatedDistance)
        && equalDouble(left.residual, right.residual);
}

MarkerSet::MarkerSet()
    : _createdAt(QDateTime::currentDateTimeUtc())
    , _updatedAt(_createdAt)
{
}

const MarkerProjection &Marker::projection(const QString &imageId) const
{
    const auto it = std::find_if(projections.cbegin(), projections.cend(), [&imageId](const MarkerProjection &item)
    {
        return item.imageId == imageId;
    });
    if (it == projections.cend())
    {
        throw MarkerModelError(QStringLiteral("未找到影像上的标记投影: %1").arg(imageId));
    }
    return *it;
}

MarkerProjection &Marker::projection(const QString &imageId)
{
    const auto it = std::find_if(projections.begin(), projections.end(), [&imageId](const MarkerProjection &item)
    {
        return item.imageId == imageId;
    });
    if (it == projections.end())
    {
        throw MarkerModelError(QStringLiteral("未找到影像上的标记投影: %1").arg(imageId));
    }
    return *it;
}

MarkerId MarkerSet::addMarker(const QString &label, MarkerRole role)
{
    const QString normalized_label = label.trimmed();
    validateUniqueLabel(normalized_label);

    Marker marker;
    marker.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
    marker.label = normalized_label;
    marker.role = role;
    _markers.push_back(marker);
    _updatedAt = QDateTime::currentDateTimeUtc();
    return marker.id;
}

void MarkerSet::removeMarker(const MarkerId &id)
{
    const auto it = findMarker(_markers, id);
    if (it == _markers.end())
    {
        throw MarkerModelError(QStringLiteral("未找到标记 UUID: %1").arg(id));
    }
    _markers.erase(it);
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void MarkerSet::renameMarker(const MarkerId &id, const QString &label)
{
    Marker &target = mutableMarker(id);
    const QString normalized_label = label.trimmed();
    validateUniqueLabel(normalized_label, id);
    target.label = normalized_label;
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void MarkerSet::setMarkerEnabled(const MarkerId &id, bool enabled)
{
    mutableMarker(id).enabled = enabled;
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void MarkerSet::setMarkerRole(const MarkerId &id, MarkerRole role)
{
    mutableMarker(id).role = role;
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void MarkerSet::setReferenceCoordinate(const MarkerId &id, const ReferenceCoordinate &coordinate)
{
    if (!std::isfinite(coordinate.x) || !std::isfinite(coordinate.y) || !std::isfinite(coordinate.z)
        || !std::isfinite(coordinate.sigmaX) || coordinate.sigmaX <= 0.0
        || !std::isfinite(coordinate.sigmaY) || coordinate.sigmaY <= 0.0
        || !std::isfinite(coordinate.sigmaZ) || coordinate.sigmaZ <= 0.0)
    {
        throw MarkerModelError(QStringLiteral("参考坐标及其精度必须为有限值，精度必须大于零"));
    }
    ReferenceCoordinate assessed = coordinate;
    AxisOrder axis_order = AxisOrder::TraditionalGis;
    if (axisOrderFromName(assessed.axisOrder, &axis_order))
    {
        assessed.axisOrder = axisOrderName(axis_order);
    }
    const ReferenceCoordinateAssessment assessment = assessReferenceCoordinate(assessed);
    assessed.referenceUsable = assessment.usable;
    assessed.referenceError = assessment.error;
    mutableMarker(id).referenceCoordinate = assessed;
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void MarkerSet::clearReferenceCoordinate(const MarkerId &id)
{
    mutableMarker(id).referenceCoordinate.reset();
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void MarkerSet::setTargetIdentity(const MarkerId &id, const TargetIdentity &identity)
{
    if (identity.family.trimmed().isEmpty() || identity.encodedId < 0
        || !std::isfinite(identity.rotationDegrees))
    {
        throw MarkerModelError(QStringLiteral("编码标靶身份必须包含 family、非负 ID 和有限旋转角"));
    }
    mutableMarker(id).targetIdentity = identity;
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void MarkerSet::clearTargetIdentity(const MarkerId &id)
{
    mutableMarker(id).targetIdentity.reset();
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void MarkerSet::upsertProjection(const MarkerId &id, const MarkerProjection &projection)
{
    validateProjection(projection);
    Marker &target = mutableMarker(id);
    const auto it = std::find_if(target.projections.begin(), target.projections.end(),
                                 [&projection](const MarkerProjection &item)
    {
        return item.imageId == projection.imageId;
    });
    if (it == target.projections.end())
    {
        target.projections.push_back(projection);
        _updatedAt = QDateTime::currentDateTimeUtc();
        return;
    }
    *it = projection;
    _updatedAt = QDateTime::currentDateTimeUtc();
}

void MarkerSet::removeProjection(const MarkerId &id, const QString &imageId)
{
    Marker &target = mutableMarker(id);
    const auto it = std::find_if(target.projections.begin(), target.projections.end(),
                                 [&imageId](const MarkerProjection &item)
    {
        return item.imageId == imageId;
    });
    if (it == target.projections.end())
    {
        throw MarkerModelError(QStringLiteral("未找到要删除的标记投影: %1").arg(imageId));
    }
    target.projections.erase(it);
    _updatedAt = QDateTime::currentDateTimeUtc();
}

ScaleBarId MarkerSet::addScaleBar(const QString &label,
                                  const MarkerId &firstMarkerId,
                                  const MarkerId &secondMarkerId,
                                  double measuredDistance,
                                  double sigma,
                                  ScaleBarRole role)
{
    const QString normalized_label = label.trimmed();
    if (normalized_label.isEmpty())
    {
        throw MarkerModelError(QStringLiteral("比例尺名称不能为空"));
    }
    if (firstMarkerId == secondMarkerId)
    {
        throw MarkerModelError(QStringLiteral("比例尺的两个端点不能相同"));
    }
    marker(firstMarkerId);
    marker(secondMarkerId);
    if (!std::isfinite(measuredDistance) || measuredDistance <= 0.0
        || !std::isfinite(sigma) || sigma <= 0.0)
    {
        throw MarkerModelError(QStringLiteral("比例尺长度和精度必须为正有限值"));
    }
    const auto duplicate = std::find_if(_scaleBars.cbegin(), _scaleBars.cend(),
                                        [&normalized_label](const ScaleBar &scaleBar)
    {
        return scaleBar.label == normalized_label;
    });
    if (duplicate != _scaleBars.cend())
    {
        throw MarkerModelError(QStringLiteral("比例尺名称已存在: %1").arg(normalized_label));
    }

    ScaleBar scale_bar;
    scale_bar.id = QUuid::createUuid().toString(QUuid::WithoutBraces);
    scale_bar.label = normalized_label;
    scale_bar.firstMarkerId = firstMarkerId;
    scale_bar.secondMarkerId = secondMarkerId;
    scale_bar.measuredDistance = measuredDistance;
    scale_bar.sigma = sigma;
    scale_bar.role = role;
    _scaleBars.push_back(scale_bar);
    _updatedAt = QDateTime::currentDateTimeUtc();
    return scale_bar.id;
}

const Marker &MarkerSet::marker(const MarkerId &id) const
{
    const auto it = findMarker(_markers, id);
    if (it == _markers.cend())
    {
        throw MarkerModelError(QStringLiteral("未找到标记 UUID: %1").arg(id));
    }
    return *it;
}

Marker &MarkerSet::mutableMarker(const MarkerId &id)
{
    const auto it = findMarker(_markers, id);
    if (it == _markers.end())
    {
        throw MarkerModelError(QStringLiteral("未找到标记 UUID: %1").arg(id));
    }
    return *it;
}

const QVector<Marker> &MarkerSet::markers() const noexcept
{
    return _markers;
}

const QVector<ScaleBar> &MarkerSet::scaleBars() const noexcept
{
    return _scaleBars;
}

int MarkerSet::schemaVersion() const noexcept
{
    return _schemaVersion;
}

QString MarkerSet::projectImageRevision() const
{
    return _projectImageRevision;
}

QDateTime MarkerSet::createdAt() const
{
    return _createdAt;
}

QDateTime MarkerSet::updatedAt() const
{
    return _updatedAt;
}

bool MarkerSet::operator==(const MarkerSet &other) const
{
    return _schemaVersion == other._schemaVersion
        && _projectImageRevision == other._projectImageRevision
        && _markers == other._markers
        && _scaleBars == other._scaleBars
        && _createdAt == other._createdAt
        && _updatedAt == other._updatedAt;
}

void MarkerSet::validateUniqueLabel(const QString &label, const MarkerId &ignoredId) const
{
    if (label.isEmpty())
    {
        throw MarkerModelError(QStringLiteral("标记名称不能为空"));
    }
    const auto duplicate = std::find_if(_markers.cbegin(), _markers.cend(),
                                        [&label, &ignoredId](const Marker &marker)
    {
        return marker.id != ignoredId && marker.label == label;
    });
    if (duplicate != _markers.cend())
    {
        throw MarkerModelError(QStringLiteral("标记名称已存在: %1").arg(label));
    }
}

} // namespace xjw::control_points
