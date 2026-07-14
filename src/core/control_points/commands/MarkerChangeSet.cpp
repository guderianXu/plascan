#include "MarkerChangeSet.h"

namespace xjw::control_points
{

MarkerChangeSet MarkerChangeSet::replaceProjection(const MarkerSet &set,
                                                    const MarkerId &markerId,
                                                    const QString &imageId,
                                                    const MarkerProjection &projection)
{
    set.marker(markerId);
    if (imageId.trimmed().isEmpty())
    {
        throw MarkerModelError(QStringLiteral("替换投影时影像 UUID 不能为空"));
    }

    MarkerSet after = set;
    MarkerProjection replacement = projection;
    replacement.imageId = imageId;
    after.upsertProjection(markerId, replacement);
    return fromSnapshots(set, std::move(after), QStringLiteral("放置标记投影"), {markerId});
}

MarkerChangeSet MarkerChangeSet::replaceProjections(
    const MarkerSet &set,
    const MarkerId &markerId,
    const QVector<MarkerProjection> &projections,
    const QString &description)
{
    set.marker(markerId);
    MarkerSet after = set;
    for (const MarkerProjection &projection : projections)
    {
        after.upsertProjection(markerId, projection);
    }
    return fromSnapshots(set, std::move(after), description, {markerId});
}

MarkerChangeSet MarkerChangeSet::createMarkerWithProjection(const MarkerSet &set,
                                                            const QString &label,
                                                            MarkerRole role,
                                                            const MarkerProjection &projection)
{
    MarkerSet after = set;
    const MarkerId marker_id = after.addMarker(label, role);
    after.upsertProjection(marker_id, projection);
    return fromSnapshots(set, std::move(after), QStringLiteral("添加标记"), {marker_id});
}

MarkerChangeSet MarkerChangeSet::removeProjection(const MarkerSet &set,
                                                  const MarkerId &markerId,
                                                  const QString &imageId)
{
    set.marker(markerId).projection(imageId);
    MarkerSet after = set;
    after.removeProjection(markerId, imageId);
    return fromSnapshots(set, std::move(after), QStringLiteral("移除标记投影"), {markerId});
}

MarkerChangeSet MarkerChangeSet::setProjectionState(const MarkerSet &set,
                                                    const MarkerId &markerId,
                                                    const QString &imageId,
                                                    ProjectionState state,
                                                    const QString &description)
{
    MarkerProjection projection = set.marker(markerId).projection(imageId);
    projection.state = state;
    MarkerSet after = set;
    after.upsertProjection(markerId, projection);
    return fromSnapshots(set, std::move(after), description, {markerId});
}

MarkerChangeSet MarkerChangeSet::updateMarker(
    const MarkerSet &set,
    const MarkerId &markerId,
    const QString &label,
    MarkerRole role,
    bool enabled,
    const std::optional<ReferenceCoordinate> &referenceCoordinate)
{
    set.marker(markerId);
    MarkerSet after = set;
    after.renameMarker(markerId, label);
    after.setMarkerRole(markerId, role);
    after.setMarkerEnabled(markerId, enabled);
    if (referenceCoordinate.has_value())
    {
        after.setReferenceCoordinate(markerId, referenceCoordinate.value());
    }
    else
    {
        after.clearReferenceCoordinate(markerId);
    }
    return fromSnapshots(set, std::move(after), QStringLiteral("更新标记属性"), {markerId});
}

MarkerChangeSet MarkerChangeSet::replaceMarkerSet(const MarkerSet &before,
                                                   const MarkerSet &after,
                                                   const QString &description,
                                                   const QVector<MarkerId> &affectedMarkerIds)
{
    return fromSnapshots(before, after, description, affectedMarkerIds);
}

MarkerChangeSet MarkerChangeSet::fromSnapshots(const MarkerSet &before,
                                               MarkerSet after,
                                               const QString &description,
                                               const QVector<MarkerId> &affectedMarkerIds)
{
    MarkerChangeSet change;
    change._before = before;
    change._after = std::move(after);
    change._description = description;
    change._affectedMarkerIds = affectedMarkerIds;
    change._valid = true;
    return change;
}

void MarkerChangeSet::apply(MarkerSet *set) const
{
    if (!set || !_valid)
    {
        throw MarkerModelError(QStringLiteral("无法应用无效的标记变更"));
    }
    *set = _after;
}

void MarkerChangeSet::revert(MarkerSet *set) const
{
    if (!set || !_valid)
    {
        throw MarkerModelError(QStringLiteral("无法撤销无效的标记变更"));
    }
    *set = _before;
}

QString MarkerChangeSet::description() const
{
    return _description;
}

QVector<MarkerId> MarkerChangeSet::affectedMarkerIds() const
{
    return _affectedMarkerIds;
}

bool MarkerChangeSet::isValid() const noexcept
{
    return _valid;
}

} // namespace xjw::control_points
