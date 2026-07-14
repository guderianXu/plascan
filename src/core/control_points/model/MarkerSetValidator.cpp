#include "MarkerSetValidator.h"

#include <QSet>

#include <cmath>

namespace xjw::control_points
{

QVector<MarkerValidationIssue> MarkerSetValidator::validate(const MarkerSet &set)
{
    QVector<MarkerValidationIssue> issues;
    QSet<QString> marker_ids;
    QSet<QString> labels;
    QSet<QString> target_identities;

    for (const Marker &marker : set.markers())
    {
        if (marker.id.trimmed().isEmpty() || marker_ids.contains(marker.id))
        {
            issues.push_back({QStringLiteral("invalid_marker_id"),
                              QStringLiteral("标记 UUID 为空或重复"), marker.id, {}});
        }
        marker_ids.insert(marker.id);

        const QString label = marker.label.trimmed();
        if (label.isEmpty() || labels.contains(label))
        {
            issues.push_back({QStringLiteral("invalid_marker_label"),
                              QStringLiteral("标记名称为空或重复"), marker.id, {}});
        }
        labels.insert(label);

        if (marker.targetIdentity.has_value())
        {
            const TargetIdentity &identity = *marker.targetIdentity;
            if (identity.family.trimmed().isEmpty() || identity.encodedId < 0
                || !std::isfinite(identity.rotationDegrees))
            {
                issues.push_back({QStringLiteral("invalid_target_identity"),
                                  QStringLiteral("编码标靶身份无效"), marker.id, {}});
            }
            else
            {
                const QString identity_key = QStringLiteral("%1\x1f%2")
                                                 .arg(identity.family.trimmed().toLower())
                                                 .arg(identity.encodedId);
                if (target_identities.contains(identity_key))
                {
                    issues.push_back({QStringLiteral("duplicate_target_identity"),
                                      QStringLiteral("编码标靶 family/ID 重复"), marker.id, {}});
                }
                target_identities.insert(identity_key);
            }
        }

        QSet<QString> image_ids;
        for (const MarkerProjection &projection : marker.projections)
        {
            if (projection.imageId.trimmed().isEmpty() || image_ids.contains(projection.imageId))
            {
                issues.push_back({QStringLiteral("invalid_projection_image"),
                                  QStringLiteral("投影影像 UUID 为空或重复"),
                                  marker.id, projection.imageId});
            }
            image_ids.insert(projection.imageId);

            if (!std::isfinite(projection.xy.x()) || !std::isfinite(projection.xy.y()))
            {
                issues.push_back({QStringLiteral("invalid_projection_coordinate"),
                                  QStringLiteral("投影像素坐标不是有限值"),
                                  marker.id, projection.imageId});
            }
            if (!std::isfinite(projection.sigmaPx) || projection.sigmaPx <= 0.0)
            {
                issues.push_back({QStringLiteral("invalid_projection_sigma"),
                                  QStringLiteral("投影精度不是正有限值"),
                                  marker.id, projection.imageId});
            }
        }
    }

    QSet<QString> scale_bar_ids;
    QSet<QString> scale_bar_labels;
    for (const ScaleBar &scaleBar : set.scaleBars())
    {
        if (scaleBar.id.trimmed().isEmpty() || scale_bar_ids.contains(scaleBar.id))
        {
            issues.push_back({QStringLiteral("invalid_scale_bar_id"),
                              QStringLiteral("比例尺 UUID 为空或重复"), {}, {}});
        }
        scale_bar_ids.insert(scaleBar.id);
        if (scaleBar.label.trimmed().isEmpty() || scale_bar_labels.contains(scaleBar.label))
        {
            issues.push_back({QStringLiteral("invalid_scale_bar_label"),
                              QStringLiteral("比例尺名称为空或重复"), {}, {}});
        }
        scale_bar_labels.insert(scaleBar.label);
        if (!marker_ids.contains(scaleBar.firstMarkerId) || !marker_ids.contains(scaleBar.secondMarkerId)
            || scaleBar.firstMarkerId == scaleBar.secondMarkerId)
        {
            issues.push_back({QStringLiteral("invalid_scale_bar_endpoints"),
                              QStringLiteral("比例尺端点无效"), {}, {}});
        }
        if (!std::isfinite(scaleBar.measuredDistance) || scaleBar.measuredDistance <= 0.0
            || !std::isfinite(scaleBar.sigma) || scaleBar.sigma <= 0.0)
        {
            issues.push_back({QStringLiteral("invalid_scale_bar_measurement"),
                              QStringLiteral("比例尺长度或精度无效"), {}, {}});
        }
    }

    return issues;
}

} // namespace xjw::control_points
