#include "MarkerCsv.h"

#include <QHash>

namespace xjw::control_points
{

namespace
{

QString escapedCsv(QString value)
{
    if (value.contains(QLatin1Char(',')) || value.contains(QLatin1Char('"'))
        || value.contains(QLatin1Char('\n')))
    {
        value.replace(QStringLiteral("\""), QStringLiteral("\"\""));
        return QStringLiteral("\"") + value + QStringLiteral("\"");
    }
    return value;
}

QString csvLine(const QStringList &cells)
{
    QStringList escaped;
    escaped.reserve(cells.size());
    for (const QString &cell : cells)
    {
        escaped.push_back(escapedCsv(cell));
    }
    return escaped.join(QLatin1Char(',')) + QLatin1Char('\n');
}

} // namespace

QString markerSetToCsv(const MarkerSet &markerSet)
{
    QString output = QStringLiteral(
        "role,id,x,y,z,sigma_m,sigma_x_m,sigma_y_m,sigma_z_m,enabled,source_crs,axis_order,"
        "vertical_datum,vertical_unit,reference_usable,reference_error,image_uuid,image_path,"
        "pixel_x,pixel_y,from_id,to_id,measured_m\n");
    QHash<MarkerId, QString> labels;
    for (const Marker &marker : markerSet.markers())
    {
        labels.insert(marker.id, marker.label);
        if (!marker.referenceCoordinate)
        {
            continue;
        }
        const ReferenceCoordinate &coordinate = *marker.referenceCoordinate;
        const QString role = marker.role == MarkerRole::ControlPoint ? QStringLiteral("control")
            : marker.role == MarkerRole::CheckPoint ? QStringLiteral("check") : QStringLiteral("tie");
        const QVector<MarkerProjection> projections = marker.projections.isEmpty()
            ? QVector<MarkerProjection>{MarkerProjection{}} : marker.projections;
        for (const MarkerProjection &projection : projections)
        {
            output += csvLine({role,
                               marker.label,
                               QString::number(coordinate.x, 'g', 17),
                               QString::number(coordinate.y, 'g', 17),
                               QString::number(coordinate.z, 'g', 17),
                               QString(),
                               QString::number(coordinate.sigmaX, 'g', 17),
                               QString::number(coordinate.sigmaY, 'g', 17),
                               QString::number(coordinate.sigmaZ, 'g', 17),
                               marker.enabled ? QStringLiteral("true") : QStringLiteral("false"),
                               coordinate.sourceCrs,
                               coordinate.axisOrder,
                               coordinate.verticalDatum,
                               coordinate.verticalUnit,
                               coordinate.referenceUsable ? QStringLiteral("true") : QStringLiteral("false"),
                               coordinate.referenceError,
                               projection.imageId,
                               projection.imagePathSnapshot,
                               projection.imageId.isEmpty()
                                   ? QString() : QString::number(projection.xy.x(), 'g', 17),
                               projection.imageId.isEmpty()
                                   ? QString() : QString::number(projection.xy.y(), 'g', 17),
                               QString(),
                               QString(),
                               QString()});
        }
    }
    for (const ScaleBar &scaleBar : markerSet.scaleBars())
    {
        output += csvLine({QStringLiteral("scale_bar"),
                           scaleBar.label,
                           QString(), QString(), QString(),
                           QString::number(scaleBar.sigma, 'g', 17),
                           QString(), QString(), QString(),
                           scaleBar.enabled ? QStringLiteral("true") : QStringLiteral("false"),
                           QString(), QString(), QString(), QString(), QString(), QString(),
                           QString(), QString(), QString(), QString(),
                           labels.value(scaleBar.firstMarkerId),
                           labels.value(scaleBar.secondMarkerId),
                           QString::number(scaleBar.measuredDistance, 'g', 17)});
    }
    return output;
}

} // namespace xjw::control_points
