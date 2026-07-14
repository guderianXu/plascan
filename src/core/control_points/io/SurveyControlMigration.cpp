#include "SurveyControlMigration.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>

#include <cmath>

namespace xjw::control_points
{

namespace
{

QString normalizePath(const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return {};
    }
    return QDir::fromNativeSeparators(QDir::cleanPath(trimmed));
}

QString imageIdForPath(const QString &path, const QHash<QString, QString> &identityByPath)
{
    const QString normalized = normalizePath(path);
    for (auto it = identityByPath.cbegin(); it != identityByPath.cend(); ++it)
    {
        if (normalizePath(it.key()).compare(normalized, Qt::CaseInsensitive) == 0)
        {
            return it.value();
        }
    }
    return {};
}

bool finiteNumber(const QJsonObject &object, const QString &key, double *value)
{
    const QJsonValue json_value = object.value(key);
    if (!json_value.isDouble() || !std::isfinite(json_value.toDouble()))
    {
        return false;
    }
    if (value) *value = json_value.toDouble();
    return true;
}

double positiveSigma(const QJsonObject &record, const QString &axisKey)
{
    double sigma = 0.0;
    if (finiteNumber(record, axisKey, &sigma) && sigma > 0.0)
    {
        return sigma;
    }
    if (finiteNumber(record, QStringLiteral("sigma_m"), &sigma) && sigma > 0.0)
    {
        return sigma;
    }
    return 1.0;
}

bool migratePoints(const QJsonArray &records,
                   MarkerRole role,
                   const QHash<QString, QString> &identityByPath,
                   MarkerSet *set,
                   QHash<QString, MarkerId> *markerIdByLegacyId,
                   SurveyControlMigrationResult *result)
{
    for (const QJsonValue &value : records)
    {
        const QJsonObject record = value.toObject();
        const QString legacy_id = record.value(QStringLiteral("id")).toString().trimmed();
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        if (legacy_id.isEmpty()
            || !finiteNumber(record, QStringLiteral("x"), &x)
            || !finiteNumber(record, QStringLiteral("y"), &y)
            || !finiteNumber(record, QStringLiteral("z"), &z))
        {
            result->error = QStringLiteral("旧控制点缺少有效 id/x/y/z");
            return false;
        }
        if (markerIdByLegacyId->contains(legacy_id))
        {
            result->error = QStringLiteral("旧控制点 ID 重复: %1").arg(legacy_id);
            return false;
        }

        const MarkerId marker_id = set->addMarker(legacy_id, role);
        set->setMarkerEnabled(marker_id, record.value(QStringLiteral("enabled")).toBool(true));
        ReferenceCoordinate coordinate;
        coordinate.x = x;
        coordinate.y = y;
        coordinate.z = z;
        coordinate.sigmaX = positiveSigma(record, QStringLiteral("sigma_x_m"));
        coordinate.sigmaY = positiveSigma(record, QStringLiteral("sigma_y_m"));
        coordinate.sigmaZ = positiveSigma(record, QStringLiteral("sigma_z_m"));
        coordinate.sourceCrs = record.value(QStringLiteral("source_crs")).toString();
        set->setReferenceCoordinate(marker_id, coordinate);
        markerIdByLegacyId->insert(legacy_id, marker_id);
        ++result->migratedMarkers;

        for (const QJsonValue &observationValue : record.value(QStringLiteral("observations")).toArray())
        {
            const QJsonObject observation = observationValue.toObject();
            const QString image_path = observation.value(QStringLiteral("image_path")).toString(
                observation.value(QStringLiteral("image")).toString());
            const QString image_id = imageIdForPath(image_path, identityByPath);
            double u = 0.0;
            double v = 0.0;
            if (image_id.isEmpty()
                || !finiteNumber(observation, QStringLiteral("u"), &u)
                || !finiteNumber(observation, QStringLiteral("v"), &v))
            {
                result->error = QStringLiteral("控制点 %1 的影像观测无法绑定稳定影像 UUID: %2")
                                    .arg(legacy_id, image_path);
                return false;
            }

            MarkerProjection projection;
            projection.imageId = image_id;
            projection.imagePathSnapshot = image_path;
            projection.xy = QPointF(u, v);
            projection.state = ProjectionState::ManualPinned;
            projection.source = QStringLiteral("legacy_survey_control");
            set->upsertProjection(marker_id, projection);
            ++result->migratedProjections;
        }
    }
    return true;
}

} // namespace

SurveyControlMigrationResult migrateSurveyControl(
    const QJsonObject &legacy,
    const QHash<QString, QString> &imageIdentityByPath)
{
    SurveyControlMigrationResult result;
    QHash<QString, MarkerId> marker_id_by_legacy_id;

    try
    {
        if (!migratePoints(legacy.value(QStringLiteral("control_points")).toArray(),
                           MarkerRole::ControlPoint,
                           imageIdentityByPath,
                           &result.markerSet,
                           &marker_id_by_legacy_id,
                           &result)
            || !migratePoints(legacy.value(QStringLiteral("check_points")).toArray(),
                              MarkerRole::CheckPoint,
                              imageIdentityByPath,
                              &result.markerSet,
                              &marker_id_by_legacy_id,
                              &result))
        {
            return result;
        }

        for (const QJsonValue &value : legacy.value(QStringLiteral("scale_bars")).toArray())
        {
            const QJsonObject record = value.toObject();
            const QString label = record.value(QStringLiteral("id")).toString().trimmed();
            const QString from_id = record.value(QStringLiteral("from_id")).toString().trimmed();
            const QString to_id = record.value(QStringLiteral("to_id")).toString().trimmed();
            double measured = 0.0;
            if (!marker_id_by_legacy_id.contains(from_id))
            {
                result.error = QStringLiteral("比例尺端点不存在: %1").arg(from_id);
                return result;
            }
            if (!marker_id_by_legacy_id.contains(to_id))
            {
                result.error = QStringLiteral("比例尺端点不存在: %1").arg(to_id);
                return result;
            }
            if (label.isEmpty() || !finiteNumber(record, QStringLiteral("measured_m"), &measured)
                || measured <= 0.0)
            {
                result.error = QStringLiteral("旧比例尺缺少有效 id/measured_m");
                return result;
            }

            const double sigma = positiveSigma(record, QStringLiteral("sigma_m"));
            result.markerSet.addScaleBar(label,
                                         marker_id_by_legacy_id.value(from_id),
                                         marker_id_by_legacy_id.value(to_id),
                                         measured,
                                         sigma,
                                         ScaleBarRole::Control);
            ++result.migratedScaleBars;
        }
    }
    catch (const MarkerModelError &exception)
    {
        result.error = QString::fromUtf8(exception.what());
        return result;
    }

    result.ok = true;
    return result;
}

} // namespace xjw::control_points
