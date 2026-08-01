#include "SurveyControlBaAdapter.h"

#include "project/ProjectMatchInputReader.h"

#include <QJsonArray>
#include <QSet>

#include <cmath>
#include <limits>

namespace xjw::core::project
{
namespace
{

bool finiteJsonDouble(const QJsonObject &object, const QString &key, double *value)
{
    if (!value || !object.contains(key))
    {
        return false;
    }
    const double parsed = object.value(key).toDouble(std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(parsed))
    {
        return false;
    }
    *value = parsed;
    return true;
}

bool pointFromSurveyRecord(const QJsonObject &record, std::array<double, 3> *point)
{
    if (!point)
    {
        return false;
    }
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!finiteJsonDouble(record, QStringLiteral("x"), &x)
        || !finiteJsonDouble(record, QStringLiteral("y"), &y)
        || !finiteJsonDouble(record, QStringLiteral("z"), &z))
    {
        return false;
    }
    *point = {{x, y, z}};
    return true;
}

double sigmaFromSurveyRecord(const QJsonObject &record)
{
    double sigma = 0.0;
    if (finiteJsonDouble(record, QStringLiteral("sigma_m"), &sigma) && sigma > 0.0)
    {
        return sigma;
    }

    double squaredSum = 0.0;
    int count = 0;
    for (const QString &key : {QStringLiteral("sigma_x_m"),
                               QStringLiteral("sigma_y_m"),
                               QStringLiteral("sigma_z_m")})
    {
        if (finiteJsonDouble(record, key, &sigma) && sigma > 0.0)
        {
            squaredSum += sigma * sigma;
            ++count;
        }
    }
    return count > 0 ? std::sqrt(squaredSum / static_cast<double>(count)) : 1.0;
}

bool measuredDistanceFromScaleBarRecord(const QJsonObject &record, double *distanceMeters)
{
    if (!distanceMeters)
    {
        return false;
    }
    for (const QString &key : {QStringLiteral("measured_m"),
                               QStringLiteral("length_m"),
                               QStringLiteral("distance_m")})
    {
        double value = 0.0;
        if (finiteJsonDouble(record, key, &value) && value > 0.0)
        {
            *distanceMeters = value;
            return true;
        }
    }
    return false;
}

} // namespace

void appendSurveyControlBaInput(const QJsonObject &meta,
                                const QMap<QString, int> &cameraIndexByPath,
                                BaInputBuildResult *result)
{
    if (!result)
    {
        return;
    }

    const QJsonObject surveyControl = meta.value(QStringLiteral("survey_control")).toObject();
    const QJsonArray controlPoints = surveyControl.value(QStringLiteral("control_points")).toArray();
    QMap<QString, int> trackIndexByControlId;

    // 控制点先建立 trackIndex 映射，后续比例尺通过稳定 ID 引用轨迹；不能直接使用
    // JSON 数组下标，因为禁用或非法控制点会被跳过。
    for (int pointIndex = 0; pointIndex < controlPoints.size(); ++pointIndex)
    {
        const QJsonObject record = controlPoints.at(pointIndex).toObject();
        if (!record.value(QStringLiteral("enabled")).toBool(true))
        {
            ++result->rejectedSurveyControlPointCount;
            continue;
        }

        std::array<double, 3> controlPoint{};
        if (!pointFromSurveyRecord(record, &controlPoint))
        {
            ++result->rejectedSurveyControlPointCount;
            continue;
        }

        const QJsonArray observations = record.value(QStringLiteral("observations")).toArray();
        if (observations.size() < 2)
        {
            ++result->rejectedSurveyControlPointCount;
            continue;
        }

        xjw::BATrack track;
        track.initialPoint = controlPoint;
        QSet<int> usedCameras;
        for (const QJsonValue &value : observations)
        {
            const QJsonObject observation = value.toObject();
            const int cameraIndex = cameraIndexForImageToken(
                observation.value(QStringLiteral("image_path")).toString(
                    observation.value(QStringLiteral("image")).toString()),
                cameraIndexByPath);
            if (cameraIndex < 0 || usedCameras.contains(cameraIndex))
            {
                continue;
            }

            double u = 0.0;
            double v = 0.0;
            if (!finiteJsonDouble(observation, QStringLiteral("u"), &u)
                || !finiteJsonDouble(observation, QStringLiteral("v"), &v))
            {
                continue;
            }

            track.observations.push_back({cameraIndex, u, v, 1.0});
            usedCameras.insert(cameraIndex);
        }

        if (track.observations.size() < 2)
        {
            ++result->rejectedSurveyControlPointCount;
            continue;
        }

        xjw::BAControlPointConstraint constraint;
        constraint.point = controlPoint;
        constraint.sigmaMeters = sigmaFromSurveyRecord(record);
        constraint.weight = 1.0;
        constraint.sourceIndex = pointIndex;
        track.controlPointConstraints.push_back(constraint);

        const QString id = record.value(QStringLiteral("id")).toString().trimmed();
        const int trackIndex = static_cast<int>(result->tracks.size());
        if (!id.isEmpty())
        {
            trackIndexByControlId.insert(id, trackIndex);
        }

        result->surveyControlObservationCount += static_cast<int>(track.observations.size());
        ++result->surveyControlTrackCount;
        result->tracks.push_back(std::move(track));
    }

    // 只有两端都成功进入 tracks 的标尺才可加入 BA，否则约束索引会指向错误物点。
    const QJsonArray scaleBars = surveyControl.value(QStringLiteral("scale_bars")).toArray();
    for (int scaleBarIndex = 0; scaleBarIndex < scaleBars.size(); ++scaleBarIndex)
    {
        const QJsonObject record = scaleBars.at(scaleBarIndex).toObject();
        if (!record.value(QStringLiteral("enabled")).toBool(true))
        {
            ++result->rejectedSurveyScaleBarCount;
            continue;
        }

        const QString fromId = record.value(QStringLiteral("from_id")).toString(
            record.value(QStringLiteral("point_a_id")).toString()).trimmed();
        const QString toId = record.value(QStringLiteral("to_id")).toString(
            record.value(QStringLiteral("point_b_id")).toString()).trimmed();
        if (fromId.isEmpty() || toId.isEmpty()
            || !trackIndexByControlId.contains(fromId)
            || !trackIndexByControlId.contains(toId)
            || trackIndexByControlId.value(fromId) == trackIndexByControlId.value(toId))
        {
            ++result->rejectedSurveyScaleBarCount;
            continue;
        }

        double measuredDistance = 0.0;
        if (!measuredDistanceFromScaleBarRecord(record, &measuredDistance))
        {
            ++result->rejectedSurveyScaleBarCount;
            continue;
        }

        xjw::BAScaleBarConstraint constraint;
        constraint.trackIndexA = trackIndexByControlId.value(fromId);
        constraint.trackIndexB = trackIndexByControlId.value(toId);
        constraint.measuredDistanceMeters = measuredDistance;
        constraint.sigmaMeters = sigmaFromSurveyRecord(record);
        constraint.weight = 1.0;
        constraint.sourceIndex = scaleBarIndex;
        result->scaleBarConstraints.push_back(constraint);
        ++result->surveyScaleBarConstraintCount;
    }
}

} // namespace xjw::core::project
