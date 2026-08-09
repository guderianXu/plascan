#include "PlanetaryLineScanBaCliOutput.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QTextStream>

#include <array>
#include <cmath>

namespace xjw
{
namespace cli
{
namespace
{

void setError(QString *errorMessage, const QString &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

QJsonArray vector3(const std::array<double, 3> &value)
{
    return QJsonArray{value[0], value[1], value[2]};
}

std::array<double, 3> localOrigin(const lidar::PlanetaryLineScanBaResult &result)
{
    std::array<double, 3> origin{};
    if (result.points.empty())
    {
        return origin;
    }
    for (const auto &point : result.points)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            // Both A/B solves start from the same triangulated network. Using
            // that stable reference makes their local PLY files directly overlayable.
            origin[axis] += point.initialBodyFixedMeters[axis];
        }
    }
    for (double &coordinate : origin)
    {
        coordinate /= result.points.size();
    }
    return origin;
}

bool writeJson(const QString &path, const QJsonObject &object, QString *errorMessage)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly))
    {
        setError(errorMessage, QStringLiteral("无法写入 JSON: %1").arg(path));
        return false;
    }
    if (file.write(QJsonDocument(object).toJson(QJsonDocument::Indented)) < 0 || !file.commit())
    {
        setError(errorMessage, QStringLiteral("提交 JSON 失败: %1").arg(path));
        return false;
    }
    return true;
}

bool writePly(const QString &path,
              const lidar::PlanetaryLineScanBaResult &result,
              bool subtractLocalOrigin,
              QString *errorMessage)
{
    const std::array<double, 3> origin = subtractLocalOrigin
        ? localOrigin(result)
        : std::array<double, 3>{};
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        setError(errorMessage, QStringLiteral("无法写入 PLY: %1").arg(path));
        return false;
    }
    QTextStream stream(&file);
    stream.setRealNumberNotation(QTextStream::ScientificNotation);
    stream.setRealNumberPrecision(15);
    const qsizetype vertexCount = static_cast<qsizetype>(result.points.size() +
                                                         result.laserShots.size());
    stream << "ply\nformat ascii 1.0\n";
    stream << "comment coordinate_frame "
           << (subtractLocalOrigin ? "LOCAL_MOON_ME_METERS" : "MOON_ME_METERS") << "\n";
    stream << "comment local_origin_m " << origin[0] << ' ' << origin[1] << ' '
           << origin[2] << "\n";
    stream << "element vertex " << vertexCount << "\n";
    stream << "property double x\nproperty double y\nproperty double z\n";
    stream << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    stream << "end_header\n";
    for (const auto &point : result.points)
    {
        stream << point.refinedBodyFixedMeters[0] - origin[0] << ' '
               << point.refinedBodyFixedMeters[1] - origin[1] << ' '
               << point.refinedBodyFixedMeters[2] - origin[2]
               << " 210 220 255\n";
    }
    for (const auto &shot : result.laserShots)
    {
        stream << shot.refinedPointBodyFixedMeters[0] - origin[0] << ' '
               << shot.refinedPointBodyFixedMeters[1] - origin[1] << ' '
               << shot.refinedPointBodyFixedMeters[2] - origin[2]
               << " 255 64 32\n";
    }
    if (stream.status() != QTextStream::Ok || !file.commit())
    {
        setError(errorMessage, QStringLiteral("提交 PLY 失败: %1").arg(path));
        return false;
    }
    return true;
}

bool writeLaserCsv(const QString &path,
                   const lidar::PlanetaryLineScanBaResult &result,
                   QString *errorMessage)
{
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        setError(errorMessage, QStringLiteral("无法写入激光残差 CSV: %1").arg(path));
        return false;
    }
    QTextStream stream(&file);
    stream.setRealNumberPrecision(15);
    stream << "shot_id,simultaneous_image,shot_et_s,used_et_s,time_delta_s,"
              "observed_range_m,initial_computed_minus_observed_m,"
              "refined_computed_minus_observed_m\n";
    for (const auto &shot : result.laserShots)
    {
        stream << QString::fromStdString(shot.id) << ','
               << QString::fromStdString(shot.simultaneousImageId) << ','
               << shot.shotEphemerisTimeSeconds << ','
               << shot.usedEphemerisTimeSeconds << ','
               << shot.imageLineTimeMinusShotTimeSeconds << ','
               << shot.observedRangeMeters << ','
               << shot.initialComputedMinusObservedMeters << ','
               << shot.refinedComputedMinusObservedMeters << '\n';
    }
    if (stream.status() != QTextStream::Ok || !file.commit())
    {
        setError(errorMessage, QStringLiteral("提交激光残差 CSV 失败: %1").arg(path));
        return false;
    }
    return true;
}

} // namespace

QJsonObject planetaryLineScanBaResultToJson(
    const lidar::PlanetaryLineScanBaResult &result)
{
    QJsonObject object;
    object[QStringLiteral("success")] = result.success;
    object[QStringLiteral("solution_usable")] = result.solutionUsable;
    object[QStringLiteral("converged")] = result.converged;
    object[QStringLiteral("termination_type")] =
        QString::fromStdString(result.terminationType);
    object[QStringLiteral("model_level")] = QStringLiteral("P0");
    object[QStringLiteral("camera_bias_model")] =
        QStringLiteral("per_image_rigid_body_fixed_6dof");
    object[QStringLiteral("nominal_trajectory_interpolation")] =
        QStringLiteral("raw_isd_hermite_position_slerp_attitude");
    object[QStringLiteral("input_isd_interpolation_method")] = QStringLiteral("lagrange");
    object[QStringLiteral("approximate_usgscsm_interpolation")] = true;
    object[QStringLiteral("range_origin_model")] =
        QStringLiteral("image_sensor_center_zero_lever_arm");
    object[QStringLiteral("control_point_types_supported")] =
        QJsonArray{QStringLiteral("Free")};
    object[QStringLiteral("laser_constraints_enabled")] = result.laserConstraintsEnabled;
    object[QStringLiteral("message")] = QString::fromStdString(result.message);
    object[QStringLiteral("solver_brief_report")] =
        QString::fromStdString(result.solverBriefReport);
    object[QStringLiteral("iterations")] = result.iterations;
    object[QStringLiteral("control_point_count")] = result.controlPointCount;
    object[QStringLiteral("image_observation_count")] = result.imageObservationCount;
    object[QStringLiteral("active_laser_range_count")] = result.activeLaserRangeCount;
    object[QStringLiteral("initial_image_rms_px")] = result.initialImageRmsPixels;
    object[QStringLiteral("refined_image_rms_px")] = result.refinedImageRmsPixels;
    object[QStringLiteral("initial_range_rms_m")] = result.initialLaserRangeRmsMeters;
    object[QStringLiteral("refined_range_rms_m")] = result.refinedLaserRangeRmsMeters;
    object[QStringLiteral("range_residual_sign")] = QStringLiteral("computed_minus_observed");

    QJsonArray cameras;
    for (const auto &camera : result.cameras)
    {
        cameras.append(QJsonObject{
            {QStringLiteral("serial_number"), QString::fromStdString(camera.serialNumber)},
            {QStringLiteral("translation_body_fixed_m"),
             vector3(camera.translationBodyFixedMeters)},
            {QStringLiteral("angle_axis_body_fixed_rad"),
             vector3(camera.angleAxisBodyFixedRadians)},
        });
    }
    object[QStringLiteral("cameras")] = cameras;

    QJsonArray points;
    for (const auto &point : result.points)
    {
        points.append(QJsonObject{
            {QStringLiteral("id"), QString::fromStdString(point.id)},
            {QStringLiteral("initial_moon_me_m"), vector3(point.initialBodyFixedMeters)},
            {QStringLiteral("refined_moon_me_m"), vector3(point.refinedBodyFixedMeters)},
            {QStringLiteral("observation_count"), point.observationCount},
            {QStringLiteral("initial_ray_separation_m"), point.initialRaySeparationMeters},
        });
    }
    object[QStringLiteral("control_points")] = points;

    QJsonArray laserShots;
    for (const auto &shot : result.laserShots)
    {
        laserShots.append(QJsonObject{
            {QStringLiteral("id"), QString::fromStdString(shot.id)},
            {QStringLiteral("simultaneous_image"),
             QString::fromStdString(shot.simultaneousImageId)},
            {QStringLiteral("shot_et_s"), shot.shotEphemerisTimeSeconds},
            {QStringLiteral("used_et_s"), shot.usedEphemerisTimeSeconds},
            {QStringLiteral("used_minus_shot_time_s"),
             shot.imageLineTimeMinusShotTimeSeconds},
            {QStringLiteral("observed_range_m"), shot.observedRangeMeters},
            {QStringLiteral("initial_computed_minus_observed_m"),
             shot.initialComputedMinusObservedMeters},
            {QStringLiteral("refined_computed_minus_observed_m"),
             shot.refinedComputedMinusObservedMeters},
            {QStringLiteral("initial_point_moon_me_m"),
             vector3(shot.initialPointBodyFixedMeters)},
            {QStringLiteral("refined_point_moon_me_m"),
             vector3(shot.refinedPointBodyFixedMeters)},
        });
    }
    object[QStringLiteral("laser_shots")] = laserShots;
    return object;
}

bool writePlanetaryLineScanBaArtifacts(
    const QString &outputDirectory,
    const QString &prefix,
    const lidar::PlanetaryLineScanBaResult &result,
    QString *errorMessage)
{
    QDir directory(outputDirectory);
    if (!directory.mkpath(QStringLiteral(".")))
    {
        setError(errorMessage, QStringLiteral("无法创建输出目录: %1").arg(outputDirectory));
        return false;
    }
    // Publish the success/result manifest last so a later artifact failure cannot
    // leave a success=true JSON beside an incomplete output set.
    return writePly(directory.filePath(prefix + QStringLiteral("_sparse_moon_me.ply")),
                    result, false, errorMessage) &&
           writePly(directory.filePath(prefix + QStringLiteral("_sparse_local.ply")),
                    result, true, errorMessage) &&
           writeLaserCsv(directory.filePath(prefix + QStringLiteral("_laser_residuals.csv")),
                         result, errorMessage) &&
           writeJson(directory.filePath(prefix + QStringLiteral("_result.json")),
                     planetaryLineScanBaResultToJson(result), errorMessage);
}

} // namespace cli
} // namespace xjw
