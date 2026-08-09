#include "PlanetaryLaserJson.h"

#include "PlanetaryLaserJsonInternal.h"
#include "io/PathIO.h"

#include <QByteArray>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>
#include <QString>

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <utility>

namespace xjw
{
namespace lidar
{
namespace
{

void setError(std::string *errorMessage, const std::string &message)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
}

std::string utf8(const QString &value)
{
    return value.toUtf8().toStdString();
}

bool rejectUnknownKeys(const QJsonObject &object,
                       std::initializer_list<const char *> allowedKeys,
                       const std::string &path,
                       std::string *errorMessage)
{
    for (auto iterator = object.begin(); iterator != object.end(); ++iterator)
    {
        bool allowed = false;
        for (const char *key : allowedKeys)
        {
            if (iterator.key() == QLatin1String(key))
            {
                allowed = true;
                break;
            }
        }
        if (!allowed)
        {
            setError(errorMessage, path + " contains unsupported field '" + utf8(iterator.key()) + "'");
            return false;
        }
    }
    return true;
}

bool requireString(const QJsonObject &object,
                   const char *key,
                   const std::string &path,
                   std::string *value,
                   std::string *errorMessage)
{
    const QJsonValue json_value = object.value(QLatin1String(key));
    if (!json_value.isString() || json_value.toString().isEmpty())
    {
        setError(errorMessage, path + "." + key + " must be a non-empty string");
        return false;
    }
    *value = utf8(json_value.toString());
    return true;
}

bool requireNumber(const QJsonObject &object,
                   const char *key,
                   const std::string &path,
                   double *value,
                   std::string *errorMessage)
{
    const QJsonValue json_value = object.value(QLatin1String(key));
    if (!json_value.isDouble() || !std::isfinite(json_value.toDouble()))
    {
        setError(errorMessage, path + "." + key + " must be a finite JSON number");
        return false;
    }
    *value = json_value.toDouble();
    return true;
}

bool parseVector3(const QJsonValue &value,
                  const std::string &path,
                  std::array<double, 3> *output,
                  std::string *errorMessage)
{
    if (!value.isArray() || value.toArray().size() != 3)
    {
        setError(errorMessage, path + " must be an array of exactly 3 numbers");
        return false;
    }
    const QJsonArray array = value.toArray();
    for (int index = 0; index < 3; ++index)
    {
        if (!array[index].isDouble() || !std::isfinite(array[index].toDouble()))
        {
            setError(errorMessage, path + " contains a non-finite or non-numeric value");
            return false;
        }
        (*output)[static_cast<std::size_t>(index)] = array[index].toDouble();
    }
    return true;
}

template <std::size_t Dimension>
bool parseSquareMatrix(const QJsonValue &value,
                       const std::string &path,
                       std::array<double, Dimension * Dimension> *output,
                       std::string *errorMessage)
{
    if (!value.isArray() || value.toArray().size() != static_cast<int>(Dimension))
    {
        setError(errorMessage, path + " must be a full " + std::to_string(Dimension)
                               + "x" + std::to_string(Dimension) + " matrix");
        return false;
    }
    const QJsonArray rows = value.toArray();
    for (std::size_t row = 0; row < Dimension; ++row)
    {
        if (!rows[static_cast<int>(row)].isArray()
            || rows[static_cast<int>(row)].toArray().size() != static_cast<int>(Dimension))
        {
            setError(errorMessage, path + " must be a full square matrix");
            return false;
        }
        const QJsonArray columns = rows[static_cast<int>(row)].toArray();
        for (std::size_t column = 0; column < Dimension; ++column)
        {
            const QJsonValue item = columns[static_cast<int>(column)];
            if (!item.isDouble() || !std::isfinite(item.toDouble()))
            {
                setError(errorMessage, path + " contains a non-finite or non-numeric value");
                return false;
            }
            (*output)[row * Dimension + column] = item.toDouble();
        }
    }
    return true;
}

bool parseStringArray(const QJsonValue &value,
                      const std::string &path,
                      std::vector<std::string> *output,
                      std::string *errorMessage)
{
    if (!value.isArray())
    {
        setError(errorMessage, path + " must be an array");
        return false;
    }
    for (const QJsonValue &item : value.toArray())
    {
        if (!item.isString() || item.toString().isEmpty())
        {
            setError(errorMessage, path + " must contain only non-empty strings");
            return false;
        }
        output->push_back(utf8(item.toString()));
    }
    return true;
}

bool parseReference(const QJsonObject &root,
                    PlanetaryLaserReferenceSystem *reference,
                    std::string *errorMessage)
{
    const QJsonValue value = root.value(QStringLiteral("reference"));
    if (!value.isObject())
    {
        setError(errorMessage, "reference must be an object");
        return false;
    }
    const QJsonObject object = value.toObject();
    if (!rejectUnknownKeys(object,
                           {"target", "body_fixed_frame", "laser_frame", "time_system",
                            "latitude_type", "longitude_direction"},
                           "reference", errorMessage))
    {
        return false;
    }
    std::string time_system;
    if (!requireString(object, "target", "reference", &reference->targetName, errorMessage)
        || !requireString(object, "body_fixed_frame", "reference", &reference->bodyFixedFrame, errorMessage)
        || !requireString(object, "laser_frame", "reference", &reference->laserFrame, errorMessage)
        || !requireString(object, "time_system", "reference", &time_system, errorMessage)
        || !requireString(object, "latitude_type", "reference", &reference->latitudeType, errorMessage)
        || !requireString(object, "longitude_direction", "reference", &reference->longitudeDirection, errorMessage))
    {
        return false;
    }
    if (time_system != "TDB_ET_SECONDS")
    {
        setError(errorMessage, "reference.time_system must be TDB_ET_SECONDS");
        return false;
    }
    reference->timeSystem = PlanetaryLaserTimeSystem::TdbEtSeconds;
    return true;
}

bool validateUnits(const QJsonObject &root, std::string *errorMessage)
{
    const QJsonValue value = root.value(QStringLiteral("units"));
    if (!value.isObject())
    {
        setError(errorMessage, "PlaScan SI JSON v1 requires an explicit units object");
        return false;
    }
    const QJsonObject units = value.toObject();
    if (!rejectUnknownKeys(units, {"length", "angle", "time", "pixel"},
                           "units", errorMessage))
    {
        return false;
    }
    const auto equals = [&units](const char *key, const char *expected)
    {
        const QJsonValue unit = units.value(QLatin1String(key));
        return unit.isString() && unit.toString() == QLatin1String(expected);
    };
    if (!equals("length", "m") || !equals("angle", "deg")
        || !equals("time", "s") || !equals("pixel", "px"))
    {
        setError(errorMessage, "PlaScan SI JSON v1 units must be length=m, angle=deg, time=s, pixel=px");
        return false;
    }
    return true;
}

bool parseDatasetSemantics(const QJsonObject &root,
                           PlanetaryLaserDataset *dataset,
                           std::string *errorMessage)
{
    std::string sensor_model;
    std::string range_type;
    if (!requireString(root, "sensor_model", "dataset", &sensor_model, errorMessage)
        || !requireString(root, "range_type", "dataset", &range_type, errorMessage))
    {
        return false;
    }

    if (sensor_model == "frame")
    {
        dataset->sensorModel = PlanetaryLaserSensorModel::Frame;
    }
    else if (sensor_model == "line_scan")
    {
        dataset->sensorModel = PlanetaryLaserSensorModel::LineScan;
    }
    else if (sensor_model != "unknown")
    {
        setError(errorMessage, "sensor_model must be frame, line_scan, or unknown");
        return false;
    }

    if (range_type == "one_way")
    {
        dataset->rangeType = PlanetaryLaserRangeType::OneWay;
    }
    else if (range_type == "round_trip")
    {
        dataset->rangeType = PlanetaryLaserRangeType::RoundTrip;
    }
    else if (range_type != "unknown")
    {
        setError(errorMessage, "range_type must be one_way, round_trip, or unknown");
        return false;
    }
    return true;
}

bool parseMeasure(const QJsonObject &object,
                  const std::string &path,
                  PlanetaryLaserImageMeasure *measure,
                  std::string *errorMessage)
{
    if (!rejectUnknownKeys(object,
                           {"image_id", "sample_px", "line_px", "kind", "covariance_px2"},
                           path, errorMessage))
    {
        return false;
    }
    std::string kind;
    if (!requireString(object, "image_id", path, &measure->imageId, errorMessage)
        || !requireNumber(object, "sample_px", path, &measure->samplePixels, errorMessage)
        || !requireNumber(object, "line_px", path, &measure->linePixels, errorMessage)
        || !requireString(object, "kind", path, &kind, errorMessage))
    {
        return false;
    }
    if (kind == "measured")
    {
        measure->kind = PlanetaryLaserImageMeasureKind::Measured;
    }
    else if (kind == "projected")
    {
        measure->kind = PlanetaryLaserImageMeasureKind::ProjectedVirtual;
    }
    else
    {
        setError(errorMessage, path + ".kind must be measured or projected");
        return false;
    }
    if (object.contains(QStringLiteral("covariance_px2")))
    {
        std::array<double, 4> covariance{};
        if (!parseSquareMatrix<2>(object.value(QStringLiteral("covariance_px2")),
                                  path + ".covariance_px2", &covariance, errorMessage))
        {
            return false;
        }
        measure->covariancePixelsSquared = covariance;
    }
    return true;
}

bool parsePlaScanShot(const QJsonObject &object,
                      std::size_t index,
                      PlanetaryLaserShot *shot,
                      std::string *errorMessage)
{
    const std::string path = "shots[" + std::to_string(index) + "]";
    if (!rejectUnknownKeys(
            object,
            {"id", "point_mode", "ephemeris_time_s", "range_m", "range_sigma_m",
             "point_body_fixed_m", "point_planetocentric", "point_covariance_body_fixed_m2",
             "simultaneous_image_ids", "image_measures", "lever_arm_sensor_m"},
            path, errorMessage))
    {
        return false;
    }
    std::string point_mode;
    if (!requireString(object, "id", path, &shot->id, errorMessage)
        || !requireString(object, "point_mode", path, &point_mode, errorMessage)
        || !requireNumber(object, "ephemeris_time_s", path, &shot->ephemerisTimeSeconds, errorMessage)
        || !requireNumber(object, "range_m", path, &shot->observedRangeMeters, errorMessage)
        || !requireNumber(object, "range_sigma_m", path, &shot->rangeSigmaMeters, errorMessage)
        || !parseVector3(object.value(QStringLiteral("lever_arm_sensor_m")),
                         path + ".lever_arm_sensor_m", &shot->leverArmSensorMeters, errorMessage))
    {
        return false;
    }
    if (point_mode == "fixed")
    {
        shot->pointMode = PlanetaryLaserPointMode::Fixed;
    }
    else if (point_mode == "constrained")
    {
        shot->pointMode = PlanetaryLaserPointMode::Constrained;
    }
    else if (point_mode == "free")
    {
        shot->pointMode = PlanetaryLaserPointMode::Free;
    }
    else
    {
        setError(errorMessage, path + ".point_mode must be fixed, constrained, or free");
        return false;
    }
    shot->leverArmSpecified = true;

    const bool has_xyz = object.contains(QStringLiteral("point_body_fixed_m"));
    const bool has_spherical = object.contains(QStringLiteral("point_planetocentric"));
    if (has_xyz == has_spherical)
    {
        setError(errorMessage, path + " must contain exactly one point representation");
        return false;
    }
    if (has_xyz)
    {
        if (!parseVector3(object.value(QStringLiteral("point_body_fixed_m")),
                          path + ".point_body_fixed_m", &shot->pointBodyFixedMeters, errorMessage))
        {
            return false;
        }
    }
    else
    {
        const QJsonValue point_value = object.value(QStringLiteral("point_planetocentric"));
        if (!point_value.isObject())
        {
            setError(errorMessage, path + ".point_planetocentric must be an object");
            return false;
        }
        const QJsonObject point = point_value.toObject();
        if (!rejectUnknownKeys(point, {"latitude_deg", "longitude_deg", "radius_m"},
                               path + ".point_planetocentric", errorMessage))
        {
            return false;
        }
        double latitude = 0.0;
        double longitude = 0.0;
        double radius = 0.0;
        if (!requireNumber(point, "latitude_deg", path + ".point_planetocentric", &latitude, errorMessage)
            || !requireNumber(point, "longitude_deg", path + ".point_planetocentric", &longitude, errorMessage)
            || !requireNumber(point, "radius_m", path + ".point_planetocentric", &radius, errorMessage))
        {
            return false;
        }
        if (latitude < -90.0 || latitude > 90.0 || radius <= 0.0)
        {
            setError(errorMessage, path + ".point_planetocentric has invalid latitude or radius");
            return false;
        }
        shot->pointBodyFixedMeters = planetocentricToBodyFixedMeters(latitude, longitude, radius);
    }

    if (object.contains(QStringLiteral("point_covariance_body_fixed_m2")))
    {
        std::array<double, 9> covariance{};
        if (!parseSquareMatrix<3>(object.value(QStringLiteral("point_covariance_body_fixed_m2")),
                                  path + ".point_covariance_body_fixed_m2", &covariance, errorMessage))
        {
            return false;
        }
        shot->pointCovarianceBodyFixedMetersSquared = covariance;
    }
    if (!parseStringArray(object.value(QStringLiteral("simultaneous_image_ids")),
                          path + ".simultaneous_image_ids", &shot->simultaneousImageIds, errorMessage))
    {
        return false;
    }
    if (object.contains(QStringLiteral("image_measures")))
    {
        const QJsonValue measures_value = object.value(QStringLiteral("image_measures"));
        if (!measures_value.isArray())
        {
            setError(errorMessage, path + ".image_measures must be an array");
            return false;
        }
        const QJsonArray measures = measures_value.toArray();
        for (int measure_index = 0; measure_index < measures.size(); ++measure_index)
        {
            if (!measures[measure_index].isObject())
            {
                setError(errorMessage, path + ".image_measures entries must be objects");
                return false;
            }
            PlanetaryLaserImageMeasure measure;
            if (!parseMeasure(measures[measure_index].toObject(),
                              path + ".image_measures[" + std::to_string(measure_index) + "]",
                              &measure, errorMessage))
            {
                return false;
            }
            shot->imageMeasures.push_back(std::move(measure));
        }
    }
    return true;
}

bool parsePlaScanJson(const QJsonObject &root,
                      PlanetaryLaserDataset *dataset,
                      std::string *errorMessage)
{
    if (!rejectUnknownKeys(root,
                           {"schema", "version", "sensor_model", "range_type",
                            "units", "reference", "shots"},
                           "dataset", errorMessage))
    {
        return false;
    }
    const QJsonValue version = root.value(QStringLiteral("version"));
    if (!version.isDouble() || version.toInt(-1) != 1 || version.toDouble() != 1.0)
    {
        setError(errorMessage, "PlaScan planetary laser JSON version must be integer 1");
        return false;
    }
    if (!validateUnits(root, errorMessage)
        || !parseDatasetSemantics(root, dataset, errorMessage)
        || !parseReference(root, &dataset->reference, errorMessage))
    {
        return false;
    }
    const QJsonValue shots_value = root.value(QStringLiteral("shots"));
    if (!shots_value.isArray())
    {
        setError(errorMessage, "shots must be an array");
        return false;
    }
    dataset->sourceFormat = PlanetaryLaserSourceFormat::PlaScanSiJsonV1;
    const QJsonArray shots = shots_value.toArray();
    for (int index = 0; index < shots.size(); ++index)
    {
        if (!shots[index].isObject())
        {
            setError(errorMessage, "shots entries must be objects");
            return false;
        }
        PlanetaryLaserShot shot;
        if (!parsePlaScanShot(shots[index].toObject(), static_cast<std::size_t>(index), &shot, errorMessage))
        {
            return false;
        }
        dataset->shots.push_back(std::move(shot));
    }
    return dataset->validate(errorMessage);
}

} // namespace

bool parsePlanetaryLaserJson(const std::string &json,
                             const PlanetaryLaserJsonParseOptions &options,
                             PlanetaryLaserDataset *dataset,
                             std::string *errorMessage)
{
    if (!dataset)
    {
        setError(errorMessage, "Planetary laser dataset output is null");
        return false;
    }
    if (json.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        setError(errorMessage, "Planetary laser JSON is too large for the JSON parser");
        return false;
    }

    QJsonParseError parse_error;
    const QJsonDocument document = QJsonDocument::fromJson(
        QByteArray(json.data(), static_cast<int>(json.size())), &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !document.isObject())
    {
        setError(errorMessage, "Invalid planetary laser JSON at byte "
                                   + std::to_string(parse_error.offset) + ": "
                                   + utf8(parse_error.errorString()));
        return false;
    }

    const QJsonObject root = document.object();
    PlanetaryLaserDataset parsed;
    const QJsonValue schema = root.value(QStringLiteral("schema"));
    bool success = false;
    if (root.contains(QStringLiteral("schema")))
    {
        if (!schema.isString())
        {
            setError(errorMessage, "Planetary laser JSON schema must be a string");
            return false;
        }
        if (schema.toString() != QStringLiteral("plascan.planetary_laser_dataset"))
        {
            setError(errorMessage, "Unsupported planetary laser JSON schema: " + utf8(schema.toString()));
            return false;
        }
        success = parsePlaScanJson(root, &parsed, errorMessage);
    }
    else if (root.value(QStringLiteral("points")).isArray())
    {
        if (!options.isisContext)
        {
            setError(errorMessage,
                     "ISIS LidarData JSON requires explicit target/frame/time-system import context");
            return false;
        }
        success = parseIsisLidarDataJson(root, *options.isisContext, &parsed, errorMessage);
    }
    else
    {
        setError(errorMessage, "JSON is neither PlaScan SI v1 nor ISIS LidarData");
        return false;
    }

    if (success)
    {
        *dataset = std::move(parsed);
    }
    return success;
}

bool loadPlanetaryLaserJsonFile(const std::string &path,
                                const PlanetaryLaserJsonParseOptions &options,
                                PlanetaryLaserDataset *dataset,
                                std::string *errorMessage)
{
    std::ifstream input = xjw::common::io::openInputFile(path);
    if (!input)
    {
        setError(errorMessage, "Cannot open planetary laser JSON: " + path);
        return false;
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof())
    {
        setError(errorMessage, "Failed while reading planetary laser JSON: " + path);
        return false;
    }
    return parsePlanetaryLaserJson(contents.str(), options, dataset, errorMessage);
}

} // namespace lidar
} // namespace xjw
