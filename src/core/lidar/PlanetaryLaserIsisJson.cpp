#include "PlanetaryLaserJsonInternal.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace xjw
{
namespace lidar
{
namespace
{

constexpr double kPi = 3.141592653589793238462643383279502884;

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

std::array<double, 9> multiply3(const std::array<double, 9> &left,
                               const std::array<double, 9> &right)
{
    std::array<double, 9> output{};
    for (std::size_t row = 0; row < 3; ++row)
    {
        for (std::size_t column = 0; column < 3; ++column)
        {
            for (std::size_t inner = 0; inner < 3; ++inner)
            {
                output[row * 3 + column] += left[row * 3 + inner] * right[inner * 3 + column];
            }
        }
    }
    return output;
}

std::array<double, 9> transpose3(const std::array<double, 9> &matrix)
{
    return {{matrix[0], matrix[3], matrix[6],
             matrix[1], matrix[4], matrix[7],
             matrix[2], matrix[5], matrix[8]}};
}

bool positiveSemidefinite3(const std::array<double, 9> &matrix)
{
    const double scale = *std::max_element(
        matrix.begin(), matrix.end(), [](double first, double second)
        {
            return std::abs(first) < std::abs(second);
        });
    const double absolute_scale = std::abs(scale);
    if (absolute_scale == 0.0)
    {
        return true;
    }
    const double c00 = matrix[0] / absolute_scale;
    const double c01 = matrix[1] / absolute_scale;
    const double c02 = matrix[2] / absolute_scale;
    const double c11 = matrix[4] / absolute_scale;
    const double c12 = matrix[5] / absolute_scale;
    const double c22 = matrix[8] / absolute_scale;
    if (c00 < -1.0e-12 || c11 < -1.0e-12 || c22 < -1.0e-12)
    {
        return false;
    }
    const auto valid_minor = [](double first, double second, double cross)
    {
        const double minor = first * second - cross * cross;
        return minor >= -1.0e-12;
    };
    if (!valid_minor(c00, c11, c01)
        || !valid_minor(c00, c22, c02)
        || !valid_minor(c11, c22, c12))
    {
        return false;
    }
    const double determinant = c00 * (c11 * c22 - c12 * c12)
        - c01 * (c01 * c22 - c12 * c02)
        + c02 * (c01 * c12 - c11 * c02);
    return determinant >= -1.0e-12;
}

bool convertIsisCovariance(const QJsonValue &value,
                           double latitudeDegrees,
                           double longitudeDegrees,
                           double radiusMeters,
                           const std::string &path,
                           std::array<double, 9> *output,
                           std::string *errorMessage)
{
    if (!value.isArray() || value.toArray().size() != 6)
    {
        setError(errorMessage, path + " must have exactly 6 upper-triangular elements");
        return false;
    }
    std::array<double, 6> packed{};
    const QJsonArray array = value.toArray();
    for (int index = 0; index < 6; ++index)
    {
        if (!array[index].isDouble() || !std::isfinite(array[index].toDouble()))
        {
            setError(errorMessage, path + " contains a non-finite or non-numeric value");
            return false;
        }
        packed[static_cast<std::size_t>(index)] = array[index].toDouble();
    }

    // ISIS LidarData::write() serializes SurfacePoint::GetSphericalMatrix()
    // with its default metre coordinate units. The angular axes are radians,
    // while the radius axis is already metres; no kilometre scaling belongs
    // here even though the separate JSON radius field is stored in kilometres.
    const std::array<double, 9> spherical{{
        packed[0], packed[1], packed[2],
        packed[1], packed[3], packed[4],
        packed[2], packed[4], packed[5]}};
    if (!std::all_of(spherical.begin(), spherical.end(), [](double item) { return std::isfinite(item); })
        || !positiveSemidefinite3(spherical))
    {
        setError(errorMessage, path + " is not a finite positive-semidefinite covariance");
        return false;
    }

    const double latitude = latitudeDegrees * kPi / 180.0;
    const double longitude = longitudeDegrees * kPi / 180.0;
    const double sin_latitude = std::sin(latitude);
    const double cos_latitude = std::cos(latitude);
    const double sin_longitude = std::sin(longitude);
    const double cos_longitude = std::cos(longitude);
    const std::array<double, 9> jacobian{{
        -radiusMeters * sin_latitude * cos_longitude,
        -radiusMeters * cos_latitude * sin_longitude,
        cos_latitude * cos_longitude,
        -radiusMeters * sin_latitude * sin_longitude,
        radiusMeters * cos_latitude * cos_longitude,
        cos_latitude * sin_longitude,
        radiusMeters * cos_latitude,
        0.0,
        sin_latitude}};
    *output = multiply3(multiply3(jacobian, spherical), transpose3(jacobian));
    if (!std::all_of(output->begin(), output->end(), [](double item) { return std::isfinite(item); }))
    {
        setError(errorMessage, path + " overflows while converting to body-fixed metres squared");
        return false;
    }
    return true;
}

bool parseIsisMeasure(const QJsonObject &object,
                      const std::string &path,
                      PlanetaryLaserImageMeasure *measure,
                      std::string *errorMessage)
{
    if (!requireString(object, "serialNumber", path, &measure->imageId, errorMessage)
        || !requireNumber(object, "sample", path, &measure->samplePixels, errorMessage)
        || !requireNumber(object, "line", path, &measure->linePixels, errorMessage))
    {
        return false;
    }
    measure->kind = PlanetaryLaserImageMeasureKind::ProjectedVirtual;
    return true;
}

bool parseIsisShot(const QJsonObject &object,
                   std::size_t index,
                   const PlanetaryLaserIsisContext &context,
                   PlanetaryLaserShot *shot,
                   std::string *errorMessage)
{
    const std::string path = "points[" + std::to_string(index) + "]";
    double range_kilometers = 0.0;
    double latitude_degrees = 0.0;
    double longitude_degrees = 0.0;
    double radius_kilometers = 0.0;
    if (!requireString(object, "id", path, &shot->id, errorMessage)
        || !requireNumber(object, "time", path, &shot->ephemerisTimeSeconds, errorMessage)
        || !requireNumber(object, "range", path, &range_kilometers, errorMessage)
        || !requireNumber(object, "sigmaRange", path, &shot->rangeSigmaMeters, errorMessage)
        || !requireNumber(object, "latitude", path, &latitude_degrees, errorMessage)
        || !requireNumber(object, "longitude", path, &longitude_degrees, errorMessage)
        || !requireNumber(object, "radius", path, &radius_kilometers, errorMessage))
    {
        return false;
    }
    if (latitude_degrees < -90.0 || latitude_degrees > 90.0
        || range_kilometers <= 0.0 || radius_kilometers <= 0.0)
    {
        setError(errorMessage, path + " has invalid ISIS range, latitude, or radius");
        return false;
    }

    shot->observedRangeMeters = range_kilometers * 1000.0;
    const double radius_meters = radius_kilometers * 1000.0;
    if (!std::isfinite(shot->observedRangeMeters) || !std::isfinite(radius_meters))
    {
        setError(errorMessage, path + " overflows while converting ISIS kilometres to metres");
        return false;
    }
    shot->pointBodyFixedMeters = planetocentricToBodyFixedMeters(
        latitude_degrees, longitude_degrees, radius_meters);
    shot->leverArmSensorMeters = *context.leverArmSensorMeters;
    shot->leverArmSpecified = true;

    if (object.contains(QStringLiteral("aprioriMatrix")))
    {
        shot->pointMode = PlanetaryLaserPointMode::Constrained;
        std::array<double, 9> covariance{};
        if (!convertIsisCovariance(object.value(QStringLiteral("aprioriMatrix")),
                                   latitude_degrees, longitude_degrees, radius_meters,
                                   path + ".aprioriMatrix", &covariance, errorMessage))
        {
            return false;
        }
        shot->pointCovarianceBodyFixedMetersSquared = covariance;
    }
    else
    {
        shot->pointMode = PlanetaryLaserPointMode::Free;
    }
    if (!parseStringArray(object.value(QStringLiteral("simultaneousImages")),
                          path + ".simultaneousImages", &shot->simultaneousImageIds, errorMessage))
    {
        return false;
    }

    const QJsonValue measures_value = object.value(QStringLiteral("measures"));
    if (!measures_value.isArray())
    {
        setError(errorMessage, path + ".measures must be an array");
        return false;
    }
    const QJsonArray measures = measures_value.toArray();
    for (int measure_index = 0; measure_index < measures.size(); ++measure_index)
    {
        if (!measures[measure_index].isObject())
        {
            setError(errorMessage, path + ".measures entries must be objects");
            return false;
        }
        PlanetaryLaserImageMeasure measure;
        if (!parseIsisMeasure(measures[measure_index].toObject(),
                              path + ".measures[" + std::to_string(measure_index) + "]",
                              &measure, errorMessage))
        {
            return false;
        }
        shot->imageMeasures.push_back(std::move(measure));
    }
    return true;
}

} // namespace

bool parseIsisLidarDataJson(const QJsonObject &root,
                            const PlanetaryLaserIsisContext &context,
                            PlanetaryLaserDataset *dataset,
                            std::string *errorMessage)
{
    if (!context.leverArmSensorMeters)
    {
        setError(errorMessage,
                 "ISIS LidarData import context must explicitly specify the laser lever arm in metres");
        return false;
    }
    dataset->sourceFormat = PlanetaryLaserSourceFormat::IsisLidarDataJson;
    dataset->sensorModel = context.sensorModel;
    dataset->rangeType = context.rangeType;
    dataset->reference = context.reference;

    const QJsonValue points_value = root.value(QStringLiteral("points"));
    if (!points_value.isArray())
    {
        setError(errorMessage, "ISIS LidarData points must be an array");
        return false;
    }
    const QJsonArray points = points_value.toArray();
    for (int index = 0; index < points.size(); ++index)
    {
        if (!points[index].isObject())
        {
            setError(errorMessage, "ISIS LidarData points entries must be objects");
            return false;
        }
        PlanetaryLaserShot shot;
        if (!parseIsisShot(points[index].toObject(), static_cast<std::size_t>(index),
                           context, &shot, errorMessage))
        {
            return false;
        }
        dataset->shots.push_back(std::move(shot));
    }
    return dataset->validate(errorMessage);
}

} // namespace lidar
} // namespace xjw
