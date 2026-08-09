#include "PlanetaryLaserShot.h"

#include <algorithm>
#include <cmath>
#include <set>
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

bool finiteVector(const std::array<double, 3> &values)
{
    return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

bool validSensorModel(PlanetaryLaserSensorModel sensorModel)
{
    switch (sensorModel)
    {
        case PlanetaryLaserSensorModel::Unknown:
        case PlanetaryLaserSensorModel::Frame:
        case PlanetaryLaserSensorModel::LineScan:
            return true;
        default:
            return false;
    }
}

bool validRangeType(PlanetaryLaserRangeType rangeType)
{
    switch (rangeType)
    {
        case PlanetaryLaserRangeType::Unknown:
        case PlanetaryLaserRangeType::OneWay:
        case PlanetaryLaserRangeType::RoundTrip:
            return true;
        default:
            return false;
    }
}

bool validPointMode(PlanetaryLaserPointMode pointMode)
{
    switch (pointMode)
    {
        case PlanetaryLaserPointMode::Fixed:
        case PlanetaryLaserPointMode::Constrained:
        case PlanetaryLaserPointMode::Free:
            return true;
        default:
            return false;
    }
}

bool validMeasureKind(PlanetaryLaserImageMeasureKind kind)
{
    switch (kind)
    {
        case PlanetaryLaserImageMeasureKind::Measured:
        case PlanetaryLaserImageMeasureKind::ProjectedVirtual:
            return true;
        default:
            return false;
    }
}

bool validateCovariance3(const std::array<double, 9> &matrix, std::string *reason)
{
    for (double value : matrix)
    {
        if (!std::isfinite(value))
        {
            setError(reason, "contains a non-finite value");
            return false;
        }
    }

    double scale = 0.0;
    for (double value : matrix)
    {
        scale = std::max(scale, std::abs(value));
    }
    const double symmetry_tolerance = std::max(1.0e-30, 1.0e-12 * scale);
    if (std::abs(matrix[1] - matrix[3]) > symmetry_tolerance
        || std::abs(matrix[2] - matrix[6]) > symmetry_tolerance
        || std::abs(matrix[5] - matrix[7]) > symmetry_tolerance)
    {
        setError(reason, "must be symmetric");
        return false;
    }
    if (scale == 0.0)
    {
        return true;
    }

    const double c00 = matrix[0] / scale;
    const double c01 = 0.5 * (matrix[1] + matrix[3]) / scale;
    const double c02 = 0.5 * (matrix[2] + matrix[6]) / scale;
    const double c11 = matrix[4] / scale;
    const double c12 = 0.5 * (matrix[5] + matrix[7]) / scale;
    const double c22 = matrix[8] / scale;
    if (c00 < -1.0e-12 || c11 < -1.0e-12 || c22 < -1.0e-12)
    {
        setError(reason, "has a negative diagonal element");
        return false;
    }

    const auto minorIsValid = [](double first, double second, double cross)
    {
        const double minor = first * second - cross * cross;
        return minor >= -1.0e-12;
    };
    if (!minorIsValid(c00, c11, c01)
        || !minorIsValid(c00, c22, c02)
        || !minorIsValid(c11, c22, c12))
    {
        setError(reason, "is not positive semidefinite");
        return false;
    }

    const double determinant = c00 * (c11 * c22 - c12 * c12)
        - c01 * (c01 * c22 - c12 * c02)
        + c02 * (c01 * c12 - c11 * c02);
    if (determinant < -1.0e-12)
    {
        setError(reason, "is not positive semidefinite");
        return false;
    }
    return true;
}

bool positiveDefinite3(const std::array<double, 9> &matrix)
{
    double scale = 0.0;
    for (double value : matrix)
    {
        scale = std::max(scale, std::abs(value));
    }
    if (!(scale > 0.0))
    {
        return false;
    }

    constexpr double kPivotTolerance = 1.0e-14;
    const double c00 = matrix[0] / scale;
    const double c10 = 0.5 * (matrix[3] + matrix[1]) / scale;
    const double c20 = 0.5 * (matrix[6] + matrix[2]) / scale;
    const double c11 = matrix[4] / scale;
    const double c21 = 0.5 * (matrix[7] + matrix[5]) / scale;
    const double c22 = matrix[8] / scale;
    if (!(c00 > kPivotTolerance))
    {
        return false;
    }

    const double l00 = std::sqrt(c00);
    const double l10 = c10 / l00;
    const double l20 = c20 / l00;
    const double pivot11 = c11 - l10 * l10;
    if (!(pivot11 > kPivotTolerance))
    {
        return false;
    }

    const double l11 = std::sqrt(pivot11);
    const double l21 = (c21 - l20 * l10) / l11;
    const double pivot22 = c22 - l20 * l20 - l21 * l21;
    return pivot22 > kPivotTolerance;
}

bool validateCovariance2(const std::array<double, 4> &matrix)
{
    for (double value : matrix)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    const double scale = std::max({std::abs(matrix[0]), std::abs(matrix[1]),
                                   std::abs(matrix[2]), std::abs(matrix[3])});
    if (std::abs(matrix[1] - matrix[2]) > std::max(1.0e-30, 1.0e-12 * scale))
    {
        return false;
    }
    if (scale == 0.0)
    {
        return true;
    }
    const double first = matrix[0] / scale;
    const double second = matrix[3] / scale;
    const double cross = 0.5 * (matrix[1] + matrix[2]) / scale;
    return first >= -1.0e-12
        && second >= -1.0e-12
        && first * second - cross * cross >= -1.0e-12;
}

} // namespace

std::array<double, 3> planetocentricToBodyFixedMeters(double latitudeDegrees,
                                                       double longitudeDegrees,
                                                       double radiusMeters)
{
    const double latitude = latitudeDegrees * kPi / 180.0;
    const double longitude = longitudeDegrees * kPi / 180.0;
    const double cos_latitude = std::cos(latitude);
    return {{radiusMeters * cos_latitude * std::cos(longitude),
             radiusMeters * cos_latitude * std::sin(longitude),
             radiusMeters * std::sin(latitude)}};
}

const char *planetaryLaserTimeSystemName(PlanetaryLaserTimeSystem timeSystem)
{
    return timeSystem == PlanetaryLaserTimeSystem::TdbEtSeconds ? "TDB_ET_SECONDS" : "UNKNOWN";
}

const char *planetaryLaserImageMeasureKindName(PlanetaryLaserImageMeasureKind kind)
{
    switch (kind)
    {
        case PlanetaryLaserImageMeasureKind::Measured:
            return "measured";
        case PlanetaryLaserImageMeasureKind::ProjectedVirtual:
            return "projected";
        default:
            return "unknown";
    }
}

const char *planetaryLaserSensorModelName(PlanetaryLaserSensorModel sensorModel)
{
    switch (sensorModel)
    {
        case PlanetaryLaserSensorModel::Frame:
            return "frame";
        case PlanetaryLaserSensorModel::LineScan:
            return "line_scan";
        default:
            return "unknown";
    }
}

const char *planetaryLaserRangeTypeName(PlanetaryLaserRangeType rangeType)
{
    switch (rangeType)
    {
        case PlanetaryLaserRangeType::OneWay:
            return "one_way";
        case PlanetaryLaserRangeType::RoundTrip:
            return "round_trip";
        default:
            return "unknown";
    }
}

const char *planetaryLaserPointModeName(PlanetaryLaserPointMode pointMode)
{
    switch (pointMode)
    {
        case PlanetaryLaserPointMode::Fixed:
            return "fixed";
        case PlanetaryLaserPointMode::Constrained:
            return "constrained";
        case PlanetaryLaserPointMode::Free:
            return "free";
        default:
            return "unknown";
    }
}

bool PlanetaryLaserDataset::validate(std::string *errorMessage) const
{
    switch (sourceFormat)
    {
        case PlanetaryLaserSourceFormat::PlaScanSiJsonV1:
        case PlanetaryLaserSourceFormat::IsisLidarDataJson:
            break;
        case PlanetaryLaserSourceFormat::Unknown:
            setError(errorMessage, "Planetary laser source format is not set");
            return false;
        default:
            setError(errorMessage, "Planetary laser source format is invalid");
            return false;
    }
    if (!validSensorModel(sensorModel) || !validRangeType(rangeType))
    {
        setError(errorMessage, "Planetary laser sensor model or range type is invalid");
        return false;
    }
    if (reference.targetName.empty() || reference.bodyFixedFrame.empty() || reference.laserFrame.empty())
    {
        setError(errorMessage, "Planetary laser target, body-fixed frame, and laser frame must be explicit");
        return false;
    }
    if (reference.timeSystem != PlanetaryLaserTimeSystem::TdbEtSeconds)
    {
        setError(errorMessage, "Planetary laser time system must be TDB_ET_SECONDS");
        return false;
    }
    if (reference.latitudeType != "planetocentric" || reference.longitudeDirection != "positive_east")
    {
        setError(errorMessage, "Planetary laser angular convention must be planetocentric/positive_east");
        return false;
    }
    if (shots.empty())
    {
        setError(errorMessage, "Planetary laser dataset has no shots");
        return false;
    }

    std::set<std::string> shot_ids;
    for (std::size_t shot_index = 0; shot_index < shots.size(); ++shot_index)
    {
        const PlanetaryLaserShot &shot = shots[shot_index];
        const std::string prefix = "Shot[" + std::to_string(shot_index) + "]";
        if (shot.id.empty() || !shot_ids.insert(shot.id).second)
        {
            setError(errorMessage, prefix + " has an empty or duplicate id");
            return false;
        }
        if (!validPointMode(shot.pointMode))
        {
            setError(errorMessage, prefix + " has an invalid point mode");
            return false;
        }
        if (!std::isfinite(shot.ephemerisTimeSeconds)
            || !std::isfinite(shot.observedRangeMeters) || shot.observedRangeMeters <= 0.0
            || !std::isfinite(shot.rangeSigmaMeters) || shot.rangeSigmaMeters <= 0.0)
        {
            setError(errorMessage, prefix + " has invalid ET, range, or range sigma");
            return false;
        }
        if (!finiteVector(shot.pointBodyFixedMeters) || !finiteVector(shot.leverArmSensorMeters))
        {
            setError(errorMessage, prefix + " has non-finite point or lever-arm coordinates");
            return false;
        }
        const double point_norm = std::hypot(
            std::hypot(shot.pointBodyFixedMeters[0], shot.pointBodyFixedMeters[1]),
            shot.pointBodyFixedMeters[2]);
        if (!(point_norm > 0.0))
        {
            setError(errorMessage, prefix + " body-fixed point is at the target center");
            return false;
        }
        if (sourceFormat == PlanetaryLaserSourceFormat::PlaScanSiJsonV1 && !shot.leverArmSpecified)
        {
            setError(errorMessage, prefix + " must explicitly specify lever_arm_sensor_m");
            return false;
        }
        if (shot.pointCovarianceBodyFixedMetersSquared)
        {
            std::string reason;
            if (!validateCovariance3(*shot.pointCovarianceBodyFixedMetersSquared, &reason))
            {
                setError(errorMessage, prefix + " point covariance " + reason);
                return false;
            }
        }
        if (shot.pointMode == PlanetaryLaserPointMode::Constrained
            && !shot.pointCovarianceBodyFixedMetersSquared)
        {
            setError(errorMessage, prefix + " constrained point requires a body-fixed covariance");
            return false;
        }
        if (shot.pointMode == PlanetaryLaserPointMode::Constrained
            && !positiveDefinite3(*shot.pointCovarianceBodyFixedMetersSquared))
        {
            setError(errorMessage, prefix + " constrained point covariance must be positive definite");
            return false;
        }
        if (shot.pointMode == PlanetaryLaserPointMode::Free
            && shot.pointCovarianceBodyFixedMetersSquared)
        {
            setError(errorMessage,
                     prefix + " free point cannot carry a soft covariance; use constrained point mode");
            return false;
        }

        std::set<std::string> simultaneous_ids;
        for (const std::string &image_id : shot.simultaneousImageIds)
        {
            if (image_id.empty() || !simultaneous_ids.insert(image_id).second)
            {
                setError(errorMessage, prefix + " has an empty or duplicate simultaneous image id");
                return false;
            }
        }
        if (simultaneous_ids.empty())
        {
            setError(errorMessage, prefix + " has no simultaneous image id");
            return false;
        }

        std::set<std::pair<std::string, PlanetaryLaserImageMeasureKind>> measure_keys;
        std::set<std::string> measure_ids;
        for (const PlanetaryLaserImageMeasure &measure : shot.imageMeasures)
        {
            const auto measure_key = std::make_pair(measure.imageId, measure.kind);
            if (!validMeasureKind(measure.kind)
                || measure.imageId.empty() || !measure_keys.insert(measure_key).second
                || !std::isfinite(measure.samplePixels) || measure.samplePixels < 0.0
                || !std::isfinite(measure.linePixels) || measure.linePixels < 0.0)
            {
                setError(errorMessage, prefix + " has an invalid or duplicate image measure");
                return false;
            }
            measure_ids.insert(measure.imageId);
            if (sourceFormat == PlanetaryLaserSourceFormat::IsisLidarDataJson
                && measure.kind != PlanetaryLaserImageMeasureKind::ProjectedVirtual)
            {
                setError(errorMessage, prefix + " ISIS image measures must remain projected/virtual");
                return false;
            }
            if (measure.covariancePixelsSquared && !validateCovariance2(*measure.covariancePixelsSquared))
            {
                setError(errorMessage, prefix + " has an invalid image covariance");
                return false;
            }
        }
        if (sourceFormat == PlanetaryLaserSourceFormat::IsisLidarDataJson)
        {
            for (const std::string &image_id : simultaneous_ids)
            {
                if (!measure_ids.contains(image_id))
                {
                    setError(errorMessage, prefix + " ISIS simultaneous image has no projected measure");
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace lidar
} // namespace xjw
