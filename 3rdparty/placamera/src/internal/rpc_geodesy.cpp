#include "placamera/rpc_camera.h"

#include <cmath>

namespace placamera
{

    namespace
    {

        constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
        constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

        bool finite(double value) noexcept
        {
            return std::isfinite(value);
        }

        bool validEllipsoid(const ReferenceEllipsoid& ellipsoid) noexcept
        {
            return finite(ellipsoid.semiMajorAxisMeters) && finite(ellipsoid.inverseFlattening) &&
                   ellipsoid.semiMajorAxisMeters > 0.0 &&
                   (ellipsoid.inverseFlattening == 0.0 || ellipsoid.inverseFlattening > 1.0);
        }

        double flattening(const ReferenceEllipsoid& ellipsoid) noexcept
        {
            return ellipsoid.inverseFlattening == 0.0 ? 0.0 : 1.0 / ellipsoid.inverseFlattening;
        }

    } // namespace

    EvaluationResult<Vector3> geodeticToCartesian(const GeodeticCoordinate& geodetic,
                                                  const ReferenceEllipsoid& ellipsoid)
    {
        if (!validEllipsoid(ellipsoid) || !finite(geodetic.longitudeDegrees) || !finite(geodetic.latitudeDegrees) ||
            !finite(geodetic.heightMeters) || geodetic.latitudeDegrees < -90.0 || geodetic.latitudeDegrees > 90.0)
        {
            return EvaluationResult<Vector3>::failure(CameraErrorCode::InvalidArgument,
                                                      "geodetic coordinate or reference ellipsoid is invalid");
        }

        const double longitude = geodetic.longitudeDegrees * kDegreesToRadians;
        const double latitude = geodetic.latitudeDegrees * kDegreesToRadians;
        const double f = flattening(ellipsoid);
        const double eccentricity_squared = f * (2.0 - f);
        const double sine_latitude = std::sin(latitude);
        const double cosine_latitude = std::cos(latitude);
        const double prime_vertical =
            ellipsoid.semiMajorAxisMeters / std::sqrt(1.0 - eccentricity_squared * sine_latitude * sine_latitude);
        const Vector3 cartesian{(prime_vertical + geodetic.heightMeters) * cosine_latitude * std::cos(longitude),
                                (prime_vertical + geodetic.heightMeters) * cosine_latitude * std::sin(longitude),
                                (prime_vertical * (1.0 - eccentricity_squared) + geodetic.heightMeters) *
                                    sine_latitude};
        if (!finite(cartesian[0]) || !finite(cartesian[1]) || !finite(cartesian[2]))
        {
            return EvaluationResult<Vector3>::failure(CameraErrorCode::OutsideModelDomain,
                                                      "geodetic conversion produced a non-finite coordinate");
        }
        return EvaluationResult<Vector3>::success(cartesian);
    }

    EvaluationResult<GeodeticCoordinate> cartesianToGeodetic(const Vector3& cartesian,
                                                             const ReferenceEllipsoid& ellipsoid)
    {
        if (!validEllipsoid(ellipsoid) || !finite(cartesian[0]) || !finite(cartesian[1]) || !finite(cartesian[2]))
        {
            return EvaluationResult<GeodeticCoordinate>::failure(
                CameraErrorCode::InvalidArgument, "Cartesian coordinate or reference ellipsoid is invalid");
        }

        const double horizontal = std::hypot(cartesian[0], cartesian[1]);
        if (horizontal < 1.0e-9 && std::abs(cartesian[2]) < 1.0e-9)
        {
            return EvaluationResult<GeodeticCoordinate>::failure(CameraErrorCode::OutsideModelDomain,
                                                                 "ellipsoid center has no unique geodetic coordinate");
        }

        const double f = flattening(ellipsoid);
        const double eccentricity_squared = f * (2.0 - f);
        const double longitude = std::atan2(cartesian[1], cartesian[0]);
        double latitude = std::atan2(cartesian[2], horizontal * (1.0 - eccentricity_squared));
        double height = 0.0;
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            const double sine_latitude = std::sin(latitude);
            const double prime_vertical =
                ellipsoid.semiMajorAxisMeters / std::sqrt(1.0 - eccentricity_squared * sine_latitude * sine_latitude);
            height = horizontal > 1.0e-9 ? horizontal / std::cos(latitude) - prime_vertical
                                         : std::abs(cartesian[2]) - prime_vertical * (1.0 - eccentricity_squared);
            const double next_latitude = std::atan2(
                cartesian[2], horizontal * (1.0 - eccentricity_squared * prime_vertical / (prime_vertical + height)));
            if (std::abs(next_latitude - latitude) < 1.0e-14)
            {
                latitude = next_latitude;
                break;
            }
            latitude = next_latitude;
        }

        const double sine_latitude = std::sin(latitude);
        const double prime_vertical =
            ellipsoid.semiMajorAxisMeters / std::sqrt(1.0 - eccentricity_squared * sine_latitude * sine_latitude);
        height = horizontal > 1.0e-9 ? horizontal / std::cos(latitude) - prime_vertical
                                     : std::abs(cartesian[2]) - prime_vertical * (1.0 - eccentricity_squared);
        const GeodeticCoordinate geodetic{longitude * kRadiansToDegrees, latitude * kRadiansToDegrees, height};
        if (!finite(geodetic.longitudeDegrees) || !finite(geodetic.latitudeDegrees) || !finite(geodetic.heightMeters))
        {
            return EvaluationResult<GeodeticCoordinate>::failure(
                CameraErrorCode::NonConvergence, "Cartesian conversion did not produce a finite geodetic coordinate");
        }
        return EvaluationResult<GeodeticCoordinate>::success(geodetic);
    }

} // namespace placamera
