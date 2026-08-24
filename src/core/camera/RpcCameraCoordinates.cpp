#include "RpcCameraModel.h"

#include <cmath>

namespace xjw
{
    namespace
    {

        constexpr double kWgs84SemiMajorMeters = 6378137.0;
        constexpr double kWgs84Flattening = 1.0 / 298.257223563;
        constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
        constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

    } // namespace

    bool RpcCameraModel::geodeticToEcef(const GeodeticCoordinate& geodetic, EcefCoordinate* ecef)
    {
        if (!ecef || !std::isfinite(geodetic[0]) || !std::isfinite(geodetic[1]) || !std::isfinite(geodetic[2]) ||
            geodetic[1] < -90.0 || geodetic[1] > 90.0)
        {
            return false;
        }
        const double longitude = geodetic[0] * kDegreesToRadians;
        const double latitude = geodetic[1] * kDegreesToRadians;
        const double eccentricity2 = kWgs84Flattening * (2.0 - kWgs84Flattening);
        const double sin_latitude = std::sin(latitude);
        const double cos_latitude = std::cos(latitude);
        const double prime_vertical =
            kWgs84SemiMajorMeters / std::sqrt(1.0 - eccentricity2 * sin_latitude * sin_latitude);
        (*ecef)[0] = (prime_vertical + geodetic[2]) * cos_latitude * std::cos(longitude);
        (*ecef)[1] = (prime_vertical + geodetic[2]) * cos_latitude * std::sin(longitude);
        (*ecef)[2] = (prime_vertical * (1.0 - eccentricity2) + geodetic[2]) * sin_latitude;
        return std::isfinite((*ecef)[0]) && std::isfinite((*ecef)[1]) && std::isfinite((*ecef)[2]);
    }

    bool RpcCameraModel::ecefToGeodetic(const EcefCoordinate& ecef, GeodeticCoordinate* geodetic)
    {
        if (!geodetic || !std::isfinite(ecef[0]) || !std::isfinite(ecef[1]) || !std::isfinite(ecef[2]))
        {
            return false;
        }
        const double horizontal = std::hypot(ecef[0], ecef[1]);
        if (horizontal < 1.0e-9 && std::abs(ecef[2]) < 1.0e-9)
        {
            return false;
        }

        const double eccentricity2 = kWgs84Flattening * (2.0 - kWgs84Flattening);
        const double longitude = std::atan2(ecef[1], ecef[0]);
        double latitude = std::atan2(ecef[2], horizontal * (1.0 - eccentricity2));
        double height = 0.0;
        for (int iteration = 0; iteration < 15; ++iteration)
        {
            const double sin_latitude = std::sin(latitude);
            const double prime_vertical =
                kWgs84SemiMajorMeters / std::sqrt(1.0 - eccentricity2 * sin_latitude * sin_latitude);
            if (horizontal > 1.0e-9)
            {
                height = horizontal / std::cos(latitude) - prime_vertical;
            }
            else
            {
                height = std::abs(ecef[2]) - prime_vertical * (1.0 - eccentricity2);
            }
            const double next_latitude =
                std::atan2(ecef[2], horizontal * (1.0 - eccentricity2 * prime_vertical / (prime_vertical + height)));
            if (std::abs(next_latitude - latitude) < 1.0e-14)
            {
                latitude = next_latitude;
                break;
            }
            latitude = next_latitude;
        }

        const double sin_latitude = std::sin(latitude);
        const double prime_vertical =
            kWgs84SemiMajorMeters / std::sqrt(1.0 - eccentricity2 * sin_latitude * sin_latitude);
        height = horizontal > 1.0e-9 ? horizontal / std::cos(latitude) - prime_vertical
                                     : std::abs(ecef[2]) - prime_vertical * (1.0 - eccentricity2);
        (*geodetic)[0] = longitude * kRadiansToDegrees;
        (*geodetic)[1] = latitude * kRadiansToDegrees;
        (*geodetic)[2] = height;
        return std::isfinite((*geodetic)[0]) && std::isfinite((*geodetic)[1]) && std::isfinite((*geodetic)[2]);
    }

} // namespace xjw
