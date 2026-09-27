#include "engine/RpcEngineInternals.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xjw::aerial_triangulation::engine
{
    namespace
    {
        double median(std::vector<double> values)
        {
            if (values.empty())
            {
                return 0.0;
            }
            std::sort(values.begin(), values.end());
            const std::size_t middle = values.size() / 2;
            return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
        }

    } // namespace
    namespace detail
    {
        RpcGeodeticCoordinate assignRpcLocalEnu(std::vector<RpcPoint>* points)
        {
            std::vector<double> longitudes;
            std::vector<double> latitudes;
            std::vector<double> heights;
            longitudes.reserve(points->size());
            latitudes.reserve(points->size());
            heights.reserve(points->size());
            for (const RpcPoint& point : *points)
            {
                longitudes.push_back(point.geodetic[0]);
                latitudes.push_back(point.geodetic[1]);
                heights.push_back(point.geodetic[2]);
            }
            const RpcGeodeticCoordinate origin{
                median(std::move(longitudes)), median(std::move(latitudes)), median(std::move(heights))};
            const auto originEcefResult = placamera::geodeticToCartesian(
                placamera::GeodeticCoordinate{origin[0], origin[1], origin[2]},
                placamera::ReferenceEllipsoid::wgs84());
            const auto originEcef = originEcefResult.value();

            constexpr double degreesToRadians = 3.14159265358979323846 / 180.0;
            const double longitude = origin[0] * degreesToRadians;
            const double latitude = origin[1] * degreesToRadians;
            const std::array<double, 3> east{-std::sin(longitude), std::cos(longitude), 0.0};
            const std::array<double, 3> north{-std::sin(latitude) * std::cos(longitude),
                                              -std::sin(latitude) * std::sin(longitude),
                                              std::cos(latitude)};
            const std::array<double, 3> up{
                std::cos(latitude) * std::cos(longitude), std::cos(latitude) * std::sin(longitude), std::sin(latitude)};
            for (RpcPoint& point : *points)
            {
                const std::array<double, 3> delta{
                    point.ecef[0] - originEcef[0], point.ecef[1] - originEcef[1], point.ecef[2] - originEcef[2]};
                const auto dot = [&delta](const std::array<double, 3>& axis)
                { return delta[0] * axis[0] + delta[1] * axis[1] + delta[2] * axis[2]; };
                point.localEnu = {
                    static_cast<float>(dot(east)), static_cast<float>(dot(north)), static_cast<float>(dot(up))};
            }
            return origin;
        }

    } // namespace detail
} // namespace xjw::aerial_triangulation::engine
