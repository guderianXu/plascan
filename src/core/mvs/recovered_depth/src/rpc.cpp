#include "metmodel/rpc.hpp"

#include <cmath>
#include <limits>

namespace metmodel
{
namespace
{

double polynomial(const std::array<double, 20>& c, double lon, double lat, double height)
{
    const double lon2 = lon * lon;
    const double lat2 = lat * lat;
    const double height2 = height * height;
    return c[0] + c[1] * lon + c[2] * lat + c[3] * height + c[4] * lon * lat +
        c[5] * lon * height + c[6] * lat * height + c[7] * lon2 + c[8] * lat2 +
        c[9] * height2 + c[10] * lon * lat * height + c[11] * lon * lon2 +
        c[12] * lon * lat2 + c[13] * lon * height2 + c[14] * lon2 * lat +
        c[15] * lat * lat2 + c[16] * lat * height2 + c[17] * lon2 * height +
        c[18] * lat2 * height + c[19] * height * height2;
}

bool finite(double value)
{
    return std::isfinite(value);
}

} // namespace

bool validate_rpc00b(const Rpc00bCamera& rpc, std::string& error)
{
    const std::array<double, 10> scales{rpc.line_off, rpc.sample_off, rpc.lat_off, rpc.lon_off,
                                        rpc.height_off, rpc.line_scale, rpc.sample_scale,
                                        rpc.lat_scale, rpc.lon_scale, rpc.height_scale};
    for (const double value : scales)
    {
        if (!finite(value))
        {
            error = "RPC00B contains a non-finite offset or scale";
            return false;
        }
    }
    if (rpc.line_scale <= 0.0 || rpc.sample_scale <= 0.0 || rpc.lat_scale <= 0.0 ||
        rpc.lon_scale <= 0.0 || rpc.height_scale <= 0.0)
    {
        error = "RPC00B scales must be positive";
        return false;
    }
    for (const auto* coefficients : {&rpc.line_num, &rpc.line_den, &rpc.sample_num, &rpc.sample_den})
    {
        for (const double value : *coefficients)
        {
            if (!finite(value))
            {
                error = "RPC00B contains a non-finite rational coefficient";
                return false;
            }
        }
    }
    const auto& correction = rpc.affine_correction;
    for (const double value : {correction.sample_offset, correction.sample_lon, correction.sample_lat,
                               correction.sample_height, correction.line_offset, correction.line_lon,
                               correction.line_lat, correction.line_height})
    {
        if (!finite(value))
        {
            error = "RPC00B contains a non-finite affine correction";
            return false;
        }
    }
    error.clear();
    return true;
}

bool rpc00b_project(const Rpc00bCamera& rpc,
                    const metalign::Vec3& world,
                    metalign::Vec2& pixel,
                    std::string& error)
{
    if (!validate_rpc00b(rpc, error) || !finite(world.x) || !finite(world.y) || !finite(world.z))
    {
        if (error.empty()) error = "RPC00B world coordinate is non-finite";
        return false;
    }
    const double lon = (world.x - rpc.lon_off) / rpc.lon_scale;
    const double lat = (world.y - rpc.lat_off) / rpc.lat_scale;
    const double height = (world.z - rpc.height_off) / rpc.height_scale;
    const double line_den = polynomial(rpc.line_den, lon, lat, height);
    const double sample_den = polynomial(rpc.sample_den, lon, lat, height);
    if (!finite(line_den) || !finite(sample_den) || std::abs(line_den) < 1.0e-14 ||
        std::abs(sample_den) < 1.0e-14)
    {
        error = "RPC00B rational denominator is singular";
        return false;
    }
    pixel.y = rpc.line_off + rpc.line_scale * polynomial(rpc.line_num, lon, lat, height) / line_den;
    pixel.x = rpc.sample_off + rpc.sample_scale * polynomial(rpc.sample_num, lon, lat, height) / sample_den;
    const auto& correction = rpc.affine_correction;
    const double delta_lon = world.x - rpc.lon_off;
    const double delta_lat = world.y - rpc.lat_off;
    const double delta_height = world.z - rpc.height_off;
    pixel.x += correction.sample_offset + correction.sample_lon * delta_lon +
        correction.sample_lat * delta_lat + correction.sample_height * delta_height;
    pixel.y += correction.line_offset + correction.line_lon * delta_lon +
        correction.line_lat * delta_lat + correction.line_height * delta_height;
    if (!finite(pixel.x) || !finite(pixel.y))
    {
        error = "RPC00B projection is non-finite";
        return false;
    }
    error.clear();
    return true;
}

bool rpc00b_unproject_at_height(const Rpc00bCamera& rpc,
                                const metalign::Vec2& pixel,
                                double height,
                                metalign::Vec3& world,
                                std::string& error)
{
    if (!validate_rpc00b(rpc, error) || !finite(pixel.x) || !finite(pixel.y) || !finite(height))
    {
        if (error.empty()) error = "RPC00B inverse input is non-finite";
        return false;
    }
    world = {rpc.lon_off, rpc.lat_off, height};
    const double delta_lon = std::max(std::abs(rpc.lon_scale) * 1.0e-6, 1.0e-8);
    const double delta_lat = std::max(std::abs(rpc.lat_scale) * 1.0e-6, 1.0e-8);
    for (int iteration = 0; iteration != 24; ++iteration)
    {
        metalign::Vec2 projected;
        metalign::Vec2 lon_projected;
        metalign::Vec2 lat_projected;
        std::string project_error;
        if (!rpc00b_project(rpc, world, projected, project_error) ||
            !rpc00b_project(rpc, {world.x + delta_lon, world.y, height}, lon_projected, project_error) ||
            !rpc00b_project(rpc, {world.x, world.y + delta_lat, height}, lat_projected, project_error))
        {
            error = project_error;
            return false;
        }
        const double residual_sample = pixel.x - projected.x;
        const double residual_line = pixel.y - projected.y;
        if (std::max(std::abs(residual_sample), std::abs(residual_line)) < 1.0e-7)
        {
            error.clear();
            return true;
        }
        const double a = (lon_projected.x - projected.x) / delta_lon;
        const double b = (lat_projected.x - projected.x) / delta_lat;
        const double c = (lon_projected.y - projected.y) / delta_lon;
        const double d = (lat_projected.y - projected.y) / delta_lat;
        const double determinant = a * d - b * c;
        if (!finite(determinant) || std::abs(determinant) < 1.0e-18)
        {
            error = "RPC00B inverse Jacobian is singular";
            return false;
        }
        world.x += (residual_sample * d - residual_line * b) / determinant;
        world.y += (a * residual_line - c * residual_sample) / determinant;
        if (!finite(world.x) || !finite(world.y))
        {
            error = "RPC00B inverse diverged";
            return false;
        }
    }
    error = "RPC00B fixed-height inverse did not converge";
    return false;
}

} // namespace metmodel
