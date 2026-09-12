#pragma once

#include "metalign/geometry.hpp"

#include <array>
#include <string>

namespace metmodel
{

// Public RPC00B representation used by the CUDA height-plane-sweep path.  It
// deliberately remains separate from the recovered type-8 compact camera: a
// standard RPC must never be silently coerced into that unrelated ABI.
struct Rpc00bAffineCorrection
{
    double sample_offset = 0.0;
    double sample_lon = 0.0;
    double sample_lat = 0.0;
    double sample_height = 0.0;
    double line_offset = 0.0;
    double line_lon = 0.0;
    double line_lat = 0.0;
    double line_height = 0.0;
};

struct Rpc00bCamera
{
    double line_off = 0.0;
    double sample_off = 0.0;
    double lat_off = 0.0;
    double lon_off = 0.0;
    double height_off = 0.0;
    double line_scale = 0.0;
    double sample_scale = 0.0;
    double lat_scale = 0.0;
    double lon_scale = 0.0;
    double height_scale = 0.0;
    std::array<double, 20> line_num{};
    std::array<double, 20> line_den{};
    std::array<double, 20> sample_num{};
    std::array<double, 20> sample_den{};
    Rpc00bAffineCorrection affine_correction{};
};

bool validate_rpc00b(const Rpc00bCamera& rpc, std::string& error);
bool rpc00b_project(const Rpc00bCamera& rpc,
                    const metalign::Vec3& world,
                    metalign::Vec2& pixel,
                    std::string& error);
bool rpc00b_unproject_at_height(const Rpc00bCamera& rpc,
                                const metalign::Vec2& pixel,
                                double height,
                                metalign::Vec3& world,
                                std::string& error);

} // namespace metmodel
