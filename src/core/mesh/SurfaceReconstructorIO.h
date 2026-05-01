#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xjw
{
namespace mesh
{
namespace detail
{

struct PointXYZRGB
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    std::uint8_t r = 200;
    std::uint8_t g = 200;
    std::uint8_t b = 200;
};

bool loadPointCloud(const std::string &path,
                    std::vector<PointXYZRGB> &points,
                    std::string *errorMessage);

} // namespace detail
} // namespace mesh
} // namespace xjw
