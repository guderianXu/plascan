#pragma once

#include <cstdint>
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

float estimateBaseVoxelStep(const std::vector<PointXYZRGB> &points,
                            int resolution);

std::vector<PointXYZRGB> voxelDownsamplePoints(const std::vector<PointXYZRGB> &points,
                                                float voxelSize);

std::vector<PointXYZRGB> statisticalDenoisePoints(const std::vector<PointXYZRGB> &points,
                                                    int k,
                                                    float stdMul,
                                                    float gridCellSize);

} // namespace detail
} // namespace mesh
} // namespace xjw
