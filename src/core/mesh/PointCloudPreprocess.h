#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <plapoint/filters/preprocessing.h>

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
    bool hasNormal = false;
    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 1.0f;
    std::uint8_t r = 200;
    std::uint8_t g = 200;
    std::uint8_t b = 200;
};

float estimateBaseVoxelStep(const std::vector<PointXYZRGB> &points,
                            int resolution);

std::vector<PointXYZRGB> voxelDownsamplePoints(const std::vector<PointXYZRGB> &points,
                                                float voxelSize,
                                                plapoint::ProcessingDevice device =
                                                    plapoint::ProcessingDevice::Auto);

std::vector<PointXYZRGB> statisticalDenoisePoints(const std::vector<PointXYZRGB> &points,
                                                    int k,
                                                    float stdMul,
                                                    float gridCellSize,
                                                    plapoint::ProcessingDevice device =
                                                        plapoint::ProcessingDevice::Auto);

std::size_t removeInvalidPoissonPoints(std::vector<PointXYZRGB> *points);

} // namespace detail
} // namespace mesh
} // namespace xjw
