#pragma once

#include "../SurfaceReconstructorIO.h"

#include <opencv2/core.hpp>

#include <vector>

namespace xjw
{
namespace mesh
{
namespace poisson
{

class PoissonPreprocessor
{
public:
    float estimateBaseVoxelStep(const std::vector<detail::PointXYZRGB> &points,
                                int resolution) const;

    std::vector<detail::PointXYZRGB> voxelDownsamplePoints(
        const std::vector<detail::PointXYZRGB> &points,
        float voxelSize) const;

    std::vector<detail::PointXYZRGB> statisticalDenoisePoints(
        const std::vector<detail::PointXYZRGB> &points,
        int k,
        float stdMul,
        float gridCellSize) const;

    std::vector<cv::Vec3f> estimateNormals(const std::vector<detail::PointXYZRGB> &points,
                                           int k,
                                           float gridCellSize) const;
};

} // namespace poisson
} // namespace mesh
} // namespace xjw
