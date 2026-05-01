#pragma once

#include "../MeshTypes.h"
#include "../SurfaceReconstructorIO.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace xjw
{
namespace mesh
{
namespace poisson
{

struct VoxelGrid
{
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float minX = 0.0f;
    float minY = 0.0f;
    float minZ = 0.0f;
    float step = 1.0f;
    std::vector<std::uint8_t> occupied;
    std::vector<float> scalarField;
    float isoLevel = 0.6f;

    bool hasScalarField() const
    {
        return !scalarField.empty() && scalarField.size() == occupied.size();
    }

    bool valid() const
    {
        return nx > 2 && ny > 2 && nz > 2 && step > 0.0f
               && (!occupied.empty() || !scalarField.empty());
    }
};

class PoissonVoxelPipeline
{
public:
    bool shouldUseVoxelReconstruction(const std::vector<detail::PointXYZRGB> &points) const;
    VoxelGrid buildVoxelGrid(const std::vector<detail::PointXYZRGB> &points,
                             const ReconstructionConfig &config) const;
    VoxelGrid buildVoxelGrid(const std::vector<detail::PointXYZRGB> &points,
                             const std::vector<cv::Vec3f> &normals,
                             const ReconstructionConfig &config) const;
    void voxelGridToMesh(const VoxelGrid &voxelGrid,
                         const std::vector<detail::PointXYZRGB> &points,
                         TriMesh *mesh) const;
};

} // namespace poisson
} // namespace mesh
} // namespace xjw
