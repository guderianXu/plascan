#pragma once

#include "../MeshTypes.h"
#include "../SurfaceReconstructorIO.h"

#include <functional>
#include <string>
#include <vector>

namespace xjw
{
namespace mesh
{
namespace poisson
{

class PoissonBranchPipeline
{
public:
    // Returns false only when a voxel reconstruction path is selected but fails fatally.
    // If no voxel/poisson path is selected, it keeps mesh empty and returns true so caller can fallback.
    bool reconstruct(const std::vector<detail::PointXYZRGB> &points,
                     const ReconstructionConfig &config,
                     TriMesh *mesh,
                     std::string *errorMessage,
                     const std::function<void(const std::string &, float)> &progress) const;
};

} // namespace poisson
} // namespace mesh
} // namespace xjw
