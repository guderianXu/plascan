#pragma once

#include "../MeshTypes.h"
#include "../SurfaceReconstructorIO.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace xjw
{
namespace mesh
{
namespace poisson
{

// Simplified in-repo Poisson-like reconstruction pipeline.
// It is class-based for readability and future extension.
class PoissonLiteReconstructor
{
public:
    bool reconstruct(const std::vector<detail::PointXYZRGB> &points,
                     const std::vector<cv::Vec3f> &normals,
                     const ReconstructionConfig &config,
                     TriMesh *mesh,
                     std::string *errorMessage) const;

private:
    bool validateInput(const std::vector<detail::PointXYZRGB> &points,
                       const std::vector<cv::Vec3f> &normals,
                       TriMesh *mesh,
                       std::string *errorMessage) const;
};

} // namespace poisson
} // namespace mesh
} // namespace xjw
