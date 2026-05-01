#include "PoissonBranchPipeline.h"

#include "../SurfaceReconstructorPostprocess.h"
#include "PoissonLite.h"
#include "PoissonPreprocess.h"
#include "PoissonVoxel.h"

#include <algorithm>
#include <cmath>

namespace xjw
{
namespace mesh
{
namespace poisson
{

bool PoissonBranchPipeline::reconstruct(const std::vector<detail::PointXYZRGB> &points,
                                        const ReconstructionConfig &config,
                                        TriMesh *mesh,
                                        std::string *errorMessage,
                                        const std::function<void(const std::string &, float)> &progress) const
{
    if (!mesh)
    {
        if (errorMessage)
        {
            *errorMessage = "Poisson 分支失败：输出网格指针为空";
        }
        return false;
    }

    const PoissonVoxelPipeline voxelPipeline;
    const bool useVoxelReconstruction = voxelPipeline.shouldUseVoxelReconstruction(points);
    const bool tryPoisson = config.forcePoisson && useVoxelReconstruction;

    const PoissonPreprocessor preprocessor;
    const float poissonVoxel = preprocessor.estimateBaseVoxelStep(points, config.resolution);

    if (tryPoisson)
    {
        progress("正在统一法线并执行 Poisson 重建...", 0.30f);
        const int adaptiveK = std::clamp(
            std::max(std::max(12, config.kNormals),
                     static_cast<int>(std::lround(std::log10(static_cast<double>(std::max<std::size_t>(points.size(), 10))) * 6.0))
                         + config.poissonDepth / 2),
            12,
            32);
        const float normalGridScale = std::clamp(2.15f - (config.poissonDepth - 8) * 0.09f, 1.35f, 2.30f);
        const auto normals = preprocessor.estimateNormals(points, adaptiveK, poissonVoxel * normalGridScale);
        const PoissonLiteReconstructor poissonReconstructor;
        if (!poissonReconstructor.reconstruct(points, normals, config, mesh, errorMessage))
        {
            progress("Poisson 失败，回退到体素重建...", 0.45f);
            mesh->vertices.clear();
            mesh->faces.clear();
        }
    }

    if (mesh->empty() && useVoxelReconstruction)
    {
        progress("正在构建 3D 体素场...", 0.12f);
        const VoxelGrid voxelGrid = voxelPipeline.buildVoxelGrid(points, config);
        if (!voxelGrid.valid())
        {
            if (errorMessage)
            {
                *errorMessage = "3D 体素场构建失败";
            }
            return false;
        }

        progress("正在提取 3D 曲面...", 0.50f);
        voxelPipeline.voxelGridToMesh(voxelGrid, points, mesh);

        progress("正在简化三角面片...", 0.62f);
        detail::simplifyVoxelMeshAdaptive(mesh, config, voxelGrid.step);
    }

    if (!mesh->empty() && tryPoisson)
    {
        // Preserve detail for Poisson output: only simplify if the mesh is extremely dense.
        const int poissonFaceCount = mesh->faceCount();
        const int hardUpperBound = std::max(120000, std::max(1, config.simplifyTargetFaces) * 4);
        if (poissonFaceCount > hardUpperBound)
        {
            progress("正在轻量简化 Poisson 网格...", 0.62f);
            ReconstructionConfig liteConfig = config;
            liteConfig.simplifyTargetFaces = std::max(config.simplifyTargetFaces * 2, 80000);
            liteConfig.voxelSimplifyFactor = std::clamp(config.voxelSimplifyFactor * 0.85f, 1.0f, 2.4f);
            detail::simplifyVoxelMeshAdaptive(mesh, liteConfig, poissonVoxel);
        }
    }

    return true;
}

} // namespace poisson
} // namespace mesh
} // namespace xjw
