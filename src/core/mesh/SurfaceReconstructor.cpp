#include "SurfaceReconstructor.h"
#include "SurfaceReconstructorHeightGrid.h"
#include "SurfaceReconstructorIO.h"
#include "SurfaceReconstructorPostprocess.h"
#include "poisson/PoissonBranchPipeline.h"
#include "poisson/PoissonPreprocess.h"

#include <algorithm>
#include <string>
#include <vector>

namespace xjw 
{
namespace mesh 
{

namespace 
{

using PointXYZRGB = detail::PointXYZRGB;

} // anonymous namespace

bool SurfaceReconstructor::reconstructFromPointCloudFile(const std::string &cloudPath,
                                                         const ReconstructionConfig &config,
                                                         TriMesh &outMesh,
                                                         std::string *errorMsg)
{
    auto progress = [&](const std::string &stage, float p) {
        if (config.progressFn) config.progressFn(stage, p);
    };

    progress("正在加载点云...", 0.02f);
    std::vector<PointXYZRGB> points;
    if (!detail::loadPointCloud(cloudPath, points, errorMsg)) {
        return false;
    }

    if (points.size() < 100) {
        if (errorMsg) *errorMsg = "点云点数过少，无法稳定重建";
        return false;
    }

    const poisson::PoissonPreprocessor preprocessor;
    const float baseVoxel = preprocessor.estimateBaseVoxelStep(points, config.resolution);
    if (config.enableDenoise)
    {
        progress("正在点云去噪...", 0.08f);
        points = preprocessor.statisticalDenoisePoints(points,
                                                       std::clamp(config.denoiseK, 8, 64),
                                                       std::clamp(config.denoiseStdMul, 0.6f, 3.0f),
                                                       baseVoxel * 2.0f);
    }

    if (config.enableDownsample)
    {
        progress("正在点云下采样...", 0.12f);
        const float voxelSize = baseVoxel * std::clamp(config.downsampleVoxelScale, 0.4f, 2.5f);
        points = preprocessor.voxelDownsamplePoints(points, voxelSize);
    }

    if (points.size() < 120)
    {
        if (errorMsg)
        {
            *errorMsg = "点云预处理后点数不足，无法稳定网格化";
        }
        return false;
    }

    TriMesh mesh;
    bool usedHeightGridFallback = false;
    const poisson::PoissonBranchPipeline poissonBranch;
    if (!poissonBranch.reconstruct(points, config, &mesh, errorMsg, progress))
    {
        return false;
    }

    if (mesh.empty())
    {
        usedHeightGridFallback = true;
        progress("正在构建高度格网...", 0.10f);
        detail::HeightGrid hg = detail::buildHeightGrid(points, config);
        if (hg.nx < 4 || hg.ny < 4) {
            if (errorMsg) *errorMsg = "高程格网构建失败（点云范围过小）";
            return false;
        }

        if (config.fillHoles) {
            progress("正在填充空洞...", 0.30f);
            detail::fillHoles(&hg, std::max(0, config.holeFillPasses));
        }

        progress("正在三角分割...", 0.50f);
        detail::heightGridToMesh(hg, points, config, &mesh);
    }

    if (mesh.empty()) {
        if (errorMsg) *errorMsg = "网格化失败（结果为空）";
        return false;
    }

    progress("正在清理退化面...", 0.64f);
    detail::removeDegenerateFaces(&mesh);

    if (config.cleanSmallComponents) {
        progress("正在清理碎片连通体...", 0.70f);
        detail::removeSmallConnectedComponents(&mesh, std::max(2, config.minComponentFaces));
        if (mesh.empty()) {
            if (errorMsg) *errorMsg = "网格清理后为空，请降低连通体阈值或提高分辨率";
            return false;
        }
    }

    progress("正在平滑网格...", 0.75f);
    int smoothIters = config.smoothIterations;
    float smoothLambda = config.smoothLambda;
    if (!usedHeightGridFallback)
    {
        // Poisson path has already done regularization, so keep this pass light to avoid blurring.
        smoothIters = std::max(0, config.smoothIterations - 1);
        smoothLambda = std::clamp(config.smoothLambda * 0.58f, 0.08f, 0.32f);
    }
    detail::taubinSmooth(&mesh, smoothIters, smoothLambda);

    progress("正在重算法线...", 0.90f);
    detail::recomputeNormals(&mesh);

    progress("网格重建完成", 1.0f);
    outMesh = std::move(mesh);
    return true;
}

} // namespace mesh
} // namespace xjw
