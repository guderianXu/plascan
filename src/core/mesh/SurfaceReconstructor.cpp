#include "SurfaceReconstructor.h"
#include "PointCloudPreprocess.h"
#include "SurfaceReconstructorHeightGrid.h"
#include "SurfaceReconstructorPostprocess.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plamatrix/dense/dense_matrix.h>

#include <algorithm>
#include <exception>
#include <string>
#include <vector>

namespace xjw
{
namespace mesh
{

namespace
{

using PlaPointCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

std::vector<detail::PointXYZRGB> cloudToPointXYZRGB(const PlaPointCloud &cloud)
{
    std::vector<detail::PointXYZRGB> points;
    points.reserve(static_cast<std::size_t>(cloud.size()));
    const bool hasColors = cloud.hasColors();
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        auto pt = cloud[i];
        detail::PointXYZRGB p;
        p.x = pt.x();
        p.y = pt.y();
        p.z = pt.z();
        if (hasColors)
        {
            p.r = pt.r();
            p.g = pt.g();
            p.b = pt.b();
        }
        points.push_back(p);
    }
    return points;
}

PlaPointCloud pointXYZRGBToCloud(const std::vector<detail::PointXYZRGB> &points)
{
    const auto n = static_cast<plamatrix::Index>(points.size());
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(n, 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(n, 3);
    for (plamatrix::Index i = 0; i < n; ++i)
    {
        pts(i, 0) = points[static_cast<std::size_t>(i)].x;
        pts(i, 1) = points[static_cast<std::size_t>(i)].y;
        pts(i, 2) = points[static_cast<std::size_t>(i)].z;
        colors(i, 0) = points[static_cast<std::size_t>(i)].r;
        colors(i, 1) = points[static_cast<std::size_t>(i)].g;
        colors(i, 2) = points[static_cast<std::size_t>(i)].b;
    }
    PlaPointCloud cloud(std::move(pts));
    cloud.setColors(std::move(colors));
    return cloud;
}

bool convertPoissonResultToMesh(const plamatrix::DenseMatrix<float, plamatrix::Device::CPU> &verts,
                                 const plamatrix::DenseMatrix<float, plamatrix::Device::CPU> &faces,
                                 TriMesh *mesh)
{
    if (!mesh)
    {
        return false;
    }

    mesh->vertices.clear();
    mesh->faces.clear();

    const int vertexCount = static_cast<int>(verts.rows());
    const int faceCount = static_cast<int>(faces.rows());

    mesh->vertices.reserve(static_cast<std::size_t>(vertexCount));
    for (int i = 0; i < vertexCount; ++i)
    {
        MeshVertex v;
        v.x = verts(static_cast<plamatrix::Index>(i), 0);
        v.y = verts(static_cast<plamatrix::Index>(i), 1);
        v.z = verts(static_cast<plamatrix::Index>(i), 2);
        v.nx = 0.0f;
        v.ny = 0.0f;
        v.nz = 1.0f;
        mesh->vertices.push_back(v);
    }

    mesh->faces.reserve(static_cast<std::size_t>(faceCount));
    for (int i = 0; i < faceCount; ++i)
    {
        Triangle t;
        t.v[0] = static_cast<int>(faces(static_cast<plamatrix::Index>(i), 0));
        t.v[1] = static_cast<int>(faces(static_cast<plamatrix::Index>(i), 1));
        t.v[2] = static_cast<int>(faces(static_cast<plamatrix::Index>(i), 2));
        mesh->faces.push_back(t);
    }

    return true;
}

} // namespace

bool SurfaceReconstructor::reconstructFromPointCloudFile(const std::string &cloudPath,
                                                         const ReconstructionConfig &config,
                                                         TriMesh &outMesh,
                                                         std::string *errorMsg)
{
    auto progress = [&](const std::string &stage, float p) {
        if (config.progressFn)
        {
            config.progressFn(stage, p);
        }
    };

    progress("正在加载点云...", 0.02f);
    auto cloudPtr = plapoint::io::readPly<float>(cloudPath);
    if (!cloudPtr || cloudPtr->size() < 100)
    {
        if (errorMsg)
        {
            *errorMsg = cloudPtr ? "点云点数过少，无法稳定重建" : "无法加载点云文件";
        }
        return false;
    }

    std::vector<detail::PointXYZRGB> points = cloudToPointXYZRGB(*cloudPtr);

    const float baseVoxel = detail::estimateBaseVoxelStep(points, config.resolution);
    if (config.enableDenoise)
    {
        progress("正在点云去噪...", 0.08f);
        points = detail::statisticalDenoisePoints(points,
                                                   std::clamp(config.denoiseK, 8, 64),
                                                   std::clamp(config.denoiseStdMul, 0.6f, 3.0f),
                                                   baseVoxel * 2.0f);
    }

    if (config.enableDownsample)
    {
        progress("正在点云下采样...", 0.12f);
        const float voxelSize = baseVoxel * std::clamp(config.downsampleVoxelScale, 0.4f, 2.5f);
        points = detail::voxelDownsamplePoints(points, voxelSize);
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

    if (config.forcePoisson)
    {
        progress("正在执行 Poisson 重建...", 0.30f);
        try
        {
            PlaPointCloud poissonCloud = pointXYZRGBToCloud(points);
            auto poissonCloudPtr = std::make_shared<const PlaPointCloud>(std::move(poissonCloud));
            plapoint::mesh::PoissonReconstruction<float> poisson;
            poisson.setInputCloud(poissonCloudPtr);
            poisson.setDepth(std::clamp(config.poissonDepth, 1, 8));
            auto [verts, faces] = poisson.reconstruct();
            convertPoissonResultToMesh(verts, faces, &mesh);
        }
        catch (const std::exception &)
        {
            mesh = TriMesh{};
            progress("Poisson 重建失败，改用高度格网...", 0.34f);
        }

        if (!mesh.empty())
        {
            // Light simplification for Poisson output if extremely dense
            const int poissonFaceCount = mesh.faceCount();
            const int hardUpperBound = std::max(120000, std::max(1, config.simplifyTargetFaces) * 4);
            if (poissonFaceCount > hardUpperBound)
            {
                progress("正在轻量简化 Poisson 网格...", 0.62f);
                ReconstructionConfig liteConfig = config;
                liteConfig.simplifyTargetFaces = std::max(config.simplifyTargetFaces * 2, 80000);
                liteConfig.voxelSimplifyFactor = std::clamp(config.voxelSimplifyFactor * 0.85f, 1.0f, 2.4f);
                detail::simplifyVoxelMeshAdaptive(&mesh, liteConfig, baseVoxel);
            }
        }
    }

    if (mesh.empty())
    {
        usedHeightGridFallback = true;
        progress("正在构建高度格网...", 0.10f);
        detail::HeightGrid hg = detail::buildHeightGrid(points, config);
        if (hg.nx < 4 || hg.ny < 4)
        {
            if (errorMsg)
            {
                *errorMsg = "高程格网构建失败（点云范围过小）";
            }
            return false;
        }

        if (config.fillHoles)
        {
            progress("正在填充空洞...", 0.30f);
            detail::fillHoles(&hg, std::max(0, config.holeFillPasses));
        }

        progress("正在三角分割...", 0.50f);
        detail::heightGridToMesh(hg, points, config, &mesh);
    }

    if (mesh.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "网格化失败（结果为空）";
        }
        return false;
    }

    progress("正在清理退化面...", 0.64f);
    detail::removeDegenerateFaces(&mesh);

    if (config.cleanSmallComponents)
    {
        progress("正在清理碎片连通体...", 0.70f);
        detail::removeSmallConnectedComponents(&mesh, std::max(2, config.minComponentFaces));
        if (mesh.empty())
        {
            if (errorMsg)
            {
                *errorMsg = "网格清理后为空，请降低连通体阈值或提高分辨率";
            }
            return false;
        }
    }

    progress("正在平滑网格...", 0.75f);
    int smoothIters = config.smoothIterations;
    float smoothLambda = config.smoothLambda;
    if (!usedHeightGridFallback)
    {
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
