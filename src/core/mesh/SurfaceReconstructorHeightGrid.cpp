#include "SurfaceReconstructorHeightGrid.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/height_grid.h>
#include <plapoint/mesh/height_grid.h>

namespace xjw
{
namespace mesh
{
namespace detail
{

namespace
{

using PlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
using PlaHeightGrid = plapoint::mesh::HeightGrid<float>;

PlaCloud toPlaCloud(const std::vector<PointXYZRGB> &points)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> xyz(
        static_cast<plamatrix::Index>(points.size()), 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
        static_cast<plamatrix::Index>(points.size()), 3);
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        xyz.setValue(row, 0, points[i].x);
        xyz.setValue(row, 1, points[i].y);
        xyz.setValue(row, 2, points[i].z);
        colors.setValue(row, 0, points[i].r);
        colors.setValue(row, 1, points[i].g);
        colors.setValue(row, 2, points[i].b);
    }

    PlaCloud cloud(std::move(xyz));
    cloud.setColors(std::move(colors));
    return cloud;
}

PlaHeightGrid toPlaGrid(const HeightGrid &heightGrid)
{
    PlaHeightGrid out;
    out.width = heightGrid.nx;
    out.height = heightGrid.ny;
    out.minX = heightGrid.minX;
    out.minY = heightGrid.minY;
    out.stepX = heightGrid.stepX;
    out.stepY = heightGrid.stepY;
    out.heights = heightGrid.heights;
    out.weights = heightGrid.sumW;
    out.valid = heightGrid.valid;
    out.fillPass = heightGrid.fillPass;
    return out;
}

HeightGrid fromPlaGrid(const PlaHeightGrid &heightGrid)
{
    HeightGrid out;
    out.nx = heightGrid.width;
    out.ny = heightGrid.height;
    out.minX = heightGrid.minX;
    out.minY = heightGrid.minY;
    out.stepX = heightGrid.stepX;
    out.stepY = heightGrid.stepY;
    out.heights = heightGrid.heights;
    out.sumW = heightGrid.weights;
    out.valid = heightGrid.valid;
    out.fillPass = heightGrid.fillPass;
    return out;
}

plapoint::mesh::HeightGridOptions<float> makeHeightGridOptions(
    const std::vector<PointXYZRGB> &points,
    const ReconstructionConfig &config)
{
    plapoint::mesh::HeightGridOptions<float> options;
    options.padding = std::max(0.0f, config.padding);
    options.maxFillPassForFaces = std::max(0, config.holeFillPasses);

    if (points.empty())
    {
        options.width = 0;
        options.height = 0;
        return options;
    }

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;
    for (const auto &point : points)
    {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    float dx = std::max(1e-6f, maxX - minX);
    float dy = std::max(1e-6f, maxY - minY);
    dx += dx * options.padding * 2.0f;
    dy += dy * options.padding * 2.0f;
    const int resolution = std::clamp(config.resolution, 32, 1024);
    const float maxSpan = std::max(dx, dy);
    options.width = std::max(8, static_cast<int>(std::round(resolution * dx / maxSpan)));
    options.height = std::max(8, static_cast<int>(std::round(resolution * dy / maxSpan)));
    return options;
}

PlaHeightGrid buildHeightGridWithRequestedDevice(
    const PlaCloud &cloud,
    const plapoint::mesh::HeightGridOptions<float> &options,
    plapoint::ProcessingDevice processingDevice)
{
    if (processingDevice == plapoint::ProcessingDevice::GPU ||
        processingDevice == plapoint::ProcessingDevice::Auto)
    {
#ifdef PLAPOINT_WITH_CUDA
        if (plapoint::gpu::hasUsableCudaDevice())
        {
            try
            {
                return plapoint::gpu::buildHeightGrid(cloud.toGpu(), options);
            }
            catch (const std::exception &)
            {
                if (processingDevice == plapoint::ProcessingDevice::GPU)
                {
                    throw;
                }
            }
        }
#endif
    }

    return plapoint::mesh::buildHeightGrid(cloud, options);
}

void assignMeshFromPlaCloud(const PlaCloud &source, TriMesh *mesh)
{
    if (!mesh)
    {
        return;
    }

    mesh->vertices.clear();
    mesh->faces.clear();
    mesh->vertices.reserve(source.size());
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        MeshVertex vertex;
        vertex.x = source.points().getValue(row, 0);
        vertex.y = source.points().getValue(row, 1);
        vertex.z = source.points().getValue(row, 2);
        vertex.nx = 0.0f;
        vertex.ny = 0.0f;
        vertex.nz = 1.0f;
        if (source.hasColors())
        {
            vertex.r = source.colors()->getValue(row, 0);
            vertex.g = source.colors()->getValue(row, 1);
            vertex.b = source.colors()->getValue(row, 2);
        }
        mesh->vertices.push_back(vertex);
    }

    if (!source.hasFaces())
    {
        return;
    }

    mesh->faces.reserve(static_cast<std::size_t>(source.faces()->rows()));
    for (plamatrix::Index row = 0; row < source.faces()->rows(); ++row)
    {
        Triangle triangle;
        triangle.v[0] = source.faces()->getValue(row, 0);
        triangle.v[1] = source.faces()->getValue(row, 1);
        triangle.v[2] = source.faces()->getValue(row, 2);
        mesh->faces.push_back(triangle);
    }
}

} // namespace

HeightGrid buildHeightGrid(const std::vector<PointXYZRGB> &points,
                           const ReconstructionConfig &config)
{
    if (points.empty())
    {
        return {};
    }

    const auto cloud = toPlaCloud(points);
    const auto options = makeHeightGridOptions(points, config);
    return fromPlaGrid(buildHeightGridWithRequestedDevice(cloud,
                                                         options,
                                                         config.preprocessingDevice));
}

void fillHoles(HeightGrid *heightGrid, int maxPasses)
{
    if (heightGrid == nullptr)
    {
        return;
    }

    auto plaGrid = toPlaGrid(*heightGrid);
    plapoint::mesh::fillHoles(plaGrid, maxPasses);
    *heightGrid = fromPlaGrid(plaGrid);
}

void heightGridToMesh(const HeightGrid &heightGrid,
                      const std::vector<PointXYZRGB> &points,
                      const ReconstructionConfig &config,
                      TriMesh *mesh)
{
    if (mesh == nullptr)
    {
        return;
    }

    const auto sourceCloud = toPlaCloud(points);
    auto options = makeHeightGridOptions(points, config);
    options.maxFillPassForFaces = std::max(0, config.holeFillPasses);
    const auto plaMesh = plapoint::mesh::heightGridToMesh(toPlaGrid(heightGrid), sourceCloud, options);
    assignMeshFromPlaCloud(plaMesh, mesh);
}

} // namespace detail
} // namespace mesh
} // namespace xjw
