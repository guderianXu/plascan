#include "SurfaceReconstructorHeightGrid.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/height_grid.h>
#include <plapoint/mesh/height_grid.h>
#ifdef PLAPOINT_WITH_OPENCL
#include <plapoint/opencl/height_grid.h>
#include <plapoint/opencl/opencl_runtime.h>
#endif

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

int checkedHeightGridDimension(double value, const char *axis)
{
    const double rounded = std::round(value);
    if (!std::isfinite(rounded) || rounded < 0.0 ||
        rounded > static_cast<double>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            std::string("SurfaceReconstructor height-grid: derived ") + axis +
            " dimension is outside the int range");
    }
    return std::max(8, static_cast<int>(rounded));
}

plapoint::mesh::HeightGridOptions<float> makeHeightGridOptions(
    const std::vector<PointXYZRGB> &points,
    const ReconstructionConfig &config)
{
    plapoint::mesh::HeightGridOptions<float> options;
    if (!std::isfinite(config.padding))
    {
        throw std::invalid_argument(
            "SurfaceReconstructor height-grid: padding must be finite");
    }
    options.padding = std::max(0.0f, config.padding);
    options.maxFillPassForFaces = std::max(0, config.holeFillPasses);

    if (points.empty())
    {
        options.width = 0;
        options.height = 0;
        return options;
    }

    double minX = static_cast<double>(points[0].x);
    double maxX = minX;
    double minY = static_cast<double>(points[0].y);
    double maxY = minY;
    for (const auto &point : points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z))
        {
            throw std::invalid_argument(
                "SurfaceReconstructor height-grid: points must be finite");
        }
        const double x = static_cast<double>(point.x);
        const double y = static_cast<double>(point.y);
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }

    const double rawDx = maxX - minX;
    const double rawDy = maxY - minY;
    const double dx = std::max(1.0e-6, rawDx);
    const double dy = std::max(1.0e-6, rawDy);
    const double padding = static_cast<double>(options.padding);
    const double paddingX = dx * padding;
    const double paddingY = dy * padding;
    const double paddedMinX = minX - paddingX;
    const double paddedMaxX = maxX + paddingX;
    const double paddedMinY = minY - paddingY;
    const double paddedMaxY = maxY + paddingY;
    const double floatLimit = static_cast<double>(std::numeric_limits<float>::max());
    if (!std::isfinite(rawDx) || !std::isfinite(rawDy) ||
        rawDx > floatLimit || rawDy > floatLimit ||
        !std::isfinite(paddingX) || !std::isfinite(paddingY) ||
        paddedMinX < -floatLimit || paddedMaxX > floatLimit ||
        paddedMinY < -floatLimit || paddedMaxY > floatLimit)
    {
        throw std::overflow_error(
            "SurfaceReconstructor height-grid: derived bounds exceed the float range");
    }

    const double paddedDx = dx + 2.0 * paddingX;
    const double paddedDy = dy + 2.0 * paddingY;
    if (!std::isfinite(paddedDx) || !std::isfinite(paddedDy) ||
        paddedDx <= 0.0 || paddedDy <= 0.0 ||
        paddedDx > floatLimit || paddedDy > floatLimit)
    {
        throw std::overflow_error(
            "SurfaceReconstructor height-grid: derived axis span must be finite and positive");
    }
    const int resolution = std::clamp(config.resolution, 32, 1024);
    const double maxSpan = std::max(paddedDx, paddedDy);
    options.width = checkedHeightGridDimension(
        static_cast<double>(resolution) * (paddedDx / maxSpan), "width");
    options.height = checkedHeightGridDimension(
        static_cast<double>(resolution) * (paddedDy / maxSpan), "height");
    return options;
}

PlaHeightGrid buildHeightGridWithRequestedDevice(
    const PlaCloud &cloud,
    const plapoint::mesh::HeightGridOptions<float> &options,
    plapoint::ProcessingDevice processingDevice)
{
    if (processingDevice == plapoint::ProcessingDevice::CUDA ||
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
                if (processingDevice == plapoint::ProcessingDevice::CUDA)
                {
                    throw;
                }
            }
        }
        else if (processingDevice == plapoint::ProcessingDevice::CUDA)
        {
            throw std::runtime_error("CUDA height-grid device is not available");
        }
#else
        if (processingDevice == plapoint::ProcessingDevice::CUDA)
        {
            throw std::runtime_error("PlaPoint was built without CUDA support");
        }
#endif
    }

    if (processingDevice == plapoint::ProcessingDevice::OpenCL ||
        processingDevice == plapoint::ProcessingDevice::Auto)
    {
#ifdef PLAPOINT_WITH_OPENCL
        if (plapoint::opencl::hasUsableOpenClDevice())
        {
            try
            {
                return plapoint::opencl::buildHeightGrid(cloud, options);
            }
            catch (const std::exception &)
            {
                if (processingDevice == plapoint::ProcessingDevice::OpenCL)
                {
                    throw;
                }
            }
        }
        else if (processingDevice == plapoint::ProcessingDevice::OpenCL)
        {
            throw std::runtime_error("OpenCL height-grid device is not available");
        }
#else
        if (processingDevice == plapoint::ProcessingDevice::OpenCL)
        {
            throw std::runtime_error("PlaPoint was built without OpenCL support");
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
