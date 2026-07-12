#include "VisualHullReconstructor.h"
#include "SurfaceReconstructorPostprocess.h"

#include <plapoint/mesh/marching_cubes.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>

#if defined(MESHING_OPENMP)
#include <omp.h>
#endif

namespace xjw::mesh
{
namespace
{

int resolveWorkerCount(int requested)
{
    if (requested > 0)
    {
        return std::clamp(requested, 1, 128);
    }
    const unsigned int hardware = std::thread::hardware_concurrency();
    return std::clamp(static_cast<int>(hardware > 0 ? hardware : 8U), 1, 128);
}

bool validBounds(const VisualHullConfig &config)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(config.boundsMin[axis]) || !std::isfinite(config.boundsMax[axis]) ||
            config.boundsMin[axis] >= config.boundsMax[axis])
        {
            return false;
        }
    }
    return true;
}

bool isOccupied(float worldX,
                float worldY,
                float worldZ,
                const std::vector<VisualHullView> &views,
                const VisualHullConfig &config)
{
    int visibleViews = 0;
    int silhouetteViolations = 0;
    int freeSpaceViolations = 0;
    for (const VisualHullView &view : views)
    {
        float pixelX = 0.0f;
        float pixelY = 0.0f;
        float cameraDepth = 0.0f;
        if (!view.camera.projectWithDepth(worldX, worldY, worldZ, pixelX, pixelY, cameraDepth))
        {
            continue;
        }

        const int column = static_cast<int>(std::lround(pixelX));
        const int row = static_cast<int>(std::lround(pixelY));
        if (row < 0 || column < 0 || row >= view.silhouetteMask.rows ||
            column >= view.silhouetteMask.cols)
        {
            continue;
        }

        ++visibleViews;
        if (view.silhouetteMask.at<std::uint8_t>(row, column) == 0)
        {
            ++silhouetteViolations;
            if (silhouetteViolations > config.allowedSilhouetteViolations)
            {
                return false;
            }
            continue;
        }

        if (config.enableDepthFreeSpaceCarving && !view.depthMap.empty() &&
            row < view.depthMap.rows && column < view.depthMap.cols)
        {
            const float measuredDepth = view.depthMap.at<float>(row, column);
            if (std::isfinite(measuredDepth) && measuredDepth > 0.0f)
            {
                const float tolerance = std::max(1.0e-6f,
                                                 measuredDepth * config.relativeDepthTolerance);
                if (cameraDepth < measuredDepth - tolerance)
                {
                    ++freeSpaceViolations;
                    if (freeSpaceViolations >= config.minimumDepthFreeSpaceViolations)
                    {
                        return false;
                    }
                }
            }
        }
    }

    return visibleViews >= config.minimumVisibleViews &&
           silhouetteViolations <= config.allowedSilhouetteViolations;
}

void recomputeNormalsAndColors(TriMesh *mesh, const std::vector<VisualHullView> &views)
{
    if (!mesh || mesh->empty())
    {
        return;
    }

    float centroidX = 0.0f;
    float centroidY = 0.0f;
    float centroidZ = 0.0f;
    for (const MeshVertex &vertex : mesh->vertices)
    {
        centroidX += vertex.x;
        centroidY += vertex.y;
        centroidZ += vertex.z;
    }
    const float inverseVertexCount = 1.0f / static_cast<float>(mesh->vertices.size());
    centroidX *= inverseVertexCount;
    centroidY *= inverseVertexCount;
    centroidZ *= inverseVertexCount;

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();
    for (const MeshVertex &vertex : mesh->vertices)
    {
        minX = std::min(minX, vertex.x);
        minY = std::min(minY, vertex.y);
        minZ = std::min(minZ, vertex.z);
        maxX = std::max(maxX, vertex.x);
        maxY = std::max(maxY, vertex.y);
        maxZ = std::max(maxZ, vertex.z);
    }
    const float visibilityTolerance =
        std::max({maxX - minX, maxY - minY, maxZ - minZ}) / 80.0f;

    std::vector<cv::Mat> visibilityDepths;
    visibilityDepths.reserve(views.size());
    for (const VisualHullView &view : views)
    {
        if (view.colorImage.empty())
        {
            visibilityDepths.emplace_back();
            continue;
        }
        cv::Mat depth(view.colorImage.size(), CV_32F,
                      cv::Scalar(std::numeric_limits<float>::infinity()));
        for (const MeshVertex &vertex : mesh->vertices)
        {
            float pixelX = 0.0f;
            float pixelY = 0.0f;
            float cameraDepth = 0.0f;
            if (!view.camera.projectWithDepth(vertex.x, vertex.y, vertex.z,
                                              pixelX, pixelY, cameraDepth))
            {
                continue;
            }
            const int column = static_cast<int>(std::lround(pixelX));
            const int row = static_cast<int>(std::lround(pixelY));
            if (row >= 0 && column >= 0 && row < depth.rows && column < depth.cols)
            {
                depth.at<float>(row, column) = std::min(depth.at<float>(row, column), cameraDepth);
            }
        }
        cv::erode(depth, depth, cv::Mat::ones(3, 3, CV_8UC1));
        visibilityDepths.push_back(std::move(depth));
    }

    for (MeshVertex &vertex : mesh->vertices)
    {
        vertex.nx = 0.0f;
        vertex.ny = 0.0f;
        vertex.nz = 0.0f;
    }
    for (const Triangle &face : mesh->faces)
    {
        MeshVertex &a = mesh->vertices[static_cast<std::size_t>(face.v[0])];
        MeshVertex &b = mesh->vertices[static_cast<std::size_t>(face.v[1])];
        MeshVertex &c = mesh->vertices[static_cast<std::size_t>(face.v[2])];
        const float ux = b.x - a.x;
        const float uy = b.y - a.y;
        const float uz = b.z - a.z;
        const float vx = c.x - a.x;
        const float vy = c.y - a.y;
        const float vz = c.z - a.z;
        const float nx = uy * vz - uz * vy;
        const float ny = uz * vx - ux * vz;
        const float nz = ux * vy - uy * vx;
        for (MeshVertex *vertex : {&a, &b, &c})
        {
            vertex->nx += nx;
            vertex->ny += ny;
            vertex->nz += nz;
        }
    }

    for (MeshVertex &vertex : mesh->vertices)
    {
        const float normalLength = std::sqrt(vertex.nx * vertex.nx +
                                             vertex.ny * vertex.ny +
                                             vertex.nz * vertex.nz);
        if (normalLength > 1.0e-8f)
        {
            vertex.nx /= normalLength;
            vertex.ny /= normalLength;
            vertex.nz /= normalLength;
        }
        const float outwardScore = vertex.nx * (vertex.x - centroidX) +
                                    vertex.ny * (vertex.y - centroidY) +
                                    vertex.nz * (vertex.z - centroidZ);
        if (outwardScore < 0.0f)
        {
            vertex.nx = -vertex.nx;
            vertex.ny = -vertex.ny;
            vertex.nz = -vertex.nz;
        }

        float accumulatedWeight = 0.0f;
        float accumulatedRed = 0.0f;
        float accumulatedGreen = 0.0f;
        float accumulatedBlue = 0.0f;
        for (std::size_t viewIndex = 0; viewIndex < views.size(); ++viewIndex)
        {
            const VisualHullView &view = views[viewIndex];
            if (view.colorImage.empty())
            {
                continue;
            }
            float pixelX = 0.0f;
            float pixelY = 0.0f;
            float depth = 0.0f;
            if (!view.camera.projectWithDepth(vertex.x, vertex.y, vertex.z,
                                              pixelX, pixelY, depth))
            {
                continue;
            }
            const int column = static_cast<int>(std::lround(pixelX));
            const int row = static_cast<int>(std::lround(pixelY));
            if (row < 0 || column < 0 || row >= view.colorImage.rows ||
                column >= view.colorImage.cols)
            {
                continue;
            }
            if (!view.silhouetteMask.empty() &&
                view.silhouetteMask.at<std::uint8_t>(row, column) == 0)
            {
                continue;
            }
            if (viewIndex < visibilityDepths.size() && !visibilityDepths[viewIndex].empty())
            {
                const float nearestDepth = visibilityDepths[viewIndex].at<float>(row, column);
                if (std::isfinite(nearestDepth) && depth > nearestDepth + visibilityTolerance)
                {
                    continue;
                }
            }

            float directionX = view.camera.C[0] - vertex.x;
            float directionY = view.camera.C[1] - vertex.y;
            float directionZ = view.camera.C[2] - vertex.z;
            const float directionLength = std::sqrt(directionX * directionX +
                                                    directionY * directionY +
                                                    directionZ * directionZ);
            if (directionLength <= 1.0e-8f)
            {
                continue;
            }
            directionX /= directionLength;
            directionY /= directionLength;
            directionZ /= directionLength;
            const float score = vertex.nx * directionX +
                                vertex.ny * directionY +
                                vertex.nz * directionZ;
            if (score <= 0.10f)
            {
                continue;
            }

            const float weight = score * score * score * score;

            if (view.colorImage.type() == CV_8UC3)
            {
                const cv::Vec3b color = view.colorImage.at<cv::Vec3b>(row, column);
                accumulatedRed += weight * color[2];
                accumulatedGreen += weight * color[1];
                accumulatedBlue += weight * color[0];
                accumulatedWeight += weight;
            }
            else if (view.colorImage.type() == CV_8UC1)
            {
                const std::uint8_t color = view.colorImage.at<std::uint8_t>(row, column);
                accumulatedRed += weight * color;
                accumulatedGreen += weight * color;
                accumulatedBlue += weight * color;
                accumulatedWeight += weight;
            }
        }
        if (accumulatedWeight > 1.0e-8f)
        {
            vertex.r = static_cast<std::uint8_t>(std::clamp(
                accumulatedRed / accumulatedWeight, 0.0f, 255.0f));
            vertex.g = static_cast<std::uint8_t>(std::clamp(
                accumulatedGreen / accumulatedWeight, 0.0f, 255.0f));
            vertex.b = static_cast<std::uint8_t>(std::clamp(
                accumulatedBlue / accumulatedWeight, 0.0f, 255.0f));
        }
    }
}

} // namespace

bool VisualHullReconstructor::reconstruct(const std::vector<VisualHullView> &views,
                                          const VisualHullConfig &config,
                                          TriMesh *mesh,
                                          std::string *errorMessage)
{
    if (!mesh)
    {
        if (errorMessage)
        {
            *errorMessage = "visual hull mesh output is null";
        }
        return false;
    }
    *mesh = TriMesh{};

    const int validViewCount = static_cast<int>(std::count_if(
        views.begin(), views.end(), [](const VisualHullView &view)
        {
            return view.camera.valid() && view.silhouetteMask.type() == CV_8UC1 &&
                   !view.silhouetteMask.empty();
        }));
    if (!validBounds(config) || config.resolution < 8 || validViewCount < 2)
    {
        if (errorMessage)
        {
            *errorMessage = "visual hull configuration or input views are invalid";
        }
        return false;
    }

    const int resolution = std::clamp(config.resolution, 8, 256);
    const int gridSize = resolution + 1;
    const std::size_t layerSize = static_cast<std::size_t>(gridSize) * gridSize;
    std::vector<float> field(layerSize * gridSize, 1.0f);
    const float stepX = (config.boundsMax[0] - config.boundsMin[0]) / resolution;
    const float stepY = (config.boundsMax[1] - config.boundsMin[1]) / resolution;
    const float stepZ = (config.boundsMax[2] - config.boundsMin[2]) / resolution;
    std::atomic_bool cancelled{false};
    const int workers = resolveWorkerCount(config.workerCount);

#if defined(MESHING_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
#endif
    for (int zIndex = 0; zIndex < gridSize; ++zIndex)
    {
        if (cancelled.load(std::memory_order_relaxed))
        {
            continue;
        }
        if (config.isCancelled && config.isCancelled())
        {
            cancelled.store(true, std::memory_order_relaxed);
            continue;
        }
        const float worldZ = config.boundsMin[2] + stepZ * zIndex;
        for (int yIndex = 0; yIndex < gridSize; ++yIndex)
        {
            const float worldY = config.boundsMin[1] + stepY * yIndex;
            for (int xIndex = 0; xIndex < gridSize; ++xIndex)
            {
                const float worldX = config.boundsMin[0] + stepX * xIndex;
                const std::size_t offset = static_cast<std::size_t>(zIndex) * layerSize +
                                           static_cast<std::size_t>(yIndex) * gridSize + xIndex;
                field[offset] = isOccupied(worldX, worldY, worldZ, views, config) ? -1.0f : 1.0f;
            }
        }
    }
    if (cancelled.load(std::memory_order_relaxed))
    {
        if (errorMessage)
        {
            *errorMessage = "visual hull reconstruction cancelled";
        }
        return false;
    }
    if (config.progressFn)
    {
        config.progressFn("正在提取视觉外壳表面...", 0.75f);
    }

    try
    {
        plapoint::mesh::MarchingCubes<float> marchingCubes;
        marchingCubes.setBounds({config.boundsMin[0], config.boundsMin[1], config.boundsMin[2]},
                                {config.boundsMax[0], config.boundsMax[1], config.boundsMax[2]});
        marchingCubes.setResolution(resolution, resolution, resolution);
        marchingCubes.setIsoLevel(0.0f);
        auto [vertices, faces] = marchingCubes.extract(
            [&](float x, float y, float z)
            {
                const int xIndex = std::clamp(static_cast<int>(std::lround((x - config.boundsMin[0]) / stepX)),
                                              0, resolution);
                const int yIndex = std::clamp(static_cast<int>(std::lround((y - config.boundsMin[1]) / stepY)),
                                              0, resolution);
                const int zIndex = std::clamp(static_cast<int>(std::lround((z - config.boundsMin[2]) / stepZ)),
                                              0, resolution);
                return field[static_cast<std::size_t>(zIndex) * layerSize +
                             static_cast<std::size_t>(yIndex) * gridSize + xIndex];
            });

        mesh->vertices.resize(static_cast<std::size_t>(vertices.rows()));
        for (plamatrix::Index row = 0; row < vertices.rows(); ++row)
        {
            MeshVertex &vertex = mesh->vertices[static_cast<std::size_t>(row)];
            vertex.x = vertices(row, 0);
            vertex.y = vertices(row, 1);
            vertex.z = vertices(row, 2);
        }
        mesh->faces.resize(static_cast<std::size_t>(faces.rows()));
        for (plamatrix::Index row = 0; row < faces.rows(); ++row)
        {
            Triangle &face = mesh->faces[static_cast<std::size_t>(row)];
            face.v[0] = static_cast<int>(std::lround(faces(row, 0)));
            face.v[1] = static_cast<int>(std::lround(faces(row, 1)));
            face.v[2] = static_cast<int>(std::lround(faces(row, 2)));
        }
    }
    catch (const std::exception &exception)
    {
        if (errorMessage)
        {
            *errorMessage = exception.what();
        }
        return false;
    }

    if (mesh->empty())
    {
        if (errorMessage)
        {
            *errorMessage = "visual hull extraction produced an empty mesh";
        }
        return false;
    }
    detail::weldCoincidentVertices(mesh, 1.0e-7f);
    detail::removeDegenerateFaces(mesh);
    detail::removeSmallConnectedComponents(mesh, 64);
    detail::taubinSmooth(mesh, 2, 0.18f);
    recomputeNormalsAndColors(mesh, views);
    if (config.progressFn)
    {
        config.progressFn("视觉外壳重建完成", 1.0f);
    }
    return true;
}

} // namespace xjw::mesh
