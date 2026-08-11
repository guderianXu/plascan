#include "VisualHullReconstructor.h"
#include "Mc33IsoSurfaceExtractor.h"
#include "SurfaceReconstructorPostprocess.h"
#include "VisualHullFieldEvaluator.h"

#include <plapoint/mesh/marching_cubes.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

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

std::uint64_t edgeKey(int first, int second)
{
    const std::uint32_t low = static_cast<std::uint32_t>(std::min(first, second));
    const std::uint32_t high = static_cast<std::uint32_t>(std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

std::vector<int> edgeConnectedFaceRoots(const TriMesh &mesh)
{
    std::vector<int> parent(mesh.faces.size());
    std::iota(parent.begin(), parent.end(), 0);
    const auto find_root = [&parent](int face)
    {
        int root = face;
        while (parent[static_cast<std::size_t>(root)] != root)
        {
            root = parent[static_cast<std::size_t>(root)];
        }
        while (parent[static_cast<std::size_t>(face)] != face)
        {
            const int next = parent[static_cast<std::size_t>(face)];
            parent[static_cast<std::size_t>(face)] = root;
            face = next;
        }
        return root;
    };
    const auto unite = [&parent, &find_root](int left, int right)
    {
        const int left_root = find_root(left);
        const int right_root = find_root(right);
        if (left_root != right_root)
        {
            parent[static_cast<std::size_t>(right_root)] = left_root;
        }
    };

    std::unordered_map<std::uint64_t, int> first_face_by_edge;
    first_face_by_edge.reserve(mesh.faces.size() * 2);
    for (int face_index = 0; face_index < static_cast<int>(mesh.faces.size()); ++face_index)
    {
        const Triangle &face = mesh.faces[static_cast<std::size_t>(face_index)];
        const bool valid =
            face.v[0] >= 0 && face.v[1] >= 0 && face.v[2] >= 0 &&
            face.v[0] < static_cast<int>(mesh.vertices.size()) &&
            face.v[1] < static_cast<int>(mesh.vertices.size()) &&
            face.v[2] < static_cast<int>(mesh.vertices.size());
        if (!valid)
        {
            continue;
        }
        const std::array<std::array<int, 2>, 3> edges{{
            {{face.v[0], face.v[1]}},
            {{face.v[1], face.v[2]}},
            {{face.v[2], face.v[0]}}}};
        for (const auto &edge : edges)
        {
            const std::uint64_t key = edgeKey(edge[0], edge[1]);
            const auto [it, inserted] = first_face_by_edge.emplace(key, face_index);
            if (!inserted)
            {
                unite(face_index, it->second);
            }
        }
    }
    for (int face_index = 0; face_index < static_cast<int>(parent.size()); ++face_index)
    {
        parent[static_cast<std::size_t>(face_index)] = find_root(face_index);
    }
    return parent;
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
        const double world[3] = {worldX, worldY, worldZ};
        double pixel[2] = {};
        double cameraDepth = 0.0;
        if (!view.camera.projectWorldPointWithDepth(world, pixel, cameraDepth))
        {
            continue;
        }

        const int column = static_cast<int>(std::lround(pixel[0]));
        const int row = static_cast<int>(std::lround(pixel[1]));
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
            const double world[3] = {vertex.x, vertex.y, vertex.z};
            double pixel[2] = {};
            double cameraDepth = 0.0;
            if (!view.camera.projectWorldPointWithDepth(world, pixel, cameraDepth))
            {
                continue;
            }
            const int column = static_cast<int>(std::lround(pixel[0]));
            const int row = static_cast<int>(std::lround(pixel[1]));
            if (row >= 0 && column >= 0 && row < depth.rows && column < depth.cols)
            {
                depth.at<float>(row, column) = std::min(
                    depth.at<float>(row, column), static_cast<float>(cameraDepth));
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
            const double world[3] = {vertex.x, vertex.y, vertex.z};
            double pixel[2] = {};
            double depth = 0.0;
            if (!view.camera.projectWorldPointWithDepth(world, pixel, depth))
            {
                continue;
            }
            const int column = static_cast<int>(std::lround(pixel[0]));
            const int row = static_cast<int>(std::lround(pixel[1]));
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

            const std::array<double, 3> center = view.camera.cameraCenter();
            float directionX = static_cast<float>(center[0]) - vertex.x;
            float directionY = static_cast<float>(center[1]) - vertex.y;
            float directionZ = static_cast<float>(center[2]) - vertex.z;
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

void closeOccupiedField(
    std::vector<float> *field,
    int gridSize,
    int iterations,
    int workers)
{
    if (field == nullptr || gridSize < 3 || iterations <= 0)
    {
        return;
    }
    const std::size_t layer_size =
        static_cast<std::size_t>(gridSize) * gridSize;
    const auto offset = [gridSize, layer_size](
                            int x, int y, int z)
    {
        return static_cast<std::size_t>(z) * layer_size +
               static_cast<std::size_t>(y) * gridSize +
               static_cast<std::size_t>(x);
    };
    std::vector<std::uint8_t> occupied(field->size(), 0);
    for (std::size_t index = 0; index < field->size(); ++index)
    {
        occupied[index] = (*field)[index] < 0.0f ? 1 : 0;
    }
    std::vector<std::uint8_t> updated(field->size(), 0);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        std::fill(updated.begin(), updated.end(), 0);
#if defined(MESHING_OPENMP)
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
        for (int z = 1; z < gridSize - 1; ++z)
        {
            for (int y = 1; y < gridSize - 1; ++y)
            {
                for (int x = 1; x < gridSize - 1; ++x)
                {
                    const std::size_t center = offset(x, y, z);
                    updated[center] =
                        occupied[center] ||
                        occupied[offset(x - 1, y, z)] ||
                        occupied[offset(x + 1, y, z)] ||
                        occupied[offset(x, y - 1, z)] ||
                        occupied[offset(x, y + 1, z)] ||
                        occupied[offset(x, y, z - 1)] ||
                        occupied[offset(x, y, z + 1)];
                }
            }
        }
        occupied.swap(updated);
    }
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        std::fill(updated.begin(), updated.end(), 0);
#if defined(MESHING_OPENMP)
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
        for (int z = 1; z < gridSize - 1; ++z)
        {
            for (int y = 1; y < gridSize - 1; ++y)
            {
                for (int x = 1; x < gridSize - 1; ++x)
                {
                    const std::size_t center = offset(x, y, z);
                    updated[center] =
                        occupied[center] &&
                        occupied[offset(x - 1, y, z)] &&
                        occupied[offset(x + 1, y, z)] &&
                        occupied[offset(x, y - 1, z)] &&
                        occupied[offset(x, y + 1, z)] &&
                        occupied[offset(x, y, z - 1)] &&
                        occupied[offset(x, y, z + 1)];
                }
            }
        }
        occupied.swap(updated);
    }
    for (std::size_t index = 0; index < field->size(); ++index)
    {
        const float magnitude = std::max(
            1.0e-6f,
            std::abs((*field)[index]));
        (*field)[index] = occupied[index]
            ? -magnitude
            : magnitude;
    }
}

} // namespace

MeshConnectivityStats VisualHullReconstructor::analyzeConnectivity(const TriMesh &mesh)
{
    MeshConnectivityStats stats;
    if (mesh.vertices.empty() || mesh.faces.empty())
    {
        return stats;
    }

    const std::vector<int> component_roots = edgeConnectedFaceRoots(mesh);

    struct ComponentAccumulator
    {
        std::size_t faceCount = 0;
        std::array<float, 3> minimum{
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()};
        std::array<float, 3> maximum{
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest()};
    };

    std::unordered_map<int, ComponentAccumulator> component_accumulators;
    for (int face_index = 0; face_index < static_cast<int>(mesh.faces.size()); ++face_index)
    {
        const Triangle &face = mesh.faces[static_cast<std::size_t>(face_index)];
        if (face.v[0] < 0 || face.v[0] >= static_cast<int>(mesh.vertices.size()))
        {
            continue;
        }
        ComponentAccumulator &component =
            component_accumulators[component_roots[static_cast<std::size_t>(face_index)]];
        ++component.faceCount;
        for (const int vertex_index : face.v)
        {
            if (vertex_index < 0 || vertex_index >= static_cast<int>(mesh.vertices.size()))
            {
                continue;
            }
            const MeshVertex &vertex = mesh.vertices[static_cast<std::size_t>(vertex_index)];
            const std::array<float, 3> point{vertex.x, vertex.y, vertex.z};
            for (int axis = 0; axis < 3; ++axis)
            {
                component.minimum[axis] = std::min(component.minimum[axis], point[axis]);
                component.maximum[axis] = std::max(component.maximum[axis], point[axis]);
            }
        }
    }
    stats.componentCount = static_cast<int>(component_accumulators.size());
    for (const auto &[root, component] : component_accumulators)
    {
        (void)root;
        stats.largestComponentFaceCount = std::max(stats.largestComponentFaceCount,
                                                   component.faceCount);
        stats.componentFaceCounts.push_back(component.faceCount);
        MeshConnectivityStats::Component diagnostic;
        diagnostic.faceCount = component.faceCount;
        diagnostic.boundsMin = component.minimum;
        diagnostic.boundsMax = component.maximum;
        const double dx = static_cast<double>(component.maximum[0] - component.minimum[0]);
        const double dy = static_cast<double>(component.maximum[1] - component.minimum[1]);
        const double dz = static_cast<double>(component.maximum[2] - component.minimum[2]);
        diagnostic.diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);
        stats.components.push_back(diagnostic);
    }
    std::sort(stats.componentFaceCounts.begin(), stats.componentFaceCounts.end(), std::greater<>());
    std::sort(stats.components.begin(), stats.components.end(),
              [](const MeshConnectivityStats::Component &left,
                 const MeshConnectivityStats::Component &right)
              {
                  return left.faceCount > right.faceCount;
              });
    stats.largestComponentFaceRatio = mesh.faces.empty()
        ? 0.0
        : static_cast<double>(stats.largestComponentFaceCount) /
              static_cast<double>(mesh.faces.size());
    return stats;
}

bool VisualHullReconstructor::requiresSilhouetteOnlyRetry(
    const MeshConnectivityStats &stats,
    double minimumLargestComponentFaceRatio,
    int maximumConnectedComponents)
{
    return stats.componentCount <= 0 ||
           stats.componentCount > std::max(1, maximumConnectedComponents) ||
           stats.largestComponentFaceRatio <
               std::clamp(minimumLargestComponentFaceRatio, 0.0, 1.0);
}

bool VisualHullReconstructor::retainLargestConnectedComponent(TriMesh *mesh)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty())
    {
        return false;
    }

    const std::vector<int> component_roots = edgeConnectedFaceRoots(*mesh);
    std::unordered_map<int, std::size_t> face_counts;
    for (int face_index = 0; face_index < static_cast<int>(mesh->faces.size()); ++face_index)
    {
        ++face_counts[component_roots[static_cast<std::size_t>(face_index)]];
    }
    const auto largest = std::max_element(
        face_counts.begin(), face_counts.end(), [](const auto &left, const auto &right)
        {
            return left.second < right.second;
        });
    if (largest == face_counts.end() || largest->second == mesh->faces.size())
    {
        return false;
    }

    std::vector<Triangle> retained_faces;
    retained_faces.reserve(largest->second);
    std::vector<bool> used_vertices(mesh->vertices.size(), false);
    for (int face_index = 0; face_index < static_cast<int>(mesh->faces.size()); ++face_index)
    {
        const Triangle &face = mesh->faces[static_cast<std::size_t>(face_index)];
        if (component_roots[static_cast<std::size_t>(face_index)] == largest->first)
        {
            retained_faces.push_back(face);
            used_vertices[static_cast<std::size_t>(face.v[0])] = true;
            used_vertices[static_cast<std::size_t>(face.v[1])] = true;
            used_vertices[static_cast<std::size_t>(face.v[2])] = true;
        }
    }

    std::vector<int> remap(mesh->vertices.size(), -1);
    std::vector<MeshVertex> retained_vertices;
    retained_vertices.reserve(mesh->vertices.size());
    for (std::size_t index = 0; index < mesh->vertices.size(); ++index)
    {
        if (used_vertices[index])
        {
            remap[index] = static_cast<int>(retained_vertices.size());
            retained_vertices.push_back(mesh->vertices[index]);
        }
    }
    for (Triangle &face : retained_faces)
    {
        face.v[0] = remap[static_cast<std::size_t>(face.v[0])];
        face.v[1] = remap[static_cast<std::size_t>(face.v[1])];
        face.v[2] = remap[static_cast<std::size_t>(face.v[2])];
    }
    mesh->vertices = std::move(retained_vertices);
    mesh->faces = std::move(retained_faces);
    return true;
}

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
            return view.camera.isValid() && view.silhouetteMask.type() == CV_8UC1 &&
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

    const int resolution = std::clamp(config.resolution, 8, 384);
    const int gridSize = resolution + 1;
    const std::size_t layerSize = static_cast<std::size_t>(gridSize) * gridSize;
    std::vector<float> field(layerSize * gridSize, 1.0f);
    const float stepX = (config.boundsMax[0] - config.boundsMin[0]) / resolution;
    const float stepY = (config.boundsMax[1] - config.boundsMin[1]) / resolution;
    const float stepZ = (config.boundsMax[2] - config.boundsMin[2]) / resolution;
    std::atomic_bool cancelled{false};
    std::atomic<int> completed_layer_count{0};
    std::mutex progress_mutex;
    int reported_progress_percent = 5;
    const int workers = resolveWorkerCount(config.workerCount);
    const std::vector<detail::PreparedVisualHullView>
        prepared_field_views =
            config.useContinuousSilhouetteField
            ? detail::prepareVisualHullFieldViews(views)
            : std::vector<detail::PreparedVisualHullView>{};

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
                const bool volume_boundary =
                    xIndex == 0 || yIndex == 0 || zIndex == 0 ||
                    xIndex == resolution || yIndex == resolution ||
                    zIndex == resolution;
                if (config.closeVolumeBoundary && volume_boundary)
                {
                    field[offset] = 1.0f;
                    continue;
                }
                field[offset] =
                    config.useContinuousSilhouetteField
                    ? detail::evaluateContinuousVisualHullField(
                          worldX,
                          worldY,
                          worldZ,
                          prepared_field_views,
                          config)
                    : (isOccupied(
                           worldX,
                           worldY,
                           worldZ,
                           views,
                           config)
                           ? -1.0f
                           : 1.0f);
            }
        }
        const int completed_layers =
            completed_layer_count.fetch_add(1, std::memory_order_relaxed) + 1;
        const int progress_percent =
            5 + completed_layers * 65 / gridSize;
        if (config.progressFn)
        {
            std::lock_guard<std::mutex> lock(progress_mutex);
            if (progress_percent > reported_progress_percent)
            {
                reported_progress_percent = progress_percent;
                config.progressFn(
                    "正在评估多视轮廓体素（" +
                        std::to_string(completed_layers) + "/" +
                        std::to_string(gridSize) + " 层）...",
                    static_cast<float>(progress_percent) / 100.0f);
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
    closeOccupiedField(
        &field,
        gridSize,
        std::clamp(config.topologyClosingIterations, 0, 3),
        workers);
    if (config.progressFn)
    {
        config.progressFn("正在提取视觉外壳表面...", 0.75f);
    }

    bool extracted = false;
    std::string topology_extraction_error;
    if (Mc33IsoSurfaceExtractor::isAvailable())
    {
        Mc33IsoSurfaceOptions extraction_options;
        extraction_options.isoLevel = 0.0f;
        extraction_options.isCancelled = config.isCancelled;
        Mc33IsoSurfaceResult extraction =
            Mc33IsoSurfaceExtractor::extract(
                config.boundsMin,
                config.boundsMax,
                {resolution, resolution, resolution},
                field,
                {},
                extraction_options);
        if (extraction.ok && !extraction.mesh.empty())
        {
            *mesh = std::move(extraction.mesh);
            extracted = true;
        }
        else
        {
            topology_extraction_error = extraction.errorMessage;
        }
    }

    try
    {
        if (!extracted)
        {
            plapoint::mesh::MarchingCubes<float> marchingCubes;
            marchingCubes.setBounds(
                {config.boundsMin[0], config.boundsMin[1], config.boundsMin[2]},
                {config.boundsMax[0], config.boundsMax[1], config.boundsMax[2]});
            marchingCubes.setResolution(resolution, resolution, resolution);
            marchingCubes.setIsoLevel(0.0f);
            auto [vertices, faces] = marchingCubes.extract(
                [&](float x, float y, float z)
                {
                    const int xIndex = std::clamp(
                        static_cast<int>(std::lround(
                            (x - config.boundsMin[0]) / stepX)),
                        0,
                        resolution);
                    const int yIndex = std::clamp(
                        static_cast<int>(std::lround(
                            (y - config.boundsMin[1]) / stepY)),
                        0,
                        resolution);
                    const int zIndex = std::clamp(
                        static_cast<int>(std::lround(
                            (z - config.boundsMin[2]) / stepZ)),
                        0,
                        resolution);
                    return field[static_cast<std::size_t>(zIndex) * layerSize +
                                 static_cast<std::size_t>(yIndex) * gridSize +
                                 xIndex];
                });

            mesh->vertices.resize(static_cast<std::size_t>(vertices.rows()));
            for (plamatrix::Index row = 0; row < vertices.rows(); ++row)
            {
                MeshVertex &vertex =
                    mesh->vertices[static_cast<std::size_t>(row)];
                vertex.x = vertices(row, 0);
                vertex.y = vertices(row, 1);
                vertex.z = vertices(row, 2);
            }
            mesh->faces.resize(static_cast<std::size_t>(faces.rows()));
            for (plamatrix::Index row = 0; row < faces.rows(); ++row)
            {
                Triangle &face =
                    mesh->faces[static_cast<std::size_t>(row)];
                face.v[0] = static_cast<int>(std::lround(faces(row, 0)));
                face.v[1] = static_cast<int>(std::lround(faces(row, 1)));
                face.v[2] = static_cast<int>(std::lround(faces(row, 2)));
            }
        }
    }
    catch (const std::exception &exception)
    {
        if (errorMessage)
        {
            *errorMessage = topology_extraction_error.empty()
                ? exception.what()
                : topology_extraction_error +
                      "; fallback extraction failed: " + exception.what();
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
    detail::removeDuplicateFaces(mesh);
    detail::removeNonManifoldFaces(mesh);
    detail::compactReferencedVertices(mesh);
    detail::removeSmallConnectedComponents(mesh, 64);
    detail::taubinSmooth(
        mesh,
        std::clamp(config.smoothingIterations, 0, 20),
        std::clamp(config.smoothingLambda, 0.0f, 0.49f));
    recomputeNormalsAndColors(mesh, views);
    if (config.progressFn)
    {
        config.progressFn("视觉外壳重建完成", 1.0f);
    }
    return true;
}

} // namespace xjw::mesh
