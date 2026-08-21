#include "NativeMeshSimplifier.h"

#include "MeshFaceOrientation.h"
#include "MeshQuadricSimplifier.h"
#include "SurfaceReconstructorPostprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xjw::mesh
{
namespace
{

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct EdgeInfo
{
    int first = -1;
    int second = -1;
    int faceCount = 0;
    Vec3 firstNormal;
    bool sharp = false;
};

std::uint64_t edgeKey(int first, int second)
{
    const std::uint32_t low = static_cast<std::uint32_t>(
        std::min(first, second));
    const std::uint32_t high = static_cast<std::uint32_t>(
        std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

Vec3 positionOf(const MeshVertex &vertex)
{
    return {vertex.x, vertex.y, vertex.z};
}

Vec3 add(const Vec3 &left, const Vec3 &right)
{
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 subtract(const Vec3 &left, const Vec3 &right)
{
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 multiply(const Vec3 &value, double scale)
{
    return {value.x * scale, value.y * scale, value.z * scale};
}

double dot(const Vec3 &left, const Vec3 &right)
{
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

Vec3 cross(const Vec3 &left, const Vec3 &right)
{
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

double length(const Vec3 &value)
{
    return std::sqrt(dot(value, value));
}

Vec3 normalized(const Vec3 &value)
{
    const double magnitude = length(value);
    return magnitude > 1.0e-18
        ? multiply(value, 1.0 / magnitude)
        : Vec3{};
}

Vec3 faceCross(const std::vector<Vec3> &positions, const Triangle &face)
{
    return cross(
        subtract(positions[static_cast<std::size_t>(face.v[1])],
                 positions[static_cast<std::size_t>(face.v[0])]),
        subtract(positions[static_cast<std::size_t>(face.v[2])],
                 positions[static_cast<std::size_t>(face.v[0])]));
}

bool isCancelled(const NativeMeshSimplifyOptions &options)
{
    return options.isCancelled && options.isCancelled();
}

void buildSmoothingTopology(
    const TriMesh &mesh,
    const std::vector<Vec3> &positions,
    float featureAngleDegrees,
    std::vector<std::vector<int>> *neighbors,
    std::vector<std::uint8_t> *features)
{
    neighbors->assign(mesh.vertices.size(), {});
    features->assign(mesh.vertices.size(), 0);
    std::unordered_map<std::uint64_t, EdgeInfo> edges;
    edges.reserve(mesh.faces.size() * 2);
    const double feature_cosine = std::cos(
        std::clamp(featureAngleDegrees, 0.0f, 180.0f) *
        0.01745329251994329577);
    for (const Triangle &face : mesh.faces)
    {
        const Vec3 normal = normalized(faceCross(positions, face));
        for (int corner = 0; corner < 3; ++corner)
        {
            const int first = face.v[corner];
            const int second = face.v[(corner + 1) % 3];
            (*neighbors)[static_cast<std::size_t>(first)].push_back(second);
            (*neighbors)[static_cast<std::size_t>(second)].push_back(first);
            EdgeInfo &edge = edges[edgeKey(first, second)];
            if (edge.faceCount == 0)
            {
                edge.first = std::min(first, second);
                edge.second = std::max(first, second);
                edge.firstNormal = normal;
            }
            else if (dot(edge.firstNormal, normal) < feature_cosine)
            {
                edge.sharp = true;
            }
            ++edge.faceCount;
        }
    }
    for (std::vector<int> &list : *neighbors)
    {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }
    for (const auto &entry : edges)
    {
        const EdgeInfo &edge = entry.second;
        if (edge.faceCount != 2 || edge.sharp)
        {
            (*features)[static_cast<std::size_t>(edge.first)] = 1;
            (*features)[static_cast<std::size_t>(edge.second)] = 1;
        }
    }
}

bool smoothMeshBounded(
    TriMesh *mesh,
    const NativeMeshSimplifyOptions &options)
{
    if (mesh == nullptr || options.smoothingIterations <= 0 ||
        !(options.smoothingMaximumDisplacement > 0.0f))
    {
        return false;
    }

    std::vector<Vec3> anchors;
    anchors.reserve(mesh->vertices.size());
    for (const MeshVertex &vertex : mesh->vertices)
    {
        anchors.push_back(positionOf(vertex));
    }
    std::vector<Vec3> current = anchors;
    std::vector<std::vector<int>> neighbors;
    std::vector<std::uint8_t> features;
    buildSmoothingTopology(
        *mesh,
        current,
        options.smoothingFeatureAngleDegrees,
        &neighbors,
        &features);

    const double maximum_displacement =
        static_cast<double>(options.smoothingMaximumDisplacement);
    bool changed = false;
    for (int iteration = 0; iteration < options.smoothingIterations; ++iteration)
    {
        if (isCancelled(options))
        {
            return false;
        }

        std::vector<Vec3> normals(current.size());
        for (const Triangle &face : mesh->faces)
        {
            const Vec3 area_normal = faceCross(current, face);
            for (int corner = 0; corner < 3; ++corner)
            {
                const std::size_t index =
                    static_cast<std::size_t>(face.v[corner]);
                normals[index] = add(normals[index], area_normal);
            }
        }
        for (Vec3 &normal : normals)
        {
            normal = normalized(normal);
        }

        std::vector<Vec3> proposed = current;
        for (std::size_t index = 0; index < current.size(); ++index)
        {
            if (features[index] || neighbors[index].empty())
            {
                continue;
            }
            Vec3 centroid;
            for (const int neighbor : neighbors[index])
            {
                centroid = add(
                    centroid,
                    current[static_cast<std::size_t>(neighbor)]);
            }
            centroid = multiply(
                centroid,
                1.0 / static_cast<double>(neighbors[index].size()));
            Vec3 displacement = subtract(centroid, current[index]);
            displacement = subtract(
                displacement,
                multiply(normals[index], dot(displacement, normals[index])));
            Vec3 candidate = add(current[index], displacement);
            const Vec3 from_anchor = subtract(candidate, anchors[index]);
            const double distance = length(from_anchor);
            if (distance > maximum_displacement)
            {
                candidate = add(
                    anchors[index],
                    multiply(from_anchor, maximum_displacement / distance));
            }
            proposed[index] = candidate;
        }

        std::vector<std::uint8_t> rejected(current.size(), 0);
        for (const Triangle &face : mesh->faces)
        {
            const Vec3 previous_cross = faceCross(current, face);
            const Vec3 proposed_cross = faceCross(proposed, face);
            if (length(previous_cross) <= 1.0e-18 ||
                length(proposed_cross) <= 1.0e-18 ||
                dot(previous_cross, proposed_cross) <= 0.0)
            {
                for (int corner = 0; corner < 3; ++corner)
                {
                    rejected[static_cast<std::size_t>(face.v[corner])] = 1;
                }
            }
        }
        for (std::size_t index = 0; index < proposed.size(); ++index)
        {
            if (rejected[index])
            {
                proposed[index] = current[index];
            }
            else if (length(subtract(proposed[index], current[index])) > 1.0e-12)
            {
                changed = true;
            }
        }
        current = std::move(proposed);
    }

    if (!changed)
    {
        return false;
    }
    for (std::size_t index = 0; index < current.size(); ++index)
    {
        MeshVertex &vertex = mesh->vertices[index];
        vertex.x = static_cast<float>(current[index].x);
        vertex.y = static_cast<float>(current[index].y);
        vertex.z = static_cast<float>(current[index].z);
    }
    detail::recomputeNormals(mesh);
    return true;
}

} // namespace

NativeMeshSimplifyStatistics simplifyMeshTopologySafe(
    TriMesh *mesh,
    const NativeMeshSimplifyOptions &options)
{
    NativeMeshSimplifyStatistics statistics;
    if (mesh == nullptr)
    {
        statistics.error = "mesh is null";
        return statistics;
    }
    statistics.inputVertexCount = mesh->vertexCount();
    statistics.inputFaceCount = mesh->faceCount();
    statistics.outputVertexCount = statistics.inputVertexCount;
    statistics.outputFaceCount = statistics.inputFaceCount;
    if (mesh->empty())
    {
        statistics.error = "mesh is empty";
        return statistics;
    }
    if (options.targetFaceCount <= 0 ||
        options.targetFaceCount >= mesh->faceCount())
    {
        statistics.error =
            "target face count must be positive and smaller than input";
        return statistics;
    }
    if (isCancelled(options))
    {
        statistics.cancelled = true;
        statistics.error = "simplification cancelled before initialization";
        return statistics;
    }

    TriMesh candidate = *mesh;
    const MeshFaceOrientationStatistics orientation =
        repairMeshFaceOrientation(&candidate);
    statistics.inconsistentSharedEdgeCountBefore =
        orientation.inconsistentSharedEdgeCountBefore;
    statistics.reorientedInputFaceCount = orientation.flippedFaceCount;
    statistics.removedContradictoryFaceCount =
        orientation.removedContradictoryFaceCount;
    statistics.orientationConflictCount = orientation.orientationConflictCount;
    if (!orientation.succeeded)
    {
        statistics.error =
            "mesh face orientation is non-manifold or contradictory";
        return statistics;
    }

    QuadricSimplifyOptions quadric_options;
    quadric_options.targetFaceCount = options.targetFaceCount;
    quadric_options.maximumPasses = options.maximumPasses;
    quadric_options.workerCount = options.workerCount;
    quadric_options.minimumFaceArea = options.minimumFaceArea;
    quadric_options.maximumResultFaceAspectRatio =
        options.maximumResultFaceAspectRatio;
    quadric_options.featureAngleDegrees = options.featureAngleDegrees;
    quadric_options.maximumNormalDeviationDegrees = std::min(
        options.maximumNormalDeviationDegrees,
        options.maximumNormalFlippingDegrees);
    quadric_options.isCancelled = options.isCancelled;
    quadric_options.progress = options.progress;

    statistics.initialized = true;
    const QuadricSimplifyStatistics quadric =
        simplifyMeshQuadric(&candidate, quadric_options);
    statistics.cancelled = quadric.cancelled;
    statistics.reachedTarget = quadric.reachedTarget;
    statistics.collapsedEdgeCount = quadric.collapsedEdgeCount;
    statistics.rejectedBoundaryEdgeCount =
        quadric.rejectedBoundaryEdgeCount;
    statistics.rejectedFeatureEdgeCount =
        quadric.rejectedFeatureEdgeCount;
    statistics.rejectedTopologyEdgeCount =
        quadric.rejectedTopologyEdgeCount;
    statistics.rejectedFlipEdgeCount = quadric.rejectedFlipEdgeCount;
    statistics.rejectedTriangleQualityEdgeCount =
        quadric.rejectedTriangleQualityEdgeCount;
    statistics.passCount = quadric.passCount;
    if (statistics.cancelled || isCancelled(options))
    {
        statistics.cancelled = true;
        statistics.error = "native mesh simplification cancelled";
        return statistics;
    }

    statistics.smoothingApplied = smoothMeshBounded(&candidate, options);
    if (isCancelled(options))
    {
        statistics.cancelled = true;
        statistics.error = "native mesh smoothing cancelled";
        return statistics;
    }
    statistics.outputVertexCount = candidate.vertexCount();
    statistics.outputFaceCount = candidate.faceCount();
    statistics.reachedTarget =
        candidate.faceCount() <= options.targetFaceCount;
    statistics.succeeded = !candidate.empty() &&
        candidate.faceCount() < statistics.inputFaceCount;
    if (!statistics.succeeded)
    {
        statistics.error =
            "native mesh simplifier could not perform a legal edge collapse";
        return statistics;
    }

    *mesh = std::move(candidate);
    return statistics;
}

} // namespace xjw::mesh
